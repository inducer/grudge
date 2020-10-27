__copyright__ = "Copyright (C) 2020 Alexandru Fikl"

__license__ = """
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""

import os

import numpy as np
import pyopencl as cl

from meshmode.array_context import PyOpenCLArrayContext
from meshmode.dof_array import thaw, flatten

from grudge import bind, sym
from pytools.obj_array import make_obj_array

import logging
logger = logging.getLogger(__name__)


# {{{ plotting (keep in sync with `var-velocity.py`)

class Plotter:
    def __init__(self, actx, discr, order, visualize=True):
        self.actx = actx
        self.ambient_dim = discr.ambient_dim
        self.dim = discr.dim

        self.visualize = visualize
        if not self.visualize:
            return

        if self.ambient_dim == 2:
            import matplotlib.pyplot as pt
            self.fig = pt.figure(figsize=(8, 8), dpi=300)

            x = thaw(actx, discr.discr_from_dd(sym.DD_VOLUME).nodes())
            self.x = actx.to_numpy(flatten(actx.np.atan2(x[1], x[0])))
        elif self.ambient_dim == 3:
            from grudge.shortcuts import make_visualizer
            self.vis = make_visualizer(discr, vis_order=order)
        else:
            raise ValueError("unsupported dimension")

    def __call__(self, evt, basename, overwrite=True):
        if not self.visualize:
            return

        if self.ambient_dim == 2:
            u = self.actx.to_numpy(flatten(evt.state_component))

            filename = "%s.png" % basename
            if not overwrite and os.path.exists(filename):
                from meshmode import FileExistsError
                raise FileExistsError("output file '%s' already exists" % filename)

            ax = self.fig.gca()
            ax.grid()
            ax.plot(self.x, u, "-")
            ax.plot(self.x, u, "k.")
            ax.set_xlabel(r"$\theta$")
            ax.set_ylabel("$u$")
            ax.set_title(f"t = {evt.t:.2f}")

            self.fig.savefig(filename)
            self.fig.clf()
        elif self.ambient_dim == 3:
            self.vis.write_vtk_file("%s.vtu" % basename, [
                ("u", evt.state_component)
                ], overwrite=overwrite)
        else:
            raise ValueError("unsupported dimension")

# }}}


def main(ctx_factory, dim=2, order=4, product_tag=None, visualize=False):
    cl_ctx = ctx_factory()
    queue = cl.CommandQueue(cl_ctx)
    actx = PyOpenCLArrayContext(queue)

    # {{{ parameters

    # sphere radius
    radius = 1.0
    # sphere resolution
    resolution = 64 if dim == 2 else 1

    # cfl
    dt_factor = 2.0
    # final time
    final_time = np.pi

    # velocity field
    sym_x = sym.nodes(dim)
    c = make_obj_array([
        -sym_x[1], sym_x[0], 0.0
        ])[:dim]
    # flux
    flux_type = "lf"

    # }}}

    # {{{ discretization

    if dim == 2:
        from meshmode.mesh.generation import make_curve_mesh, ellipse
        mesh = make_curve_mesh(
                lambda t: radius * ellipse(1.0, t),
                np.linspace(0.0, 1.0, resolution + 1),
                order)
    elif dim == 3:
        from meshmode.mesh.generation import generate_icosphere
        mesh = generate_icosphere(radius, order=4 * order,
                uniform_refinement_rounds=resolution)
    else:
        raise ValueError("unsupported dimension")

    quad_tag_to_group_factory = {}
    if product_tag == "none":
        product_tag = None

    from meshmode.discretization.poly_element import \
            PolynomialWarpAndBlendGroupFactory, \
            QuadratureSimplexGroupFactory

    quad_tag_to_group_factory[sym.QTAG_NONE] = \
            PolynomialWarpAndBlendGroupFactory(order)

    if product_tag:
        quad_tag_to_group_factory[product_tag] = \
                QuadratureSimplexGroupFactory(order=4*order)

    from grudge import DGDiscretizationWithBoundaries
    discr = DGDiscretizationWithBoundaries(actx, mesh,
            quad_tag_to_group_factory=quad_tag_to_group_factory)

    volume_discr = discr.discr_from_dd(sym.DD_VOLUME)
    logger.info("ndofs:     %d", volume_discr.ndofs)
    logger.info("nelements: %d", volume_discr.mesh.nelements)

    # }}}

    # {{{ symbolic operators

    def f_initial_condition(x):
        return x[0]

    from grudge.models.advection import SurfaceAdvectionOperator
    op = SurfaceAdvectionOperator(c,
        flux_type=flux_type,
        quad_tag=product_tag)

    bound_op = bind(discr, op.sym_operator())
    u0 = bind(discr, f_initial_condition(sym_x))(actx, t=0)

    def rhs(t, u):
        return bound_op(actx, t=t, u=u)

    # check velocity is tangential
    sym_normal = sym.surface_normal(dim, dim=dim - 1, dd=sym.DD_VOLUME).as_vector()
    error = bind(discr, sym.norm(2, c.dot(sym_normal)))(actx)
    logger.info("u_dot_n:   %.5e", error)

    # }}}

    # {{{ time stepping

    # compute time step
    h_min = bind(discr,
            sym.h_max_from_volume(discr.ambient_dim, dim=discr.dim))(actx)
    dt = dt_factor * h_min/order**2
    nsteps = int(final_time // dt) + 1
    dt = final_time/nsteps + 1.0e-15

    logger.info("dt:        %.5e", dt)
    logger.info("nsteps:    %d", nsteps)

    from grudge.shortcuts import set_up_rk4
    dt_stepper = set_up_rk4("u", dt, u0, rhs)
    plot = Plotter(actx, discr, order, visualize=visualize)

    norm = bind(discr, sym.norm(2, sym.var("u")))
    norm_u = norm(actx, u=u0)

    step = 0

    event = dt_stepper.StateComputed(0.0, 0, 0, u0)
    plot(event, "fld-surface-%04d" % 0)

    if visualize and dim == 3:
        from grudge.shortcuts import make_visualizer
        vis = make_visualizer(discr, vis_order=order)
        vis.write_vtk_file("fld-surface-velocity.vtu", [
            ("u", bind(discr, c)(actx)),
            ("n", bind(discr, sym_normal)(actx))
            ], overwrite=True)

        df = sym.DOFDesc(sym.FACE_RESTR_INTERIOR)
        face_discr = discr.connection_from_dds(sym.DD_VOLUME, df).to_discr

        face_normal = bind(discr, sym.normal(
            df, face_discr.ambient_dim, dim=face_discr.dim))(actx)

        from meshmode.discretization.visualization import make_visualizer
        vis = make_visualizer(actx, face_discr, vis_order=order)
        vis.write_vtk_file("fld-surface-face-normals.vtu", [
            ("n", face_normal)
            ], overwrite=True)

    for event in dt_stepper.run(t_end=final_time, max_steps=nsteps + 1):
        if not isinstance(event, dt_stepper.StateComputed):
            continue

        step += 1
        if step % 10 == 0:
            norm_u = norm(actx, u=event.state_component)
            plot(event, "fld-surface-%04d" % step)

        logger.info("[%04d] t = %.5f |u| = %.5e", step, event.t, norm_u)

    plot(event, "fld-surface-%04d" % step)

    # }}}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dim", choices=[2, 3], default=2, type=int)
    parser.add_argument("--qtag", choices=["none", "product"], default="none")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    main(cl.create_some_context,
            dim=args.dim,
            product_tag=args.qtag)
