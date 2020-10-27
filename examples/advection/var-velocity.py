__copyright__ = "Copyright (C) 2017 Bogdan Enache"

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
from pytools.obj_array import flat_obj_array

import logging
logger = logging.getLogger(__name__)


# {{{ plotting (keep in sync with `weak.py`)

class Plotter:
    def __init__(self, actx, discr, order, visualize=True, ylim=None):
        self.actx = actx
        self.dim = discr.ambient_dim

        self.visualize = visualize
        if not self.visualize:
            return

        if self.dim == 1:
            import matplotlib.pyplot as pt
            self.fig = pt.figure(figsize=(8, 8), dpi=300)
            self.ylim = ylim

            volume_discr = discr.discr_from_dd(sym.DD_VOLUME)
            self.x = actx.to_numpy(flatten(thaw(actx, volume_discr.nodes()[0])))
        else:
            from grudge.shortcuts import make_visualizer
            self.vis = make_visualizer(discr, vis_order=order)

    def __call__(self, evt, basename, overwrite=True):
        if not self.visualize:
            return

        if self.dim == 1:
            u = self.actx.to_numpy(flatten(evt.state_component))

            filename = "%s.png" % basename
            if not overwrite and os.path.exists(filename):
                from meshmode import FileExistsError
                raise FileExistsError("output file '%s' already exists" % filename)

            ax = self.fig.gca()
            ax.plot(self.x, u, "-")
            ax.plot(self.x, u, "k.")
            if self.ylim is not None:
                ax.set_ylim(self.ylim)

            ax.set_xlabel("$x$")
            ax.set_ylabel("$u$")
            ax.set_title(f"t = {evt.t:.2f}")
            self.fig.savefig(filename)
            self.fig.clf()
        else:
            self.vis.write_vtk_file("%s.vtu" % basename, [
                ("u", evt.state_component)
                ], overwrite=overwrite)

# }}}


def main(ctx_factory, dim=2, order=4, product_tag=None, visualize=False):
    cl_ctx = ctx_factory()
    queue = cl.CommandQueue(cl_ctx)
    actx = PyOpenCLArrayContext(queue)

    # {{{ parameters

    # domain [0, d]^dim
    d = 1.0
    # number of points in each dimension
    npoints = 25
    # grid spacing
    h = d / npoints

    # cfl
    dt_factor = 1.0
    # finale time
    final_time = 0.5
    # time steps
    dt = dt_factor * h/order**2
    nsteps = int(final_time // dt) + 1
    dt = final_time/nsteps + 1.0e-15

    # flux
    flux_type = "upwind"
    # velocity field
    sym_x = sym.nodes(dim)
    if dim == 1:
        c = sym_x
    else:
        # solid body rotation
        c = flat_obj_array(
                np.pi * (d/2 - sym_x[1]),
                np.pi * (sym_x[0] - d/2),
                0)[:dim]

    # }}}

    # {{{ discretization

    from meshmode.mesh.generation import generate_regular_rect_mesh
    mesh = generate_regular_rect_mesh(
            a=(0,)*dim, b=(d,)*dim,
            n=(npoints,)*dim,
            order=order)

    from meshmode.discretization.poly_element import \
            QuadratureSimplexGroupFactory

    if product_tag:
        quad_tag_to_group_factory = {
                product_tag: QuadratureSimplexGroupFactory(order=4*order)
                }
    else:
        quad_tag_to_group_factory = {}

    from grudge import DGDiscretizationWithBoundaries
    discr = DGDiscretizationWithBoundaries(actx, mesh, order=order,
            quad_tag_to_group_factory=quad_tag_to_group_factory)

    # }}}

    # {{{ symbolic operators

    # gaussian parameters
    source_center = np.array([0.5, 0.75, 0.0])[:dim]
    source_width = 0.05
    dist_squared = np.dot(sym_x - source_center, sym_x - source_center)

    def f_gaussian(x):
        return sym.exp(-dist_squared / source_width**2)

    def f_step(x):
        return sym.If(sym.Comparison(
            dist_squared, "<", (4*source_width) ** 2),
            1, 0)

    def u_bc(x):
        return 0.0

    from grudge.models.advection import VariableCoefficientAdvectionOperator
    op = VariableCoefficientAdvectionOperator(
            c,
            u_bc(sym.nodes(dim, sym.BTAG_ALL)),
            quad_tag=product_tag,
            flux_type=flux_type)

    bound_op = bind(discr, op.sym_operator())
    u = bind(discr, f_gaussian(sym.nodes(dim)))(actx, t=0)

    def rhs(t, u):
        return bound_op(t=t, u=u)

    # }}}

    # {{{ time stepping

    from grudge.shortcuts import set_up_rk4
    dt_stepper = set_up_rk4("u", dt, u, rhs)
    plot = Plotter(actx, discr, order, visualize=visualize,
            ylim=[-0.1, 1.1])

    step = 0
    norm = bind(discr, sym.norm(2, sym.var("u")))
    for event in dt_stepper.run(t_end=final_time):
        if not isinstance(event, dt_stepper.StateComputed):
            continue

        if step % 10 == 0:
            norm_u = norm(u=event.state_component)
            plot(event, "fld-var-velocity-%04d" % step)

        step += 1
        logger.info("[%04d] t = %.5f |u| = %.5e", step, event.t, norm_u)

    # }}}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dim", default=2, type=int)
    parser.add_argument("--qtag", default="product")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    main(cl.create_some_context,
            dim=args.dim,
            product_tag=args.qtag)
