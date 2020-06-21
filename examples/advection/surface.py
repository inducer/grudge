from __future__ import division, absolute_import

__copyright__ = "Copyright (C) 2020 Alexandru Fikl"

__license__ = """
This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

import os
import numpy as np
import numpy.linalg as la

import pyopencl as cl
import pyopencl.array
import pyopencl.clmath

from grudge import bind, sym
from pytools.obj_array import make_obj_array

import logging
logger = logging.getLogger(__name__)


# {{{ plotting (keep in sync with `var-velocity.py`)

class Plotter:
    def __init__(self, queue, discr, order, visualize=True):
        self.queue = queue
        self.ambient_dim = discr.ambient_dim
        self.dim = discr.dim

        self.visualize = visualize
        if not self.visualize:
            return

        if self.ambient_dim == 2:
            import matplotlib.pyplot as pt
            self.fig = pt.figure(figsize=(8, 8), dpi=300)

            volume_discr = discr.discr_from_dd(sym.DD_VOLUME)
            x = volume_discr.nodes().with_queue(queue)
            self.x = (2.0 * np.pi * (x[1] < 0) + cl.clmath.atan2(x[1], x[0])).get(queue)
        elif self.ambient_dim == 3:
            from grudge.shortcuts import make_visualizer
            self.vis = make_visualizer(discr, vis_order=order)
        else:
            raise ValueError("unsupported dimension")

    def __call__(self, evt, basename, overwrite=True):
        if not self.visualize:
            return

        if self.ambient_dim == 2:
            u = evt.state_component.get(self.queue)

            filename = "%s.png" % basename
            if not overwrite and os.path.exists(filename):
                from meshmode import FileExistsError
                raise FileExistsError("output file '%s' already exists" % filename)

            ax = self.fig.gca()
            ax.grid()
            ax.plot(self.x, u, '-')
            ax.plot(self.x, u, 'k.')
            ax.set_xlabel(r"$\theta$")
            ax.set_ylabel("$u$")
            ax.set_title("t = {:.2f}".format(evt.t))

            self.fig.savefig(filename)
            self.fig.clf()
        elif self.ambient_dim == 3:
            self.vis.write_vtk_file("%s.vtu" % basename, [
                ("u", evt.state_component)
                ], overwrite=overwrite)
        else:
            raise ValueError("unsupported dimension")

# }}}


def main(ctx_factory, dim=2, order=4, product_tag=None, visualize=True):
    cl_ctx = ctx_factory()
    queue = cl.CommandQueue(cl_ctx)

    # {{{ parameters

    # sphere radius
    radius = 1.0
    # sphere resolution
    resolution = 64 if dim == 2 else 1

    # cfl
    dt_factor = 1.0
    # final time
    final_time = 1.0

    # velocity field
    sym_x = sym.nodes(dim)
    c = make_obj_array([
        -sym_x[1], sym_x[0], 0.0
        ])[:dim]
    norm_c = sym.sqrt((c**2).sum())
    # flux
    flux_type = "central"

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
        return generate_icosphere(radius, order=order,
                uniform_refinement_rounds=resolution)
    else:
        raise ValueError("unsupported dimension")

    quad_tag_to_group_factory = {}
    if product_tag == "none":
        product_tag = None

    if product_tag:
        from meshmode.discretization.poly_element import \
                QuadratureSimplexGroupFactory
        quad_tag_to_group_factory = {
                product_tag: QuadratureSimplexGroupFactory(order=4*order)
                }

    from grudge import DGDiscretizationWithBoundaries
    discr = DGDiscretizationWithBoundaries(cl_ctx, mesh, order=order,
            quad_tag_to_group_factory=quad_tag_to_group_factory)

    # }}}

    # {{{ symbolic operators

    def f_gaussian(x):
        return x[0]

    from grudge.models.advection import SurfaceAdvectionOperator
    op = SurfaceAdvectionOperator(c,
        flux_type=flux_type,
        quad_tag=product_tag)

    bound_op = bind(discr, op.sym_operator())
    u = bind(discr, f_gaussian(sym_x))(queue, t=0)

    def rhs(t, u):
        return bound_op(queue, t=t, u=u)

    # }}}

    # {{{ time stepping

    # compute time step
    h_min = bind(discr,
            sym.h_max_from_volume(discr.ambient_dim, dim=discr.dim))(queue)
    dt = dt_factor * h_min/order**2
    nsteps = int(final_time // dt) + 1
    dt = final_time/nsteps + 1.0e-15

    from grudge.shortcuts import set_up_rk4
    dt_stepper = set_up_rk4("u", dt, u, rhs)
    plot = Plotter(queue, discr, order, visualize=visualize)

    norm = bind(discr, sym.norm(2, sym.var("u")))

    step = 0
    norm_u = 0.0
    for event in dt_stepper.run(t_end=final_time):
        if not isinstance(event, dt_stepper.StateComputed):
            continue

        if step % 1 == 0:
            norm_u = norm(queue, u=event.state_component)
            plot(event, "fld-surface-%04d" % step)

        step += 1
        logger.info("[%04d] t = %.5f |u| = %.5e", step, event.t, norm_u)

    # }}}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dim", default=2, type=int)
    parser.add_argument("--qtag", default="none")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    main(cl.create_some_context,
            dim=args.dim,
            product_tag=args.qtag)
