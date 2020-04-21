# Copyright (C) 2007 Andreas Kloeckner
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import numpy as np
import numpy.linalg as la

import pyopencl as cl
import pyopencl.array
import pyopencl.clmath

from grudge import bind, sym

import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def main(ctx_factory, dim=1, order=4, visualize=True):
    cl_ctx = ctx_factory()
    queue = cl.CommandQueue(cl_ctx)

    # {{{ parameters

    # domain side [-d/2, d/2]^dim
    d = 1.0
    # number of points in each dimension
    npoints = 20
    # grid spacing
    h = d / npoints
    # cfl?
    dt_factor = 4
    # final time
    final_time = 1.0

    # velocity field
    c = np.array([0.1] * dim)
    norm_c = la.norm(c)
    # flux
    flux_type = "central"

    # compute number of steps
    dt = dt_factor * h / order**2
    nsteps = int(final_time // dt) + 1
    dt = final_time/nsteps + 1.0e-15

    # }}}

    # {{{ discretization

    from meshmode.mesh.generation import generate_box_mesh
    mesh = generate_box_mesh(
            [np.linspace(-d/2, d/2, npoints) for _ in range(dim)],
            order=order)

    from grudge import DGDiscretizationWithBoundaries
    discr = DGDiscretizationWithBoundaries(cl_ctx, mesh, order=order)

    # }}}

    # {{{ solve advection

    def f(x):
        return sym.sin(3 * x)

    def u_analytic(x):
        t = sym.var("t", sym.DD_SCALAR)
        return f(-np.dot(c, x) / norm_c + t * norm_c)

    from grudge.models.advection import WeakAdvectionOperator
    op = WeakAdvectionOperator(c,
        inflow_u=u_analytic(sym.nodes(dim, sym.BTAG_ALL)),
        flux_type=flux_type)

    bound_op = bind(discr, op.sym_operator())
    u = bind(discr, u_analytic(sym.nodes(dim)))(queue, t=0)

    return

    def rhs(t, u):
        return bound_op(queue, t=t, u=u)

    from grudge.shortcuts import set_up_rk4
    dt_stepper = set_up_rk4("u", dt, u, rhs)

    if dim == 1:
        import matplotlib.pyplot as pt
        pt.figure(figsize=(8, 8), dpi=300)
    else:
        from grudge.shortcuts import make_visualizer
        vis = make_visualizer(discr, vis_order=order)

    step = 0
    norm = bind(discr, sym.norm(2, sym.var("u")))
    for event in dt_stepper.run(t_end=final_time):
        if not isinstance(event, dt_stepper.StateComputed):
            continue

        step += 1
        norm_u = norm(queue, u=event.state_component)
        logger.info("[%04d] t = %.5f |u| = %.5e", step, event.t, norm_u)

        if dim == 1:
            x = discr.discr_from_dd(sym.DTAG_VOLUME_ALL).nodes().get(queue)
            u = event.state_component.get(queue)

            pt.plot(x, u)
            pt.xlabel("$x$")
            pt.ylabel("$u$")
            pt.title("t = {:.2f}".format(event.t))
            pt.savefig("fld-weak-%04d.png" % step)
            pt.clf()
        else:
            vis.write_vtk_file("fld-weak-%04d.vtu" % step, [
                ("u", event.state_component)
                ], overwrite=True)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dim", default=2, type=int)
    args = parser.parse_args()

    main(cl.create_some_context,
            dim=args.dim)
