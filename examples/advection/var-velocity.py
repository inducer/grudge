__copyright__ = """
Copyright (C) 2017 Bogdan Enache
Copyright (C) 2021 University of Illinois Board of Trustees
"""

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

import logging
import os

import numpy as np

import pyopencl as cl
import pyopencl.tools as cl_tools
from arraycontext import flatten
from meshmode.mesh import BTAG_ALL
from pytools.obj_array import flat_obj_array

import grudge.dof_desc as dof_desc
import grudge.op as op
from grudge.array_context import PyOpenCLArrayContext


logger = logging.getLogger(__name__)


# {{{ plotting (keep in sync with `weak.py`)

class Plotter:
    def __init__(self, actx, dcoll, order, visualize=True, ylim=None):
        self.actx = actx
        self.dim = dcoll.ambient_dim

        self.visualize = visualize
        if not self.visualize:
            return

        if self.dim == 1:
            import matplotlib.pyplot as pt
            self.fig = pt.figure(figsize=(8, 8), dpi=300)
            self.ylim = ylim

            volume_discr = dcoll.discr_from_dd(dof_desc.DD_VOLUME_ALL)
            self.x = actx.to_numpy(flatten(volume_discr.nodes()[0], self.actx))
        else:
            from grudge.shortcuts import make_visualizer
            self.vis = make_visualizer(dcoll)

    def __call__(self, evt, basename, overwrite=True):
        if not self.visualize:
            return

        if self.dim == 1:
            u = self.actx.to_numpy(flatten(evt.state_component, self.actx))

            filename = f"{basename}.png"
            if not overwrite and os.path.exists(filename):
                from meshmode import FileExistsError
                raise FileExistsError(f"output file '{filename}' already exists")

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
            self.vis.write_vtk_file(f"{basename}.vtu", [
                ("u", evt.state_component)
                ], overwrite=overwrite)

# }}}


def main(ctx_factory, dim=2, order=4, use_quad=False, visualize=False,
        flux_type="upwind"):
    cl_ctx = ctx_factory()
    queue = cl.CommandQueue(cl_ctx)
    actx = PyOpenCLArrayContext(
        queue,
        allocator=cl_tools.MemoryPool(cl_tools.ImmediateAllocator(queue)),
        force_device_scalars=True,
    )

    # {{{ parameters

    # domain [0, d]^dim
    d = 1.0
    # number of points in each dimension
    npoints = 25

    # final time
    final_time = 1

    if use_quad:
        qtag = dof_desc.DISCR_TAG_QUAD
    else:
        qtag = None

    # }}}

    # {{{ discretization

    from meshmode.mesh.generation import generate_regular_rect_mesh
    mesh = generate_regular_rect_mesh(
            a=(0,)*dim, b=(d,)*dim,
            npoints_per_axis=(npoints,)*dim,
            order=order)

    from meshmode.discretization.poly_element import QuadratureSimplexGroupFactory

    if use_quad:
        discr_tag_to_group_factory = {
            qtag: QuadratureSimplexGroupFactory(order=4*order)
        }
    else:
        discr_tag_to_group_factory = {}

    from grudge.discretization import make_discretization_collection

    dcoll = make_discretization_collection(
        actx, mesh, order=order,
        discr_tag_to_group_factory=discr_tag_to_group_factory
    )

    # }}}

    # {{{ advection operator

    # gaussian parameters

    def f_halfcircle(x):
        source_center = np.array([d/2, d/2, d/2])[:dim]
        dist = x - source_center
        return (
                (0.5+0.5*actx.np.tanh(500*(-np.dot(dist, dist) + 0.4**2)))
                * (0.5+0.5*actx.np.tanh(500*(dist[0]))))

    def zero_inflow_bc(dtag, t=0):
        dd = dof_desc.as_dofdesc(dtag, qtag)
        return dcoll.discr_from_dd(dd).zeros(actx)

    from grudge.models.advection import VariableCoefficientAdvectionOperator

    x = actx.thaw(dcoll.nodes())

    # velocity field
    if dim == 1:
        c = x
    else:
        # solid body rotation
        c = flat_obj_array(
            np.pi * (d/2 - x[1]),
            np.pi * (x[0] - d/2),
            0
        )[:dim]

    adv_operator = VariableCoefficientAdvectionOperator(
        dcoll,
        c,
        inflow_u=lambda t: zero_inflow_bc(BTAG_ALL, t),
        quad_tag=qtag,
        flux_type=flux_type
    )

    u = f_halfcircle(x)

    def rhs(t, u):
        return adv_operator.operator(t, u)

    dt = actx.to_numpy(adv_operator.estimate_rk4_timestep(actx, dcoll, fields=u))

    logger.info("Timestep size: %g", dt)

    # }}}

    # {{{ time stepping

    from grudge.shortcuts import set_up_rk4
    dt_stepper = set_up_rk4("u", float(dt), u, rhs)
    plot = Plotter(actx, dcoll, order, visualize=visualize,
            ylim=[-0.1, 1.1])

    step = 0
    for event in dt_stepper.run(t_end=final_time):
        if not isinstance(event, dt_stepper.StateComputed):
            continue

        if step % 10 == 0:
            norm_u = actx.to_numpy(op.norm(dcoll, event.state_component, 2))
            plot(event, f"fld-var-velocity-{step:04d}")

            logger.info("[%04d] t = %.5f |u| = %.5e", step, event.t, norm_u)
            # NOTE: These are here to ensure the solution is bounded for the
            # time interval specified
            assert norm_u < 1

        step += 1

    # }}}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dim", default=2, type=int)
    parser.add_argument("--order", default=4, type=int)
    parser.add_argument("--use-quad", action="store_true")
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--flux", default="upwind",
            help="'central' or 'upwind'. Run with central to observe aliasing "
            "instability. Add --use-quad to fix that instability.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    main(cl.create_some_context,
         dim=args.dim,
         order=args.order,
         use_quad=args.use_quad,
         visualize=args.visualize,
         flux_type=args.flux)
