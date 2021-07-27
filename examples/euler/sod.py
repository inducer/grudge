"""Minimal example of a grudge driver."""

__copyright__ = """
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

import os
import numpy as np

import pyopencl as cl
import pyopencl.tools as cl_tools

from arraycontext import thaw
from grudge.array_context import PyOpenCLArrayContext

from meshmode.dof_array import flatten
from meshmode.mesh import BTAG_ALL

from pytools.obj_array import make_obj_array

import grudge.dof_desc as dof_desc
import grudge.op as op

import logging
logger = logging.getLogger(__name__)


# {{{ plotting (keep in sync with `var-velocity.py`)

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

            volume_discr = dcoll.discr_from_dd(dof_desc.DD_VOLUME)
            self.x = actx.to_numpy(flatten(thaw(volume_discr.nodes()[0], actx)))
        else:
            from grudge.shortcuts import make_visualizer
            self.vis = make_visualizer(dcoll)

    def __call__(self, evt, basename, overwrite=True):
        if not self.visualize:
            return

        if self.dim == 1:
            q = self.actx.to_numpy(flatten(evt.state_component))

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


def main(ctx_factory, dim=1, order=4, visualize=False):
    cl_ctx = ctx_factory()
    queue = cl.CommandQueue(cl_ctx)
    actx = PyOpenCLArrayContext(
        queue,
        allocator=cl_tools.MemoryPool(cl_tools.ImmediateAllocator(queue)),
        force_device_scalars=True,
    )

    # {{{ parameters

    # domain [0, 1]
    d = 1.0
    # number of points in each dimension
    npoints = 100

    # final time
    final_time = 0.2

    # flux
    flux_type = "lf"

    # eos-related parameters
    gamma = 1.4
    gas_const = 287.1

    # }}}

    # {{{ discretization

    from meshmode.mesh.generation import generate_box_mesh
    mesh = generate_box_mesh(
        [np.linspace(0, d, npoints) for _ in range(dim)],
        order=order)

    from grudge import DiscretizationCollection

    dcoll = DiscretizationCollection(actx, mesh, order=order)

    # }}}

    # {{{ Euler operator

    def sod_initial_condition(nodes, t=0):
        gmn1 = 1.0 / (gamma - 1.0)
        x = nodes[0]
        zeros = 0*x

        _rhol = 0.125
        _rhor = 1.0
        _x0 = 0.5
        pleft = 0.1
        pright = 1.0

        rhor = zeros + _rhor
        rhol = zeros + _rhol
        x0 = zeros + _x0
        energyl = zeros + gmn1 * pleft
        energyr = zeros + gmn1 * pright
        yesno = actx.np.greater(x, x0)
        density = actx.np.where(yesno, rhor, rhol)
        energy = actx.np.where(yesno, energyr, energyl)
        mom = make_obj_array([zeros for i in range(dim)])

        from grudge.models.euler import EulerState

        return EulerState(density=density,
                          total_energy=energy,
                          momentum=mom)

    from grudge.models.euler import EulerOperator

    euler_operator = EulerOperator(
        dcoll,
        # NOTE: BC interface is hard-coded in the operator class for now
        bdry_fcts={BTAG_ALL: None},
        flux_type=flux_type,
        gamma=gamma,
        gas_const=gas_const,
    )

    nodes = thaw(dcoll.nodes(), actx)
    q_init = sod_initial_condition(nodes, t=0)

    def rhs(t, q):
        return euler_operator.operator(t, q)

    dt = euler_operator.estimate_rk4_timestep(actx, dcoll, state=q_init)

    logger.info("Timestep size: %g", dt)

    # }}}

    # {{{ time stepping

    from grudge.shortcuts import set_up_rk4
    dt_stepper = set_up_rk4("q", dt, q_init.join(), rhs)
    plot = Plotter(actx, dcoll, order, visualize=visualize, ylim=[0.0, 1.2])

    step = 0
    norm_q = 0.0
    for event in dt_stepper.run(t_end=final_time):
        if not isinstance(event, dt_stepper.StateComputed):
            continue

        if step % 10 == 0:
            norm_q = actx.to_numpy(op.norm(dcoll, event.state_component, 2))
            plot(event, "fld-sod-%04d" % step)

        step += 1
        logger.info("[%04d] t = %.5f |q| = %.5e", step, event.t, norm_q)

        # NOTE: These are here to ensure the solution is bounded for the
        # time interval specified
        assert norm_u < 1

    # }}}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dim", default=2, type=int)
    parser.add_argument("--order", default=4, type=int)
    parser.add_argument("--visualize", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    main(cl.create_some_context,
         dim=args.dim,
         order=args.order,
         visualize=args.visualize)
