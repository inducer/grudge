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

        assert self.dim == 1

        state = evt.state_component
        density = self.actx.to_numpy(flatten(state[0]))
        energy = self.actx.to_numpy(flatten(state[1]))
        momentum = self.actx.to_numpy(flatten(state[2]))

        filename = "%s.png" % basename
        if not overwrite and os.path.exists(filename):
            from meshmode import FileExistsError
            raise FileExistsError("output file '%s' already exists" % filename)

        ax = self.fig.gca()
        ax.plot(self.x, density, "-")
        ax.plot(self.x, density, "k.")
        if self.ylim is not None:
            ax.set_ylim(self.ylim)

        ax.set_xlabel("$x$")
        ax.set_ylabel("$\\rho$")
        ax.set_title(f"t = {evt.t:.2f}")

        self.fig.savefig(filename)
        self.fig.clf()

# }}}


def main(ctx_factory, dim=2, order=4, visualize=False):
    cl_ctx = ctx_factory()
    queue = cl.CommandQueue(cl_ctx)
    actx = PyOpenCLArrayContext(
        queue,
        allocator=cl_tools.MemoryPool(cl_tools.ImmediateAllocator(queue)),
        force_device_scalars=True,
    )

    # {{{ parameters

    # number of points in each dimension
    nel_1d = 16

    # final time
    final_time = 0.2

    # flux
    flux_type = "lf"

    # eos-related parameters
    gamma = 1.4
    gas_const = 287.1

    # }}}

    # {{{ discretization

    from meshmode.mesh.generation import generate_regular_rect_mesh

    box_ll = -5.0
    box_ur = 5.0
    mesh = generate_regular_rect_mesh(
        a=(box_ll,)*dim,
        b=(box_ur,)*dim,
        nelements_per_axis=(nel_1d,)*dim)

    from grudge import DiscretizationCollection

    dcoll = DiscretizationCollection(actx, mesh, order=order)

    # }}}

    # {{{ Euler operator

    def vortex_initial_condition(x_vec, t=0):
        _beta = 5
        _center = np.zeros(shape=(dim,))
        _velocity = np.zeros(shape=(dim,))

        vortex_loc = _center + t * _velocity

        # Coordinates relative to vortex center
        x_rel = x_vec[0] - vortex_loc[0]
        y_rel = x_vec[1] - vortex_loc[1]
        actx = x_vec[0].array_context
 
        r = actx.np.sqrt(x_rel ** 2 + y_rel ** 2)
        expterm = _beta * actx.np.exp(1 - r ** 2)
        u = _velocity[0] - expterm * y_rel / (2 * np.pi)
        v = _velocity[1] + expterm * x_rel / (2 * np.pi)
        velocity = make_obj_array([u, v])
        mass = (1 - (gamma - 1) / (16 * gamma * np.pi ** 2)
                * expterm ** 2) ** (1 / (gamma - 1))
        momentum = mass * velocity
        p = mass ** gamma

        energy = p / (gamma - 1) + mass / 2 * (u ** 2 + v ** 2)

        from grudge.models.euler import EulerState

        return EulerState(density=mass,
                          total_energy=energy,
                          momentum=momentum)

    from grudge.models.euler import EulerOperator

    euler_operator = EulerOperator(
        dcoll,
        bdry_fcts={BTAG_ALL: vortex_initial_condition},
        flux_type=flux_type,
        gamma=gamma,
        gas_const=gas_const,
    )

    def rhs(t, q):
        return euler_operator.operator(t, q)

    q_init = vortex_initial_condition(thaw(dcoll.nodes(), actx))
    dt = 1/10 * euler_operator.estimate_rk4_timestep(actx, dcoll, state=q_init)

    logger.info("Timestep size: %g", dt)

    # }}}

    from grudge.shortcuts import make_visualizer

    step = 0
    vis = make_visualizer(dcoll)

    if visualize:
        vis.write_vtk_file(
            f"fld-vortex-{step:04d}.vtu",
            [
                ("rho", q_init.density),
                ("energy", q_init.total_energy),
                ("momentum", q_init.momentum),
            ]
        )

    # {{{ time stepping

    from grudge.shortcuts import set_up_rk4
    dt_stepper = set_up_rk4("q", dt, q_init.join(), rhs)

    norm_q = 0.0
    for event in dt_stepper.run(t_end=final_time):
        if not isinstance(event, dt_stepper.StateComputed):
            continue

        step += 1

        if step % 1 == 0:
            state = event.state_component
            import ipdb; ipdb.set_trace()
            if visualize:
                vis.write_vtk_file(
                    f"fld-vortex-{step:04d}.vtu",
                    [
                        ("rho", state[0]),
                        ("energy", state[1]),
                        ("momentum", state[2:]),
                    ]
                )

        norm_q = actx.to_numpy(op.norm(dcoll, event.state_component, 2))
        logger.info("[%04d] t = %.5f |q| = %.5e", step, event.t, norm_q)

        assert norm_q < 100

    # }}}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--order", default=4, type=int)
    parser.add_argument("--visualize", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    main(cl.create_some_context,
         order=args.order,
         visualize=args.visualize)
