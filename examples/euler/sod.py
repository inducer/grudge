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

    def __call__(self, state, basename, overwrite=True):
        if not self.visualize:
            return

        assert self.dim == 1

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
        # ax.set_title(f"t = {evt.t:.2f}")

        self.fig.savefig(filename)
        self.fig.clf()

# }}}


def main(ctx_factory, order=4, visualize=False, esdg=False):
    cl_ctx = ctx_factory()
    queue = cl.CommandQueue(cl_ctx)
    actx = PyOpenCLArrayContext(
        queue,
        allocator=cl_tools.MemoryPool(cl_tools.ImmediateAllocator(queue)),
        force_device_scalars=True,
    )

    # {{{ parameters

    dim = 1

    # domain [0, 1]
    d = 1.0
    # number of points in each dimension
    npoints = 25

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
    from grudge.dof_desc import DISCR_TAG_BASE, DISCR_TAG_QUAD
    from meshmode.discretization.poly_element import \
        (default_simplex_group_factory,
         QuadratureSimplexGroupFactory)

    dcoll = DiscretizationCollection(
        actx, mesh,
        discr_tag_to_group_factory={
            DISCR_TAG_BASE: default_simplex_group_factory(dim, order),
            DISCR_TAG_QUAD: default_simplex_group_factory(dim, order)
        }
    )

    # }}}

    # {{{ Euler operator

    def sod_initial_condition(nodes, t=0):
        gmn1 = 1.0 / (gamma - 1.0)
        x = nodes[0]
        zeros = 0*x

        _x0 = 0.5
        _rhoin = 1.0
        _rhoout = 0.125
        _pin = 1.0
        _pout = 0.1

        rhoin = zeros + _rhoin
        rhoout = zeros + _rhoout

        energyin = zeros + gmn1 * _pin
        energyout = zeros + gmn1 * _pout

        x0 = zeros + _x0
        yesno = actx.np.greater(x, x0)

        mass = actx.np.where(yesno, rhoout, rhoin)
        energy = actx.np.where(yesno, energyout, energyin)
        mom = make_obj_array([zeros for i in range(dim)])

        result = np.empty((2+dim,), dtype=object)
        result[0] = mass
        result[1] = energy
        result[2:dim+2] = mom
        return result

    nodes = thaw(dcoll.nodes(), actx)
    q_init = sod_initial_condition(nodes)

    from grudge.models.euler import \
        EntropyStableEulerOperator, EulerOperator

    if esdg:
        operator_cls = EntropyStableEulerOperator
    else:
        operator_cls = EulerOperator

    euler_operator = operator_cls(
        dcoll,
        bdry_fcts={BTAG_ALL: sod_initial_condition},
        flux_type=flux_type,
        gamma=gamma,
        gas_const=gas_const,
    )

    def rhs(t, q):
        return euler_operator.operator(t, q)

    dt = 1/4 * euler_operator.estimate_rk4_timestep(actx, dcoll, state=q_init)

    logger.info("Timestep size: %g", dt)

    # }}}

    # {{{ time stepping

    from grudge.shortcuts import set_up_rk4
    dt_stepper = set_up_rk4("q", dt, q_init, rhs)
    plot = Plotter(actx, dcoll, order, visualize=visualize, ylim=[0.0, 1.2])

    step = 0
    plot(q_init, "fld-sod-init")
    norm_q = 0.0
    for event in dt_stepper.run(t_end=final_time):
        if not isinstance(event, dt_stepper.StateComputed):
            continue

        if step % 1 == 0:
            norm_q = actx.to_numpy(op.norm(dcoll, event.state_component, 2))
            plot(event.state_component, "fld-sod-%04d" % step)

        step += 1
        logger.info("[%04d] t = %.5f |q| = %.5e", step, event.t, norm_q)

        assert norm_q < 100

    # }}}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--order", default=4, type=int)
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--esdg", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    main(cl.create_some_context,
         order=args.order,
         visualize=args.visualize,
         esdg=args.esdg)
