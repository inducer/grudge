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
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme()

import logging
logger = logging.getLogger(__name__)


# {{{ plotting (keep in sync with `var-velocity.py`)

class Plotter:
    def __init__(self, actx, dcoll, order, npoints, visualize=True, ylim=None):
        self.actx = actx
        self.dim = dcoll.ambient_dim
        self.order = order
        self.npoints = npoints

        self.visualize = visualize
        if not self.visualize:
            return

        self.mplcontext = plt.subplots(2, 2, sharex=True)
        self.ylim = ylim

        volume_discr = dcoll.discr_from_dd(dof_desc.DD_VOLUME)
        self.x = actx.to_numpy(flatten(thaw(volume_discr.nodes()[0], actx)))

    def __call__(self, state, basename, overwrite=True, t=0.0):
        if not self.visualize:
            return

        assert self.dim == 1

        density = self.actx.to_numpy(flatten(state[0]))
        energy = self.actx.to_numpy(flatten(state[1]))
        momentum = self.actx.to_numpy(flatten(state[2]))
        # Hard-coded...
        gamma = 1.4
        velocity = momentum / density
        pressure = (gamma - 1) * (energy - 0.5 * (momentum * velocity))

        filename = "%s.pdf" % basename
        if not overwrite and os.path.exists(filename):
            from meshmode import FileExistsError
            raise FileExistsError("output file '%s' already exists" % filename)

        fontsize = 11
        linewidth = 2
        markersize = 2
        # NOTE: For some reason, this does NOT work:
        # fig, axs = self.mplcontext
        # So using the wasteful approach....
        fig, axs = plt.subplots(2, 2, sharex=True)

        axs[0, 0].plot(self.x, density, ":",
                       marker='o',
                       label='density',
                       linewidth=linewidth,
                       markersize=markersize)
        axs[1, 0].plot(self.x, pressure, ":",
                       marker='o',
                       label='pressure',
                       linewidth=linewidth,
                       markersize=markersize)
        axs[0, 1].plot(self.x, velocity, ":",
                       marker='o',
                       label='velocity',
                       linewidth=linewidth,
                       markersize=markersize)
        axs[1, 1].plot(self.x, energy, ":",
                       marker='o',
                       label='energy',
                       linewidth=linewidth,
                       markersize=markersize)

        # if self.ylim is not None:
        #     ax.set_ylim(self.ylim)
        # ax.legend(prop={'size': fontsize})

        axs[0, 0].set_xlabel("$x$", fontsize=fontsize)
        axs[0, 0].set_ylabel("$\\rho$", fontsize=fontsize)
        # axs[0, 0].tick_params(axis='x', labelsize=fontsize)
        # axs[0, 0].tick_params(axis='y', labelsize=fontsize)
        axs[1, 0].set_xlabel("$x$", fontsize=fontsize)
        axs[1, 0].set_ylabel("$p$", fontsize=fontsize)
        # axs[1, 0].tick_params(axis='x', labelsize=fontsize)
        # axs[1, 0].tick_params(axis='y', labelsize=fontsize)
        axs[0, 1].set_xlabel("$x$", fontsize=fontsize)
        axs[0, 1].set_ylabel("$u$", fontsize=fontsize)
        # axs[0, 1].tick_params(axis='x', labelsize=fontsize)
        # axs[0, 1].tick_params(axis='y', labelsize=fontsize)
        axs[1, 1].set_xlabel("$x$", fontsize=fontsize)
        axs[1, 1].set_ylabel("$\\rho e$", fontsize=fontsize)
        # axs[1, 1].tick_params(axis='x', labelsize=fontsize)
        # axs[1, 1].tick_params(axis='y', labelsize=fontsize)
        fig.suptitle(
            f"N = {self.order}, Npt = {self.npoints}, t = {t:.3f}",
            fontsize=fontsize
        )

        fig.savefig(filename, bbox_inches='tight')
        fig.clf()

# }}}


def main(ctx_factory, order=4, visualize=False, esdg=False, overintegration=False):
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
    npoints = 20

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

    interpolation_grp_factory = default_simplex_group_factory(dim, order)
    if overintegration:
        discr_tag_to_group_factory = {
            DISCR_TAG_BASE: interpolation_grp_factory,
            DISCR_TAG_QUAD: QuadratureSimplexGroupFactory(order + 2)
        }
    else:
        discr_tag_to_group_factory = {
            DISCR_TAG_BASE: interpolation_grp_factory,
            DISCR_TAG_QUAD: interpolation_grp_factory
        }

    dcoll = DiscretizationCollection(
        actx, mesh,
        discr_tag_to_group_factory=discr_tag_to_group_factory
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
        initial_condition=sod_initial_condition,
        flux_type=flux_type,
        gamma=gamma,
        gas_const=gas_const,
    )

    def rhs(t, q):
        return euler_operator.operator(t, q)

    dt = 1/5 * euler_operator.estimate_rk4_timestep(actx, dcoll, state=q_init)

    logger.info("Timestep size: %g", dt)

    # }}}

    # {{{ time stepping

    from grudge.shortcuts import set_up_rk4
    dt_stepper = set_up_rk4("q", dt, q_init, rhs)
    plot = Plotter(actx, dcoll, order, npoints,
                   visualize=visualize, ylim=[-0.2, 1.2])

    step = 0
    plot(q_init, "fld-sod-init")
    norm_q = 0.0
    for event in dt_stepper.run(t_end=final_time):
        if not isinstance(event, dt_stepper.StateComputed):
            continue

        if step % 1 == 0:
            norm_q = actx.to_numpy(op.norm(dcoll, event.state_component, 2))
            plot(event.state_component, "fld-sod-%04d" % step, t=event.t)

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
    parser.add_argument("--overintegration", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    main(cl.create_some_context,
         order=args.order,
         visualize=args.visualize,
         esdg=args.esdg,
         overintegration=args.overintegration)
