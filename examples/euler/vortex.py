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


import numpy as np

import pyopencl as cl
import pyopencl.tools as cl_tools

from dataclasses import dataclass

from arraycontext import thaw, freeze
from meshmode.array_context import (
    SingleGridWorkBalancingPytatoArrayContext as PytatoPyOpenCLArrayContext
)
from grudge.models.euler import (
    EulerState,
    EulerOperator,
    EntropyStableEulerOperator
)

from pytools.obj_array import make_obj_array

import grudge.op as op

import matplotlib.pyplot as pt

import logging
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LSRKCoefficients:
    """Dataclass which defines a given low-storage Runge-Kutta (LSRK) scheme.
    The methods are determined by the provided `A`, `B` and `C` coefficient arrays.
    """

    A: np.ndarray
    B: np.ndarray
    C: np.ndarray


LSRK54CarpenterKennedyCoefs = LSRKCoefficients(
    A=np.array([
        0.,
        -567301805773/1357537059087,
        -2404267990393/2016746695238,
        -3550918686646/2091501179385,
        -1275806237668/842570457699]),
    B=np.array([
        1432997174477/9575080441755,
        5161836677717/13612068292357,
        1720146321549/2090206949498,
        3134564353537/4481467310338,
        2277821191437/14882151754819]),
    C=np.array([
        0.,
        1432997174477/9575080441755,
        2526269341429/6820363962896,
        2006345519317/3224310063776,
        2802321613138/2924317926251]))


def lsrk_step(state, t, dt, rhs):
    """Take one step using a low-storage Runge-Kutta method."""
    k = 0.0 * state
    coefs = LSRK54CarpenterKennedyCoefs
    for i in range(len(coefs.A)):
        k = coefs.A[i]*k + dt*rhs(t + coefs.C[i]*dt, state)
        state += coefs.B[i]*k

    return state


def vortex_initial_condition(x_vec, t=0):
    M = 0.5  # Mach number
    _x0 = 5
    epsilon = 1   # vortex strength
    gamma = 1.4
    x, y = x_vec
    actx = x.array_context

    fxyt = 1 - (((x - _x0) - t)**2 + y**2)
    expterm = actx.np.exp(fxyt/2)

    u = 1 - (epsilon*y/(2*np.pi)) * expterm
    v = ((epsilon*(x - _x0) - t)/(2*np.pi)) * expterm

    velocity = make_obj_array([u, v])
    mass = (
        1 - ((epsilon**2 * (gamma - 1) * M**2)/(8*np.pi**2)) * actx.np.exp(fxyt)
    ) ** (1 / (gamma - 1))
    momentum = mass * velocity
    p = (mass ** gamma)/(gamma * M**2)

    energy = p / (gamma - 1) + mass / 2 * (u ** 2 + v ** 2)

    return EulerState(mass=mass, energy=energy, momentum=momentum)


def run_vortex(actx, order=3, resolution=8, final_time=5,
               nodal_dg=False,
               flux_type="central",
               visualize=False):

    logger.info(
        """
        Isentropic vortex parameters:\n
        order: %s\n
        final_time: %s\n
        nodal_dg: %s\n
        resolution: %s\n
        flux_type: %s\n
        visualize: %s
        """,
        order, final_time, nodal_dg, resolution, flux_type, visualize
    )

    # eos-related parameters
    gamma = 1.4
    gas_const = 287.1

    # {{{ discretization

    from meshmode.mesh.generation import generate_regular_rect_mesh

    mesh = generate_regular_rect_mesh(
        a=(0, -5),
        b=(20, 5),
        nelements_per_axis=(2*resolution, resolution),
        periodic=(True, True))

    from grudge import DiscretizationCollection
    from grudge.dof_desc import as_dofdesc, DISCR_TAG_BASE, DISCR_TAG_QUAD
    from meshmode.discretization.poly_element import \
        (PolynomialWarpAndBlend2DRestrictingGroupFactory,
         QuadratureSimplexGroupFactory)

    dcoll = DiscretizationCollection(
        actx, mesh,
        discr_tag_to_group_factory={
            DISCR_TAG_BASE: PolynomialWarpAndBlend2DRestrictingGroupFactory(order),
            DISCR_TAG_QUAD: QuadratureSimplexGroupFactory(2*order)
        }
    )

    # }}}

    # {{{ Euler operator

    if nodal_dg:
        operator_cls = EulerOperator
        exp_name = f"fld-vortex-{flux_type}"
    else:
        operator_cls = EntropyStableEulerOperator
        exp_name = f"fld-esdg-vortex-{flux_type}"

    euler_operator = operator_cls(
        dcoll,
        flux_type=flux_type,
        gamma=gamma,
        gas_const=gas_const,
    )

    def rhs(t, q):
        return euler_operator.operator(t, q)

    compiled_rhs = actx.compile(rhs)

    fields = vortex_initial_condition(thaw(dcoll.nodes(), actx))

    from grudge.dt_utils import h_min_from_volume

    cfl = 0.01
    cn = 0.5*(order + 1)**2
    dt = cfl * actx.to_numpy(h_min_from_volume(dcoll)) / cn

    logger.info("Timestep size: %g", dt)

    # }}}

    from grudge.shortcuts import make_visualizer

    vis = make_visualizer(dcoll)

    # {{{ time stepping

    step = 0
    t = 0.0

    dq = as_dofdesc("vol").with_discr_tag(DISCR_TAG_QUAD)
    entropy = euler_operator.state_to_mathematical_entropy(fields)
    entropy_int0 =  actx.to_numpy(op.integral(dcoll, dq, entropy))

    fig = pt.figure(figsize=(8, 8), dpi=300)
    taxis = [0]
    entropy_rel_diff = [0]

    while t < final_time:
        fields = thaw(freeze(fields, actx), actx)
        fields = lsrk_step(fields, t, dt, compiled_rhs)
        t += dt

        if step % 10 == 0:
            norm_q = actx.to_numpy(op.norm(dcoll, fields, 2))
            logger.info("[%04d] t = %.5f |q| = %.5e", step, t, norm_q)
            # analytic = \
            #     thaw(
            #         freeze(
            #             vortex_initial_condition(thaw(dcoll.nodes(), actx), t=t),
            #             actx
            #         ),
            #         actx
            #     )
            entropy = euler_operator.state_to_mathematical_entropy(fields)
            entropy_integral = actx.to_numpy(op.integral(dcoll, dq, entropy))
            int_diff = entropy_integral - entropy_int0
            rel_diff = abs(int_diff)/abs(entropy_int0)

            taxis.append(t)
            entropy_rel_diff.append(rel_diff)
            logger.info("|∫η1 - ∫η0|/|∫η0|: %s", rel_diff)

            if visualize:
                vis.write_vtk_file(
                    f"{exp_name}-{step:04d}.vtu",
                    [
                        ("rho", fields.mass),
                        ("energy", fields.energy),
                        ("momentum", fields.momentum),
                        # ("analytic rho", analytic.mass),
                        # ("analytic energy", analytic.energy),
                        # ("analytic momentum", analytic.momentum)
                    ]
                )
            assert norm_q < 10000

        step += 1

    # }}}

    # Plot entropy integral rel. diff over time:
    filename = \
        f"{exp_name}-entropy-diff-cfl{cfl}-r{resolution}-deg{order}.png"
    pt.rcParams.update({'font.size': 20})
    ax = fig.gca()
    ax.grid()
    ax.set_yscale('log')
    ax.plot(taxis, entropy_rel_diff, "-")
    ax.plot(taxis, entropy_rel_diff, "k.")
    ax.set_xlabel("time")
    ax.set_ylabel("|∫η1 - ∫η0| /|∫η0|")
    pt.title(f'Rel. change in entropy: cfl={cfl}')
    fig.savefig(filename, bbox_inches="tight")


def run_convergence_test_vortex(
        actx, order=3, final_time=5,
        nodal_dg=False,
        flux_type="central"):

    logger.info(
        """
        Isentropic vortex convergence test parameters:\n
        order: %s\n
        final_time: %s\n
        nodal_dg: %s\n
        flux_type: %s
        """,
        order, final_time, nodal_dg, flux_type
    )

    # eos-related parameters
    gamma = 1.4
    gas_const = 287.1

    from meshmode.mesh.generation import generate_regular_rect_mesh

    dim = 2

    from grudge import DiscretizationCollection
    from grudge.dof_desc import DISCR_TAG_BASE, DISCR_TAG_QUAD
    from meshmode.discretization.poly_element import \
        (PolynomialWarpAndBlend2DRestrictingGroupFactory,
         QuadratureSimplexGroupFactory)
    from pytools.convergence import EOCRecorder
    from grudge.dt_utils import h_max_from_volume

    eoc_rec = EOCRecorder()

    for resolution in [8, 16, 32, 64]:

        # {{{ discretization

        mesh = generate_regular_rect_mesh(
            a=(0, -5),
            b=(20, 5),
            nelements_per_axis=(2*resolution, resolution),
            periodic=(True, True))

        discr_tag_to_group_factory = {
            DISCR_TAG_BASE: PolynomialWarpAndBlend2DRestrictingGroupFactory(order),
            DISCR_TAG_QUAD: QuadratureSimplexGroupFactory(2*order)
        }

        dcoll = DiscretizationCollection(
            actx, mesh,
            discr_tag_to_group_factory=discr_tag_to_group_factory
        )
        h_max = actx.to_numpy(h_max_from_volume(dcoll, dim=dcoll.ambient_dim))
        nodes = thaw(dcoll.nodes(), actx)

        # }}}

        if nodal_dg:
            operator_cls = EulerOperator
        else:
            operator_cls = EntropyStableEulerOperator

        euler_operator = operator_cls(
            dcoll,
            flux_type=flux_type,
            gamma=gamma,
            gas_const=gas_const,
        )

        def rhs(t, q):
            return euler_operator.operator(t, q)

        compiled_rhs = actx.compile(rhs)

        fields = vortex_initial_condition(nodes)

        from grudge.dt_utils import h_min_from_volume

        cfl = 0.125
        cn = 0.5*(order + 1)**2
        dt = cfl * actx.to_numpy(h_min_from_volume(dcoll)) / cn

        logger.info("Timestep size: %g", dt)

        # {{{ time stepping

        step = 0
        t = 0.0
        last_q = None
        while t < final_time:
            fields = thaw(freeze(fields, actx), actx)
            fields = lsrk_step(fields, t, dt, compiled_rhs)
            t += dt
            logger.info("[%04d] t = %.5f", step, t)
            last_q = fields
            last_t = t
            step += 1

        # }}}

        error_l2 = op.norm(
            dcoll,
            last_q - vortex_initial_condition(nodes, t=last_t),
            2
        )
        error_l2 = actx.to_numpy(error_l2)
        logger.info("h_max %.5e error %.5e", h_max, error_l2)
        eoc_rec.add_data_point(h_max, error_l2)

    logger.info("\n%s", eoc_rec.pretty_print(abscissa_label="h",
                                             error_label="L2 Error"))


def main(ctx_factory, order=3, final_time=5, resolution=8,
         nodal_dg=False,
         lf_stabilization=False,
         visualize=False,
         test_convergence=False):
    cl_ctx = ctx_factory()
    queue = cl.CommandQueue(cl_ctx)
    actx = PytatoPyOpenCLArrayContext(
        queue,
        allocator=cl_tools.MemoryPool(cl_tools.ImmediateAllocator(queue))
    )

    if lf_stabilization:
        flux_type = "lf"
    else:
        flux_type = "central"

    if test_convergence:
        run_convergence_test_vortex(
            actx, order=order,
            final_time=5,
            nodal_dg=nodal_dg,
            flux_type=flux_type)
    else:
        run_vortex(
            actx, order=order, resolution=resolution,
            final_time=final_time,
            nodal_dg=nodal_dg,
            flux_type=flux_type,
            visualize=visualize)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--order", default=3, type=int)
    parser.add_argument("--tfinal", default=5, type=float)
    parser.add_argument("--resolution", default=8, type=int)
    parser.add_argument("--nodaldg", action="store_true")
    parser.add_argument("--lfflux", action="store_true")
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--convergence", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    main(cl.create_some_context,
         order=args.order,
         final_time=args.tfinal,
         resolution=args.resolution,
         nodal_dg=args.nodaldg,
         lf_stabilization=args.lfflux,
         visualize=args.visualize,
         test_convergence=args.convergence)
