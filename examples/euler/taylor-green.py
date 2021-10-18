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
from grudge.models.euler import EulerState, EntropyStableEulerOperator

from pytools.obj_array import make_obj_array

import grudge.op as op

import logging
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LSRKCoefficients:
    A: np.ndarray
    B: np.ndarray
    C: np.ndarray


LSRK144NiegemannDiehlBuschCoefs = LSRKCoefficients(
    A=np.array([
        0.,
        -0.7188012108672410,
        -0.7785331173421570,
        -0.0053282796654044,
        -0.8552979934029281,
        -3.9564138245774565,
        -1.5780575380587385,
        -2.0837094552574054,
        -0.7483334182761610,
        -0.7032861106563359,
        0.0013917096117681,
        -0.0932075369637460,
        -0.9514200470875948,
        -7.1151571693922548]),
    B=np.array([
        0.0367762454319673,
        0.3136296607553959,
        0.1531848691869027,
        0.0030097086818182,
        0.3326293790646110,
        0.2440251405350864,
        0.3718879239592277,
        0.6204126221582444,
        0.1524043173028741,
        0.0760894927419266,
        0.0077604214040978,
        0.0024647284755382,
        0.0780348340049386,
        5.5059777270269628]),
    C=np.array([
        0.,
        0.0367762454319673,
        0.1249685262725025,
        0.2446177702277698,
        0.2476149531070420,
        0.2969311120382472,
        0.3978149645802642,
        0.5270854589440328,
        0.6981269994175695,
        0.8190890835352128,
        0.8527059887098624,
        0.8604711817462826,
        0.8627060376969976,
        0.8734213127600976]))


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


def lsrk144_step(state, t, dt, rhs):
    k = 0.0 * state
    coefs = LSRK144NiegemannDiehlBuschCoefs
    for i in range(len(coefs.A)):
        k = coefs.A[i]*k + dt*rhs(t + coefs.C[i]*dt, state)
        state = state + coefs.B[i]*k
    return state


def lsrk54_step(state, t, dt, rhs):
    k = 0.0 * state
    coefs = LSRK54CarpenterKennedyCoefs
    for i in range(len(coefs.A)):
        k = coefs.A[i]*k + dt*rhs(t + coefs.C[i]*dt, state)
        state = state + coefs.B[i]*k
    return state


def tg_vortex_initial_condition(dcoll, x_vec, t=0):
    # Parameters chosen from https://arxiv.org/pdf/1909.12546.pdf
    M = 0.05
    L = 1.
    gamma = 1.4
    v0 = 1.
    p0 = 1.
    rho0 = gamma * M ** 2

    x, y, z = x_vec
    actx = x.array_context

    p = p0 + rho0 * (v0 ** 2) / 16 * (
        actx.np.cos(2*x / L + actx.np.cos(2*y / L))) * actx.np.cos(2*z / L + 2)
    u = v0 * actx.np.sin(x / L) * actx.np.cos(y / L) * actx.np.cos(z / L)
    v = -v0 * actx.np.cos(x / L) * actx.np.sin(y / L) * actx.np.cos(z / L)
    w = 0 * z
    momentum = rho0 * make_obj_array([u, v, w])
    energy = p / (gamma - 1) + rho0 / 2 * (u ** 2 + v ** 2 + w ** 2)

    return EulerState(mass=rho0 * (dcoll.zeros(actx) + 1),
                      energy=energy,
                      momentum=momentum)


def run_tg_vortex(actx, order=3, resolution=16, dtscaling=1, final_time=20,
                  dumpfreq=50, timestepper="lsrk54", visualize=False):
    logger.info(
        """
        Taylor-Green vortex parameters:\n
        order: %s, resolution: %s, dt scaling factor: %s, \n
        final time: %s, \n
        dumpfreq: %s,
        timestepper: %s, visualize: %s
        """,
        order, resolution, dtscaling,
        final_time, dumpfreq, timestepper, visualize
    )

    # eos-related parameters
    gamma = 1.4
    gas_const = 287.1

    # {{{ discretization

    from meshmode.mesh.generation import generate_regular_rect_mesh

    dim = 3
    box_ll = -np.pi
    box_ur = np.pi
    mesh = generate_regular_rect_mesh(
        a=(box_ll,)*dim,
        b=(box_ur,)*dim,
        nelements_per_axis=(resolution,)*dim,
        periodic=(True,)*dim)

    from grudge import DiscretizationCollection
    from grudge.dof_desc import DISCR_TAG_BASE, DISCR_TAG_QUAD
    from meshmode.discretization.poly_element import \
        (PolynomialWarpAndBlend3DRestrictingGroupFactory,
         QuadratureSimplexGroupFactory)

    dcoll = DiscretizationCollection(
        actx, mesh,
        discr_tag_to_group_factory={
            DISCR_TAG_BASE: PolynomialWarpAndBlend3DRestrictingGroupFactory(order),
            DISCR_TAG_QUAD: QuadratureSimplexGroupFactory(2*order)
        }
    )

    # }}}

    # {{{ Euler operator

    euler_operator = EntropyStableEulerOperator(
        dcoll,
        flux_type="lf",
        gamma=gamma,
        gas_const=gas_const,
    )

    def rhs(t, q):
        return euler_operator.operator(t, q)

    compiled_rhs = actx.compile(rhs)

    fields = tg_vortex_initial_condition(dcoll, thaw(dcoll.nodes(), actx))

    if timestepper == "lsrk54":
        stepper = lsrk54_step
    else:
        assert timestepper == "lsrk144"
        stepper = lsrk144_step

    dt = dtscaling * euler_operator.estimate_rk4_timestep(actx, dcoll, state=fields)

    logger.info("Timestep size: %g", dt)

    # }}}

    from grudge.shortcuts import make_visualizer

    vis = make_visualizer(dcoll)

    # {{{ time stepping

    step = 0
    t = 0.0
    while t < final_time:
        fields = thaw(freeze(fields, actx), actx)
        fields = stepper(fields, t, dt, compiled_rhs)

        if step % dumpfreq == 0:
            norm_q = actx.to_numpy(op.norm(dcoll, fields.join(), 2))
            logger.info("[%04d] t = %.5f |q| = %.5e", step, t, norm_q)
            if visualize:
                vis.write_vtk_file(
                    f"fld-taylor-green-vortex-{step:04d}.vtu",
                    [
                        ("rho", fields.mass),
                        ("energy", fields.energy),
                        ("momentum", fields.momentum)
                    ]
                )

        t += dt
        step += 1

    # }}}


def main(ctx_factory, order=3, final_time=20,
         resolution=16, dtscaling=1,
         dumpfreq=50, timestepper="lsrk54", visualize=False):
    cl_ctx = ctx_factory()
    queue = cl.CommandQueue(cl_ctx)
    actx = PytatoPyOpenCLArrayContext(
        queue,
        allocator=cl_tools.MemoryPool(cl_tools.ImmediateAllocator(queue))
    )
    run_tg_vortex(
        actx,
        order=order,
        resolution=resolution,
        dtscaling=dtscaling,
        final_time=final_time,
        dumpfreq=dumpfreq,
        timestepper=timestepper,
        visualize=visualize
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--order", default=3, type=int)
    parser.add_argument("--tfinal", default=20., type=float)
    parser.add_argument("--resolution", default=8, type=int)
    parser.add_argument("--dumpfreq", default=50, type=int)
    parser.add_argument("--dtscaling", default=1., type=float)
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--timestepper", default="lsrk54",
                        choices=['lsrk54', 'lsrk144'])
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    main(cl.create_some_context,
         order=args.order,
         final_time=args.tfinal,
         resolution=args.resolution,
         dtscaling=args.dtscaling,
         dumpfreq=args.dumpfreq,
         timestepper=args.timestepper,
         visualize=args.visualize)
