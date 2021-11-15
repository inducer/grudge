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

from arraycontext import thaw, freeze
from grudge.array_context import (  # noqa: F401
    PyOpenCLArrayContext
)
from meshmode.array_context import (
    SingleGridWorkBalancingPytatoArrayContext as PytatoPyOpenCLArrayContext
)
from grudge.models.euler import (
    EulerState,
    EulerOperator,
    EntropyStableEulerOperator,
    AdiabaticSlipBC,
    PrescribedBC
)

from meshmode.mesh import BTAG_ALL

from pytools.obj_array import make_obj_array

import grudge.op as op

import logging
logger = logging.getLogger(__name__)

from dataclasses import dataclass

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


def sod_shock_initial_condition(nodes, t=0):
    gamma = 1.4
    dim = len(nodes)
    gmn1 = 1.0 / (gamma - 1.0)
    x = nodes[0]
    actx = x.array_context
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
    sigma = 0.05
    xtanh = 1.0/sigma*(x - x0)

    mass = (rhoin/2.0*(actx.np.tanh(-xtanh) + 1.0)
            + rhoout/2.0*(actx.np.tanh(xtanh) + 1.0))
    energy = (energyin/2.0*(actx.np.tanh(-xtanh) + 1.0)
              + energyout/2.0*(actx.np.tanh(xtanh) + 1.0))
    momentum = make_obj_array([zeros for _ in range(dim)])

    return EulerState(mass=mass, energy=energy, momentum=momentum)


def run_sod_shock_tube(actx,
                       order=3,
                       resolution=16,
                       final_time=0.2,
                       nodal_dg=False,
                       visualize=False):

    # eos-related parameters
    gamma = 1.4
    gas_const = 287.1

    # {{{ discretization

    from meshmode.mesh.generation import generate_regular_rect_mesh

    dim = 2
    box_ll = 0.0
    box_ur = 1.0
    mesh = generate_regular_rect_mesh(
        a=(box_ll,)*dim,
        b=(box_ur,)*dim,
        nelements_per_axis=(resolution,)*dim,
        periodic=(False, True),
        boundary_tag_to_face={
            "prescribed": ["+x", "-x"],
            # "slip": ["+y", "-y"]
        }
    )

    from grudge import DiscretizationCollection
    from grudge.dof_desc import \
        as_dofdesc, DISCR_TAG_BASE, DISCR_TAG_QUAD, DTAG_BOUNDARY
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

    # bcs = [
    #     AdiabaticSlipBC(
    #         dd=as_dofdesc(BTAG_ALL).with_discr_tag(DISCR_TAG_QUAD)
    #     )
    # ]
    dd_slip = DTAG_BOUNDARY("slip")
    dd_prescribe = DTAG_BOUNDARY("prescribed")
    bcs = [
        # AdiabaticSlipBC(
        #     dd=as_dofdesc(dd_slip).with_discr_tag(DISCR_TAG_QUAD)
        # ),
        PrescribedBC(dd=as_dofdesc(dd_prescribe).with_discr_tag(DISCR_TAG_QUAD),
                     prescribed_state=sod_shock_initial_condition)
    ]

    if nodal_dg:
        operator_cls = EulerOperator
        exp_name = "fld-sod-2d"
    else:
        operator_cls = EntropyStableEulerOperator
        exp_name = "fld-esdg-sod-2d"

    euler_operator = operator_cls(
        dcoll,
        bdry_conditions=bcs,
        flux_type="lf",
        gamma=gamma,
        gas_const=gas_const,
    )

    def rhs(t, q):
        return euler_operator.operator(t, q)

    compiled_rhs = actx.compile(rhs)

    fields = sod_shock_initial_condition(thaw(dcoll.nodes(), actx))

    from grudge.dt_utils import h_min_from_volume

    cfl = 0.125
    cn = 0.5*(order + 1)**2
    dt = cfl * actx.to_numpy(h_min_from_volume(dcoll)) / cn

    logger.info("Timestep size: %g", dt)

    # }}}

    from grudge.shortcuts import make_visualizer

    vis = make_visualizer(dcoll)

    # {{{ time stepping

    step = 0
    t = 0.0
    while t < final_time:
        fields = thaw(freeze(fields, actx), actx)
        fields = lsrk_step(fields, t, dt, compiled_rhs)

        if step % 10 == 0:
            norm_q = actx.to_numpy(op.norm(dcoll, fields, 2))
            logger.info("[%04d] t = %.5f |q| = %.5e", step, t, norm_q)
            if visualize:
                vis.write_vtk_file(
                    f"{exp_name}-{step:04d}.vtu",
                    [
                        ("rho", fields.mass),
                        ("energy", fields.energy),
                        ("momentum", fields.momentum)
                    ]
                )
            assert norm_q < 10000

        t += dt
        step += 1

    # }}}


def main(ctx_factory, order=3, final_time=0.2, resolution=16,
         nodal_dg=False, visualize=False):
    cl_ctx = ctx_factory()
    queue = cl.CommandQueue(cl_ctx)
    actx = PytatoPyOpenCLArrayContext(
        queue,
        allocator=cl_tools.MemoryPool(cl_tools.ImmediateAllocator(queue)),
    )

    run_sod_shock_tube(
        actx, order=order, resolution=resolution,
        final_time=final_time,
        nodal_dg=nodal_dg,
        visualize=visualize)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--order", default=3, type=int)
    parser.add_argument("--tfinal", default=0.2, type=float)
    parser.add_argument("--resolution", default=16, type=int)
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--nodaldg", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    main(cl.create_some_context,
         order=args.order,
         final_time=args.tfinal,
         resolution=args.resolution,
         nodal_dg=args.nodaldg,
         visualize=args.visualize)
