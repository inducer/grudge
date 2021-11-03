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


def ssprk43_step(y, t, h, f):

    def f_update(t, y):
        return y + h*f(t, y)

    y1 = 1/2*y + 1/2*f_update(t, y)
    y2 = 1/2*y1 + 1/2*f_update(t + h/2, y1)
    y3 = 2/3*y + 1/6*y2 + 1/6*f_update(t + h, y2)

    return 1/2*y3 + 1/2*f_update(t + h/2, y3)


def rk4_step(y, t, h, f):
    k1 = f(t, y)
    k2 = f(t+h/2, y + h/2*k1)
    k3 = f(t+h/2, y + h/2*k2)
    k4 = f(t+h, y + h*k3)
    return y + h/6*(k1 + 2*k2 + 2*k3 + k4)


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
        boundary_tag_to_face={
            "prescribed": ["+x", "-x"],
            "slip": ["+y", "-y"]
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
        AdiabaticSlipBC(
            dd=as_dofdesc(dd_slip).with_discr_tag(DISCR_TAG_QUAD)
        ),
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
    dt = actx.to_numpy(
        1/6 * euler_operator.estimate_rk4_timestep(actx, dcoll, state=fields))

    logger.info("Timestep size: %g", dt)

    # }}}

    from grudge.shortcuts import make_visualizer

    vis = make_visualizer(dcoll)

    # {{{ time stepping

    step = 0
    t = 0.0
    while t < final_time:
        fields = thaw(freeze(fields, actx), actx)
        fields = rk4_step(fields, t, dt, compiled_rhs)

        if step % 1 == 0:
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
