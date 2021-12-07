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
from meshmode.array_context import (
    SingleGridWorkBalancingPytatoArrayContext as PytatoPyOpenCLArrayContext
)
from grudge.models.euler import (
    EulerState,
    EulerOperator,
    EntropyStableEulerOperator,
    PrescribedBC
)
from grudge.shortcuts import lsrk54_step

from pytools.obj_array import make_obj_array

import grudge.op as op

import logging
logger = logging.getLogger(__name__)


def sine_shock_initial_condition(nodes, t=0):
    gamma = 1.4
    dim = len(nodes)
    gmn1 = 1.0 / (gamma - 1.0)
    x = nodes[0]
    actx = x.array_context
    zeros = 0*x
    epsilon = 0.2

    _x0 = -4
    _rhoin = 3.857143
    _rhoout = 1 + epsilon*actx.np.sin(5*x)
    _uin = 2.629369
    _uout = 0.0
    _pin = 10.33333
    _pout = 1

    rhoin = zeros + _rhoin
    rhoout = _rhoout

    energyin = zeros + gmn1 * _pin
    energyout = zeros + gmn1 * _pout

    x0 = zeros + _x0
    sigma = 0.05
    xtanh = 1.0/sigma*(x - x0)
    weight = 0.5*(1.0 - actx.np.tanh(xtanh))

    mass = rhoout + (rhoin - rhoout)*weight

    uin = zeros + _uin
    uout = zeros + _uout
    momentum = mass * make_obj_array(
        [uout + (uin - uout)*weight for _ in range(dim)]
    )

    energy = energyout + (energyin - energyout)*weight

    return EulerState(mass=mass, energy=energy, momentum=momentum)


def run_sine_shock_problem(actx,
                           order=4,
                           resolution=40,
                           final_time=1.8,
                           overintegration=False,
                           nodal_dg=False,
                           visualize=False):

    # eos-related parameters
    gamma = 1.4
    gas_const = 287.1

    # {{{ discretization

    from meshmode.mesh.generation import generate_regular_rect_mesh

    dim = 1
    box_ll = -5.0
    box_ur = 5.0
    mesh = generate_regular_rect_mesh(
        a=(box_ll,)*dim,
        b=(box_ur,)*dim,
        nelements_per_axis=(resolution,)*dim,
        boundary_tag_to_face={
            "prescribed": ["+x", "-x"],
        }
    )

    from grudge import DiscretizationCollection
    from grudge.dof_desc import \
        as_dofdesc, DISCR_TAG_BASE, DISCR_TAG_QUAD, DTAG_BOUNDARY
    from meshmode.discretization.poly_element import \
        (default_simplex_group_factory,
         QuadratureSimplexGroupFactory)

    if nodal_dg:
        operator_cls = EulerOperator
        exp_name = f"fld-sine-shock-N{order}-K{resolution}"
    else:
        operator_cls = EntropyStableEulerOperator
        exp_name = f"fld-esdg-sine-shock-N{order}-K{resolution}"

    if overintegration:
        exp_name += "-overintegrated"
        quad_tag = DISCR_TAG_QUAD
    else:
        quad_tag = None

    dcoll = DiscretizationCollection(
        actx, mesh,
        discr_tag_to_group_factory={
            DISCR_TAG_BASE: default_simplex_group_factory(dim, order),
            DISCR_TAG_QUAD: QuadratureSimplexGroupFactory(order + 2)
        }
    )

    # }}}

    # {{{ Euler operator

    dd_prescribe = DTAG_BOUNDARY("prescribed")
    bcs = [
        PrescribedBC(dd=as_dofdesc(dd_prescribe).with_discr_tag(DISCR_TAG_QUAD),
                     prescribed_state=sine_shock_initial_condition)
    ]

    euler_operator = operator_cls(
        dcoll,
        bdry_conditions=bcs,
        flux_type="lf",
        gamma=gamma,
        gas_const=gas_const,
        quadrature_tag=quad_tag
    )

    def rhs(t, q):
        return euler_operator.operator(t, q)

    compiled_rhs = actx.compile(rhs)

    fields = sine_shock_initial_condition(thaw(dcoll.nodes(), actx))

    from grudge.dt_utils import h_min_from_volume

    cfl = 0.01
    cn = 0.5*(order + 1)**2
    dt = cfl * actx.to_numpy(h_min_from_volume(dcoll)) / cn

    logger.info("Timestep size: %g", dt)

    # }}}

    from grudge.shortcuts import make_visualizer

    vis = make_visualizer(dcoll)

    # {{{ time stepping

    from grudge.models.euler import conservative_to_primitive_vars

    step = 0
    t = 0.0
    while t < final_time:
        if step % 10 == 0:
            norm_q = actx.to_numpy(op.norm(dcoll, fields, 2))
            logger.info("[%04d] t = %.5f |q| = %.5e", step, t, norm_q)
            if visualize:
                _, velocity, pressure = \
                    conservative_to_primitive_vars(fields, gamma=gamma)
                vis.write_vtk_file(
                    f"{exp_name}-{step:04d}.vtu",
                    [
                        ("rho", fields.mass),
                        ("energy", fields.energy),
                        ("momentum", fields.momentum),
                        ("velocity", velocity),
                        ("pressure", pressure)
                    ]
                )
            assert norm_q < 10000

        fields = thaw(freeze(fields, actx), actx)
        fields = lsrk54_step(fields, t, dt, compiled_rhs)
        t += dt
        step += 1

    # }}}


def main(ctx_factory, order=4, final_time=1.8, resolution=40,
         overintegration=False, nodal_dg=False, visualize=False):
    cl_ctx = ctx_factory()
    queue = cl.CommandQueue(cl_ctx)
    actx = PytatoPyOpenCLArrayContext(
        queue,
        allocator=cl_tools.MemoryPool(cl_tools.ImmediateAllocator(queue)),
    )

    run_sine_shock_problem(
        actx, order=order, resolution=resolution,
        final_time=final_time,
        overintegration=overintegration,
        nodal_dg=nodal_dg,
        visualize=visualize)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--order", default=4, type=int)
    parser.add_argument("--tfinal", default=1.8, type=float)
    parser.add_argument("--resolution", default=40, type=int)
    parser.add_argument("--oi", action="store_true")
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--nodaldg", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    main(cl.create_some_context,
         order=args.order,
         final_time=args.tfinal,
         resolution=args.resolution,
         overintegration=args.oi,
         nodal_dg=args.nodaldg,
         visualize=args.visualize)
