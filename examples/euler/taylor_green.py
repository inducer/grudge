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
from grudge.models.euler import EulerState, EntropyStableEulerOperator
from grudge.shortcuts import lsrk54_step, lsrk144_step

from pytools.obj_array import make_obj_array

import grudge.op as op

import logging
logger = logging.getLogger(__name__)


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

    dt = dtscaling * euler_operator.estimate_rk4_timestep(
        actx, dcoll, state=fields)

    logger.info("Timestep size: %g", dt)

    # }}}

    from grudge.shortcuts import make_visualizer

    vis = make_visualizer(dcoll)

    # {{{ time stepping

    step = 0
    t = 0.0
    while t < final_time:
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

        fields = thaw(freeze(fields, actx), actx)
        fields = stepper(fields, t, dt, compiled_rhs)
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
