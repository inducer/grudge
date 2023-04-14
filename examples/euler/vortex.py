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


import pyopencl as cl
import pyopencl.tools as cl_tools

from grudge.array_context import PytatoPyOpenCLArrayContext, PyOpenCLArrayContext
from grudge.models.euler import (
    vortex_initial_condition,
    EulerOperator,
    EntropyStableEulerOperator
)
from grudge.shortcuts import rk4_step

import grudge.op as op

import logging
logger = logging.getLogger(__name__)


def run_vortex(actx, order=3, resolution=8, final_time=5,
               esdg=False,
               overintegration=False,
               flux_type="central",
               visualize=False):

    logger.info(
        """
        Isentropic vortex parameters:\n
        order: %s\n
        final_time: %s\n
        resolution: %s\n
        entropy stable: %s\n
        overintegration: %s\n
        flux_type: %s\n
        visualize: %s
        """,
        order, final_time, resolution, esdg,
        overintegration, flux_type, visualize
    )

    # eos-related parameters
    gamma = 1.4

    # {{{ discretization

    from meshmode.mesh.generation import generate_regular_rect_mesh

    mesh = generate_regular_rect_mesh(
        a=(0, -5),
        b=(20, 5),
        nelements_per_axis=(2*resolution, resolution),
        periodic=(True, True))

    from grudge import DiscretizationCollection
    from grudge.dof_desc import DISCR_TAG_BASE, DISCR_TAG_QUAD
    from meshmode.discretization.poly_element import \
        default_simplex_group_factory, QuadratureSimplexGroupFactory

    if esdg:
        case = "esdg-vortex"
        operator_cls = EntropyStableEulerOperator
    else:
        case = "vortex"
        operator_cls = EulerOperator

    exp_name = f"fld-{case}-N{order}-K{resolution}-{flux_type}"

    if overintegration:
        exp_name += "-overintegrated"
        quad_tag = DISCR_TAG_QUAD
    else:
        quad_tag = None

    dcoll = DiscretizationCollection(
        actx, mesh,
        discr_tag_to_group_factory={
            DISCR_TAG_BASE: default_simplex_group_factory(
                base_dim=mesh.dim, order=order),
            DISCR_TAG_QUAD: QuadratureSimplexGroupFactory(2*order)
        }
    )

    # }}}

    # {{{ Euler operator

    euler_operator = operator_cls(
        dcoll,
        flux_type=flux_type,
        gamma=gamma,
        quadrature_tag=quad_tag
    )

    def rhs(t, q):
        return euler_operator.operator(t, q)

    compiled_rhs = actx.compile(rhs)

    fields = vortex_initial_condition(actx.thaw(dcoll.nodes()))

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
    while t < final_time:
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
            assert norm_q < 200

        fields = actx.thaw(actx.freeze(fields))
        fields = rk4_step(fields, t, dt, compiled_rhs)
        t += dt
        step += 1

    # }}}


def main(ctx_factory, order=3, final_time=5, resolution=8,
         esdg=False,
         overintegration=False,
         lf_stabilization=False,
         visualize=False,
         lazy=False):
    cl_ctx = ctx_factory()
    queue = cl.CommandQueue(cl_ctx)

    if lazy:
        actx = PytatoPyOpenCLArrayContext(
            queue,
            allocator=cl_tools.MemoryPool(cl_tools.ImmediateAllocator(queue)),
        )
    else:
        actx = PyOpenCLArrayContext(
            queue,
            allocator=cl_tools.MemoryPool(cl_tools.ImmediateAllocator(queue)),
            force_device_scalars=True,
        )

    if not actx.supports_nonscalar_broadcasting and esdg is True:
        raise RuntimeError(
            "Cannot use ESDG with an array context that cannot perform "
            "nonscalar broadcasting. Run with --lazy instead."
        )

    if lf_stabilization:
        flux_type = "lf"
    else:
        flux_type = "central"

    run_vortex(
        actx,
        order=order,
        resolution=resolution,
        final_time=final_time,
        esdg=esdg,
        overintegration=overintegration,
        flux_type=flux_type,
        visualize=visualize
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--order", default=3, type=int)
    parser.add_argument("--tfinal", default=0.015, type=float)
    parser.add_argument("--resolution", default=8, type=int)
    parser.add_argument("--esdg", action="store_true",
                        help="use entropy stable dg")
    parser.add_argument("--oi", action="store_true",
                        help="use overintegration")
    parser.add_argument("--lf", action="store_true",
                        help="turn on lax-friedrichs dissipation")
    parser.add_argument("--visualize", action="store_true",
                        help="write out vtk output")
    parser.add_argument("--lazy", action="store_true",
                        help="switch to a lazy computation mode")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    main(cl.create_some_context,
         order=args.order,
         final_time=args.tfinal,
         resolution=args.resolution,
         esdg=args.esdg,
         overintegration=args.oi,
         lf_stabilization=args.lf,
         visualize=args.visualize,
         lazy=args.lazy)
