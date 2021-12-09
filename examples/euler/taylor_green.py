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
from grudge.array_context import PytatoPyOpenCLArrayContext
from grudge.models.euler import (
    primitive_to_conservative_vars,
    conservative_to_primitive_vars,
    EntropyStableEulerOperator
)
from grudge.shortcuts import lsrk54_step, lsrk144_step

from pytools.obj_array import make_obj_array

import grudge.op as op

import logging
logger = logging.getLogger(__name__)


def tg_vortex_initial_condition(x_vec, t=0):
    # Parameters chosen from https://arxiv.org/pdf/1909.12546.pdf
    M = 0.05
    L = 1.
    gamma = 1.4
    v0 = 1.
    p0 = 1.
    rho0 = gamma * (M ** 2)

    x, y, z = x_vec
    actx = x.array_context
    ones = 0*x + 1

    p = p0 + rho0 * (v0 ** 2) / 16 * (
        actx.np.cos(2*x / L + actx.np.cos(2*y / L))) * actx.np.cos(2*z / L + 2)
    u = v0 * actx.np.sin(x / L) * actx.np.cos(y / L) * actx.np.cos(z / L)
    v = -v0 * actx.np.cos(x / L) * actx.np.sin(y / L) * actx.np.cos(z / L)
    w = 0 * z
    velocity = make_obj_array([u, v, w])
    rho = rho0 * ones

    return primitive_to_conservative_vars((rho, velocity, p), gamma=gamma)


def run_tg_vortex(actx,
                  order=3,
                  resolution=16,
                  cfl=1.0,
                  final_time=20,
                  dumpfreq=50,
                  overintegration=False,
                  timestepper="lsrk54",
                  visualize=False):

    logger.info(
        """
        Taylor-Green vortex parameters:\n
        order: %s\n
        final_time: %s\n
        resolution: %s\n
        cfl: %s\n
        dumpfreq: %s\n
        overintegration: %s\n
        timestepper: %s\n
        visualize: %s
        """,
        order, final_time, resolution, cfl,
        dumpfreq, overintegration, timestepper, visualize
    )

    # eos-related parameters
    gamma = 1.4

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
        (default_simplex_group_factory,
         QuadratureSimplexGroupFactory)

    exp_name = f"fld-esdg-taylorgreen-N{order}-K{resolution}-cfl{cfl}"

    if overintegration:
        exp_name += "-overintegrated"
        quad_tag = DISCR_TAG_QUAD
    else:
        quad_tag = None

    dcoll = DiscretizationCollection(
        actx, mesh,
        discr_tag_to_group_factory={
            DISCR_TAG_BASE: default_simplex_group_factory(dim, order),
            DISCR_TAG_QUAD: QuadratureSimplexGroupFactory(2*order)
        }
    )

    # }}}

    # {{{ Euler operator

    euler_operator = EntropyStableEulerOperator(
        dcoll,
        flux_type="lf",
        gamma=gamma,
        quadrature_tag=quad_tag
    )

    def rhs(t, q):
        return euler_operator.operator(t, q)

    compiled_rhs = actx.compile(rhs)

    fields = tg_vortex_initial_condition(thaw(dcoll.nodes(), actx))

    if timestepper == "lsrk54":
        stepper = lsrk54_step
    else:
        assert timestepper == "lsrk144"
        stepper = lsrk144_step

    from grudge.dt_utils import h_min_from_volume

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
        if step % dumpfreq == 0:
            norm_q = actx.to_numpy(op.norm(dcoll, fields, 2))
            logger.info("[%04d] t = %.5f |q| = %.5e", step, t, norm_q)
            if visualize:
                rho, velocity, pressure = \
                    conservative_to_primitive_vars(fields, gamma=gamma)
                vis.write_vtk_file(
                    f"fld-taylor-green-vortex-{step:04d}.vtu",
                    [
                        ("rho", rho),
                        ("energy", fields.energy),
                        ("momentum", fields.momentum),
                        ("velocity", velocity),
                        ("pressure", pressure)
                    ]
                )
            assert norm_q < 10000

        fields = thaw(freeze(fields, actx), actx)
        fields = stepper(fields, t, dt, compiled_rhs)
        t += dt
        step += 1

    # }}}


def main(ctx_factory, order=3, final_time=20,
         resolution=16, cfl=1,
         dumpfreq=50, overintegration=False,
         timestepper="lsrk54", visualize=False):
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
        cfl=cfl,
        final_time=final_time,
        dumpfreq=dumpfreq,
        overintegration=overintegration,
        timestepper=timestepper,
        visualize=visualize
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--order", default=3, type=int)
    parser.add_argument("--tfinal", default=20., type=float,
                        help="specify final time for the simulation")
    parser.add_argument("--resolution", default=8, type=int,
                        help="resolution in each spatial direction")
    parser.add_argument("--dumpfreq", default=50, type=int,
                        help="step frequency for writing output or logging info")
    parser.add_argument("--cfl", default=1., type=float,
                        help="specify cfl value to use in dt computation")
    parser.add_argument("--oi", action="store_true",
                        help="use overintegration")
    parser.add_argument("--visualize", action="store_true",
                        help="write out vtk output")
    parser.add_argument("--timestepper", default="lsrk54",
                        choices=['lsrk54', 'lsrk144'],
                        help="specify timestepper method")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    main(cl.create_some_context,
         order=args.order,
         final_time=args.tfinal,
         resolution=args.resolution,
         cfl=args.cfl,
         dumpfreq=args.dumpfreq,
         overintegration=args.oi,
         timestepper=args.timestepper,
         visualize=args.visualize)
