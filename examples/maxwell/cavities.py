__copyright__ = """
Copyright (C) 2015 Andreas Kloeckner
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


import logging

import numpy as np

import pyopencl as cl
import pyopencl.tools as cl_tools

from grudge import op
from grudge.array_context import PyOpenCLArrayContext
from grudge.discretization import make_discretization_collection
from grudge.models.em import get_rectangular_cavity_mode
from grudge.shortcuts import set_up_rk4


logger = logging.getLogger(__name__)


def main(ctx_factory, dim=3, order=4, visualize=False):
    cl_ctx = ctx_factory()
    queue = cl.CommandQueue(cl_ctx)
    actx = PyOpenCLArrayContext(
        queue,
        allocator=cl_tools.MemoryPool(cl_tools.ImmediateAllocator(queue)),
        force_device_scalars=True,
    )

    from meshmode.mesh.generation import generate_regular_rect_mesh
    mesh = generate_regular_rect_mesh(
            a=(0.0,)*dim,
            b=(1.0,)*dim,
            nelements_per_axis=(4,)*dim)

    dcoll = make_discretization_collection(actx, mesh, order=order)

    if 0:
        epsilon0 = 8.8541878176e-12  # C**2 / (N m**2)
        mu0 = 4*np.pi*1e-7  # N/A**2.
        epsilon = 1*epsilon0
        mu = 1*mu0
    else:
        epsilon = 1
        mu = 1

    from grudge.models.em import MaxwellOperator

    maxwell_operator = MaxwellOperator(
        dcoll,
        epsilon,
        mu,
        flux_type=0.5,
        dimensions=dim
    )

    def cavity_mode(x, t=0):
        if dim == 3:
            return get_rectangular_cavity_mode(actx, x, t, 1, (1, 2, 2))
        else:
            return get_rectangular_cavity_mode(actx, x, t, 1, (2, 3))

    fields = cavity_mode(actx.thaw(dcoll.nodes()), t=0)

    maxwell_operator.check_bc_coverage(mesh)

    def rhs(t, w):
        return maxwell_operator.operator(t, w)

    dt = actx.to_numpy(
        maxwell_operator.estimate_rk4_timestep(actx, dcoll, fields=fields))

    dt_stepper = set_up_rk4("w", dt, fields, rhs)

    target_steps = 60
    final_t = dt * target_steps
    nsteps = int(final_t/dt) + 1

    logger.info("dt = %g nsteps = %d", dt, nsteps)

    from grudge.shortcuts import make_visualizer
    vis = make_visualizer(dcoll)

    step = 0

    def norm(u):
        return op.norm(dcoll, u, 2)

    e, h = maxwell_operator.split_eh(fields)

    if visualize:
        vis.write_vtk_file(
            f"fld-cavities-{step:04d}.vtu",
            [
                ("e", e),
                ("h", h),
            ]
        )

    for event in dt_stepper.run(t_end=final_t):
        if isinstance(event, dt_stepper.StateComputed):
            assert event.component_id == "w"

            step += 1
            e, h = maxwell_operator.split_eh(event.state_component)

            norm_e0 = actx.to_numpy(norm(u=e[0]))
            norm_e1 = actx.to_numpy(norm(u=e[1]))
            norm_h0 = actx.to_numpy(norm(u=h[0]))
            norm_h1 = actx.to_numpy(norm(u=h[1]))

            logger.info(
                "[%04d] t = %.5f |e0| = %.5e, |e1| = %.5e, |h0| = %.5e, |h1| = %.5e",
                step, event.t,
                norm_e0, norm_e1, norm_h0, norm_h1
            )

            if step % 10 == 0:
                if visualize:
                    vis.write_vtk_file(
                        f"fld-cavities-{step:04d}.vtu",
                        [
                            ("e", e),
                            ("h", h),
                        ]
                    )

            # NOTE: These are here to ensure the solution is bounded for the
            # time interval specified
            assert norm_e0 < 0.5
            assert norm_e1 < 0.5
            assert norm_h0 < 0.5
            assert norm_h1 < 0.5


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dim", default=3, type=int)
    parser.add_argument("--order", default=4, type=int)
    parser.add_argument("--visualize", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    main(cl.create_some_context,
         dim=args.dim,
         order=args.order,
         visualize=args.visualize)
