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
from pytools.obj_array import flat_obj_array

from grudge import op
from grudge.array_context import PyOpenCLArrayContext
from grudge.discretization import make_discretization_collection
from grudge.shortcuts import set_up_rk4


logger = logging.getLogger(__name__)


def main(ctx_factory, dim=2, order=4, visualize=False):
    cl_ctx = ctx_factory()
    queue = cl.CommandQueue(cl_ctx)
    actx = PyOpenCLArrayContext(
        queue,
        allocator=cl_tools.MemoryPool(cl_tools.ImmediateAllocator(queue)),
        force_device_scalars=True,
    )

    from meshmode.mesh.generation import generate_regular_rect_mesh
    mesh = generate_regular_rect_mesh(
            a=(-0.5,)*dim,
            b=(0.5,)*dim,
            nelements_per_axis=(20,)*dim)

    dcoll = make_discretization_collection(actx, mesh, order=order)

    def source_f(actx, dcoll, t=0):
        source_center = np.array([0.1, 0.22, 0.33])[:dcoll.dim]
        source_width = 0.05
        source_omega = 3
        nodes = actx.thaw(dcoll.nodes())
        source_center_dist = flat_obj_array(
            [nodes[i] - source_center[i] for i in range(dcoll.dim)]
        )
        return (
            np.sin(source_omega*t)
            * actx.np.exp(
                -np.dot(source_center_dist, source_center_dist)
                / source_width**2
            )
        )

    x = actx.thaw(dcoll.nodes())
    ones = dcoll.zeros(actx) + 1
    c = actx.np.where(np.dot(x, x) < 0.15, 0.1 * ones, 0.2 * ones)

    from meshmode.mesh import BTAG_ALL, BTAG_NONE

    from grudge.models.wave import VariableCoefficientWeakWaveOperator

    wave_op = VariableCoefficientWeakWaveOperator(
        dcoll,
        actx.freeze(c),
        source_f=source_f,
        dirichlet_tag=BTAG_NONE,
        neumann_tag=BTAG_NONE,
        radiation_tag=BTAG_ALL,
        flux_type="upwind"
    )

    fields = flat_obj_array(
        dcoll.zeros(actx),
        [dcoll.zeros(actx) for i in range(dcoll.dim)]
    )

    wave_op.check_bc_coverage(mesh)

    def rhs(t, w):
        return wave_op.operator(t, w)

    dt = actx.to_numpy(
        2/3 * wave_op.estimate_rk4_timestep(actx, dcoll, fields=fields))
    dt_stepper = set_up_rk4("w", dt, fields, rhs)

    final_t = 1
    nsteps = int(final_t/dt) + 1

    logger.info("dt=%g nsteps=%d", dt, nsteps)

    from grudge.shortcuts import make_visualizer
    vis = make_visualizer(dcoll)

    step = 0

    def norm(u):
        return op.norm(dcoll, u, 2)

    from time import time
    t_last_step = time()

    if visualize:
        u = fields[0]
        v = fields[1:]
        vis.write_vtk_file(
            f"fld-var-propogation-speed-{step:04d}.vtu",
            [
                ("u", u),
                ("v", v),
                ("c", c),
            ]
        )

    for event in dt_stepper.run(t_end=final_t):
        if isinstance(event, dt_stepper.StateComputed):
            assert event.component_id == "w"

            step += 1

            if step % 10 == 0:
                logger.info("step: %d t: %.8e L2: %.8e",
                            step, time() - t_last_step,
                            norm(event.state_component[0]))
                if visualize:
                    vis.write_vtk_file(
                        f"fld-var-propogation-speed-{step:04d}.vtu",
                        [
                            ("u", event.state_component[0]),
                            ("v", event.state_component[1:]),
                            ("c", c),
                        ]
                    )
            t_last_step = time()

            # NOTE: These are here to ensure the solution is bounded for the
            # time interval specified
            assert norm(u=event.state_component[0]) < 1


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dim", default=2, type=int)
    parser.add_argument("--order", default=4, type=int)
    parser.add_argument("--visualize", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    main(cl.create_some_context,
         dim=args.dim,
         order=args.order,
         visualize=args.visualize)
