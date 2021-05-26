"""Minimal example of a grudge driver."""

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


import numpy as np
import pyopencl as cl
import pyopencl.tools as cl_tools

from arraycontext import PyOpenCLArrayContext, thaw

from grudge.shortcuts import set_up_rk4
from grudge import DiscretizationCollection

from grudge.models.em import get_rectangular_cavity_mode

import grudge.op as op


STEPS = 60


def main(dims, write_output=False, order=4):
    cl_ctx = cl.create_some_context()
    queue = cl.CommandQueue(cl_ctx)
    actx = PyOpenCLArrayContext(
        queue,
        allocator=cl_tools.MemoryPool(cl_tools.ImmediateAllocator(queue))
    )

    from meshmode.mesh.generation import generate_regular_rect_mesh
    mesh = generate_regular_rect_mesh(
            a=(0.0,)*dims,
            b=(1.0,)*dims,
            nelements_per_axis=(4,)*dims)

    dcoll = DiscretizationCollection(actx, mesh, order=order)

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
        dimensions=dims
    )

    def cavity_mode(x, t=0):
        if dims == 3:
            return get_rectangular_cavity_mode(actx, x, t, 1, (1, 2, 2))
        else:
            return get_rectangular_cavity_mode(actx, x, t, 1, (2, 3))

    fields = cavity_mode(thaw(op.nodes(dcoll), actx), t=0)

    # FIXME
    # dt = maxwell_operator.estimate_rk4_timestep(dcoll, fields=fields)

    maxwell_operator.check_bc_coverage(mesh)

    def rhs(t, w):
        return maxwell_operator.operator(t, w)

    if mesh.dim == 2:
        dt = 0.004
    elif mesh.dim == 3:
        dt = 0.002

    dt_stepper = set_up_rk4("w", dt, fields, rhs)

    final_t = dt * STEPS
    nsteps = int(final_t/dt)

    print("dt=%g nsteps=%d" % (dt, nsteps))

    from grudge.shortcuts import make_visualizer
    vis = make_visualizer(dcoll)

    step = 0

    def norm(u):
        return op.norm(dcoll, u, 2)

    from time import time
    t_last_step = time()

    e, h = maxwell_operator.split_eh(fields)

    if write_output:
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

            print(step, event.t, norm(u=e[0]), norm(u=e[1]),
                    norm(u=h[0]), norm(u=h[1]),
                    time()-t_last_step)
            if step % 10 == 0:
                if write_output:
                    e, h = maxwell_operator.split_eh(event.state_component)
                    vis.write_vtk_file(
                        f"fld-cavities-{step:04d}.vtu",
                        [
                            ("e", e),
                            ("h", h),
                        ]
                    )
            t_last_step = time()


if __name__ == "__main__":
    main(3)
