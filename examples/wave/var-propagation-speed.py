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

from pytools.obj_array import flat_obj_array

import grudge.op as op


def main(write_output=False, order=4):
    cl_ctx = cl.create_some_context()
    queue = cl.CommandQueue(cl_ctx)
    actx = PyOpenCLArrayContext(
        queue,
        allocator=cl_tools.MemoryPool(cl_tools.ImmediateAllocator(queue))
    )

    dims = 2
    from meshmode.mesh.generation import generate_regular_rect_mesh
    mesh = generate_regular_rect_mesh(
            a=(-0.5,)*dims,
            b=(0.5,)*dims,
            nelements_per_axis=(20,)*dims)

    dcoll = DiscretizationCollection(actx, mesh, order=order)

    def source_f(actx, dcoll, t=0):
        source_center = np.array([0.1, 0.22, 0.33])[:dcoll.dim]
        source_width = 0.05
        source_omega = 3
        nodes = thaw(op.nodes(dcoll), actx)
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

    x = thaw(op.nodes(dcoll), actx)
    ones = dcoll.zeros(actx) + 1
    c = actx.np.where(np.dot(x, x) < 0.15, 0.1 * ones, 0.2 * ones)

    from grudge.models.wave import VariableCoefficientWeakWaveOperator
    from meshmode.mesh import BTAG_ALL, BTAG_NONE

    wave_op = VariableCoefficientWeakWaveOperator(
        dcoll,
        c,
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

    if mesh.dim == 2:
        dt = 0.04 * 0.3
    elif mesh.dim == 3:
        dt = 0.02 * 0.1

    dt_stepper = set_up_rk4("w", dt, fields, rhs)

    final_t = 1
    nsteps = int(final_t/dt)
    print("dt=%g nsteps=%d" % (dt, nsteps))

    from grudge.shortcuts import make_visualizer
    vis = make_visualizer(dcoll)

    step = 0

    def norm(u):
        return op.norm(dcoll, u, 2)

    from time import time
    t_last_step = time()

    if write_output:
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
                print(f"step: {step} t: {time()-t_last_step} "
                      f"L2: {norm(u=event.state_component[0])}")
                if write_output:
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
    main()
