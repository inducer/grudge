"""Minimal example of a grudge driver."""

from __future__ import division, print_function

__copyright__ = "Copyright (C) 2015 Andreas Kloeckner"

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
from grudge.shortcuts import make_discretization, set_up_rk4


def main(write_output=True):
    from meshmode.mesh.generation import generate_regular_rect_mesh
    mesh = generate_regular_rect_mesh(a=(-0.5, -0.5), b=(0.5, 0.5))

    discr = make_discretization(mesh, order=4)

    #from grudge.visualization import VtkVisualizer
    #vis = VtkVisualizer(discr, None, "fld")

    source_center = np.array([0.1, 0.22])
    source_width = 0.05
    source_omega = 3

    from grudge import sym
    sym_x = sym.nodes(2)
    sym_source_center_dist = sym_x - source_center
    sym_sin = sym.CFunction("sin")
    sym_exp = sym.CFunction("sin")
    sym_t = sym.ScalarParameter("t")

    from grudge.models.wave import StrongWaveOperator
    from meshmode.mesh import BTAG_ALL, BTAG_NONE
    op = StrongWaveOperator(-0.1, discr.dim,
            source_f=(
                sym_sin(source_omega*sym_t)
                * sym_exp(
                    -np.dot(sym_source_center_dist, sym_source_center_dist)
                    / source_width**2)),
            dirichlet_tag=BTAG_NONE,
            neumann_tag=BTAG_NONE,
            radiation_tag=BTAG_ALL,
            flux_type="upwind")

    queue = cl.CommandQueue(discr.cl_context)
    from pytools.obj_array import join_fields
    fields = join_fields(discr.zeros(queue),
            [discr.zeros(queue) for i in range(discr.dim)])

    # FIXME
    #dt = op.estimate_rk4_timestep(discr, fields=fields)

    dt = 0.001
    dt_stepper = set_up_rk4(dt, fields)

    final_t = 10
    nsteps = int(final_t/dt)
    print("dt=%g nsteps=%d" % (dt, nsteps))

    step = 0

    for event in dt_stepper.run(t_end=final_t):
        if isinstance(event, dt_stepper.StateComputed):
            assert event.component_id == "y"

            step += 1

            # if step % 10 == 0 and write_output:
            #     print(step, event.t, discr.norm(fields[0]))
            #     visf = vis.make_file("fld-%04d" % step)

            #     vis.add_data(visf,
            #             [("u", fields[0]), ("v", fields[1:]), ],
            #             time=event.t, step=step)
            #     visf.close()

    #vis.close()


if __name__ == "__main__":
    main()
