"""Minimal example of a grudge driver."""

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
from grudge.shortcuts import set_up_rk4
from grudge import sym, bind, DGDiscretizationWithBoundaries

from meshmode.array_context import PyOpenCLArrayContext


def main(write_output=True, order=4):
    cl_ctx = cl.create_some_context()
    queue = cl.CommandQueue(cl_ctx)
    actx = PyOpenCLArrayContext(queue)

    dims = 2
    from meshmode.mesh.generation import generate_regular_rect_mesh
    mesh = generate_regular_rect_mesh(
            a=(-0.5,)*dims,
            b=(0.5,)*dims,
            n=(20,)*dims)

    discr = DGDiscretizationWithBoundaries(actx, mesh, order=order)

    source_center = np.array([0.1, 0.22, 0.33])[:mesh.dim]
    source_width = 0.05
    source_omega = 3

    sym_x = sym.nodes(mesh.dim)
    sym_source_center_dist = sym_x - source_center
    sym_t = sym.ScalarVariable("t")
    c = sym.If(sym.Comparison(
                np.dot(sym_x, sym_x), "<", 0.15),
                np.float32(0.1), np.float32(0.2))

    from grudge.models.wave import VariableCoefficientWeakWaveOperator
    from meshmode.mesh import BTAG_ALL, BTAG_NONE
    op = VariableCoefficientWeakWaveOperator(c,
            discr.dim,
            source_f=(
                sym.sin(source_omega*sym_t)
                * sym.exp(
                    -np.dot(sym_source_center_dist, sym_source_center_dist)
                    / source_width**2)),
            dirichlet_tag=BTAG_NONE,
            neumann_tag=BTAG_NONE,
            radiation_tag=BTAG_ALL,
            flux_type="upwind")

    from pytools.obj_array import flat_obj_array
    fields = flat_obj_array(discr.zeros(actx),
            [discr.zeros(actx) for i in range(discr.dim)])

    op.check_bc_coverage(mesh)

    c_eval = bind(discr, c)(actx)

    bound_op = bind(discr, op.sym_operator())

    def rhs(t, w):
        return bound_op(t=t, w=w)

    if mesh.dim == 2:
        dt = 0.04 * 0.3
    elif mesh.dim == 3:
        dt = 0.02 * 0.1

    dt_stepper = set_up_rk4("w", dt, fields, rhs)

    final_t = 1
    nsteps = int(final_t/dt)
    print("dt=%g nsteps=%d" % (dt, nsteps))

    from grudge.shortcuts import make_visualizer
    vis = make_visualizer(discr, vis_order=order)

    step = 0

    norm = bind(discr, sym.norm(2, sym.var("u")))

    from time import time
    t_last_step = time()

    for event in dt_stepper.run(t_end=final_t):
        if isinstance(event, dt_stepper.StateComputed):
            assert event.component_id == "w"

            step += 1

            print(step, event.t, norm(u=event.state_component[0]),
                    time()-t_last_step)
            if step % 10 == 0:
                vis.write_vtk_file("fld-var-propogation-speed-%04d.vtu" % step,
                        [
                            ("u", event.state_component[0]),
                            ("v", event.state_component[1:]),
                            ("c", c_eval),
                            ])
            t_last_step = time()


if __name__ == "__main__":
    main()
