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
from grudge.shortcuts import set_up_rk4
from grudge import sym, bind, Discretization

from analytic_solutions import (
        get_rectangular_3D_cavity_mode,
        get_rectangular_2D_cavity_mode,
        )


def main(dims, write_output=True, order=4):
    cl_ctx = cl.create_some_context()
    queue = cl.CommandQueue(cl_ctx)

    from meshmode.mesh.generation import generate_regular_rect_mesh
    mesh = generate_regular_rect_mesh(
            a=(0.0,)*dims,
            b=(1.0,)*dims,
            n=(5,)*dims)

    discr = Discretization(cl_ctx, mesh, order=order)

    if 0:
        epsilon0 = 8.8541878176e-12  # C**2 / (N m**2)
        mu0 = 4*np.pi*1e-7  # N/A**2.
        epsilon = 1*epsilon0
        mu = 1*mu0
    else:
        epsilon = 1
        mu = 1

    from grudge.models.em import MaxwellOperator
    op = MaxwellOperator(epsilon, mu, flux_type=0.5, dimensions=dims)

    queue = cl.CommandQueue(discr.cl_context)

    if dims == 3:
        sym_mode = get_rectangular_3D_cavity_mode(1, (1, 2, 2))
        fields = bind(discr, sym_mode)(queue, t=0, epsilon=epsilon, mu=mu)
    else:
        sym_mode = get_rectangular_2D_cavity_mode(1, (2, 3))
        fields = bind(discr, sym_mode)(queue, t=0)

    # FIXME
    #dt = op.estimate_rk4_timestep(discr, fields=fields)

    op.check_bc_coverage(mesh)

    # print(sym.pretty(op.sym_operator()))
    bound_op = bind(discr, op.sym_operator())

    def rhs(t, w):
        return bound_op(queue, t=t, w=w)

    if mesh.dim == 2:
        dt = 0.004
    elif mesh.dim == 3:
        dt = 0.002

    dt_stepper = set_up_rk4("w", dt, fields, rhs)

    final_t = dt * 600
    nsteps = int(final_t/dt)

    print("dt=%g nsteps=%d" % (dt, nsteps))

    from grudge.shortcuts import make_visualizer
    vis = make_visualizer(discr, vis_order=order)

    step = 0

    norm = bind(discr, sym.norm(2, sym.var("u")))

    from time import time
    t_last_step = time()

    e, h = op.split_eh(fields)

    if 1:
        vis.write_vtk_file("fld-%04d.vtu" % step,
                [
                    ("e", e),
                    ("h", h),
                    ])

    for event in dt_stepper.run(t_end=final_t):
        if isinstance(event, dt_stepper.StateComputed):
            assert event.component_id == "w"

            step += 1

            print(step, event.t, norm(queue, u=e[0]), norm(queue, u=e[1]),
                    norm(queue, u=h[0]), norm(queue, u=h[1]),
                    time()-t_last_step)
            if step % 10 == 0:
                e, h = op.split_eh(event.state_component)
                vis.write_vtk_file("fld-%04d.vtu" % step,
                        [
                            ("e", e),
                            ("h", h),
                            ])
            t_last_step = time()


if __name__ == "__main__":
    main(3)
