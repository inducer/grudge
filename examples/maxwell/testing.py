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
import sumpy.point_calculus as spy
from grudge.shortcuts import set_up_rk4
from grudge import sym, bind, Discretization

from analytic_solutions import (
        get_rectangular_3D_cavity_mode,
        get_rectangular_2D_cavity_mode,
        )


def analytic_test(dims, n, order, cl_ctx, queue):

    from meshmode.mesh.generation import generate_regular_rect_mesh
    mesh = generate_regular_rect_mesh(
            a=(0.0,)*dims,
            b=(1.0,)*dims,
            n=(n,)*dims)

    discr = Discretization(cl_ctx, mesh, order=order)

    if 0:
        epsilon0 = 8.8541878176e-12  # C**2 / (N m**2)
        mu0 = 4*np.pi*1e-7  # N/A**2.
        epsilon = 1*epsilon0
        mu = 1*mu0
    else:
        epsilon = 1
        mu = 1

    if dims == 2:
        sym_mode = get_rectangular_2D_cavity_mode(1, (1, 2))
    else:
        sym_mode = get_rectangular_3D_cavity_mode(1, (1, 2, 2))

    analytic_sol = bind(discr, sym_mode)
    fields = analytic_sol(queue, t=0, epsilon=epsilon, mu=mu)

    from grudge.models.em import MaxwellOperator
    op = MaxwellOperator(epsilon, mu, flux_type=0.5, dimensions=dims)
    op.check_bc_coverage(mesh)
    bound_op = bind(discr, op.sym_operator())

    def rhs(t, w):
        return bound_op(queue, t=t, w=w)

    dt = 0.002
    final_t = dt * 300
    nsteps = int(final_t/dt)

    dt_stepper = set_up_rk4("w", dt, fields, rhs)

    print("dt=%g nsteps=%d" % (dt, nsteps))

    norm = bind(discr, sym.norm(2, sym.var("u")))

    step = 0
    for event in dt_stepper.run(t_end=final_t):
        if isinstance(event, dt_stepper.StateComputed):
            assert event.component_id == "w"
            esc = event.state_component

            step += 1

            if step % 10 == 0:
                print(step)

            if step == 20:
                sol = analytic_sol(queue, mu=mu, epsilon=epsilon, t=step * dt)
                print(norm(queue, u=(esc[0] - sol[0])) / norm(queue, u=sol[0]))
                print(norm(queue, u=(esc[1] - sol[1])) / norm(queue, u=sol[1]))
                print(norm(queue, u=(esc[2] - sol[2])) / norm(queue, u=sol[2]))
                print(norm(queue, u=(esc[3] - sol[3])) / norm(queue, u=sol[3]))
                print(norm(queue, u=(esc[4] - sol[4])) / norm(queue, u=sol[4]))
                print(norm(queue, u=(esc[5] - sol[5])) / norm(queue, u=sol[5]))
                return


def sumpy_test_3D(order, cl_ctx, queue):

    epsilon = 1
    mu = 1

    kx, ky, kz = factors = [n*np.pi/a for n,  a in zip((1, 2, 2), (1, 1, 1))]
    k = np.sqrt(sum(f**2 for f in factors))

    patch = spy.CalculusPatch((0.4, 0.3, 0.4))

    from grudge.discretization import PointsDiscretization
    pdiscr = PointsDiscretization(cl.array.to_device(queue, patch.points))

    sym_mode = get_rectangular_3D_cavity_mode(1, (1, 2, 2))
    fields = bind(pdiscr, sym_mode)(queue, t=0, epsilon=epsilon, mu=mu)

    for i in range(len(fields)):
        if isinstance(fields[i], (int, float)):
            fields[i] = np.zeros(patch.points.shape[1])
        else:
            fields[i] = fields[i].get()

    e = np.array([fields[0], fields[1], fields[2]])
    h = np.array([fields[3], fields[4], fields[5]])
    print(frequency_domain_maxwell(patch, e, h, k))
    return


def frequency_domain_maxwell(cpatch, e, h, k):
    mu = 1
    epsilon = 1
    c = 1/np.sqrt(mu*epsilon)
    omega = k*c
    b = mu*h
    d = epsilon*e
    # https://en.wikipedia.org/w/index.php?title=Maxwell%27s_equations&oldid=798940325#Macroscopic_formulation
    # assumed time dependence exp(-1j*omega*t)
    # Agrees with Jackson, Third Ed., (8.16)
    resid_faraday = np.vstack(cpatch.curl(e)) - 1j * omega * b
    resid_ampere = np.vstack(cpatch.curl(h)) + 1j * omega * d
    resid_div_e = cpatch.div(e)
    resid_div_h = cpatch.div(h)
    return (resid_faraday, resid_ampere,  resid_div_e,    resid_div_h)


if __name__ == "__main__":
    cl_ctx = cl.create_some_context()
    queue = cl.CommandQueue(cl_ctx)

    print("Doing Sumpy Test")
    sumpy_test_3D(4, cl_ctx, queue)
    print("Doing 2D")
    analytic_test(2, 8, 4, cl_ctx, queue)
    analytic_test(2, 16, 4, cl_ctx, queue)
    print("Doing 3D")
    analytic_test(3, 4, 4, cl_ctx, queue)
    analytic_test(3, 8, 4, cl_ctx, queue)
