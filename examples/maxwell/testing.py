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
from grudge import bind


def sumpy_test_3D(order, cl_ctx, queue):

    epsilon = 1
    mu = 1

    kx, ky, kz = factors = [n*np.pi for n in (1, 2, 2)]
    k = np.sqrt(sum(f**2 for f in factors))

    patch = spy.CalculusPatch((0.4, 0.3, 0.4))

    from grudge.discretization import PointsDiscretization
    pdiscr = PointsDiscretization(cl.array.to_device(queue, patch.points))

    from grudge.models.em import get_rectangular_cavity_mode
    sym_mode = get_rectangular_cavity_mode(1, (1, 2, 2))
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
