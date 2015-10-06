"""Minimal example of a grudge driver."""

from __future__ import division, print_function

__copyright__ = "Copyright (C) 2009 Andreas Kloeckner"

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


def make_discretization(mesh, order, **kwargs):
    from meshmode.discretization import Discretization
    from meshmode.discretization.poly_element import \
            PolynomialWarpAndBlendGroupFactory

    cl_ctx = kwargs.pop("cl_ctx", None)
    if cl_ctx is None:
        import pyopencl as cl
        cl_ctx = cl.create_some_context()

    return Discretization(cl_ctx, mesh,
            PolynomialWarpAndBlendGroupFactory(order=order))


def set_up_rk4(dt, fields, t_start=0):
    from leap.method.rk import LSRK4Method
    from leap.vm.codegen import PythonCodeGenerator

    dt_method = LSRK4Method(component_id="y")
    dt_stepper = PythonCodeGenerator().get_class(dt_method.generate())

    dt_stepper.set_up(t_start=t_start, dt_start=dt, context={"y": fields})

    return dt_stepper
