"""Minimal example of a grudge driver."""

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


def rk4_step(y, t, h, f):
    k1 = f(t, y)
    k2 = f(t+h/2, y + h/2*k1)
    k3 = f(t+h/2, y + h/2*k2)
    k4 = f(t+h, y + h*k3)
    return y + h/6*(k1 + 2*k2 + 2*k3 + k4)


def set_up_rk4(field_var_name, dt, fields, rhs, t_start=0.0):
    from leap.rk import LSRK4MethodBuilder
    from dagrt.codegen import PythonCodeGenerator

    dt_method = LSRK4MethodBuilder(component_id=field_var_name)
    dt_code = dt_method.generate()
    dt_stepper_class = PythonCodeGenerator("TimeStep").get_class(dt_code)
    dt_stepper = dt_stepper_class({"<func>"+dt_method.component_id: rhs})

    dt_stepper.set_up(
            t_start=t_start, dt_start=dt,
            context={dt_method.component_id: fields})

    return dt_stepper


def make_visualizer(dcoll, vis_order=None, **kwargs):
    from meshmode.discretization.visualization import make_visualizer
    return make_visualizer(
            dcoll._setup_actx,
            dcoll.discr_from_dd("vol"), vis_order, **kwargs)


def make_boundary_visualizer(dcoll, vis_order=None, **kwargs):
    from meshmode.discretization.visualization import make_visualizer
    from meshmode.mesh import BTAG_ALL

    return make_visualizer(
            dcoll._setup_actx, dcoll.discr_from_dd(BTAG_ALL),
            vis_order, **kwargs)
