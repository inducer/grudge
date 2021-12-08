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

import numpy as np

from dataclasses import dataclass


@dataclass(frozen=True)
class LSRKCoefficients:
    """Dataclass which defines a given low-storage Runge-Kutta (LSRK) scheme.
    The methods are determined by the provided `A`, `B` and `C` coefficient arrays.
    """

    A: np.ndarray
    B: np.ndarray
    C: np.ndarray


LSRK144NiegemannDiehlBuschCoefs = LSRKCoefficients(
    A=np.array([
        0.,
        -0.7188012108672410,
        -0.7785331173421570,
        -0.0053282796654044,
        -0.8552979934029281,
        -3.9564138245774565,
        -1.5780575380587385,
        -2.0837094552574054,
        -0.7483334182761610,
        -0.7032861106563359,
        0.0013917096117681,
        -0.0932075369637460,
        -0.9514200470875948,
        -7.1151571693922548]),
    B=np.array([
        0.0367762454319673,
        0.3136296607553959,
        0.1531848691869027,
        0.0030097086818182,
        0.3326293790646110,
        0.2440251405350864,
        0.3718879239592277,
        0.6204126221582444,
        0.1524043173028741,
        0.0760894927419266,
        0.0077604214040978,
        0.0024647284755382,
        0.0780348340049386,
        5.5059777270269628]),
    C=np.array([
        0.,
        0.0367762454319673,
        0.1249685262725025,
        0.2446177702277698,
        0.2476149531070420,
        0.2969311120382472,
        0.3978149645802642,
        0.5270854589440328,
        0.6981269994175695,
        0.8190890835352128,
        0.8527059887098624,
        0.8604711817462826,
        0.8627060376969976,
        0.8734213127600976]))


LSRK54CarpenterKennedyCoefs = LSRKCoefficients(
    A=np.array([
        0.,
        -567301805773/1357537059087,
        -2404267990393/2016746695238,
        -3550918686646/2091501179385,
        -1275806237668/842570457699]),
    B=np.array([
        1432997174477/9575080441755,
        5161836677717/13612068292357,
        1720146321549/2090206949498,
        3134564353537/4481467310338,
        2277821191437/14882151754819]),
    C=np.array([
        0.,
        1432997174477/9575080441755,
        2526269341429/6820363962896,
        2006345519317/3224310063776,
        2802321613138/2924317926251]))


EulerCoefs = LSRKCoefficients(
    A=np.array([0.]),
    B=np.array([1.]),
    C=np.array([0.]))


def lsrk_step(coefs, state, t, dt, rhs):
    """Take one step using a low-storage Runge-Kutta method."""
    k = 0.0 * state
    for i in range(len(coefs.A)):
        k = coefs.A[i]*k + dt*rhs(t + coefs.C[i]*dt, state)
        state += coefs.B[i]*k
    return state


def lsrk144_step(state, t, dt, rhs):
    """Take one step using an explicit 14-stage, 4th-order, LSRK method.

    This method is derived by Niegemann, Diehl, and Busch (2012), with
    an optimal stability region for advection-dominated flows.
    """
    return lsrk_step(LSRK144NiegemannDiehlBuschCoefs, state, t, dt, rhs)


def lsrk54_step(state, t, dt, rhs):
    """Take one step using an explicit 5-stage, 4th-order, LSRK method."""
    return lsrk_step(LSRK54CarpenterKennedyCoefs, state, t, dt, rhs)


def euler_step(state, t, dt, rhs):
    """Take one step using the explicit, 1st-order accurate, Euler method."""
    return lsrk_step(EulerCoefs, state, t, dt, rhs)


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
