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

from collections.abc import Callable, Iterator
from functools import partial
from typing import Any, ClassVar, NamedTuple

from arraycontext import BcastUntilActxArray
from arraycontext.context import ArrayContext
from pytools import memoize_in

from grudge.dof_desc import DD_VOLUME_ALL


# {{{ legacy leap-like interface

class StateComputedEvent(NamedTuple):
    t: float
    time_id: str
    component_id: str
    state_component: Any


class LSRK4Method:
    StateComputed: ClassVar[type[StateComputedEvent]] = StateComputedEvent

    def __init__(self,
                 f: Callable[[float, Any], Any],
                 y0: Any,
                 component_id: str,
                 dt: float,
                 tstart: float = 0.0) -> None:
        self.f = f
        self.y0 = y0
        self.dt = dt
        self.component_id = component_id
        self.tstart = tstart

    def run(self,
            t_end: float | None = None,
            max_steps: int | None = None) -> Iterator[StateComputedEvent]:
        t = self.tstart
        y = self.y0

        nsteps = 0
        while True:
            if t_end is not None and t >= t_end:
                return

            if max_steps is not None and nsteps >= max_steps:
                return

            y = lsrk4_step(y, t, self.dt, self.f)
            if t_end is not None:
                t += min(self.dt, t_end - t)
            else:
                t += self.dt

            yield StateComputedEvent(t=t,
                                     time_id="",
                                     component_id=self.component_id,
                                     state_component=y)

            nsteps += 1


def stability_function(rk_a, rk_b):
    """Given the matrix *A* and the 'output coefficients' *b* from a Butcher
    tableau, return the stability function of the method as a
    :class:`sympy.core.expr.Expr`.
    """
    import sympy as sp

    num_stages = len(rk_a)
    rk_a_mat = [
            list(row) + [0]*(num_stages-len(row))
            for row in rk_a]
    a = sp.Matrix(rk_a_mat)
    b = sp.Matrix(rk_b)
    eye = sp.eye(num_stages)
    ones = sp.ones(num_stages, 1)

    h_lambda = sp.Symbol("h_lambda")

    # https://en.wikipedia.org/w/index.php?title=Runge%E2%80%93Kutta_methods&oldid=1065948515#Stability
    return h_lambda, (
            (eye - h_lambda * a + h_lambda * ones * b.T).det()
            / (eye - h_lambda * a).det())


RK4_A = (
    (),
    (1/2,),
    (0, 1/2,),
    (0, 0, 1,),
)

RK4_B = (1/6, 1/3, 1/3, 1/6)

LSRK4_A = (
    0.0,
    -567301805773 / 1357537059087,
    -2404267990393 / 2016746695238,
    -3550918686646 / 2091501179385,
    -1275806237668 / 842570457699,
    )

LSRK4_B = (
    1432997174477 / 9575080441755,
    5161836677717 / 13612068292357,
    1720146321549 / 2090206949498,
    3134564353537 / 4481467310338,
    2277821191437 / 14882151754819,
    )

LSRK4_C = (
    0.0,
    1432997174477/9575080441755,
    2526269341429/6820363962896,
    2006345519317/3224310063776,
    2802321613138/2924317926251,
    # 1,
    )

# }}}


def rk4_step(y, t, h, f):
    k1 = f(t, y)
    k2 = f(t+h/2, y + h/2*k1)
    k3 = f(t+h/2, y + h/2*k2)
    k4 = f(t+h, y + h*k3)
    return y + h/6*(k1 + 2*k2 + 2*k3 + k4)


def lsrk4_step(y, t, h, f):
    p = k = y
    for a, b, c in zip(LSRK4_A, LSRK4_B, LSRK4_C, strict=True):
        k = a * k + h * f(t + c * h, p)
        p = p + b * k

    return p


def _lsrk45_update(actx: ArrayContext, y, a, b, h, rhs_val, residual=None):
    bcast = partial(BcastUntilActxArray, actx)
    if residual is None:
        residual = bcast(h) * rhs_val
    else:
        residual = bcast(a) * residual + bcast(h) * rhs_val

    y = y + bcast(b) * residual
    from pytools.obj_array import make_obj_array
    return make_obj_array([y, residual])


def compiled_lsrk45_step(actx: ArrayContext, y, t, h, f):
    @memoize_in(actx, (compiled_lsrk45_step, "update"))
    def get_state_updater():
        return actx.compile(partial(_lsrk45_update, actx))

    update = get_state_updater()

    residual = None

    for a, b, c in zip(LSRK4_A, LSRK4_B, LSRK4_C, strict=True):
        rhs_val = f(t + c*h, y)
        if residual is None:
            y, residual = update(y, a, b, h, rhs_val)
        else:
            y, residual = update(y, a, b, h, rhs_val, residual)

    return y


def set_up_rk4(field_var_name, dt, fields, rhs, t_start=0.0):
    return LSRK4Method(
        rhs,
        fields,
        field_var_name,
        dt,
        tstart=t_start)


def make_visualizer(dcoll, vis_order=None, volume_dd=None, **kwargs):
    from meshmode.discretization.visualization import make_visualizer
    if volume_dd is None:
        volume_dd = DD_VOLUME_ALL

    return make_visualizer(
            dcoll._setup_actx,
            dcoll.discr_from_dd(volume_dd), vis_order, **kwargs)


def make_boundary_visualizer(dcoll, vis_order=None, **kwargs):
    from meshmode.discretization.visualization import make_visualizer
    from meshmode.mesh import BTAG_ALL

    return make_visualizer(
            dcoll._setup_actx, dcoll.discr_from_dd(BTAG_ALL),
            vis_order, **kwargs)
