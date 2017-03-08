# grudge - the Hybrid'n'Easy DG Environment
# Copyright (C) 2007 Andreas Kloeckner
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.


import numpy as np
import pyopencl as cl  # noqa
import pyopencl.array  # noqa
import pyopencl.clmath  # noqa

import pytest  # noqa

from pyopencl.tools import (  # noqa
                pytest_generate_tests_for_pyopencl as pytest_generate_tests)

import logging
logger = logging.getLogger(__name__)

from grudge import sym, bind, Discretization

import numpy.linalg as la




def main(write_output=True, order=4):
    cl_ctx = cl.create_some_context()
    queue = cl.CommandQueue(cl_ctx)

    dim = 2


    from meshmode.mesh.generation import generate_regular_rect_mesh
    mesh = generate_regular_rect_mesh(a=(-0.5, -0.5), b=(0.5, 0.5),
            n=(20, 20), order=order)

    dt_factor = 4
    h = 1/20

    discr = Discretization(cl_ctx, mesh, order=order)

    c = np.array([0.1,0.1])
    norm_c = la.norm(c)


    flux_type = "central"
         

    def f(x):
        return sym.sin(3*x)

    def u_analytic(x):
        return f(-np.dot(c, x)/norm_c+sym.var("t", sym.DD_SCALAR)*norm_c)

    from grudge.models.advection import WeakAdvectionOperator
    from meshmode.mesh import BTAG_ALL, BTAG_NONE
    
    discr = Discretization(cl_ctx, mesh, order=order)
    op = WeakAdvectionOperator(c,
        inflow_u=u_analytic(sym.nodes(dim, sym.BTAG_ALL)),
        flux_type=flux_type)

    bound_op = bind(discr, op.sym_operator())

    u = bind(discr, u_analytic(sym.nodes(dim)))(queue, t=0)

    def rhs(t, u):
        return bound_op(queue, t=t, u=u)

    final_time = 0.3

    dt = dt_factor * h/order**2
    nsteps = (final_time // dt) + 1
    dt = final_time/nsteps + 1e-15


    from grudge.shortcuts import set_up_rk4
    dt_stepper = set_up_rk4("u", dt, u, rhs)

    last_u = None

    from grudge.shortcuts import make_visualizer
    vis = make_visualizer(discr, vis_order=order)

    step = 0

    norm = bind(discr, sym.norm(2, sym.var("u")))

    for event in dt_stepper.run(t_end=final_time):
        if isinstance(event, dt_stepper.StateComputed):

            step += 1

            #print(step, event.t, norm(queue, u=event.state_component[0]))
            vis.write_vtk_file("fld-%04d.vtu" % step,
                    [  ("u", event.state_component) ])







if __name__ == "__main__":
    main()


