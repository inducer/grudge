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

from grudge import sym, bind, DGDiscretizationWithBoundaries

import numpy.linalg as la




def main(order=1):
    cl_ctx = cl.create_some_context()
    queue = cl.CommandQueue(cl_ctx)

    dim = 2


    from meshmode.mesh.generation import n_gon, generate_regular_rect_mesh
    #mesh = n_gon(3,np.array([0.1,0.2,0.4,0.5,0.6,0.7,0.9]))
    mesh = generate_regular_rect_mesh(a=(-0.5, -0.5), b=(0.5, 0.5),
            n=(3, 3), order=7)




    pquad_dd = sym.DOFDesc("vol", "product")
    to_pquad = sym.interp("vol", pquad_dd)
    from_pquad_stiffness_t = sym.stiffness_t(dim, pquad_dd, "vol")
    from_pquad_mass = sym.MassOperator()

    def lin(x):
        return x[0]

    def non_lin(x):
        return x[1] * x[0] * x[0] + 3* x[0] * x[0] - 4

    def f(x, fun):
        #return sym.MassOperator()(x)
        #return np.dot(sym.stiffness_t(2), to_pquad(fun(x)))
        return np.dot(from_pquad_stiffness_t, fun(to_pquad(x)))

    def reg(x, fun):
        #return sym.MassOperator()(x)
        #return np.dot(sym.stiffness_t(2), to_pquad(fun(x)))
        return np.dot(sym.stiffness_t(2), fun(x))
    
    discr = DGDiscretizationWithBoundaries(cl_ctx, mesh, order=order, quad_min_degrees={"product": 2 * order} )


    #ones = bind(discr, sym.nodes(dim)[0] / sym.nodes(dim)[0])(queue, t=0)
    nds = sym.nodes(dim)
    #u = bind(discr, to_pquad(nds))(queue, t=0)
    #u = bind(discr, reg(nds, lin))(queue, t=0)
    u = bind(discr, f(nds, lin))(queue, t=0)

    print(u)




    #vis.write_vtk_file("fld-000o.vtu", [  ("u", o) ])
    #vis.write_vtk_file("fld-000u.vtu", [  ("u", u) ])







if __name__ == "__main__":
    main(4)


