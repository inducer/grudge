from __future__ import division, absolute_import

__copyright__ = "Copyright (C) 2017 Bogdan Enache"

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
import pyopencl as cl  # noqa
import pyopencl.array  # noqa
import pyopencl.clmath  # noqa

import pytest  # noqa

from pyopencl.tools import (  # noqa
                pytest_generate_tests_for_pyopencl as pytest_generate_tests)

import logging
logger = logging.getLogger(__name__)

from grudge import sym, bind, DGDiscretizationWithBoundaries
from pytools.obj_array import join_fields


def main(write_output=True, order=4):
    cl_ctx = cl.create_some_context()
    queue = cl.CommandQueue(cl_ctx)

    dim = 2

    resolution = 10
    from meshmode.mesh.generation import generate_regular_rect_mesh
    mesh = generate_regular_rect_mesh(a=(-0.5, -0.5), b=(0.5, 0.5),
            n=(resolution, resolution), order=order)

    dt_factor = 5
    h = 1/resolution

    sym_x = sym.nodes(2)

    advec_v = join_fields(-1*sym_x[1], sym_x[0]) / 2

    flux_type = "upwind"

    source_center = np.array([0.1, 0.1])
    source_width = 0.05

    sym_x = sym.nodes(2)
    sym_source_center_dist = sym_x - source_center

    def f(x):
        return sym.exp(
                    -np.dot(sym_source_center_dist, sym_source_center_dist)
                    / source_width**2)

    def u_analytic(x):
        return 0

    from grudge.models.advection import VariableCoefficientAdvectionOperator
    from meshmode.discretization.poly_element import QuadratureSimplexGroupFactory  # noqa

    discr = DGDiscretizationWithBoundaries(cl_ctx, mesh, order=order,
            quad_tag_to_group_factory={
                #"product": None,
                "product": QuadratureSimplexGroupFactory(order=4*order)
                })

    op = VariableCoefficientAdvectionOperator(2, advec_v,
        u_analytic(sym.nodes(dim, sym.BTAG_ALL)), quad_tag="product",
        flux_type=flux_type)

    bound_op = bind(
            discr, op.sym_operator(),
            #debug_flags=["dump_sym_operator_stages"]
            )

    u = bind(discr, f(sym.nodes(dim)))(queue, t=0)

    def rhs(t, u):
        return bound_op(queue, t=t, u=u)

    final_time = 50
    dt = dt_factor * h/order**2
    nsteps = (final_time // dt) + 1
    dt = final_time/nsteps + 1e-15

    from grudge.shortcuts import set_up_rk4
    dt_stepper = set_up_rk4("u", dt, u, rhs)

    from grudge.shortcuts import make_visualizer
    vis = make_visualizer(discr, vis_order=order)

    step = 0

    for event in dt_stepper.run(t_end=final_time):
        if isinstance(event, dt_stepper.StateComputed):

            step += 1
            if step % 10 == 0:
                print(step)

                vis.write_vtk_file("fld-%04d.vtu" % step,
                        [("u", event.state_component)])


if __name__ == "__main__":
    main()
