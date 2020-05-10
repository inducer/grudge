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

import pyopencl as cl
import pyopencl.array
import pyopencl.clmath

from grudge import bind, sym
from pytools.obj_array import join_fields

import logging
logger = logging.getLogger(__name__)


def main(ctx_factory, dim=2, order=4, product_tag=None, visualize=True):
    cl_ctx = ctx_factory()
    queue = cl.CommandQueue(cl_ctx)

    # {{{ parameters

    # domain [0, d]^dim
    d = 1.0
    # number of points in each dimension
    npoints = 25
    # grid spacing
    h = d / npoints

    # cfl
    dt_factor = 1.0
    # finale time
    final_time = 0.5
    # time steps
    dt = dt_factor * h/order**2
    nsteps = int(final_time // dt) + 1
    dt = final_time/nsteps + 1.0e-15

    # flux
    flux_type = "upwind"
    # velocity field
    sym_x = sym.nodes(dim)
    if dim == 1:
        c = sym_x
    else:
        # solid body rotation
        c = join_fields(
                np.pi * (d/2 - sym_x[1]),
                np.pi * (sym_x[0] - d/2),
                0)[:dim]

    # }}}

    # {{{ discretization

    from meshmode.mesh.generation import generate_regular_rect_mesh
    mesh = generate_regular_rect_mesh(
            a=(0,)*dim, b=(d,)*dim,
            n=(npoints,)*dim,
            order=order)

    from meshmode.discretization.poly_element import \
            QuadratureSimplexGroupFactory

    if product_tag:
        quad_tag_to_group_factory = {
                product_tag: QuadratureSimplexGroupFactory(order=4*order)
                }
    else:
        quad_tag_to_group_factory = {}

    from grudge import DGDiscretizationWithBoundaries
    discr = DGDiscretizationWithBoundaries(cl_ctx, mesh, order=order,
            quad_tag_to_group_factory=quad_tag_to_group_factory)

    # }}}

    # {{{ symbolic operators

    # gaussian parameters
    source_center = np.array([0.5, 0.75, 0.0])[:dim]
    source_width = 0.05
    dist_squared = np.dot(sym_x - source_center, sym_x - source_center)

    def f_gaussian(x):
        return sym.exp(-dist_squared / source_width**2)

    def u_bc(x):
        return 0.0

    from grudge.models.advection import VariableCoefficientAdvectionOperator
    op = VariableCoefficientAdvectionOperator(
            c,
            u_bc(sym.nodes(dim, sym.BTAG_ALL)),
            quad_tag=product_tag,
            flux_type=flux_type)

    bound_op = bind(discr, op.sym_operator())
    u = bind(discr, f_gaussian(sym.nodes(dim)))(queue, t=0)

    def rhs(t, u):
        return bound_op(queue, t=t, u=u)

    # }}}

    # {{{ time stepping

    from grudge.shortcuts import set_up_rk4
    dt_stepper = set_up_rk4("u", dt, u, rhs)

    from weak import Plotter
    plot = Plotter(queue, discr, order, visualize=visualize)

    step = 0
    norm = bind(discr, sym.norm(2, sym.var("u")))
    for event in dt_stepper.run(t_end=final_time):
        if not isinstance(event, dt_stepper.StateComputed):
            continue

        if step % 5 == 0:
            norm_u = norm(queue, u=event.state_component)

        step += 1
        logger.info("[%04d] t = %.5f |u| = %.5e", step, event.t, norm_u)
        plot(event, "fld-var-velocity-%04d" % step)

    # }}}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dim", default=2, type=int)
    parser.add_argument("--qtag", default="product")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    main(cl.create_some_context,
            dim=args.dim,
            product_tag=args.qtag)
