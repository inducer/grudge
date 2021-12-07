"""Minimal example of a grudge driver."""

__copyright__ = """
Copyright (C) 2021 University of Illinois Board of Trustees
"""

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


import pyopencl as cl
import pyopencl.tools as cl_tools

from arraycontext import thaw, freeze
from meshmode.array_context import (
    SingleGridWorkBalancingPytatoArrayContext as PytatoPyOpenCLArrayContext
)
from grudge.models.euler import (
    EulerOperator,
    PrescribedBC,
    conservative_to_primitive_vars,
    primitive_to_conservative_vars
)

from pytools.obj_array import make_obj_array

import grudge.op as op

import logging
logger = logging.getLogger(__name__)


def ssprk43_step(y, t, h, f, limiter=None):

    def f_update(t, y):
        return y + h*f(t, y)

    y1 = 1/2*y + 1/2*f_update(t, y)
    if limiter is not None:
        y1 = limiter(y1)

    y2 = 1/2*y1 + 1/2*f_update(t + h/2, y1)
    if limiter is not None:
        y2 = limiter(y2)

    y3 = 2/3*y + 1/6*y2 + 1/6*f_update(t + h, y2)
    if limiter is not None:
        y3 = limiter(y3)

    result = 1/2*y3 + 1/2*f_update(t + h/2, y3)
    if limiter is not None:
        result = limiter(result)

    return result


def sod_shock_initial_condition(nodes, t=0):
    gamma = 1.4
    dim = len(nodes)
    x = nodes[0]
    actx = x.array_context
    zeros = 0*x

    _x0 = 0.5
    _rhoin = 1.0
    _rhoout = 0.125
    _pin = 1.0
    _pout = 0.1
    rhoin = zeros + _rhoin
    rhoout = zeros + _rhoout

    x0 = zeros + _x0
    sigma = 1e-13
    weight = 0.5 * (1.0 - actx.np.tanh(1.0/sigma * (x - x0)))

    rho = rhoout + (rhoin - rhoout)*weight
    p = _pout + (_pin - _pout)*weight
    u = make_obj_array([zeros for _ in range(dim)])

    return primitive_to_conservative_vars((rho, u, p), gamma=gamma)


def run_sod_shock_tube(actx,
                       order=4,
                       resolution=32,
                       final_time=0.2,
                       overintegration=False,
                       visualize=False):

    # eos-related parameters
    gamma = 1.4

    # {{{ discretization

    from meshmode.mesh.generation import generate_regular_rect_mesh

    dim = 1
    box_ll = 0.0
    box_ur = 1.0
    mesh = generate_regular_rect_mesh(
        a=(box_ll,)*dim,
        b=(box_ur,)*dim,
        nelements_per_axis=(resolution,)*dim,
        boundary_tag_to_face={
            "prescribed": ["+x", "-x"],
        }
    )

    from grudge import DiscretizationCollection
    from grudge.dof_desc import \
        DISCR_TAG_BASE, DISCR_TAG_QUAD, DTAG_BOUNDARY
    from meshmode.discretization.poly_element import \
        (default_simplex_group_factory,
         QuadratureSimplexGroupFactory)

    exp_name = f"fld-sod-1d-N{order}-K{resolution}"

    if overintegration:
        exp_name += "-overintegrated"
        quad_tag = DISCR_TAG_QUAD
    else:
        quad_tag = None

    dcoll = DiscretizationCollection(
        actx, mesh,
        discr_tag_to_group_factory={
            DISCR_TAG_BASE: default_simplex_group_factory(dim, order),
            DISCR_TAG_QUAD: QuadratureSimplexGroupFactory(order + 2)
        }
    )

    # }}}

    # {{{ Euler operator

    dd_prescribe = DTAG_BOUNDARY("prescribed")
    bcs = {
        dd_prescribe: PrescribedBC(prescribed_state=sod_shock_initial_condition)
    }

    euler_operator = EulerOperator(
        dcoll,
        bdry_conditions=bcs,
        flux_type="lf",
        gamma=gamma,
        quadrature_tag=quad_tag
    )

    def rhs(t, q):
        return euler_operator.operator(t, q)

    compiled_rhs = actx.compile(rhs)

    fields = sod_shock_initial_condition(thaw(dcoll.nodes(), actx))

    from grudge.dt_utils import h_min_from_volume

    cfl = 0.01
    cn = 0.5*(order + 1)**2
    dt = cfl * actx.to_numpy(h_min_from_volume(dcoll)) / cn

    logger.info("Timestep size: %g", dt)

    # }}}

    from grudge.shortcuts import make_visualizer

    vis = make_visualizer(dcoll)

    # {{{ time stepping

    from grudge.models.euler import limiter_zhang_shu
    from functools import partial

    limiter = partial(limiter_zhang_shu, dcoll, DISCR_TAG_QUAD)

    step = 0
    t = 0.0
    while t < final_time:
        if step % 10 == 0:
            norm_q = actx.to_numpy(op.norm(dcoll, fields, 2))
            logger.info("[%04d] t = %.5f |q| = %.5e", step, t, norm_q)
            if visualize:
                _, velocity, pressure = \
                    conservative_to_primitive_vars(fields, gamma=gamma)
                vis.write_vtk_file(
                    f"{exp_name}-{step:04d}.vtu",
                    [
                        ("rho", fields.mass),
                        ("energy", fields.energy),
                        ("momentum", fields.momentum),
                        ("velocity", velocity),
                        ("pressure", pressure)
                    ]
                )
            assert norm_q < 10000

        fields = thaw(freeze(fields, actx), actx)
        fields = ssprk43_step(fields, t, dt, compiled_rhs, limiter=limiter)
        t += dt
        step += 1

    # }}}


def main(ctx_factory, order=4, final_time=0.2, resolution=32,
         overintegration=False, visualize=False):
    cl_ctx = ctx_factory()
    queue = cl.CommandQueue(cl_ctx)
    actx = PytatoPyOpenCLArrayContext(
        queue,
        allocator=cl_tools.MemoryPool(cl_tools.ImmediateAllocator(queue)),
    )

    run_sod_shock_tube(
        actx, order=order, resolution=resolution,
        final_time=final_time,
        overintegration=overintegration,
        visualize=visualize)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--order", default=4, type=int)
    parser.add_argument("--tfinal", default=0.2, type=float)
    parser.add_argument("--resolution", default=32, type=int)
    parser.add_argument("--oi", action="store_true")
    parser.add_argument("--visualize", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    main(cl.create_some_context,
         order=args.order,
         final_time=args.tfinal,
         resolution=args.resolution,
         overintegration=args.oi,
         visualize=args.visualize)
