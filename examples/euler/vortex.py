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


import numpy as np

import pyopencl as cl
import pyopencl.tools as cl_tools

from arraycontext import thaw, freeze
from grudge.array_context import PyOpenCLArrayContext, PytatoPyOpenCLArrayContext
from grudge.models.euler import EulerState, EntropyStableEulerOperator

from meshmode.mesh import BTAG_ALL

from pytools.obj_array import make_obj_array

import grudge.op as op

import logging
logger = logging.getLogger(__name__)


def rk4_step(y, t, h, f):
    k1 = f(t, y)
    k2 = f(t+h/2, y + h/2*k1)
    k3 = f(t+h/2, y + h/2*k2)
    k4 = f(t+h, y + h*k3)
    return y + h/6*(k1 + 2*k2 + 2*k3 + k4)


def vortex_initial_condition(x_vec, t=0):
    _beta = 5
    _center = np.zeros(shape=(2,))
    _velocity = np.zeros(shape=(2,))
    gamma = 1.4

    vortex_loc = _center + t * _velocity

    # Coordinates relative to vortex center
    x_rel = x_vec[0] - vortex_loc[0]
    y_rel = x_vec[1] - vortex_loc[1]
    actx = x_vec[0].array_context

    r = actx.np.sqrt(x_rel ** 2 + y_rel ** 2)
    expterm = _beta * actx.np.exp(1 - r ** 2)
    u = _velocity[0] - expterm * y_rel / (2 * np.pi)
    v = _velocity[1] + expterm * x_rel / (2 * np.pi)
    velocity = make_obj_array([u, v])
    mass = (1 - (gamma - 1) / (16 * gamma * np.pi ** 2)
            * expterm ** 2) ** (1 / (gamma - 1))
    momentum = mass * velocity
    p = mass ** gamma

    energy = p / (gamma - 1) + mass / 2 * (u ** 2 + v ** 2)

    return EulerState(mass=mass, energy=energy, momentum=momentum)


def run_vortex(actx, order=3, resolution=8, final_time=50,
               flux_type="central",
               visualize=False):

    # eos-related parameters
    gamma = 1.4
    gas_const = 287.1

    # {{{ discretization

    from meshmode.mesh.generation import generate_regular_rect_mesh

    dim = 2
    box_ll = -5.0
    box_ur = 5.0
    mesh = generate_regular_rect_mesh(
        a=(box_ll,)*dim,
        b=(box_ur,)*dim,
        nelements_per_axis=(resolution,)*dim)

    from grudge import DiscretizationCollection
    from grudge.dof_desc import DISCR_TAG_BASE, DISCR_TAG_QUAD
    from meshmode.discretization.poly_element import \
        (PolynomialWarpAndBlend2DRestrictingGroupFactory,
         QuadratureSimplexGroupFactory)

    dcoll = DiscretizationCollection(
        actx, mesh,
        discr_tag_to_group_factory={
            DISCR_TAG_BASE: PolynomialWarpAndBlend2DRestrictingGroupFactory(order),
            DISCR_TAG_QUAD: QuadratureSimplexGroupFactory(2*order)
        }
    )

    # }}}

    # {{{ Euler operator

    euler_operator = EntropyStableEulerOperator(
        dcoll,
        bdry_fcts={BTAG_ALL: vortex_initial_condition},
        initial_condition=vortex_initial_condition,
        flux_type=flux_type,
        gamma=gamma,
        gas_const=gas_const,
    )

    def rhs(t, q):
        return euler_operator.operator(t, q)

    compiled_rhs = actx.compile(rhs)

    fields = vortex_initial_condition(thaw(dcoll.nodes(), actx))
    dt = 1/3 * euler_operator.estimate_rk4_timestep(actx, dcoll, state=fields)

    logger.info("Timestep size: %g", dt)

    # }}}

    from grudge.shortcuts import make_visualizer

    vis = make_visualizer(dcoll)

    # {{{ time stepping

    step = 0
    t = 0.0
    while t < final_time:
        fields = thaw(freeze(fields, actx), actx)
        fields = rk4_step(fields, t, dt, compiled_rhs)

        if step % 10 == 0:
            norm_q = actx.to_numpy(op.norm(dcoll, fields.join(), 2))
            logger.info("[%04d] t = %.5f |q| = %.5e", step, t, norm_q)
            if visualize:
                vis.write_vtk_file(
                    f"fld-vortex-{step:04d}.vtu",
                    [
                        ("rho", fields.mass),
                        ("energy", fields.energy),
                        ("momentum", fields.momentum)
                    ]
                )
            assert norm_q < 100

        t += dt
        step += 1

    # }}}


def run_convergence_test_vortex(
        actx, order=3, final_time=1,
        flux_type="central"):

    # eos-related parameters
    gamma = 1.4
    gas_const = 287.1

    from meshmode.mesh.generation import generate_regular_rect_mesh

    dim = 2
    box_ll = -5.0
    box_ur = 5.0

    from grudge import DiscretizationCollection
    from grudge.dof_desc import DISCR_TAG_BASE, DISCR_TAG_QUAD
    from meshmode.discretization.poly_element import \
        (PolynomialWarpAndBlend2DRestrictingGroupFactory,
         QuadratureSimplexGroupFactory)
    from pytools.convergence import EOCRecorder
    from grudge.dt_utils import h_max_from_volume

    eoc_rec = EOCRecorder()

    for resolution in [8, 16, 32, 64, 128]:

        # {{{ discretization

        mesh = generate_regular_rect_mesh(
            a=(box_ll,)*dim,
            b=(box_ur,)*dim,
            nelements_per_axis=(resolution,)*dim)

        discr_tag_to_group_factory = {
            DISCR_TAG_BASE: PolynomialWarpAndBlend2DRestrictingGroupFactory(order),
            DISCR_TAG_QUAD: QuadratureSimplexGroupFactory(2*order)
        }

        dcoll = DiscretizationCollection(
            actx, mesh,
            discr_tag_to_group_factory=discr_tag_to_group_factory
        )
        h_max = h_max_from_volume(dcoll, dim=dcoll.ambient_dim)
        nodes = thaw(dcoll.nodes(), actx)

        # }}}

        euler_operator = EntropyStableEulerOperator(
            dcoll,
            bdry_fcts={BTAG_ALL: vortex_initial_condition},
            initial_condition=vortex_initial_condition,
            flux_type=flux_type,
            gamma=gamma,
            gas_const=gas_const,
        )

        def rhs(t, q):
            return euler_operator.operator(t, q)

        fields = vortex_initial_condition(nodes)
        dt = 1/3 * euler_operator.estimate_rk4_timestep(actx, dcoll, state=fields)

        logger.info("Timestep size: %g", dt)

        # {{{ time stepping

        step = 0
        t = 0.0
        last_q = None
        while t < final_time:
            fields = rk4_step(fields, t, dt, rhs)
            t += dt
            logger.info("[%04d] t = %.5f", step, t)
            last_q = fields
            last_t = t
            step += 1

        # }}}

        error_l2 = op.norm(
            dcoll,
            (last_q - vortex_initial_condition(nodes, t=last_t)).join(),
            2
        )
        error_l2 = actx.to_numpy(error_l2)
        logger.info("h_max %.5e error %.5e", h_max, error_l2)
        eoc_rec.add_data_point(h_max, error_l2)

    logger.info("\n%s", eoc_rec.pretty_print(abscissa_label="h",
                                             error_label="L2 Error"))


def main(ctx_factory, order=3, final_time=10, resolution=8,
         lf_stabilization=False, visualize=False,
         test_convergence=False):
    cl_ctx = ctx_factory()
    queue = cl.CommandQueue(cl_ctx)
    actx = PytatoPyOpenCLArrayContext(
        queue,
        allocator=cl_tools.MemoryPool(cl_tools.ImmediateAllocator(queue)),
        # force_device_scalars=True,
    )

    if lf_stabilization:
        flux_type = "lf"
    else:
        flux_type = "central"

    if test_convergence:
        run_convergence_test_vortex(
            actx, order=order,
            final_time=0.25,
            flux_type=flux_type)
    else:
        run_vortex(
            actx, order=order, resolution=resolution,
            final_time=final_time,
            flux_type=flux_type,
            visualize=visualize)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--order", default=3, type=int)
    parser.add_argument("--tfinal", default=10.0, type=float)
    parser.add_argument("--resolution", default=8, type=int)
    parser.add_argument("--lfflux", action="store_true")
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--convergence", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    main(cl.create_some_context,
         order=args.order,
         final_time=args.tfinal,
         resolution=args.resolution,
         lf_stabilization=args.lfflux,
         visualize=args.visualize,
         test_convergence=args.convergence)
