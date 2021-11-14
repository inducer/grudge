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
from grudge.array_context import (  # noqa: F401
    PyOpenCLArrayContext
)
from meshmode.array_context import (
    SingleGridWorkBalancingPytatoArrayContext as PytatoPyOpenCLArrayContext
)
from grudge.models.euler import (
    EulerState,
    EulerOperator,
    EntropyStableEulerOperator,
    AdiabaticSlipBC
)

from meshmode.mesh import BTAG_ALL

from pytools.obj_array import make_obj_array

import grudge.op as op

import logging
logger = logging.getLogger(__name__)


def ssprk43_step(y, t, h, f):

    def f_update(t, y):
        return y + h*f(t, y)

    y1 = 1/2*y + 1/2*f_update(t, y)
    y2 = 1/2*y1 + 1/2*f_update(t + h/2, y1)
    y3 = 2/3*y + 1/6*y2 + 1/6*f_update(t + h, y2)

    return 1/2*y3 + 1/2*f_update(t + h/2, y3)


def gaussian_profile(
    x_vec, t=0, rho0=1.0, rhoamp=1.0, p0=1.0, gamma=1.4,
    center=None, velocity=None):

    dim = len(x_vec)
    if center is None:
        center = np.zeros(shape=(dim,))
    if velocity is None:
        velocity = np.zeros(shape=(dim,))

    lump_loc = center + t * velocity

    # coordinates relative to lump center
    rel_center = make_obj_array(
        [x_vec[i] - lump_loc[i] for i in range(dim)]
    )
    actx = x_vec[0].array_context
    r = actx.np.sqrt(np.dot(rel_center, rel_center))
    expterm = rhoamp * actx.np.exp(1 - r ** 2)

    mass = expterm + rho0
    mom = velocity * mass
    energy = (p0 / (gamma - 1.0)) + np.dot(mom, mom) / (2.0 * mass)

    return EulerState(mass=mass, energy=energy, momentum=mom)


def make_pulse(amplitude, r0, w, r):
    dim = len(r)
    r_0 = np.zeros(dim)
    r_0 = r_0 + r0
    rel_center = make_obj_array(
        [r[i] - r_0[i] for i in range(dim)]
    )
    actx = r[0].array_context
    rms2 = w * w
    r2 = np.dot(rel_center, rel_center) / rms2
    return amplitude * actx.np.exp(-.5 * r2)


def acoustic_pulse_condition(x_vec, t=0):
    dim = len(x_vec)
    vel = np.zeros(shape=(dim,))
    orig = np.zeros(shape=(dim,))
    uniform_gaussian = gaussian_profile(
        x_vec, t=t, center=orig, velocity=vel, rhoamp=0.0)

    amplitude = 1.0
    width = 0.1
    pulse = make_pulse(amplitude, orig, width, x_vec)

    return EulerState(mass=uniform_gaussian.mass,
                      energy=uniform_gaussian.energy + pulse,
                      momentum=uniform_gaussian.momentum)


def run_acoustic_pulse(actx,
                       order=3,
                       resolution=16,
                       final_time=0.1,
                       nodal_dg=False,
                       visualize=False):

    # eos-related parameters
    gamma = 1.4
    gas_const = 287.1

    # {{{ discretization

    from meshmode.mesh.generation import generate_regular_rect_mesh

    dim = 2
    box_ll = -0.5
    box_ur = 0.5
    mesh = generate_regular_rect_mesh(
        a=(box_ll,)*dim,
        b=(box_ur,)*dim,
        nelements_per_axis=(resolution,)*dim)

    from grudge import DiscretizationCollection
    from grudge.dof_desc import as_dofdesc, DISCR_TAG_BASE, DISCR_TAG_QUAD
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

    bcs = [
        AdiabaticSlipBC(
            dd=as_dofdesc(BTAG_ALL).with_discr_tag(DISCR_TAG_QUAD)
        )
    ]

    if nodal_dg:
        operator_cls = EulerOperator
        exp_name = "fld-acoustic-pulse"
    else:
        operator_cls = EntropyStableEulerOperator
        exp_name = "fld-esdg-acoustic-pulse"

    euler_operator = operator_cls(
        dcoll,
        bdry_conditions=bcs,
        flux_type="lf",
        gamma=gamma,
        gas_const=gas_const,
    )

    def rhs(t, q):
        return euler_operator.operator(t, q)

    compiled_rhs = actx.compile(rhs)

    fields = acoustic_pulse_condition(thaw(dcoll.nodes(), actx))
    dt = actx.to_numpy(
        1/3 * euler_operator.estimate_rk4_timestep(actx, dcoll, state=fields))

    logger.info("Timestep size: %g", dt)

    # }}}

    from grudge.shortcuts import make_visualizer

    vis = make_visualizer(dcoll)

    # {{{ time stepping

    step = 0
    t = 0.0
    while t < final_time:
        fields = thaw(freeze(fields, actx), actx)
        fields = ssprk43_step(fields, t, dt, compiled_rhs)

        if step % 10 == 0:
            norm_q = actx.to_numpy(op.norm(dcoll, fields, 2))
            logger.info("[%04d] t = %.5f |q| = %.5e", step, t, norm_q)
            if visualize:
                vis.write_vtk_file(
                    f"{exp_name}-{step:04d}.vtu",
                    [
                        ("rho", fields.mass),
                        ("energy", fields.energy),
                        ("momentum", fields.momentum)
                    ]
                )
            assert norm_q < 10000

        t += dt
        step += 1

    # }}}


def main(ctx_factory, order=3, final_time=0.1, resolution=16,
         nodal_dg=False, visualize=False):
    cl_ctx = ctx_factory()
    queue = cl.CommandQueue(cl_ctx)
    actx = PytatoPyOpenCLArrayContext(
        queue,
        allocator=cl_tools.MemoryPool(cl_tools.ImmediateAllocator(queue)),
    )

    run_acoustic_pulse(
        actx, order=order, resolution=resolution,
        final_time=final_time,
        nodal_dg=nodal_dg,
        visualize=visualize)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--order", default=3, type=int)
    parser.add_argument("--tfinal", default=0.1, type=float)
    parser.add_argument("--resolution", default=16, type=int)
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--nodaldg", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    main(cl.create_some_context,
         order=args.order,
         final_time=args.tfinal,
         resolution=args.resolution,
         nodal_dg=args.nodaldg,
         visualize=args.visualize)
