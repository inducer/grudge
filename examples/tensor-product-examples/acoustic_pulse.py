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


from meshmode.mesh import TensorProductElementGroup
import numpy as np

import pyopencl as cl
import pyopencl.tools as cl_tools

from grudge.array_context import (
    PyOpenCLArrayContext,
    PytatoPyOpenCLArrayContext
)
from grudge.models.euler import (
    ConservedEulerField,
    EulerOperator,
    InviscidWallBC
)
from grudge.shortcuts import rk4_step

from meshmode.mesh import BTAG_ALL

from pytools.obj_array import make_obj_array

import grudge.op as op

import logging
logger = logging.getLogger(__name__)


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

    return ConservedEulerField(mass=mass, energy=energy, momentum=mom)


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

    return ConservedEulerField(
        mass=uniform_gaussian.mass,
        energy=uniform_gaussian.energy + pulse,
        momentum=uniform_gaussian.momentum
    )


def run_acoustic_pulse(actx,
                       order=3,
                       final_time=1,
                       resolution=4,
                       overintegration=False,
                       visualize=False):

    # eos-related parameters
    gamma = 1.4

    # {{{ discretization

    from meshmode.mesh.generation import generate_regular_rect_mesh

    dim = 3
    box_ll = -0.5
    box_ur = 0.5
    mesh = generate_regular_rect_mesh(
        a=(box_ll,)*dim,
        b=(box_ur,)*dim,
        nelements_per_axis=(resolution,)*dim,
        group_cls=TensorProductElementGroup)

    from grudge import DiscretizationCollection
    from grudge.dof_desc import DISCR_TAG_BASE, DISCR_TAG_QUAD
    from meshmode.discretization.poly_element import \
        LegendreGaussLobattoTensorProductGroupFactory as LGL

    exp_name = f"fld-acoustic-pulse-N{order}-K{resolution}"
    if overintegration:
        exp_name += "-overintegrated"
        quad_tag = DISCR_TAG_QUAD
    else:
        quad_tag = None

    dcoll = DiscretizationCollection(
        actx, mesh,
        discr_tag_to_group_factory={
            DISCR_TAG_BASE: LGL(order)
        }
    )

    # }}}

    # {{{ Euler operator

    euler_operator = EulerOperator(
        dcoll,
        bdry_conditions={BTAG_ALL: InviscidWallBC()},
        flux_type="lf",
        gamma=gamma,
        quadrature_tag=quad_tag
    )

    def rhs(t, q):
        return euler_operator.operator(t, q)

    compiled_rhs = actx.compile(rhs)

    from grudge.dt_utils import h_min_from_volume

    cfl = 0.125
    cn = 0.5*(order + 1)**2
    dt = cfl * actx.to_numpy(h_min_from_volume(dcoll)) / cn

    fields = acoustic_pulse_condition(actx.thaw(dcoll.nodes()))

    logger.info("Timestep size: %g", dt)

    # }}}

    from grudge.shortcuts import make_visualizer

    vis = make_visualizer(dcoll)

    # {{{ time stepping

    step = 0
    t = 0.0
    while t < final_time:
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
            assert norm_q < 5

        fields = actx.thaw(actx.freeze(fields))
        fields = rk4_step(fields, t, dt, compiled_rhs)
        t += dt
        step += 1

    # }}}


def main(ctx_factory, order=3, final_time=1, resolution=16,
         overintegration=False, visualize=False, lazy=False):
    cl_ctx = ctx_factory()
    queue = cl.CommandQueue(cl_ctx)

    if lazy:
        from grudge.array_context import PytatoTensorProductArrayContext
        actx = PytatoTensorProductArrayContext(
            queue,
            allocator=cl_tools.MemoryPool(cl_tools.ImmediateAllocator(queue)),
        )
    else:
        from grudge.array_context import TensorProductArrayContext
        actx = TensorProductArrayContext(
            queue,
            allocator=cl_tools.MemoryPool(cl_tools.ImmediateAllocator(queue)),
            force_device_scalars=True,
        )

    run_acoustic_pulse(
        actx,
        order=order,
        resolution=resolution,
        overintegration=overintegration,
        final_time=final_time,
        visualize=visualize
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--order", default=3, type=int)
    parser.add_argument("--tfinal", default=0.1, type=float)
    parser.add_argument("--resolution", default=16, type=int)
    parser.add_argument("--oi", action="store_true",
                        help="use overintegration")
    parser.add_argument("--visualize", action="store_true",
                        help="write out vtk output")
    parser.add_argument("--lazy", action="store_true",
                        help="switch to a lazy computation mode")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    main(cl.create_some_context,
         order=args.order,
         final_time=args.tfinal,
         resolution=args.resolution,
         overintegration=args.oi,
         visualize=args.visualize,
         lazy=args.lazy)
