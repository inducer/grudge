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
    EntropyStableEulerOperator,
    PrescribedBC,
    AdiabaticSlipBC
)

from pytools.obj_array import make_obj_array

import grudge.op as op

import logging
logger = logging.getLogger(__name__)


def get_doublemach_mesh():
    """Generate or import a grid using `gmsh`.
    Input required:
        doubleMach.msh (read existing mesh)
    This routine will generate a new grid if it does
    not find the grid file (doubleMach.msh).
    """
    from meshmode.mesh.io import (
        read_gmsh,
        generate_gmsh,
        ScriptSource,
    )
    import os
    meshfile = "doubleMach.msh"
    if not os.path.exists(meshfile):
        mesh = generate_gmsh(
            ScriptSource("""
                x0=1.0/6.0;
                setsize=0.025;
                Point(1) = {0, 0, 0, setsize};
                Point(2) = {x0,0, 0, setsize};
                Point(3) = {4, 0, 0, setsize};
                Point(4) = {4, 1, 0, setsize};
                Point(5) = {0, 1, 0, setsize};
                Line(1) = {1, 2};
                Line(2) = {2, 3};
                Line(5) = {3, 4};
                Line(6) = {4, 5};
                Line(7) = {5, 1};
                Line Loop(8) = {-5, -6, -7, -1, -2};
                Plane Surface(8) = {8};
                Physical Surface('domain') = {8};
                Physical Curve('ic1') = {6};
                Physical Curve('ic2') = {7};
                Physical Curve('ic3') = {1};
                Physical Curve('wall') = {2};
                Physical Curve('out') = {5};
        """, "geo"), force_ambient_dim=2, dimensions=2, target_unit="M",
            output_file_name=meshfile)
    else:
        mesh = read_gmsh(meshfile, force_ambient_dim=2)

    return mesh


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


def doublemach_reflection_initial_condition(x_vec, t=0):
    shock_speed = 10
    shock_location = 1/6
    gamma = 1.4
    gm1 = gamma - 1.0
    gp1 = gamma + 1.0
    gmn1 = 1.0 / gm1
    x_rel = x_vec[0]
    y_rel = x_vec[1]
    actx = x_rel.array_context

    # Normal Shock Relations
    shock_speed_2 = shock_speed * shock_speed
    rho_jump = gp1 * shock_speed_2 / (gm1 * shock_speed_2 + 2.)
    p_jump = (2. * gamma * shock_speed_2 - gm1) / gp1
    up = 2. * (shock_speed_2 - 1.) / (gp1 * shock_speed)

    rhol = gamma * rho_jump
    rhor = gamma
    ul = up * np.cos(np.pi/6.0)
    ur = 0.0
    vl = - up * np.sin(np.pi/6.0)
    vr = 0.0
    rhoel = gmn1 * p_jump
    rhoer = gmn1 * 1.0

    xinter = shock_location + y_rel/np.sqrt(3.0) + 2.0*shock_speed*t/np.sqrt(3.0)
    sigma = 0.05
    xtanh = 1.0/sigma*(x_rel-xinter)
    mass = rhol/2.0*(actx.np.tanh(-xtanh)+1.0)+rhor/2.0*(actx.np.tanh(xtanh)+1.0)
    rhoe = (rhoel/2.0*(actx.np.tanh(-xtanh)+1.0)
            + rhoer/2.0*(actx.np.tanh(xtanh)+1.0))
    u = ul/2.0*(actx.np.tanh(-xtanh)+1.0)+ur/2.0*(actx.np.tanh(xtanh)+1.0)
    v = vl/2.0*(actx.np.tanh(-xtanh)+1.0)+vr/2.0*(actx.np.tanh(xtanh)+1.0)

    vel = make_obj_array([u, v])
    momentum = mass * vel
    energy = rhoe + .5*mass*np.dot(vel, vel)

    return EulerState(mass=mass, energy=energy, momentum=momentum)


def run_doublemach_reflection(
        actx, order=3, final_time=0.2, visualize=False):

    from grudge import DiscretizationCollection
    from grudge.dof_desc import (
        as_dofdesc,
        DISCR_TAG_BASE,
        DISCR_TAG_QUAD,
        DTAG_BOUNDARY
    )
    from meshmode.discretization.poly_element import \
        (PolynomialWarpAndBlend2DRestrictingGroupFactory,
         QuadratureSimplexGroupFactory)

    # eos-related parameters
    gamma = 1.4
    gas_const = 287.1

    # {{{ discretization

    mesh = get_doublemach_mesh()

    dcoll = DiscretizationCollection(
        actx, mesh,
        discr_tag_to_group_factory={
            DISCR_TAG_BASE: PolynomialWarpAndBlend2DRestrictingGroupFactory(order),
            DISCR_TAG_QUAD: QuadratureSimplexGroupFactory(2*order)
        }
    )

    # }}}

    # {{{ Euler operator

    # Create boundary descriptors
    dd_ic1 = as_dofdesc(DTAG_BOUNDARY("ic1")).with_discr_tag(DISCR_TAG_QUAD)
    dd_ic2 = as_dofdesc(DTAG_BOUNDARY("ic2")).with_discr_tag(DISCR_TAG_QUAD)
    dd_ic3 = as_dofdesc(DTAG_BOUNDARY("ic3")).with_discr_tag(DISCR_TAG_QUAD)
    dd_wall = as_dofdesc(DTAG_BOUNDARY("wall")).with_discr_tag(DISCR_TAG_QUAD)
    dd_out = as_dofdesc(DTAG_BOUNDARY("out")).with_discr_tag(DISCR_TAG_QUAD)

    bcs = [
        PrescribedBC(dd=dd_ic1,
                     prescribed_state=doublemach_reflection_initial_condition),
        PrescribedBC(dd=dd_ic2,
                     prescribed_state=doublemach_reflection_initial_condition),
        PrescribedBC(dd=dd_ic3,
                     prescribed_state=doublemach_reflection_initial_condition),
        AdiabaticSlipBC(dd=dd_wall),
        AdiabaticSlipBC(dd=dd_out)
    ]

    euler_operator = EntropyStableEulerOperator(
        dcoll,
        bdry_conditions=bcs,
        flux_type="lf",
        gamma=gamma,
        gas_const=gas_const,
    )

    def rhs(t, q):
        return euler_operator.operator(t, q)

    compiled_rhs = actx.compile(rhs)

    fields = doublemach_reflection_initial_condition(thaw(dcoll.nodes(), actx))
    dt = actx.to_numpy(
        1/5 * euler_operator.estimate_rk4_timestep(actx, dcoll, state=fields))

    logger.info("Timestep size: %g", dt)

    # }}}

    from grudge.shortcuts import make_visualizer

    vis = make_visualizer(dcoll)

    # {{{ time stepping

    step = 0
    t = 0.0

    from grudge.models.euler import positivity_preserving_limiter
    from functools import partial

    limiter = partial(positivity_preserving_limiter, dcoll)

    while t < final_time:
        if step % 10 == 0:
            norm_q = actx.to_numpy(op.norm(dcoll, fields, 2))
            logger.info("[%04d] t = %.5f |q| = %.5e", step, t, norm_q)
            if visualize:
                vis.write_vtk_file(
                    f"fld-doublemach-{step:04d}.vtu",
                    [
                        ("rho", fields.mass),
                        ("energy", fields.energy),
                        ("momentum", fields.momentum)
                    ]
                )
            assert norm_q < 10000

        fields = thaw(freeze(fields, actx), actx)
        fields = ssprk43_step(fields, t, dt, compiled_rhs, limiter=limiter)
        t += dt
        step += 1

    # }}}


def main(ctx_factory, order=3, final_time=0.2, visualize=False):
    cl_ctx = ctx_factory()
    queue = cl.CommandQueue(cl_ctx)
    actx = PytatoPyOpenCLArrayContext(
        queue,
        allocator=cl_tools.MemoryPool(cl_tools.ImmediateAllocator(queue))
    )
    run_doublemach_reflection(
        actx, order=order, final_time=final_time, visualize=visualize)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--order", default=3, type=int)
    parser.add_argument("--tfinal", default=10.0, type=float)
    parser.add_argument("--visualize", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    main(cl.create_some_context,
         order=args.order,
         final_time=args.tfinal,
         visualize=args.visualize)
