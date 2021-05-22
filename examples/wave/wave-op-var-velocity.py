"""Minimal example of a grudge driver."""

__copyright__ = """
Copyright (C) 2020 Andreas Kloeckner
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
import numpy.linalg as la  # noqa
import pyopencl as cl
import pyopencl.tools as cl_tools

from arraycontext.impl.pyopencl import PyOpenCLArrayContext
from arraycontext.container.traversal import thaw

from pytools.obj_array import flat_obj_array

from meshmode.mesh import BTAG_ALL, BTAG_NONE  # noqa

from grudge.discretization import DiscretizationCollection
from grudge.dof_desc import DISCR_TAG_BASE, DISCR_TAG_QUAD, DOFDesc
from grudge.shortcuts import make_visualizer

import grudge.op as op


# {{{ wave equation bits

def wave_flux(dcoll, c, w_tpair):
    dd = w_tpair.dd
    dd_quad = dd.with_discr_tag(DISCR_TAG_QUAD)

    u = w_tpair[0]
    v = w_tpair[1:]

    normal = thaw(op.normal(dcoll, dd), u.int.array_context)

    flux_weak = flat_obj_array(
            np.dot(v.avg, normal),
            normal*u.avg,
            )

    # upwind
    flux_weak += flat_obj_array(
            0.5*(u.ext-u.int),
            0.5*normal*np.dot(normal, v.ext-v.int),
            )

    # FIXME this flux is only correct for continuous c
    dd_allfaces_quad = dd_quad.with_dtag("all_faces")
    c_quad = op.project(dcoll, "vol", dd_quad, c)
    flux_quad = op.project(dcoll, dd, dd_quad, flux_weak)

    return op.project(dcoll, dd_quad, dd_allfaces_quad, c_quad*flux_quad)


def wave_operator(dcoll, c, w):
    u = w[0]
    v = w[1:]

    dir_u = op.project(dcoll, "vol", BTAG_ALL, u)
    dir_v = op.project(dcoll, "vol", BTAG_ALL, v)
    dir_bval = flat_obj_array(dir_u, dir_v)
    dir_bc = flat_obj_array(-dir_u, dir_v)

    dd_quad = DOFDesc("vol", DISCR_TAG_QUAD)
    c_quad = op.project(dcoll, "vol", dd_quad, c)
    w_quad = op.project(dcoll, "vol", dd_quad, w)
    u_quad = w_quad[0]
    v_quad = w_quad[1:]

    dd_allfaces_quad = DOFDesc("all_faces", DISCR_TAG_QUAD)

    return (
        op.inverse_mass(
            dcoll,
            flat_obj_array(
                -op.weak_local_div(dcoll, dd_quad, c_quad*v_quad),
                -op.weak_local_grad(dcoll, dd_quad, c_quad*u_quad) \
                # pylint: disable=E1130
            ) + op.face_mass(
                dcoll,
                dd_allfaces_quad,
                wave_flux(
                    dcoll, c=c,
                    w_tpair=op.bdry_trace_pair(dcoll,
                                               BTAG_ALL,
                                               interior=dir_bval,
                                               exterior=dir_bc)
                ) + sum(
                    wave_flux(dcoll, c=c, w_tpair=tpair)
                    for tpair in op.interior_trace_pairs(dcoll, w)
                )
            )
        )
    )

# }}}


def rk4_step(y, t, h, f):
    k1 = f(t, y)
    k2 = f(t+h/2, y + h/2*k1)
    k3 = f(t+h/2, y + h/2*k2)
    k4 = f(t+h, y + h*k3)
    return y + h/6*(k1 + 2*k2 + 2*k3 + k4)


def bump(actx, dcoll, t=0, width=0.05, center=None):
    if center is None:
        center = np.array([0.2, 0.35, 0.1])

    center = center[:dcoll.dim]
    source_omega = 3

    nodes = thaw(op.nodes(dcoll), actx)
    center_dist = flat_obj_array([
        nodes[i] - center[i]
        for i in range(dcoll.dim)
        ])

    return (
        np.cos(source_omega*t)
        * actx.np.exp(
            -np.dot(center_dist, center_dist)
            / width**2))


def main(write_output=False):
    cl_ctx = cl.create_some_context()
    queue = cl.CommandQueue(cl_ctx)
    actx = PyOpenCLArrayContext(
        queue,
        allocator=cl_tools.MemoryPool(cl_tools.ImmediateAllocator(queue))
    )

    dim = 2
    nel_1d = 16
    from meshmode.mesh.generation import generate_regular_rect_mesh
    mesh = generate_regular_rect_mesh(
            a=(-0.5,)*dim,
            b=(0.5,)*dim,
            nelements_per_axis=(nel_1d,)*dim)

    order = 3

    if dim == 2:
        # no deep meaning here, just a fudge factor
        dt = 0.75/(nel_1d*order**2)
    elif dim == 3:
        # no deep meaning here, just a fudge factor
        dt = 0.45/(nel_1d*order**2)
    else:
        raise ValueError("don't have a stable time step guesstimate")

    print("%d elements" % mesh.nelements)

    from meshmode.discretization.poly_element import \
            QuadratureSimplexGroupFactory, \
            PolynomialWarpAndBlendGroupFactory
    dcoll = DiscretizationCollection(
        actx, mesh,
        discr_tag_to_group_factory={
            DISCR_TAG_BASE: PolynomialWarpAndBlendGroupFactory(order),
            DISCR_TAG_QUAD: QuadratureSimplexGroupFactory(3*order),
        }
    )

    # bounded above by 1
    c = 0.2 + 0.8*bump(actx, dcoll, center=np.zeros(3), width=0.5)

    fields = flat_obj_array(
            bump(actx, dcoll, ),
            [dcoll.zeros(actx) for i in range(dcoll.dim)]
            )

    vis = make_visualizer(dcoll)

    def rhs(t, w):
        return wave_operator(dcoll, c=c, w=w)

    t = 0
    t_final = 3
    istep = 0
    while t < t_final:
        fields = rk4_step(fields, t, dt, rhs)

        if istep % 10 == 0:
            print(f"step: {istep} t: {t} "
                  f"L2: {op.norm(dcoll, fields[0], 2)} "
                  f"Linf: {op.norm(dcoll, fields[0], np.inf)} "
                  f"sol max: {op.nodal_maximum(dcoll, fields[0])} "
                  f"sol min: {op.nodal_minimum(dcoll, fields[0])}")
            if write_output:
                vis.write_vtk_file(
                    "fld-wave-eager-var-velocity-%04d.vtu" % istep,
                    [
                        ("c", c),
                        ("u", fields[0]),
                        ("v", fields[1:]),
                    ]
                )

        t += dt
        istep += 1

        assert op.norm(dcoll, fields[0], 2) < 1


if __name__ == "__main__":
    main()

# vim: foldmethod=marker
