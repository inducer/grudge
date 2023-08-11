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

from arraycontext import (
    with_container_arithmetic,
    dataclass_array_container
)

from dataclasses import dataclass

from pytools.obj_array import flat_obj_array, make_obj_array

from meshmode.dof_array import DOFArray
from meshmode.mesh import BTAG_ALL, BTAG_NONE  # noqa

from grudge.dof_desc import as_dofdesc, DOFDesc, DISCR_TAG_BASE, DISCR_TAG_QUAD
from grudge.trace_pair import TracePair
from grudge.discretization import DiscretizationCollection
from grudge.shortcuts import make_visualizer, compiled_lsrk45_step

import grudge.op as op
import grudge.geometry as geo

import logging
logger = logging.getLogger(__name__)

from mpi4py import MPI


# {{{ wave equation bits

@with_container_arithmetic(bcast_obj_array=True, rel_comparison=True,
        _cls_has_array_context_attr=True)
@dataclass_array_container
@dataclass(frozen=True)
class WaveState:
    u: DOFArray
    v: np.ndarray  # [object array]

    def __post_init__(self):
        assert isinstance(self.v, np.ndarray) and self.v.dtype.char == "O"

    @property
    def array_context(self):
        return self.u.array_context


def wave_flux(actx, dcoll, c, w_tpair):
    u = w_tpair.u
    v = w_tpair.v
    dd = w_tpair.dd

    normal = geo.normal(actx, dcoll, dd)

    flux_weak = WaveState(
        u=v.avg @ normal,
        v=u.avg * normal
    )

    # upwind
    v_jump = v.diff @ normal
    flux_weak += WaveState(
        u=0.5 * u.diff,
        v=0.5 * v_jump * normal,
    )

    return op.project(dcoll, dd, dd.with_dtag("all_faces"), c*flux_weak)


class _WaveStateTag:
    pass


def wave_operator(actx, dcoll, c, w, quad_tag=None):
    dd_base = as_dofdesc("vol")
    dd_vol = DOFDesc("vol", quad_tag)
    dd_faces = DOFDesc("all_faces", quad_tag)
    dd_btag = as_dofdesc(BTAG_ALL).with_discr_tag(quad_tag)

    def interp_to_surf_quad(utpair):
        local_dd = utpair.dd
        local_dd_quad = local_dd.with_discr_tag(quad_tag)
        return TracePair(
            local_dd_quad,
            interior=op.project(dcoll, local_dd, local_dd_quad, utpair.int),
            exterior=op.project(dcoll, local_dd, local_dd_quad, utpair.ext)
        )

    w_quad = op.project(dcoll, dd_base, dd_vol, w)
    u = w_quad.u
    v = w_quad.v

    dir_w = op.project(dcoll, dd_base, dd_btag, w)
    dir_u = dir_w.u
    dir_v = dir_w.v
    dir_bval = WaveState(u=dir_u, v=dir_v)
    dir_bc = WaveState(u=-dir_u, v=dir_v)

    return (
        op.inverse_mass(
            dcoll,
            WaveState(
                u=-c*op.weak_local_div(dcoll, dd_vol, v),
                v=-c*op.weak_local_grad(dcoll, dd_vol, u)
            )
            + op.face_mass(
                dcoll,
                dd_faces,
                wave_flux(
                    actx,
                    dcoll, c=c,
                    w_tpair=op.bdry_trace_pair(dcoll,
                                               dd_btag,
                                               interior=dir_bval,
                                               exterior=dir_bc)
                ) + sum(
                    wave_flux(actx, dcoll, c=c, w_tpair=interp_to_surf_quad(tpair))
                    for tpair in op.interior_trace_pairs(dcoll, w,
                        comm_tag=_WaveStateTag)
                )
            )
        )
    )

# }}}


def estimate_rk4_timestep(actx, dcoll, c):
    from grudge.dt_utils import characteristic_lengthscales

    local_dts = characteristic_lengthscales(actx, dcoll) / c

    return op.nodal_min(dcoll, "vol", local_dts)


def bump(actx, dcoll, t=0):
    source_center = np.array([0.2, 0.35, 0.1])[:dcoll.dim]
    source_width = 0.05
    source_omega = 3

    nodes = actx.thaw(dcoll.nodes())
    center_dist = flat_obj_array([
        nodes[i] - source_center[i]
        for i in range(dcoll.dim)
        ])

    return (
        np.cos(source_omega*t)
        * actx.np.exp(
            -np.dot(center_dist, center_dist)
            / source_width**2))


def main(ctx_factory, dim=2, order=3,
         visualize=False, lazy=False, use_quad=False, use_nonaffine_mesh=False):
    cl_ctx = ctx_factory()
    queue = cl.CommandQueue(cl_ctx)

    comm = MPI.COMM_WORLD
    num_parts = comm.size

    from grudge.array_context import get_reasonable_array_context_class
    actx_class = get_reasonable_array_context_class(lazy=lazy, distributed=True)
    if lazy:
        actx = actx_class(comm, queue, mpi_base_tag=15000)
    else:
        actx = actx_class(comm, queue,
                allocator=cl_tools.MemoryPool(cl_tools.ImmediateAllocator(queue)),
                force_device_scalars=True)

    from meshmode.distributed import get_partition_by_pymetis, membership_list_to_map
    from meshmode.mesh.processing import partition_mesh

    nel_1d = 16

    if comm.rank == 0:
        if use_nonaffine_mesh:
            from meshmode.mesh.generation import generate_warped_rect_mesh
            # FIXME: *generate_warped_rect_mesh* in meshmode warps a
            # rectangle domain with hard-coded vertices at (-0.5,)*dim
            # and (0.5,)*dim. Should extend the function interface to allow
            # for specifying the end points directly.
            mesh = generate_warped_rect_mesh(dim=dim, order=order,
                                             nelements_side=nel_1d)
        else:
            from meshmode.mesh.generation import generate_regular_rect_mesh
            mesh = generate_regular_rect_mesh(
                    a=(-0.5,)*dim,
                    b=(0.5,)*dim,
                    nelements_per_axis=(nel_1d,)*dim)

        logger.info("%d elements", mesh.nelements)

        part_id_to_part = partition_mesh(mesh,
                       membership_list_to_map(
                           get_partition_by_pymetis(mesh, num_parts)))
        parts = [part_id_to_part[i] for i in range(num_parts)]

        local_mesh = comm.scatter(parts)

        del mesh

    else:
        local_mesh = comm.scatter(None)

    from meshmode.discretization.poly_element import \
            QuadratureSimplexGroupFactory, \
            default_simplex_group_factory
    dcoll = DiscretizationCollection(
        actx, local_mesh,
        discr_tag_to_group_factory={
            DISCR_TAG_BASE: default_simplex_group_factory(base_dim=dim, order=order),
            # High-order quadrature to integrate inner products of polynomials
            # on warped geometry exactly (metric terms are also polynomial)
            DISCR_TAG_QUAD: QuadratureSimplexGroupFactory(3*order),
        })

    if use_quad:
        quad_tag = DISCR_TAG_QUAD
    else:
        quad_tag = None

    fields = WaveState(
        u=bump(actx, dcoll),
        v=make_obj_array([dcoll.zeros(actx) for i in range(dcoll.dim)])
    )

    c = 1

    # FIXME: Sketchy, empirically determined fudge factor
    # 5/4 to account for larger LSRK45 stability region
    dt = actx.to_numpy(0.45 * estimate_rk4_timestep(actx, dcoll, c)) * 5/4

    vis = make_visualizer(dcoll)

    def rhs(t, w):
        return wave_operator(actx, dcoll, c=c, w=w, quad_tag=quad_tag)

    compiled_rhs = actx.compile(rhs)

    if comm.rank == 0:
        logger.info("dt = %g", dt)

    import time
    start = time.time()

    t = 0
    t_final = 3
    istep = 0
    while t < t_final:
        start = time.time()

        fields = compiled_lsrk45_step(actx, fields, t, dt, compiled_rhs)

        if istep % 10 == 0:
            stop = time.time()
            if args.no_diagnostics:
                if comm.rank == 0:
                    logger.info(f"step: {istep} t: {t} "
                                f"wall: {stop-start} ")
            else:
                l2norm = actx.to_numpy(op.norm(dcoll, fields.u, 2))

                # NOTE: These are here to ensure the solution is bounded for the
                # time interval specified
                assert l2norm < 1

                linfnorm = actx.to_numpy(op.norm(dcoll, fields.u, np.inf))
                nodalmax = actx.to_numpy(op.nodal_max(dcoll, "vol", fields.u))
                nodalmin = actx.to_numpy(op.nodal_min(dcoll, "vol", fields.u))
                if comm.rank == 0:
                    logger.info(f"step: {istep} t: {t} "
                                f"L2: {l2norm} "
                                f"Linf: {linfnorm} "
                                f"sol max: {nodalmax} "
                                f"sol min: {nodalmin} "
                                f"wall: {stop-start} ")
            if visualize:
                vis.write_parallel_vtk_file(
                    comm,
                    f"fld-wave-eager-mpi-{{rank:03d}}-{istep:04d}.vtu",
                    [
                        ("u", fields.u),
                        ("v", fields.v),
                    ]
                )
            start = stop

        t += dt
        istep += 1


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dim", default=2, type=int)
    parser.add_argument("--order", default=3, type=int)
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--lazy", action="store_true",
                        help="switch to a lazy computation mode")
    parser.add_argument("--quad", action="store_true")
    parser.add_argument("--nonaffine", action="store_true")
    parser.add_argument("--no-diagnostics", action="store_true")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    main(cl.create_some_context,
         dim=args.dim,
         order=args.order,
         visualize=args.visualize,
         lazy=args.lazy,
         use_quad=args.quad,
         use_nonaffine_mesh=args.nonaffine)

# vim: foldmethod=marker
