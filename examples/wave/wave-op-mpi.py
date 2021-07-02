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

from arraycontext import thaw
from grudge.array_context import PyOpenCLArrayContext

from pytools.obj_array import flat_obj_array

from meshmode.mesh import BTAG_ALL, BTAG_NONE  # noqa

from grudge.discretization import DiscretizationCollection
from grudge.shortcuts import make_visualizer

import grudge.op as op

import logging
logger = logging.getLogger(__name__)

from mpi4py import MPI


# {{{ wave equation bits

def wave_flux(dcoll, c, w_tpair):
    u = w_tpair[0]
    v = w_tpair[1:]

    normal = thaw(dcoll.normal(w_tpair.dd), u.int.array_context)

    flux_weak = flat_obj_array(
            np.dot(v.avg, normal),
            normal*u.avg,
            )

    # upwind
    v_jump = np.dot(normal, v.ext-v.int)
    flux_weak += flat_obj_array(
            0.5*(u.ext-u.int),
            0.5*normal*v_jump,
            )

    return op.project(dcoll, w_tpair.dd, "all_faces", c*flux_weak)


def wave_operator(dcoll, c, w):
    u = w[0]
    v = w[1:]

    dir_u = op.project(dcoll, "vol", BTAG_ALL, u)
    dir_v = op.project(dcoll, "vol", BTAG_ALL, v)
    dir_bval = flat_obj_array(dir_u, dir_v)
    dir_bc = flat_obj_array(-dir_u, dir_v)

    return (
        op.inverse_mass(
            dcoll,
            flat_obj_array(
                -c*op.weak_local_div(dcoll, v),
                -c*op.weak_local_grad(dcoll, u)
            )
            + op.face_mass(
                dcoll,
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


def estimate_rk4_timestep(actx, dcoll, c):
    from grudge.dt_utils import characteristic_lengthscales

    local_dts = characteristic_lengthscales(actx, dcoll) / c

    return op.nodal_min(dcoll, "vol", local_dts)


def bump(actx, dcoll, t=0):
    source_center = np.array([0.2, 0.35, 0.1])[:dcoll.dim]
    source_width = 0.05
    source_omega = 3

    nodes = thaw(dcoll.nodes(), actx)
    center_dist = flat_obj_array([
        nodes[i] - source_center[i]
        for i in range(dcoll.dim)
        ])

    return (
        np.cos(source_omega*t)
        * actx.np.exp(
            -np.dot(center_dist, center_dist)
            / source_width**2))


def main(ctx_factory, dim=2, order=3, visualize=False):
    cl_ctx = ctx_factory()
    queue = cl.CommandQueue(cl_ctx)
    actx = PyOpenCLArrayContext(
        queue,
        allocator=cl_tools.MemoryPool(cl_tools.ImmediateAllocator(queue)),
        force_device_scalars=True,
    )

    comm = MPI.COMM_WORLD
    num_parts = comm.Get_size()

    from meshmode.distributed import MPIMeshDistributor, get_partition_by_pymetis
    mesh_dist = MPIMeshDistributor(comm)

    nel_1d = 16

    if mesh_dist.is_mananger_rank():
        from meshmode.mesh.generation import generate_regular_rect_mesh
        mesh = generate_regular_rect_mesh(
                a=(-0.5,)*dim,
                b=(0.5,)*dim,
                nelements_per_axis=(nel_1d,)*dim)

        logger.info("%d elements", mesh.nelements)

        part_per_element = get_partition_by_pymetis(mesh, num_parts)

        local_mesh = mesh_dist.send_mesh_parts(mesh, part_per_element, num_parts)

        del mesh

    else:
        local_mesh = mesh_dist.receive_mesh_part()

    dcoll = DiscretizationCollection(actx, local_mesh, order=order,
                    mpi_communicator=comm)

    fields = flat_obj_array(
            bump(actx, dcoll),
            [dcoll.zeros(actx) for i in range(dcoll.dim)]
            )

    c = 1
    dt = 0.45 * estimate_rk4_timestep(actx, dcoll, c)

    vis = make_visualizer(dcoll)

    def rhs(t, w):
        return wave_operator(dcoll, c=c, w=w)

    if comm.rank == 0:
        logger.info("dt = %g", dt)

    t = 0
    t_final = 3
    istep = 0
    while t < t_final:
        fields = rk4_step(fields, t, dt, rhs)

        l2norm = op.norm(dcoll, fields[0], 2)

        if istep % 10 == 0:
            linfnorm = op.norm(dcoll, fields[0], np.inf)
            nodalmax = op.nodal_max(dcoll, "vol", fields[0])
            nodalmin = op.nodal_min(dcoll, "vol", fields[0])
            if comm.rank == 0:
                logger.info(f"step: {istep} t: {t} "
                            f"L2: {l2norm} "
                            f"Linf: {linfnorm} "
                            f"sol max: {nodalmax} "
                            f"sol min: {nodalmin}")
            if visualize:
                vis.write_parallel_vtk_file(
                    comm,
                    f"fld-wave-eager-mpi-{{rank:03d}}-{istep:04d}.vtu",
                    [
                        ("u", fields[0]),
                        ("v", fields[1:]),
                    ]
                )

        t += dt
        istep += 1

        # NOTE: These are here to ensure the solution is bounded for the
        # time interval specified
        assert l2norm < 1


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dim", default=2, type=int)
    parser.add_argument("--order", default=3, type=int)
    parser.add_argument("--visualize", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    main(cl.create_some_context,
         dim=args.dim,
         order=args.order,
         visualize=args.visualize)

# vim: foldmethod=marker
