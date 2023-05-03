__copyright__ = """
Copyright (C) 2017 Ellis Hoag
Copyright (C) 2017 Andreas Kloeckner
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

import pytest
import os
import numpy as np
import pyopencl as cl
import logging
import sys

from grudge.array_context import MPIPyOpenCLArrayContext, MPIPytatoArrayContext

logger = logging.getLogger(__name__)
logging.basicConfig()
logger.setLevel(logging.INFO)

from grudge import DiscretizationCollection
from grudge.shortcuts import rk4_step

from meshmode.dof_array import flat_norm

from pytools.obj_array import flat_obj_array

import grudge.op as op
import grudge.dof_desc as dof_desc


class SimpleTag:
    pass


# {{{ mpi test infrastructure

DISTRIBUTED_ACTXS = [MPIPyOpenCLArrayContext, MPIPytatoArrayContext]


def run_test_with_mpi(num_ranks, f, *args):
    import pytest
    pytest.importorskip("mpi4py")

    from pickle import dumps
    from base64 import b64encode

    invocation_info = b64encode(dumps((f, args))).decode()
    from subprocess import check_call

    # NOTE: CI uses OpenMPI; -x to pass env vars. MPICH uses -env
    check_call([
        "mpiexec", "-np", str(num_ranks),
        "-x", "RUN_WITHIN_MPI=1",
        "-x", f"INVOCATION_INFO={invocation_info}",
        sys.executable, "-m", "mpi4py", __file__])


def run_test_with_mpi_inner():
    from pickle import loads
    from base64 import b64decode
    f, (actx_class, *args) = loads(b64decode(os.environ["INVOCATION_INFO"].encode()))

    cl_context = cl.create_some_context()
    queue = cl.CommandQueue(cl_context)

    from mpi4py import MPI
    comm = MPI.COMM_WORLD

    if actx_class is MPIPytatoArrayContext:
        actx = actx_class(comm, queue, mpi_base_tag=15000)
    elif actx_class is MPIPyOpenCLArrayContext:
        actx = actx_class(comm, queue, force_device_scalars=True)
    else:
        raise ValueError("unknown actx_class")

    f(actx, *args)

# }}}


# {{{ func_comparison

@pytest.mark.parametrize("actx_class", DISTRIBUTED_ACTXS)
@pytest.mark.parametrize("num_ranks", [2])
def test_func_comparison_mpi(actx_class, num_ranks):
    run_test_with_mpi(
            num_ranks, _test_func_comparison_mpi_communication_entrypoint,
            actx_class)


def _test_func_comparison_mpi_communication_entrypoint(actx):
    """Discretize a function, communicate it, check that it matches the
    function discretized by the other end.
    """

    comm = actx.mpi_communicator

    from meshmode.distributed import (
            get_partition_by_pymetis, membership_list_to_map)
    from meshmode.mesh import BTAG_ALL
    from meshmode.mesh.processing import partition_mesh

    num_parts = comm.size

    if comm.rank == 0:
        from meshmode.mesh.generation import generate_regular_rect_mesh
        mesh = generate_regular_rect_mesh(a=(-1,)*2,
                                          b=(1,)*2,
                                          nelements_per_axis=(2,)*2)

        part_id_to_part = partition_mesh(mesh,
                       membership_list_to_map(
                           get_partition_by_pymetis(mesh, num_parts)))
        parts = [part_id_to_part[i] for i in range(num_parts)]
        local_mesh = comm.scatter(parts)
    else:
        local_mesh = comm.scatter(None)

    dcoll = DiscretizationCollection(actx, local_mesh, order=5)

    x = actx.thaw(dcoll.nodes())
    myfunc = actx.np.sin(np.dot(x, [2, 3]))

    from grudge.dof_desc import as_dofdesc

    dd_int = as_dofdesc("int_faces")
    dd_vol = as_dofdesc("vol")
    dd_af = as_dofdesc("all_faces")

    all_faces_func = op.project(dcoll, dd_vol, dd_af, myfunc)
    int_faces_func = op.project(dcoll, dd_vol, dd_int, myfunc)
    bdry_faces_func = op.project(dcoll, BTAG_ALL, dd_af,
                                 op.project(dcoll, dd_vol, BTAG_ALL, myfunc))

    def hopefully_zero():
        return (
            op.project(
                dcoll, "int_faces", "all_faces",
                dcoll.opposite_face_connection(
                    dof_desc.BoundaryDomainTag(
                        dof_desc.FACE_RESTR_INTERIOR, dof_desc.VTAG_ALL)
                    )(int_faces_func)
            )
            + sum(op.project(dcoll, tpair.dd, "all_faces", tpair.ext)
                  for tpair in op.cross_rank_trace_pairs(dcoll, myfunc,
                      comm_tag=SimpleTag))
        ) - (all_faces_func - bdry_faces_func)

    hopefully_zero_result = actx.compile(hopefully_zero)()

    error = actx.to_numpy(flat_norm(hopefully_zero_result, ord=np.inf))

    with np.printoptions(threshold=100000000, suppress=True):
        logger.debug(hopefully_zero)
    logger.info("error: %.5e", error)

    assert error < 1e-14


# }}}


# {{{ wave operator

@pytest.mark.parametrize("actx_class", DISTRIBUTED_ACTXS)
@pytest.mark.parametrize("num_ranks", [2])
def test_mpi_wave_op(actx_class, num_ranks):
    run_test_with_mpi(num_ranks, _test_mpi_wave_op_entrypoint, actx_class)


def _test_mpi_wave_op_entrypoint(actx, visualize=False):
    comm = actx.mpi_communicator
    num_parts = comm.size

    from meshmode.distributed import (
            get_partition_by_pymetis, membership_list_to_map)
    from meshmode.mesh.processing import partition_mesh

    dim = 2
    order = 4

    if comm.rank == 0:
        from meshmode.mesh.generation import generate_regular_rect_mesh
        mesh = generate_regular_rect_mesh(a=(-0.5,)*dim,
                                          b=(0.5,)*dim,
                                          nelements_per_axis=(16,)*dim)

        part_id_to_part = partition_mesh(mesh,
                       membership_list_to_map(
                           get_partition_by_pymetis(mesh, num_parts)))
        parts = [part_id_to_part[i] for i in range(num_parts)]
        local_mesh = comm.scatter(parts)

        del mesh
    else:
        local_mesh = comm.scatter(None)

    dcoll = DiscretizationCollection(actx, local_mesh, order=order)

    def source_f(actx, dcoll, t=0):
        source_center = np.array([0.1, 0.22, 0.33])[:dcoll.dim]
        source_width = 0.05
        source_omega = 3
        nodes = actx.thaw(dcoll.nodes())
        source_center_dist = flat_obj_array(
            [nodes[i] - source_center[i] for i in range(dcoll.dim)]
        )
        return (
            actx.np.sin(source_omega*t)
            * actx.np.exp(
                -np.dot(source_center_dist, source_center_dist)
                / source_width**2
            )
        )

    from grudge.models.wave import WeakWaveOperator
    from meshmode.mesh import BTAG_ALL, BTAG_NONE

    wave_op = WeakWaveOperator(
        dcoll,
        0.1,
        source_f=source_f,
        dirichlet_tag=BTAG_NONE,
        neumann_tag=BTAG_NONE,
        radiation_tag=BTAG_ALL,
        flux_type="upwind",
        comm_tag=SimpleTag,
    )

    fields = flat_obj_array(
        dcoll.zeros(actx),
        [dcoll.zeros(actx) for i in range(dcoll.dim)]
    )

    dt = actx.to_numpy(
        wave_op.estimate_rk4_timestep(actx, dcoll, fields=fields))

    wave_op.check_bc_coverage(local_mesh)

    from logpyle import LogManager, \
            add_general_quantities, \
            add_run_info
    log_filename = None
    # NOTE: LogManager hangs when using a file on a shared directory.
    # log_filename = "grudge_log.dat"
    logmgr = LogManager(log_filename, "w", comm)
    add_run_info(logmgr)
    add_general_quantities(logmgr)

    def rhs(t, w):
        return wave_op.operator(t, w)

    compiled_rhs = actx.compile(rhs)

    final_t = 4
    nsteps = int(final_t/dt)
    logger.info("[%04d] dt %.5e nsteps %4d", comm.rank, dt, nsteps)

    step = 0

    from time import time
    t_last_step = time()

    if visualize:
        from grudge.shortcuts import make_visualizer
        vis = make_visualizer(dcoll)

    logmgr.tick_before()
    for step in range(nsteps):
        t = step*dt
        fields = rk4_step(fields, t=t, h=dt, f=compiled_rhs)
        fields = actx.thaw(actx.freeze(fields))

        norm = actx.to_numpy(op.norm(dcoll, fields, 2))
        logger.info("[%04d] t = %.5e |u| = %.5e elapsed %.5e",
                    step, t, norm, time() - t_last_step)

        if visualize:
            vis.write_parallel_vtk_file(
                comm,
                f"fld-wave-mpi-{type(actx).__name__}-{{rank:03d}}-{step:04d}.vtu",
                [
                    ("u", fields[0]),
                    ("v", fields[1:]),
                ]
            )
        assert norm < 1

        t_last_step = time()
        logmgr.tick_after()
        logmgr.tick_before()

    logmgr.tick_after()
    logmgr.close()
    logger.info("Rank %d exiting", comm.rank)

# }}}


if __name__ == "__main__":
    if "RUN_WITHIN_MPI" in os.environ:
        run_test_with_mpi_inner()
    elif len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        from pytest import main
        main([__file__])

# vim: fdm=marker
