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

from grudge.array_context import PyOpenCLArrayContext
from arraycontext.container.traversal import thaw

logger = logging.getLogger(__name__)
logging.basicConfig()
logger.setLevel(logging.INFO)

from grudge import DiscretizationCollection
from grudge.shortcuts import set_up_rk4

from meshmode.dof_array import flat_norm

from pytools.obj_array import flat_obj_array

import grudge.op as op


def simple_mpi_communication_entrypoint():
    cl_ctx = cl.create_some_context()
    queue = cl.CommandQueue(cl_ctx)
    actx = PyOpenCLArrayContext(queue, force_device_scalars=True)

    from meshmode.distributed import MPIMeshDistributor, get_partition_by_pymetis
    from meshmode.mesh import BTAG_ALL

    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    num_parts = comm.Get_size()

    mesh_dist = MPIMeshDistributor(comm)

    if mesh_dist.is_mananger_rank():
        from meshmode.mesh.generation import generate_regular_rect_mesh
        mesh = generate_regular_rect_mesh(a=(-1,)*2,
                                          b=(1,)*2,
                                          nelements_per_axis=(2,)*2)

        part_per_element = get_partition_by_pymetis(mesh, num_parts)

        local_mesh = mesh_dist.send_mesh_parts(mesh, part_per_element, num_parts)
    else:
        local_mesh = mesh_dist.receive_mesh_part()

    dcoll = DiscretizationCollection(actx, local_mesh, order=5,
            mpi_communicator=comm)

    x = thaw(dcoll.nodes(), actx)
    myfunc = actx.np.sin(np.dot(x, [2, 3]))

    from grudge.dof_desc import as_dofdesc

    dd_int = as_dofdesc("int_faces")
    dd_vol = as_dofdesc("vol")
    dd_af = as_dofdesc("all_faces")

    all_faces_func = op.project(dcoll, dd_vol, dd_af, myfunc)
    int_faces_func = op.project(dcoll, dd_vol, dd_int, myfunc)
    bdry_faces_func = op.project(dcoll, BTAG_ALL, dd_af,
                                 op.project(dcoll, dd_vol, BTAG_ALL, myfunc))

    hopefully_zero = (
        op.project(
            dcoll, "int_faces", "all_faces",
            dcoll.opposite_face_connection()(int_faces_func)
        )
        + sum(op.project(dcoll, tpair.dd, "all_faces", tpair.int)
              for tpair in op.cross_rank_trace_pairs(dcoll, myfunc))
    ) - (all_faces_func - bdry_faces_func)

    error = flat_norm(hopefully_zero, ord=np.inf)

    print(__file__)
    with np.printoptions(threshold=100000000, suppress=True):
        logger.debug(hopefully_zero)
    logger.info("error: %.5e", error)

    assert error < 1e-14


def mpi_communication_entrypoint():
    cl_ctx = cl.create_some_context()
    queue = cl.CommandQueue(cl_ctx)
    actx = PyOpenCLArrayContext(queue, force_device_scalars=True)

    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    i_local_rank = comm.Get_rank()
    num_parts = comm.Get_size()

    from meshmode.distributed import MPIMeshDistributor, get_partition_by_pymetis
    mesh_dist = MPIMeshDistributor(comm)

    dim = 2
    order = 4

    if mesh_dist.is_mananger_rank():
        from meshmode.mesh.generation import generate_regular_rect_mesh
        mesh = generate_regular_rect_mesh(a=(-0.5,)*dim,
                                          b=(0.5,)*dim,
                                          nelements_per_axis=(16,)*dim)

        part_per_element = get_partition_by_pymetis(mesh, num_parts)

        local_mesh = mesh_dist.send_mesh_parts(mesh, part_per_element, num_parts)

        del mesh
    else:
        local_mesh = mesh_dist.receive_mesh_part()

    dcoll = DiscretizationCollection(actx, local_mesh, order=order,
                                     mpi_communicator=comm)

    def source_f(actx, dcoll, t=0):
        source_center = np.array([0.1, 0.22, 0.33])[:dcoll.dim]
        source_width = 0.05
        source_omega = 3
        nodes = thaw(dcoll.nodes(), actx)
        source_center_dist = flat_obj_array(
            [nodes[i] - source_center[i] for i in range(dcoll.dim)]
        )
        return (
            np.sin(source_omega*t)
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
        flux_type="upwind"
    )

    fields = flat_obj_array(
        dcoll.zeros(actx),
        [dcoll.zeros(actx) for i in range(dcoll.dim)]
    )

    dt = actx.to_numpy(
        2/3 * wave_op.estimate_rk4_timestep(actx, dcoll, fields=fields))

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

    dt_stepper = set_up_rk4("w", dt, fields, rhs)

    final_t = 4
    nsteps = int(final_t/dt)
    logger.info("[%04d] dt %.5e nsteps %4d", i_local_rank, dt, nsteps)

    step = 0

    def norm(u):
        return op.norm(dcoll, u, 2)

    from time import time
    t_last_step = time()

    logmgr.tick_before()
    for event in dt_stepper.run(t_end=final_t):
        if isinstance(event, dt_stepper.StateComputed):
            assert event.component_id == "w"

            step += 1
            logger.info("[%04d] t = %.5e |u| = %.5e ellapsed %.5e",
                        step, event.t,
                        norm(u=event.state_component[0]),
                        time() - t_last_step)

            t_last_step = time()
            logmgr.tick_after()
            logmgr.tick_before()

    logmgr.tick_after()
    logmgr.close()
    logger.info("Rank %d exiting", i_local_rank)


# {{{ MPI test pytest entrypoint

@pytest.mark.mpi
@pytest.mark.parametrize("num_ranks", [2])
def test_mpi(num_ranks):
    pytest.importorskip("mpi4py")
    pytest.importorskip("pymetis")

    from subprocess import check_call
    import sys
    # NOTE: CI uses OpenMPI; -x to pass env vars. MPICH uses -env
    check_call([
        "mpiexec", "-np", str(num_ranks),
        "-x", "RUN_WITHIN_MPI=1",
        "-x", "TEST_MPI_COMMUNICATION=1",
        sys.executable, __file__])


@pytest.mark.mpi
@pytest.mark.parametrize("num_ranks", [2])
def test_simple_mpi(num_ranks):
    pytest.importorskip("mpi4py")
    pytest.importorskip("pymetis")

    from subprocess import check_call
    import sys
    # NOTE: CI uses OpenMPI; -x to pass env vars. MPICH uses -env
    check_call([
        "mpiexec", "-np", str(num_ranks),
        "-x", "RUN_WITHIN_MPI=1",
        "-x", "TEST_SIMPLE_MPI_COMMUNICATION=1",
        # https://mpi4py.readthedocs.io/en/stable/mpi4py.run.html
        sys.executable, "-m", "mpi4py.run", __file__])

# }}}


if __name__ == "__main__":
    if "RUN_WITHIN_MPI" in os.environ:
        if "TEST_MPI_COMMUNICATION" in os.environ:
            mpi_communication_entrypoint()
        elif "TEST_SIMPLE_MPI_COMMUNICATION" in os.environ:
            simple_mpi_communication_entrypoint()
    else:
        import sys
        if len(sys.argv) > 1:
            exec(sys.argv[1])
        else:
            from pytest import main
            main([__file__])
# vim: fdm=marker
