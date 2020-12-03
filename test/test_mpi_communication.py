__copyright__ = """
Copyright (C) 2017 Ellis Hoag
Copyright (C) 2017 Andreas Kloeckner
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

from meshmode.array_context import PyOpenCLArrayContext

logger = logging.getLogger(__name__)
logging.basicConfig()
logger.setLevel(logging.INFO)

from grudge import sym, bind, DGDiscretizationWithBoundaries
from grudge.shortcuts import set_up_rk4


def simple_mpi_communication_entrypoint():
    cl_ctx = cl.create_some_context()
    queue = cl.CommandQueue(cl_ctx)
    actx = PyOpenCLArrayContext(queue)

    from meshmode.distributed import MPIMeshDistributor, get_partition_by_pymetis

    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    num_parts = comm.Get_size()

    mesh_dist = MPIMeshDistributor(comm)

    if mesh_dist.is_mananger_rank():
        from meshmode.mesh.generation import generate_regular_rect_mesh
        mesh = generate_regular_rect_mesh(a=(-1,)*2,
                                          b=(1,)*2,
                                          n=(3,)*2)

        part_per_element = get_partition_by_pymetis(mesh, num_parts)

        local_mesh = mesh_dist.send_mesh_parts(mesh, part_per_element, num_parts)
    else:
        local_mesh = mesh_dist.receive_mesh_part()

    vol_discr = DGDiscretizationWithBoundaries(actx, local_mesh, order=5,
            mpi_communicator=comm)

    sym_x = sym.nodes(local_mesh.dim)
    myfunc_symb = sym.sin(np.dot(sym_x, [2, 3]))
    myfunc = bind(vol_discr, myfunc_symb)(actx)

    sym_all_faces_func = sym.cse(
        sym.project("vol", "all_faces")(sym.var("myfunc")))
    sym_int_faces_func = sym.cse(
        sym.project("vol", "int_faces")(sym.var("myfunc")))
    sym_bdry_faces_func = sym.cse(
        sym.project(sym.BTAG_ALL, "all_faces")(
            sym.project("vol", sym.BTAG_ALL)(sym.var("myfunc"))))

    bound_face_swap = bind(vol_discr,
        sym.project("int_faces", "all_faces")(
            sym.OppositeInteriorFaceSwap("int_faces")(
                sym_int_faces_func)
            ) - (sym_all_faces_func - sym_bdry_faces_func)
            )

    hopefully_zero = bound_face_swap(myfunc=myfunc)
    error = actx.np.linalg.norm(hopefully_zero, ord=np.inf)

    print(__file__)
    with np.printoptions(threshold=100000000, suppress=True):
        logger.debug(hopefully_zero)
    logger.info("error: %.5e", error)

    assert error < 1e-14


def mpi_communication_entrypoint():
    cl_ctx = cl.create_some_context()
    queue = cl.CommandQueue(cl_ctx)
    actx = PyOpenCLArrayContext(queue)

    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    i_local_rank = comm.Get_rank()
    num_parts = comm.Get_size()

    from meshmode.distributed import MPIMeshDistributor, get_partition_by_pymetis
    mesh_dist = MPIMeshDistributor(comm)

    dim = 2
    dt = 0.04
    order = 4

    if mesh_dist.is_mananger_rank():
        from meshmode.mesh.generation import generate_regular_rect_mesh
        mesh = generate_regular_rect_mesh(a=(-0.5,)*dim,
                                          b=(0.5,)*dim,
                                          n=(16,)*dim)

        part_per_element = get_partition_by_pymetis(mesh, num_parts)

        local_mesh = mesh_dist.send_mesh_parts(mesh, part_per_element, num_parts)
    else:
        local_mesh = mesh_dist.receive_mesh_part()

    vol_discr = DGDiscretizationWithBoundaries(actx, local_mesh, order=order,
                                               mpi_communicator=comm)

    source_center = np.array([0.1, 0.22, 0.33])[:local_mesh.dim]
    source_width = 0.05
    source_omega = 3

    sym_x = sym.nodes(local_mesh.dim)
    sym_source_center_dist = sym_x - source_center
    sym_t = sym.ScalarVariable("t")

    from grudge.models.wave import WeakWaveOperator
    from meshmode.mesh import BTAG_ALL, BTAG_NONE
    op = WeakWaveOperator(0.1, vol_discr.dim,
            source_f=(
                sym.sin(source_omega*sym_t)
                * sym.exp(
                    -np.dot(sym_source_center_dist, sym_source_center_dist)
                    / source_width**2)),
            dirichlet_tag=BTAG_NONE,
            neumann_tag=BTAG_NONE,
            radiation_tag=BTAG_ALL,
            flux_type="upwind")

    from pytools.obj_array import flat_obj_array
    fields = flat_obj_array(vol_discr.zeros(actx),
            [vol_discr.zeros(actx) for i in range(vol_discr.dim)])

    # FIXME
    # dt = op.estimate_rk4_timestep(vol_discr, fields=fields)

    # FIXME: Should meshmode consider BTAG_PARTITION to be a boundary?
    #           Fails because: "found faces without boundary conditions"
    # op.check_bc_coverage(local_mesh)

    from logpyle import LogManager, \
            add_general_quantities, \
            add_run_info, \
            IntervalTimer, EventCounter
    log_filename = None
    # NOTE: LogManager hangs when using a file on a shared directory.
    # log_filename = "grudge_log.dat"
    logmgr = LogManager(log_filename, "w", comm)
    add_run_info(logmgr)
    add_general_quantities(logmgr)
    log_quantities =\
        {"rank_data_swap_timer": IntervalTimer("rank_data_swap_timer",
        "Time spent evaluating RankDataSwapAssign"),
        "rank_data_swap_counter": EventCounter("rank_data_swap_counter",
        "Number of RankDataSwapAssign instructions evaluated"),
        "exec_timer": IntervalTimer("exec_timer",
        "Total time spent executing instructions"),
        "insn_eval_timer": IntervalTimer("insn_eval_timer",
        "Time spend evaluating instructions"),
        "future_eval_timer": IntervalTimer("future_eval_timer",
        "Time spent evaluating futures"),
        "busy_wait_timer": IntervalTimer("busy_wait_timer",
        "Time wasted doing busy wait")}
    for quantity in log_quantities.values():
        logmgr.add_quantity(quantity)

    logger.debug("\n%s", sym.pretty(op.sym_operator()))
    bound_op = bind(vol_discr, op.sym_operator())

    def rhs(t, w):
        val, rhs.profile_data = bound_op(profile_data=rhs.profile_data,
                                         log_quantities=log_quantities,
                                         t=t, w=w)
        return val
    rhs.profile_data = {}

    dt_stepper = set_up_rk4("w", dt, fields, rhs)

    final_t = 4
    nsteps = int(final_t/dt)
    logger.info("[%04d] dt %.5e nsteps %4d", i_local_rank, dt, nsteps)

    # from grudge.shortcuts import make_visualizer
    # vis = make_visualizer(vol_discr, vis_order=order)

    step = 0

    norm = bind(vol_discr, sym.norm(2, sym.var("u")))

    from time import time
    t_last_step = time()

    logmgr.tick_before()
    for event in dt_stepper.run(t_end=final_t):
        if isinstance(event, dt_stepper.StateComputed):
            assert event.component_id == "w"

            step += 1
            logger.debug("[%04d] t = %.5e |u| = %.5e ellapsed %.5e",
                    step, event.t,
                    norm(u=event.state_component[0]),
                    time() - t_last_step)

            # if step % 10 == 0:
            #     vis.write_vtk_file("rank%d-fld-%04d.vtu" % (i_local_rank, step),
            #                        [("u", event.state_component[0]),
            #                         ("v", event.state_component[1:])])
            t_last_step = time()
            logmgr.tick_after()
            logmgr.tick_before()
    logmgr.tick_after()

    def print_profile_data(data):
        logger.info("""execute() for rank %d:\n
            \tInstruction Evaluation: %g\n
            \tFuture Evaluation: %g\n
            \tBusy Wait: %g\n
            \tTotal: %g seconds""",
            i_local_rank,
            data["insn_eval_time"] / data["total_time"] * 100,
            data["future_eval_time"] / data["total_time"] * 100,
            data["busy_wait_time"] / data["total_time"] * 100,
            data["total_time"])

    print_profile_data(rhs.profile_data)
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
