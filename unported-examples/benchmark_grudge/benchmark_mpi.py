import os
import numpy as np
import pyopencl as cl

from grudge import sym, bind, DGDiscretizationWithBoundaries
from grudge.shortcuts import set_up_rk4


def simple_wave_entrypoint(dim=2, num_elems=256, order=4, num_steps=30,
                           log_filename="grudge.dat"):
    cl_ctx = cl.create_some_context()
    queue = cl.CommandQueue(cl_ctx)

    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    num_parts = comm.Get_size()
    n = int(num_elems ** (1./dim))

    from meshmode.distributed import MPIMeshDistributor
    mesh_dist = MPIMeshDistributor(comm)

    if mesh_dist.is_mananger_rank():
        from meshmode.mesh.generation import generate_regular_rect_mesh
        mesh = generate_regular_rect_mesh(a=(-0.5,)*dim,
                                          b=(0.5,)*dim,
                                          n=(n,)*dim)

        from pymetis import part_graph
        _, p = part_graph(num_parts,
                          xadj=mesh.nodal_adjacency.neighbors_starts.tolist(),
                          adjncy=mesh.nodal_adjacency.neighbors.tolist())
        part_per_element = np.array(p)

        local_mesh = mesh_dist.send_mesh_parts(mesh, part_per_element, num_parts)
    else:
        local_mesh = mesh_dist.receive_mesh_part()

    vol_discr = DGDiscretizationWithBoundaries(cl_ctx, local_mesh, order=order,
                                               mpi_communicator=comm)

    source_center = np.array([0.1, 0.22, 0.33])[:local_mesh.dim]
    source_width = 0.05
    source_omega = 3

    sym_x = sym.nodes(local_mesh.dim)
    sym_source_center_dist = sym_x - source_center
    sym_t = sym.ScalarVariable("t")

    from grudge.models.wave import StrongWaveOperator
    from meshmode.mesh import BTAG_ALL, BTAG_NONE
    op = StrongWaveOperator(-0.1, vol_discr.dim,
            source_f=(
                sym.sin(source_omega*sym_t)
                * sym.exp(
                    -np.dot(sym_source_center_dist, sym_source_center_dist)
                    / source_width**2)),
            dirichlet_tag=BTAG_NONE,
            neumann_tag=BTAG_NONE,
            radiation_tag=BTAG_ALL,
            flux_type="upwind")

    from pytools.obj_array import join_fields
    fields = join_fields(vol_discr.zeros(queue),
            [vol_discr.zeros(queue) for i in range(vol_discr.dim)])

    from pytools.log import LogManager, \
            add_general_quantities, \
            add_run_info, \
            IntervalTimer, EventCounter
    # NOTE: LogManager hangs when using a file on a shared directory.
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

    bound_op = bind(vol_discr, op.sym_operator())

    def rhs(t, w):
        val, rhs.profile_data = bound_op(queue, profile_data=rhs.profile_data,
                                                log_quantities=log_quantities,
                                                t=t, w=w)
        return val
    rhs.profile_data = {}

    dt = 0.04
    dt_stepper = set_up_rk4("w", dt, fields, rhs)

    logmgr.tick_before()
    for event in dt_stepper.run(t_end=dt * num_steps):
        if isinstance(event, dt_stepper.StateComputed):
            logmgr.tick_after()
            logmgr.tick_before()
    logmgr.tick_after()

    def print_profile_data(data):
        print("""execute() for rank %d:
            \tInstruction Evaluation: %f%%
            \tFuture Evaluation: %f%%
            \tBusy Wait: %f%%
            \tTotal: %f seconds""" %
            (comm.Get_rank(),
             data['insn_eval_time'] / data['total_time'] * 100,
             data['future_eval_time'] / data['total_time'] * 100,
             data['busy_wait_time'] / data['total_time'] * 100,
             data['total_time']))

    print_profile_data(rhs.profile_data)
    logmgr.close()


if __name__ == "__main__":
    assert "RUN_WITHIN_MPI" in os.environ, "Must run within mpi"
    import sys
    assert len(sys.argv) == 5, \
        "Usage: %s %s num_elems order num_steps logfile" \
        % (sys.executable, sys.argv[0])
    simple_wave_entrypoint(num_elems=int(sys.argv[1]),
                           order=int(sys.argv[2]),
                           num_steps=int(sys.argv[3]),
                           log_filename=sys.argv[4])
