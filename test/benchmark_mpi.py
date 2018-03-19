import os
import numpy as np
import pyopencl as cl

from grudge import sym, bind, DGDiscretizationWithBoundaries
from grudge.shortcuts import set_up_rk4



def simple_wave_entrypoint(dim=2, order=4, n=256):
    cl_ctx = cl.create_some_context()
    queue = cl.CommandQueue(cl_ctx)

    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    i_local_rank = comm.Get_rank()
    num_parts = comm.Get_size()

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

    bound_op = bind(vol_discr, op.sym_operator())

    def rhs(t, w):
        return bound_op(queue, t=t, w=w)

    dt = 0.04
    dt_stepper = set_up_rk4("w", dt, fields, rhs)

    final_t = 4
    nsteps = int(final_t/dt)

    for event in dt_stepper.run(t_end=final_t):
        pass

if __name__ == "__main__":
    if "RUN_WITHIN_MPI" in os.environ:
        import sys
        mesh_size = 64
        if len(sys.argv) == 2:
            mesh_size = int(sys.argv[1])
        simple_wave_entrypoint(n=mesh_size)
    else:
        assert 0, "Must run within mpi"
