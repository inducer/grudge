from __future__ import division, absolute_import, print_function

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
logger = logging.getLogger(__name__)

from grudge import sym, bind, DGDiscretizationWithBoundaries
from grudge.shortcuts import set_up_rk4


def simple_mpi_communication_entrypoint():
    cl_ctx = cl.create_some_context()
    queue = cl.CommandQueue(cl_ctx)
    from meshmode.distributed import MPIMeshDistributor

    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    num_parts = comm.Get_size()

    mesh_dist = MPIMeshDistributor(comm)

    if mesh_dist.is_mananger_rank():
        from meshmode.mesh.generation import generate_regular_rect_mesh
        mesh = generate_regular_rect_mesh(a=(-1,)*2,
                                          b=(1,)*2,
                                          n=(3,)*2)

        from pymetis import part_graph
        _, p = part_graph(num_parts,
                          xadj=mesh.nodal_adjacency.neighbors_starts.tolist(),
                          adjncy=mesh.nodal_adjacency.neighbors.tolist())
        part_per_element = np.array(p)

        local_mesh = mesh_dist.send_mesh_parts(mesh, part_per_element, num_parts)
    else:
        local_mesh = mesh_dist.receive_mesh_part()

    vol_discr = DGDiscretizationWithBoundaries(cl_ctx, local_mesh, order=5,
            mpi_communicator=comm)

    sym_x = sym.nodes(local_mesh.dim)
    myfunc_symb = sym.sin(np.dot(sym_x, [2, 3]))
    myfunc = bind(vol_discr, myfunc_symb)(queue)

    sym_all_faces_func = sym.cse(
        sym.interp("vol", "all_faces")(sym.var("myfunc")))
    sym_int_faces_func = sym.cse(
        sym.interp("vol", "int_faces")(sym.var("myfunc")))
    sym_bdry_faces_func = sym.cse(
        sym.interp(sym.BTAG_ALL, "all_faces")(
            sym.interp("vol", sym.BTAG_ALL)(sym.var("myfunc"))))

    bound_face_swap = bind(vol_discr,
        sym.interp("int_faces", "all_faces")(
            sym.OppositeInteriorFaceSwap("int_faces")(
                sym_int_faces_func)
            ) - (sym_all_faces_func - sym_bdry_faces_func)
            )

    print(bound_face_swap)
    # 1/0

    hopefully_zero = bound_face_swap(queue, myfunc=myfunc)
    import numpy.linalg as la
    error = la.norm(hopefully_zero.get())

    np.set_printoptions(threshold=100000000, suppress=True)
    print(hopefully_zero)
    print(error)

    assert error < 1e-14


def mpi_communication_entrypoint():
    cl_ctx = cl.create_some_context()
    queue = cl.CommandQueue(cl_ctx)
    from meshmode.distributed import MPIMeshDistributor

    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    num_parts = comm.Get_size()

    mesh_dist = MPIMeshDistributor(comm)

    dims = 2
    dt = 0.04
    order = 4

    if mesh_dist.is_mananger_rank():
        from meshmode.mesh.generation import generate_regular_rect_mesh
        mesh = generate_regular_rect_mesh(a=(-0.5,)*dims,
                                          b=(0.5,)*dims,
                                          n=(16,)*dims)

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

    # FIXME
    # dt = op.estimate_rk4_timestep(vol_discr, fields=fields)

    # FIXME: Should meshmode consider BTAG_PARTITION to be a boundary?
    #           Fails because: "found faces without boundary conditions"
    # op.check_bc_coverage(local_mesh)

    # print(sym.pretty(op.sym_operator()))
    bound_op = bind(vol_discr, op.sym_operator())
    # print(bound_op)
    # 1/0

    def rhs(t, w):
        return bound_op(queue, t=t, w=w)

    dt_stepper = set_up_rk4("w", dt, fields, rhs)

    final_t = 10
    nsteps = int(final_t/dt)
    print("rank=%d dt=%g nsteps=%d" % (rank, dt, nsteps))

    from grudge.shortcuts import make_visualizer
    vis = make_visualizer(vol_discr, vis_order=order)

    step = 0

    norm = bind(vol_discr, sym.norm(2, sym.var("u")))

    from time import time
    t_last_step = time()

    for event in dt_stepper.run(t_end=final_t):
        if isinstance(event, dt_stepper.StateComputed):
            assert event.component_id == "w"

            step += 1

            print(step, event.t, norm(queue, u=event.state_component[0]),
                    time()-t_last_step)
            if step % 10 == 0:
                vis.write_vtk_file("rank%d-fld-%04d.vtu" % (rank, step),
                        [
                            ("u", event.state_component[0]),
                            ("v", event.state_component[1:]),
                            ])
            t_last_step = time()
    logger.debug("Rank %d exiting", rank)


# {{{ MPI test pytest entrypoint

@pytest.mark.mpi
@pytest.mark.parametrize("num_ranks", [2])
def test_mpi(num_ranks):
    pytest.importorskip("mpi4py")

    from subprocess import check_call
    import sys
    newenv = os.environ.copy()
    newenv["RUN_WITHIN_MPI"] = "1"
    newenv["TEST_MPI_COMMUNICATION"] = "1"
    check_call([
        "mpiexec", "-np", str(num_ranks), "-x", "RUN_WITHIN_MPI",
        sys.executable, __file__],
        env=newenv)


@pytest.mark.mpi
@pytest.mark.parametrize("num_ranks", [2])
def test_simple_mpi(num_ranks):
    pytest.importorskip("mpi4py")

    from subprocess import check_call
    import sys
    newenv = os.environ.copy()
    newenv["RUN_WITHIN_MPI"] = "1"
    newenv["TEST_SIMPLE_MPI_COMMUNICATION"] = "1"
    check_call([
        "mpiexec", "-np", str(num_ranks), "-x", "RUN_WITHIN_MPI",
        sys.executable, __file__],
        env=newenv)

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
            from py.test.cmdline import main
            main([__file__])

# vim: fdm=marker
