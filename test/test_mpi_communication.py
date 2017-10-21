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

import logging
logger = logging.getLogger(__name__)

import numpy as np


def mpi_communication_entrypoint():
    from meshmode.distributed import MPIMeshDistributor, MPIBoundaryCommunicator

    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    num_parts = comm.Get_size()

    mesh_dist = MPIMeshDistributor(comm)

    if mesh_dist.is_mananger_rank():
        np.random.seed(42)
        from meshmode.mesh.generation import generate_warped_rect_mesh
        meshes = [generate_warped_rect_mesh(3, order=4, n=4) for _ in range(2)]

        from meshmode.mesh.processing import merge_disjoint_meshes
        mesh = merge_disjoint_meshes(meshes)

        part_per_element = np.random.randint(num_parts, size=mesh.nelements)

        local_mesh = mesh_dist.send_mesh_parts(mesh, part_per_element, num_parts)
    else:
        local_mesh = mesh_dist.receive_mesh_part()

    from meshmode.discretization.poly_element\
                    import PolynomialWarpAndBlendGroupFactory
    group_factory = PolynomialWarpAndBlendGroupFactory(4)
    import pyopencl as cl
    cl_ctx = cl.create_some_context()
    queue = cl.CommandQueue(cl_ctx)

    from meshmode.discretization import Discretization
    vol_discr = Discretization(cl_ctx, local_mesh, group_factory)

    logger.debug("Rank %d exiting", rank)


# {{{ MPI test pytest entrypoint

@pytest.mark.mpi
@pytest.mark.parametrize("num_partitions", [3, 4])
def test_mpi_communication(num_partitions):
    pytest.importorskip("mpi4py")

    num_ranks = num_partitions
    from subprocess import check_call
    import sys
    newenv = os.environ.copy()
    newenv["RUN_WITHIN_MPI"] = "1"
    check_call([
        "mpiexec", "-np", str(num_ranks), "-x", "RUN_WITHIN_MPI",
        sys.executable, __file__],
        env=newenv)

# }}}

if __name__ == "__main__":
    if "RUN_WITHIN_MPI" in os.environ:
        mpi_communication_entrypoint()
    else:
        import sys
        if len(sys.argv) > 1:
            exec(sys.argv[1])
        else:
            from py.test.cmdline import main
            main([__file__])

# vim: fdm=marker
