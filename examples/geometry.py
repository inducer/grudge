"""Minimal example of viewing geometric quantities."""

__copyright__ = """
Copyright (C) 2015 Andreas Kloeckner
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


import numpy as np  # noqa
import pyopencl as cl
import pyopencl.tools as cl_tools

from grudge.array_context import PyOpenCLArrayContext

from grudge import shortcuts
from grudge import geometry
from grudge.discretization import make_discretization_collection


def main(write_output=True):
    cl_ctx = cl.create_some_context()
    queue = cl.CommandQueue(cl_ctx)
    actx = PyOpenCLArrayContext(
        queue,
        allocator=cl_tools.MemoryPool(cl_tools.ImmediateAllocator(queue)),
        force_device_scalars=True,
    )

    from meshmode.mesh import BTAG_ALL
    from meshmode.mesh.generation import generate_warped_rect_mesh

    mesh = generate_warped_rect_mesh(dim=2, order=4, nelements_side=6)
    dcoll = make_discretization_collection(actx, mesh, order=4)

    nodes = actx.thaw(dcoll.nodes())
    bdry_nodes = actx.thaw(dcoll.nodes(dd=BTAG_ALL))
    bdry_normals = geometry.normal(actx, dcoll, dd=BTAG_ALL)

    if write_output:
        vis = shortcuts.make_visualizer(dcoll)
        vis.write_vtk_file("geo.vtu", [("nodes", nodes)])

        bvis = shortcuts.make_boundary_visualizer(dcoll)
        bvis.write_vtk_file("bgeo.vtu", [("bdry normals", bdry_normals),
                                         ("bdry nodes", bdry_nodes)])


if __name__ == "__main__":
    main()
