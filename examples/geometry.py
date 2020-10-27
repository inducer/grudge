"""Minimal example of a grudge driver."""

__copyright__ = "Copyright (C) 2015 Andreas Kloeckner"

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
from grudge import sym, bind, DGDiscretizationWithBoundaries, shortcuts

from meshmode.array_context import PyOpenCLArrayContext


def main(write_output=True):
    cl_ctx = cl.create_some_context()
    queue = cl.CommandQueue(cl_ctx)
    actx = PyOpenCLArrayContext(queue)

    from meshmode.mesh.generation import generate_warped_rect_mesh
    mesh = generate_warped_rect_mesh(dim=2, order=4, n=6)

    discr = DGDiscretizationWithBoundaries(actx, mesh, order=4)

    sym_op = sym.normal(sym.BTAG_ALL, mesh.dim)
    #sym_op = sym.nodes(mesh.dim, where=sym.BTAG_ALL)
    print(sym.pretty(sym_op))
    op = bind(discr, sym_op)
    print()
    print(op.eval_code)

    vec = op(actx)

    vis = shortcuts.make_visualizer(discr, 4)
    vis.write_vtk_file("geo.vtu", [
        ])

    bvis = shortcuts.make_boundary_visualizer(discr, 4)
    bvis.write_vtk_file("bgeo.vtu", [
        ("normals", vec)
        ])


if __name__ == "__main__":
    main()
