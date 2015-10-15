"""Minimal example of a grudge driver."""

from __future__ import division, print_function

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
from grudge import sym, bind, Discretization, shortcuts


def main(write_output=True):
    cl_ctx = cl.create_some_context()
    queue = cl.CommandQueue(cl_ctx)

    from meshmode.mesh.generation import generate_regular_rect_mesh
    mesh = generate_regular_rect_mesh(a=(-0.5, -0.5), b=(0.5, 0.5))

    discr = Discretization(cl_ctx, mesh, order=4)

    sym_op = sym.normal(sym.BTAG_ALL, mesh.dim)
    #sym_op = sym.nodes(mesh.dim, where=sym.BTAG_ALL)
    print(sym.pretty(sym_op))
    op = bind(discr, sym_op)
    print()
    print(op.code)

    vec = op(queue)

    bvis = shortcuts.make_boundary_visualizer(discr, 4)

    print(vec[0].shape)
    print(discr.zeros(queue).shape)
    bvis.write_vtk_file("geo.vtu", [
        ("normals", vec)
        ])


if __name__ == "__main__":
    main()
