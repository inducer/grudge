"""Helpers for estimating a stable time step."""

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

from pytools import memoize_on_first_arg
from meshmode.discretization.poly_element import PolynomialWarpAndBlendElementGroup
import numpy.linalg as la


class WarpAndBlendTimestepInfo:
    @staticmethod
    def dt_non_geometric_factor(discr, grp):
        if grp.dim == 1:
            if grp.order == 0:
                return 1
            else:
                unodes = grp.unit_nodes
                return la.norm(unodes[0] - unodes[1]) * 0.85
        else:
            unodes = grp.unit_nodes
            vertex_indices = grp.vertex_indices()
            return 2 / 3 * \
                    min(min(min(
                        la.norm(unodes[face_node_index]-unodes[vertex_index])
                        for vertex_index in vertex_indices
                        if vertex_index != face_node_index)
                        for face_node_index in face_indices)
                        for face_indices in self.face_indices())

    @staticmethod
    def dt_geometric_factor(discr, grp):
        if grp.dim == 1:
            return abs(el.map.jacobian())

        elif grp.dim == 2:
            area = abs(2 * el.map.jacobian())
            semiperimeter = sum(la.norm(vertices[vi1] - vertices[vi2])
                    for vi1, vi2 in [(0, 1), (1, 2), (2, 0)])/2
            return area / semiperimeter

        elif grp.dim == 3:
            result = abs(el.map.jacobian())/max(abs(fj) for fj in el.face_jacobians)
            if grp.order in [1, 2]:
                from warnings import warn
                warn("cowardly halving timestep for order 1 and 2 tets "
                        "to avoid CFL issues")
                result /= 2

            return result

        else:
            raise NotImplementedError("cannot estimate timestep for "
                    "%d-dimensional elements" % grp.dim)


_GROUP_TYPE_TO_INFO = {
        PolynomialWarpAndBlendElementGroup: WarpAndBlendTimestepInfo
        }


@memoize_on_first_arg
def dt_non_geometric_factor(discr):
    return min(
            _GROUP_TYPE_TO_INFO[type(grp)].dt_non_geometric_factor(discr, grp)
            for grp in discr.groups)


@memoize_on_first_arg
def dt_geometric_factor(discr):
    return min(
            _GROUP_TYPE_TO_INFO[type(grp)].dt_geometric_factor(discr, grp)
            for grp in discr.groups)
