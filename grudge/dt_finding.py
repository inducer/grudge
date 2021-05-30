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
import numpy as np
import numpy.linalg as la

def symmetric_eigenvalues(actx, amat):
    """*amat* must be complex-valued, or ``actx.np.sqrt`` must automatically
    up-cast to complex data.
    """

    # https://gist.github.com/inducer/75ede170638c389c387e72e0ef1f0ef4
    sqrt = actx.np.sqrt

    if amat.shape == (1, 1):
        (a,), = amat
        return a
    elif amat.shape == (2, 2):
        (a, b), (_b, c) = amat
        x0 = sqrt(a**2 - 2*a*c + 4*b**2 + c**2)/2
        x1 = a/2 + c/2
        return [-x0 + x1, x0 + x1]
    elif amat.shape == (3, 3):
        # This is likely awful numerically, but *shrug*, we're only using
        # it for time step estimation.
        (a, b, c), (_b, d, e), (_c, _e, f) = amat
        x0 = a*d
        x1 = f*x0
        x2 = b*c*e
        x3 = e**2
        x4 = a*x3
        x5 = b**2
        x6 = f*x5
        x7 = c**2
        x8 = d*x7
        x9 = -a - d - f
        x10 = x9**3
        x11 = a*f
        x12 = d*f
        x13 = (-9*a - 9*d - 9*f)*(x0 + x11 + x12 - x3 - x5 - x7)
        x14 = -3*x0 - 3*x11 - 3*x12 + 3*x3 + 3*x5 + 3*x7 + x9**2
        x15_0 = (-4*x14**3 + (-27*x1 + 2*x10 - x13 - 54*x2 + 27*x4 + 27*x6
                    + 27*x8)**2)
        x15_1 = sqrt(x15_0)
        x15_2 =(-27*x1/2 + x10 - x13/2 - 27*x2 + 27*x4/2 + 27*x6/2 + 27*x8/2
                + x15_1/2)
        x15 = x15_2**(1/3)
        x16 = x15/3
        x17 = x14/(3*x15)
        x18 = a/3 + d/3 + f/3
        x19 = 3**(1/2)*1j/2
        x20 = x19 - 1/2
        x21 = -x19 - 1/2
        return [-x16 - x17 + x18, -x16*x20 - x17/x20 + x18, -x16*x21 - x17/x21 + x18]
    else:
        raise NotImplementedError(
                "unsupported shape ({amat.shape}) for eigenvalue finding")


def min_singular_value_of_mapping_jacobian(actx, dcoll, dd=None):
    from grudge.geometry import forward_metric_derivative_mat

    if dd is None:
        from grudge.dof_desc import DD_VOLUME
        dd = DD_VOLUME

    fmd = forward_metric_derivative_mat(actx, dcoll, dd=dd)
    ata = fmd @ fmd.T

    complex_dtype = dcoll.discr_from_dd(dd).complex_dtype
    from arraycontext import rec_map_array_container
    ata_complex = rec_map_array_container(
            lambda ary: ary.astype(complex_dtype),
            ata)

    sing_values = [
            actx.np.sqrt(abs(v))
            for v in symmetric_eigenvalues(actx, ata_complex)]

    from functools import reduce
    return reduce(actx.np.minimum, sing_values)


# {{{ old, busted stuff

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

# }}}

# vim: foldmethod=marker
