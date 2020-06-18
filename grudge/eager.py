from __future__ import division, print_function

__copyright__ = "Copyright (C) 2020 Andreas Kloeckner"

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
from grudge.discretization import DGDiscretizationWithBoundaries
from pytools import memoize_method
from pytools.obj_array import obj_array_vectorize
import pyopencl.array as cla  # noqa
from grudge import sym, bind

from meshmode.mesh import BTAG_ALL, BTAG_NONE  # noqa
from meshmode.dof_array import freeze, DOFArray


class EagerDGDiscretization(DGDiscretizationWithBoundaries):
    def interp(self, src, tgt, vec):
        if (isinstance(vec, np.ndarray)
                and vec.dtype.char == "O"
                and not isinstance(vec, DOFArray)):
            return obj_array_vectorize(
                    lambda el: self.interp(src, tgt, el), vec)

        return self.connection_from_dds(src, tgt)(vec)

    def nodes(self):
        return self._volume_discr.nodes()

    @memoize_method
    def _bound_grad(self):
        return bind(self, sym.nabla(self.dim) * sym.Variable("u"), local_only=True)

    def grad(self, vec):
        return self._bound_grad()(u=vec)

    def div(self, vecs):
        return sum(
                self.grad(vec_i)[i] for i, vec_i in enumerate(vecs))

    @memoize_method
    def _bound_weak_grad(self):
        return bind(self, sym.stiffness_t(self.dim) * sym.Variable("u"),
                local_only=True)

    def weak_grad(self, vec):
        return self._bound_weak_grad()(u=vec)

    def weak_div(self, vecs):
        return sum(
                self.weak_grad(vec_i)[i] for i, vec_i in enumerate(vecs))

    @memoize_method
    def normal(self, dd):
        surface_discr = self.discr_from_dd(dd)
        actx = surface_discr._setup_actx
        return freeze(
                bind(self,
                    sym.normal(dd, surface_discr.ambient_dim, surface_discr.dim),
                    local_only=True)
                (array_context=actx))

    @memoize_method
    def _bound_inverse_mass(self):
        return bind(self, sym.InverseMassOperator()(sym.Variable("u")),
                local_only=True)

    def inverse_mass(self, vec):
        if (isinstance(vec, np.ndarray)
                and vec.dtype.char == "O"
                and not isinstance(vec, DOFArray)):
            return obj_array_vectorize(
                    lambda el: self.inverse_mass(el), vec)

        return self._bound_inverse_mass()(u=vec)

    @memoize_method
    def _bound_face_mass(self):
        u = sym.Variable("u", dd=sym.as_dofdesc("all_faces"))
        return bind(self, sym.FaceMassOperator()(u), local_only=True)

    def face_mass(self, vec):
        if (isinstance(vec, np.ndarray)
                and vec.dtype.char == "O"
                and not isinstance(vec, DOFArray)):
            return obj_array_vectorize(
                    lambda el: self.face_mass(el), vec)

        return self._bound_face_mass()(u=vec)

    @memoize_method
    def _norm(self, p):
        return bind(self, sym.norm(p, sym.var("arg")), local_only=True)

    def norm(self, vec, p=2):
        return self._norm(p)(arg=vec)


def interior_trace_pair(discr, vec):
    i = discr.interp("vol", "int_faces", vec)

    if (isinstance(vec, np.ndarray)
            and vec.dtype.char == "O"
            and not isinstance(vec, DOFArray)):
        e = obj_array_vectorize(
                lambda el: discr.opposite_face_connection()(el),
                i)

    from grudge.symbolic.primitives import TracePair
    return TracePair("int_faces", i, e)


# vim: foldmethod=marker
