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
from pytools.obj_array import (
        with_object_array_or_scalar,
        is_obj_array)
import pyopencl as cl
import pyopencl.array as cla  # noqa
from grudge import sym, bind
from meshmode.mesh import BTAG_ALL, BTAG_NONE  # noqa


def with_queue(queue, ary):
    return with_object_array_or_scalar(
            lambda x: x.with_queue(queue), ary)


def without_queue(ary):
    return with_queue(None, ary)


class EagerDGDiscretization(DGDiscretizationWithBoundaries):
    def interp(self, src, tgt, vec):
        from warnings import warn
        warn("using 'interp' is deprecated, use 'project' instead.",
                DeprecationWarning, stacklevel=1)

        return self.project(src, tgt, vec)

    def project(self, src, tgt, vec):
        if is_obj_array(vec):
            return with_object_array_or_scalar(
                    lambda el: self.project(src, tgt, el), vec)

        return self.connection_from_dds(src, tgt)(vec.queue, vec)

    def nodes(self):
        return self._volume_discr.nodes()

    @memoize_method
    def _bound_grad(self):
        return bind(self, sym.nabla(self.dim) * sym.Variable("u"))

    def grad(self, vec):
        return self._bound_grad()(vec.queue, u=vec)

    def div(self, vecs):
        return sum(
                self.grad(vec_i)[i] for i, vec_i in enumerate(vecs))

    @memoize_method
    def _bound_weak_grad(self):
        return bind(self, sym.stiffness_t(self.dim) * sym.Variable("u"))

    def weak_grad(self, vec):
        return self._bound_weak_grad()(vec.queue, u=vec)

    def weak_div(self, vecs):
        return sum(
                self.weak_grad(vec_i)[i] for i, vec_i in enumerate(vecs))

    @memoize_method
    def normal(self, dd):
        with cl.CommandQueue(self.cl_context) as queue:
            surface_discr = self.discr_from_dd(dd)
            return without_queue(
                    bind(self, sym.normal(
                        dd, surface_discr.ambient_dim, surface_discr.dim))(queue))

    @memoize_method
    def _bound_inverse_mass(self):
        return bind(self, sym.InverseMassOperator()(sym.Variable("u")))

    def inverse_mass(self, vec):
        if is_obj_array(vec):
            return with_object_array_or_scalar(
                    lambda el: self.inverse_mass(el), vec)

        return self._bound_inverse_mass()(vec.queue, u=vec)

    @memoize_method
    def _bound_face_mass(self):
        u = sym.Variable("u", dd=sym.as_dofdesc("all_faces"))
        return bind(self, sym.FaceMassOperator()(u))

    def face_mass(self, vec):
        if is_obj_array(vec):
            return with_object_array_or_scalar(
                    lambda el: self.face_mass(el), vec)

        return self._bound_face_mass()(vec.queue, u=vec)

# vim: foldmethod=marker
