"""
.. autoclass:: EagerDGDiscretization
"""
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

import grudge.op as op
from grudge.discretization import DiscretizationCollection


class EagerDGDiscretization(DiscretizationCollection):
    """
    This class is deprecated and only part of the documentation in order to
    avoid breaking depending documentation builds.
    Use :class:`~grudge.discretization.DiscretizationCollection` instead in
    new code.
    """

    def __init__(self, *args, **kwargs):
        from warnings import warn
        warn("EagerDGDiscretization is deprecated and will go away in 2022. "
                "Use the base DiscretizationCollection with grudge.op "
                "instead.",
                DeprecationWarning, stacklevel=2)

        super().__init__(*args, **kwargs)

    def project(self, src, tgt, vec):
        return op.project(self, src, tgt, vec)

    def grad(self, *args):
        return op.local_grad(self, *args)

    def d_dx(self, xyz_axis, *args):
        return op.local_d_dx(self, xyz_axis, *args)

    def div(self, *args):
        return op.local_div(self, *args)

    def weak_grad(self, *args):
        return op.weak_local_grad(self, *args)

    def weak_d_dx(self, *args):
        return op.weak_local_d_dx(self, *args)

    def weak_div(self, *args):
        return op.weak_local_div(self, *args)

    def mass(self, *args):
        return op.mass(self, *args)

    def inverse_mass(self, *args):
        return op.inverse_mass(self, *args)

    def face_mass(self, *args):
        return op.face_mass(self, *args)

    def norm(self, vec, p=2, dd=None):
        return op.norm(self, vec, p, dd)

    def nodal_sum(self, dd, vec):
        return op.nodal_sum(self, dd, vec)

    def nodal_min(self, dd, vec):
        return op.nodal_min(self, dd, vec)

    def nodal_max(self, dd, vec):
        return op.nodal_max(self, dd, vec)


# FIXME: Deprecate connected_ranks instead of removing
connected_parts = op.connected_parts
interior_trace_pair = op.interior_trace_pair
cross_rank_trace_pairs = op.cross_rank_trace_pairs

# vim: foldmethod=marker
