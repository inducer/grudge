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
from pytools import memoize_method
from pytools.obj_array import obj_array_vectorize, make_obj_array
import pyopencl.array as cla  # noqa
from grudge import sym, bind

from meshmode.mesh import BTAG_ALL, BTAG_NONE, BTAG_PARTITION  # noqa
from meshmode.dof_array import freeze, flatten, unflatten

from grudge.discretization import DGDiscretizationWithBoundaries
from grudge.symbolic.primitives import TracePair


__doc__ = """
.. autoclass:: EagerDGDiscretization
.. autofunction:: interior_trace_pair
.. autofunction:: cross_rank_trace_pairs
"""


class EagerDGDiscretization(DGDiscretizationWithBoundaries):
    """
    Inherits from :class:`~grudge.discretization.DGDiscretizationWithBoundaries`.

    .. automethod:: __init__
    .. automethod:: project
    .. automethod:: nodes

    .. automethod:: grad
    .. automethod:: d_dx
    .. automethod:: div

    .. automethod:: weak_grad
    .. automethod:: weak_d_dx
    .. automethod:: weak_div

    .. automethod:: normal
    .. automethod:: mass
    .. automethod:: inverse_mass
    .. automethod:: face_mass

    .. automethod:: norm
    .. automethod:: nodal_sum
    .. automethod:: nodal_min
    .. automethod:: nodal_max
    """

    def interp(self, src, tgt, vec):
        from warnings import warn
        warn("using 'interp' is deprecated, use 'project' instead.",
                DeprecationWarning, stacklevel=2)

        return self.project(src, tgt, vec)

    def project(self, src, tgt, vec):
        """Project from one discretization to another, e.g. from the
        volume to the boundary, or from the base to the an overintegrated
        quadrature discretization.

        :arg src: a :class:`~grudge.sym.DOFDesc`, or a value convertible to one
        :arg tgt: a :class:`~grudge.sym.DOFDesc`, or a value convertible to one
        :arg vec: a :class:`~meshmode.dof_array.DOFArray`
        """
        src = sym.as_dofdesc(src)
        tgt = sym.as_dofdesc(tgt)
        if src == tgt:
            return vec

        if isinstance(vec, np.ndarray):
            return obj_array_vectorize(
                    lambda el: self.project(src, tgt, el), vec)

        return self.connection_from_dds(src, tgt)(vec)

    def nodes(self, dd=None):
        r"""Return the nodes of a discretization.

        :arg dd: a :class:`~grudge.sym.DOFDesc`, or a value convertible to one.
            Defaults to the base volume discretization.
        :returns: an object array of :class:`~meshmode.dof_array.DOFArray`\ s
        """
        if dd is None:
            return self._volume_discr.nodes()
        else:
            return self.discr_from_dd(dd).nodes()

    # {{{ derivatives

    @memoize_method
    def _bound_grad(self):
        return bind(self, sym.nabla(self.dim) * sym.Variable("u"), local_only=True)

    def grad(self, vec):
        r"""Return the gradient of the volume function represented by *vec*.

        :arg vec: a :class:`~meshmode.dof_array.DOFArray`
        :returns: an object array of :class:`~meshmode.dof_array.DOFArray`\ s
        """
        return self._bound_grad()(u=vec)

    @memoize_method
    def _bound_d_dx(self, xyz_axis):
        return bind(self, sym.nabla(self.dim)[xyz_axis] * sym.Variable("u"),
                local_only=True)

    def d_dx(self, xyz_axis, vec):
        r"""Return the derivative along axis *xyz_axis* of the volume function
        represented by *vec*.

        :arg xyz_axis: an integer indicating the axis along which the derivative
            is taken
        :arg vec: a :class:`~meshmode.dof_array.DOFArray`
        :returns: a :class:`~meshmode.dof_array.DOFArray`\ s
        """
        return self._bound_d_dx(xyz_axis)(u=vec)

    def _div_helper(self, diff_func, vecs):
        if not isinstance(vecs, np.ndarray):
            raise TypeError("argument must be an object array")
        assert vecs.dtype == np.object

        if vecs.shape[-1] != self.ambient_dim:
            raise ValueError("last dimension of *vecs* argument must match "
                    "ambient dimension")

        if len(vecs.shape) == 1:
            return sum(diff_func(i, vec_i) for i, vec_i in enumerate(vecs))
        else:
            result = np.zeros(vecs.shape[:-1], dtype=object)
            for idx in np.ndindex(vecs.shape[:-1]):
                result[idx] = sum(
                        diff_func(i, vec_i) for i, vec_i in enumerate(vecs[idx]))
            return result

    def div(self, vecs):
        r"""Return the divergence of the vector volume function
        represented by *vecs*.

        :arg vec: an object array of
            a :class:`~meshmode.dof_array.DOFArray`\ s,
            where the last axis of the array must have length
            matching the volume dimension.
        :returns: a :class:`~meshmode.dof_array.DOFArray`
        """

        return self._div_helper(
                lambda i, subvec: self.d_dx(i, subvec),
                vecs)

    @memoize_method
    def _bound_weak_grad(self, dd):
        return bind(self,
                sym.stiffness_t(self.dim, dd_in=dd) * sym.Variable("u", dd),
                local_only=True)

    def weak_grad(self, *args):
        r"""Return the "weak gradient" of the volume function represented by
        *vec*.

        May be called with ``(vecs)`` or ``(dd, vecs)``.

        :arg dd: a :class:`~grudge.sym.DOFDesc`, or a value convertible to one.
            Defaults to the base volume discretization if not provided.
        :arg vec: a :class:`~meshmode.dof_array.DOFArray`
        :returns: an object array of :class:`~meshmode.dof_array.DOFArray`\ s
        """
        if len(args) == 1:
            vec, = args
            dd = sym.DOFDesc("vol", sym.QTAG_NONE)
        elif len(args) == 2:
            dd, vec = args
        else:
            raise TypeError("invalid number of arguments")

        return self._bound_weak_grad(dd)(u=vec)

    @memoize_method
    def _bound_weak_d_dx(self, dd, xyz_axis):
        return bind(self,
                sym.stiffness_t(self.dim, dd_in=dd)[xyz_axis]
                * sym.Variable("u", dd),
                local_only=True)

    def weak_d_dx(self, *args):
        r"""Return the derivative along axis *xyz_axis* of the volume function
        represented by *vec*.

        May be called with ``(xyz_axis, vecs)`` or ``(dd, xyz_axis, vecs)``.

        :arg xyz_axis: an integer indicating the axis along which the derivative
            is taken
        :arg vec: a :class:`~meshmode.dof_array.DOFArray`
        :returns: a :class:`~meshmode.dof_array.DOFArray`\ s
        """
        if len(args) == 2:
            xyz_axis, vec = args
            dd = sym.DOFDesc("vol", sym.QTAG_NONE)
        elif len(args) == 3:
            dd, xyz_axis, vec = args
        else:
            raise TypeError("invalid number of arguments")

        return self._bound_weak_d_dx(dd, xyz_axis)(u=vec)

    def weak_div(self, *args):
        r"""Return the "weak divergence" of the vector volume function
        represented by *vecs*.

        May be called with ``(vecs)`` or ``(dd, vecs)``.

        :arg dd: a :class:`~grudge.sym.DOFDesc`, or a value convertible to one.
            Defaults to the base volume discretization if not provided.
        :arg vec: a object array of
            a :class:`~meshmode.dof_array.DOFArray`\ s,
            where the last axis of the array must have length
            matching the volume dimension.
        :returns: a :class:`~meshmode.dof_array.DOFArray`
        """
        if len(args) == 1:
            vecs, = args
            dd = sym.DOFDesc("vol", sym.QTAG_NONE)
        elif len(args) == 2:
            dd, vecs = args
        else:
            raise TypeError("invalid number of arguments")

        return self._div_helper(
                lambda i, subvec: self.weak_d_dx(dd, i, subvec),
                vecs)

    # }}}

    @memoize_method
    def normal(self, dd):
        """Get unit normal to specified surface discretization, *dd*.

        :arg dd: a :class:`~grudge.sym.DOFDesc` as the surface discretization.
        :returns: an object array of :class:`~meshmode.dof_array.DOFArray`.
        """
        surface_discr = self.discr_from_dd(dd)
        actx = surface_discr._setup_actx
        return freeze(
                bind(self,
                    sym.normal(dd, surface_discr.ambient_dim, surface_discr.dim),
                    local_only=True)
                (array_context=actx))

    @memoize_method
    def _bound_mass(self, dd):
        return bind(self, sym.MassOperator(dd_in=dd)(sym.Variable("u", dd)),
                local_only=True)

    def mass(self, *args):
        if len(args) == 1:
            vec, = args
            dd = sym.DOFDesc("vol", sym.QTAG_NONE)
        elif len(args) == 2:
            dd, vec = args
        else:
            raise TypeError("invalid number of arguments")

        if isinstance(vec, np.ndarray):
            return obj_array_vectorize(
                    lambda el: self.mass(dd, el), vec)

        return self._bound_mass(dd)(u=vec)

    @memoize_method
    def _bound_inverse_mass(self):
        return bind(self, sym.InverseMassOperator()(sym.Variable("u")),
                local_only=True)

    def inverse_mass(self, vec):
        if isinstance(vec, np.ndarray):
            return obj_array_vectorize(
                    lambda el: self.inverse_mass(el), vec)

        return self._bound_inverse_mass()(u=vec)

    @memoize_method
    def _bound_face_mass(self, dd):
        u = sym.Variable("u", dd=dd)
        return bind(self, sym.FaceMassOperator(dd_in=dd)(u), local_only=True)

    def face_mass(self, *args):
        if len(args) == 1:
            vec, = args
            dd = sym.DOFDesc("all_faces", sym.QTAG_NONE)
        elif len(args) == 2:
            dd, vec = args
        else:
            raise TypeError("invalid number of arguments")

        if isinstance(vec, np.ndarray):
            return obj_array_vectorize(
                    lambda el: self.face_mass(dd, el), vec)

        return self._bound_face_mass(dd)(u=vec)

    @memoize_method
    def _norm(self, p, dd):
        return bind(self,
                sym.norm(p, sym.var("arg", dd=dd), dd=dd),
                local_only=True)

    def norm(self, vec, p=2, dd=None):
        if dd is None:
            dd = "vol"

        dd = sym.as_dofdesc(dd)

        if isinstance(vec, np.ndarray):
            if p == 2:
                return sum(
                        self.norm(vec[idx], dd=dd)**2
                        for idx in np.ndindex(vec.shape))**0.5
            elif p == np.inf:
                return max(
                        self.norm(vec[idx], np.inf, dd=dd)
                        for idx in np.ndindex(vec.shape))
            else:
                raise ValueError("unsupported norm order")

        return self._norm(p, dd)(arg=vec)

    @memoize_method
    def _nodal_reduction(self, operator, dd):
        return bind(self, operator(dd)(sym.var("arg")), local_only=True)

    def nodal_sum(self, dd, vec):
        return self._nodal_reduction(sym.NodalSum, dd)(arg=vec)

    def nodal_min(self, dd, vec):
        return self._nodal_reduction(sym.NodalMin, dd)(arg=vec)

    def nodal_max(self, dd, vec):
        return self._nodal_reduction(sym.NodalMax, dd)(arg=vec)

    @memoize_method
    def connected_ranks(self):
        from meshmode.distributed import get_connected_partitions
        return get_connected_partitions(self._volume_discr.mesh)


def interior_trace_pair(discrwb, vec):
    """Return a :class:`grudge.sym.TracePair` for the interior faces of
    *discrwb*.
    """
    i = discrwb.project("vol", "int_faces", vec)
    e = obj_array_vectorize(lambda el: discrwb.opposite_face_connection()(el), i)
    return TracePair("int_faces", interior=i, exterior=e)


# {{{ distributed-memory functionality

class _RankBoundaryCommunication:
    base_tag = 1273

    def __init__(self, discrwb, remote_rank, vol_field, tag=None):
        self.tag = self.base_tag
        if tag is not None:
            self.tag += tag

        self.discrwb = discrwb
        self.array_context = vol_field.array_context
        self.remote_btag = BTAG_PARTITION(remote_rank)

        self.bdry_discr = discrwb.discr_from_dd(self.remote_btag)
        self.local_dof_array = discrwb.project("vol", self.remote_btag, vol_field)

        local_data = self.array_context.to_numpy(flatten(self.local_dof_array))

        comm = self.discrwb.mpi_communicator

        self.send_req = comm.Isend(
                local_data, remote_rank, tag=self.tag)

        self.remote_data_host = np.empty_like(local_data)
        self.recv_req = comm.Irecv(self.remote_data_host, remote_rank, self.tag)

    def finish(self):
        self.recv_req.Wait()

        actx = self.array_context
        remote_dof_array = unflatten(self.array_context, self.bdry_discr,
                actx.from_numpy(self.remote_data_host))

        bdry_conn = self.discrwb.get_distributed_boundary_swap_connection(
                sym.as_dofdesc(sym.DTAG_BOUNDARY(self.remote_btag)))
        swapped_remote_dof_array = bdry_conn(remote_dof_array)

        self.send_req.Wait()

        return TracePair(self.remote_btag,
                interior=self.local_dof_array,
                exterior=swapped_remote_dof_array)


def _cross_rank_trace_pairs_scalar_field(discrwb, vec, tag=None):
    rbcomms = [_RankBoundaryCommunication(discrwb, remote_rank, vec, tag=tag)
            for remote_rank in discrwb.connected_ranks()]
    return [rbcomm.finish() for rbcomm in rbcomms]


def cross_rank_trace_pairs(discrwb, vec, tag=None):
    if isinstance(vec, np.ndarray):

        n, = vec.shape
        result = {}
        for ivec in range(n):
            for rank_tpair in _cross_rank_trace_pairs_scalar_field(
                    discrwb, vec[ivec]):
                assert isinstance(rank_tpair.dd.domain_tag, sym.DTAG_BOUNDARY)
                assert isinstance(rank_tpair.dd.domain_tag.tag, BTAG_PARTITION)
                result[rank_tpair.dd.domain_tag.tag.part_nr, ivec] = rank_tpair

        return [
            TracePair(
                dd=sym.as_dofdesc(sym.DTAG_BOUNDARY(BTAG_PARTITION(remote_rank))),
                interior=make_obj_array([
                    result[remote_rank, i].int for i in range(n)]),
                exterior=make_obj_array([
                    result[remote_rank, i].ext for i in range(n)])
                )
            for remote_rank in discrwb.connected_ranks()]
    else:
        return _cross_rank_trace_pairs_scalar_field(discrwb, vec, tag=tag)

# }}}


# vim: foldmethod=marker
