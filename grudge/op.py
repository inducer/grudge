"""
.. autofunction:: project
.. autofunction:: nodes

.. autofunction:: grad
.. autofunction:: d_dx
.. autofunction:: div

.. autofunction:: weak_grad
.. autofunction:: weak_d_dx
.. autofunction:: weak_div

.. autofunction:: normal
.. autofunction:: mass
.. autofunction:: inverse_mass
.. autofunction:: face_mass

.. autofunction:: norm
.. autofunction:: nodal_sum
.. autofunction:: nodal_min
.. autofunction:: nodal_max

.. autofunction:: interior_trace_pair
.. autofunction:: cross_rank_trace_pairs
"""

__copyright__ = "Copyright (C) 2021 Andreas Kloeckner"

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


from numbers import Number
from pytools import memoize_on_first_arg

import numpy as np  # noqa
from pytools.obj_array import obj_array_vectorize, make_obj_array
import pyopencl.array as cla  # noqa
from grudge import sym, bind

from meshmode.mesh import BTAG_ALL, BTAG_NONE, BTAG_PARTITION  # noqa
from meshmode.dof_array import freeze, flatten, unflatten

from grudge.symbolic.primitives import TracePair


# def interp(discrwb, src, tgt, vec):
#     from warnings import warn
#     warn("using 'interp' is deprecated, use 'project' instead.",
#             DeprecationWarning, stacklevel=2)
#
#     return discrwb.project(src, tgt, vec)


def project(discrwb, src, tgt, vec):
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
                lambda el: project(discrwb, src, tgt, el), vec)

    if isinstance(vec, Number):
        return vec

    return discrwb.connection_from_dds(src, tgt)(vec)


# {{{ geometric properties

def nodes(discrwb, dd=None):
    r"""Return the nodes of a discretization.

    :arg dd: a :class:`~grudge.sym.DOFDesc`, or a value convertible to one.
        Defaults to the base volume discretization.
    :returns: an object array of :class:`~meshmode.dof_array.DOFArray`\ s
    """
    if dd is None:
        return discrwb._volume_discr.nodes()
    else:
        return discrwb.discr_from_dd(dd).nodes()


@memoize_on_first_arg
def normal(discrwb, dd):
    """Get unit normal to specified surface discretization, *dd*.

    :arg dd: a :class:`~grudge.sym.DOFDesc` as the surface discretization.
    :returns: an object array of :class:`~meshmode.dof_array.DOFArray`.
    """
    surface_discr = discrwb.discr_from_dd(dd)
    actx = surface_discr._setup_actx
    return freeze(
            bind(discrwb,
                sym.normal(dd, surface_discr.ambient_dim, surface_discr.dim),
                local_only=True)
            (array_context=actx))

# }}}


# {{{ derivatives

@memoize_on_first_arg
def _bound_grad(discrwb):
    return bind(discrwb, sym.nabla(discrwb.dim) * sym.Variable("u"), local_only=True)


def grad(discrwb, vec):
    r"""Return the gradient of the volume function represented by *vec*.

    :arg vec: a :class:`~meshmode.dof_array.DOFArray`
    :returns: an object array of :class:`~meshmode.dof_array.DOFArray`\ s
    """
    return _bound_grad(discrwb)(u=vec)


@memoize_on_first_arg
def _bound_d_dx(discrwb, xyz_axis):
    return bind(discrwb, sym.nabla(discrwb.dim)[xyz_axis] * sym.Variable("u"),
            local_only=True)


def d_dx(discrwb, xyz_axis, vec):
    r"""Return the derivative along axis *xyz_axis* of the volume function
    represented by *vec*.

    :arg xyz_axis: an integer indicating the axis along which the derivative
        is taken
    :arg vec: a :class:`~meshmode.dof_array.DOFArray`
    :returns: a :class:`~meshmode.dof_array.DOFArray`\ s
    """
    return _bound_d_dx(discrwb, xyz_axis)(u=vec)


def _div_helper(discrwb, diff_func, vecs):
    if not isinstance(vecs, np.ndarray):
        raise TypeError("argument must be an object array")
    assert vecs.dtype == object

    if vecs.shape[-1] != discrwb.ambient_dim:
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


def div(discrwb, vecs):
    r"""Return the divergence of the vector volume function
    represented by *vecs*.

    :arg vec: an object array of
        a :class:`~meshmode.dof_array.DOFArray`\ s,
        where the last axis of the array must have length
        matching the volume dimension.
    :returns: a :class:`~meshmode.dof_array.DOFArray`
    """

    return _div_helper(discrwb,
            lambda i, subvec: d_dx(discrwb, i, subvec),
            vecs)


@memoize_on_first_arg
def _bound_weak_grad(discrwb, dd):
    return bind(discrwb,
            sym.stiffness_t(discrwb.dim, dd_in=dd) * sym.Variable("u", dd),
            local_only=True)


def weak_grad(discrwb, *args):
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

    return _bound_weak_grad(discrwb, dd)(u=vec)


@memoize_on_first_arg
def _bound_weak_d_dx(discrwb, dd, xyz_axis):
    return bind(discrwb,
            sym.stiffness_t(discrwb.dim, dd_in=dd)[xyz_axis]
            * sym.Variable("u", dd),
            local_only=True)


def weak_d_dx(discrwb, *args):
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

    return _bound_weak_d_dx(discrwb, dd, xyz_axis)(u=vec)


def weak_div(discrwb, *args):
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

    return _div_helper(discrwb,
            lambda i, subvec: weak_d_dx(discrwb, dd, i, subvec),
            vecs)

# }}}


# {{{ mass-like

@memoize_on_first_arg
def _bound_mass(discrwb, dd):
    return bind(discrwb, sym.MassOperator(dd_in=dd)(sym.Variable("u", dd)),
            local_only=True)


def mass(discrwb, *args):
    if len(args) == 1:
        vec, = args
        dd = sym.DOFDesc("vol", sym.QTAG_NONE)
    elif len(args) == 2:
        dd, vec = args
    else:
        raise TypeError("invalid number of arguments")

    if isinstance(vec, np.ndarray):
        return obj_array_vectorize(
                lambda el: mass(discrwb, dd, el), vec)

    return _bound_mass(discrwb, dd)(u=vec)


@memoize_on_first_arg
def _bound_inverse_mass(discrwb):
    return bind(discrwb, sym.InverseMassOperator()(sym.Variable("u")),
            local_only=True)


def inverse_mass(discrwb, vec):
    if isinstance(vec, np.ndarray):
        return obj_array_vectorize(
                lambda el: inverse_mass(discrwb, el), vec)

    return _bound_inverse_mass(discrwb)(u=vec)


@memoize_on_first_arg
def _bound_face_mass(discrwb, dd):
    u = sym.Variable("u", dd=dd)
    return bind(discrwb, sym.FaceMassOperator(dd_in=dd)(u), local_only=True)


def face_mass(discrwb, *args):
    if len(args) == 1:
        vec, = args
        dd = sym.DOFDesc("all_faces", sym.QTAG_NONE)
    elif len(args) == 2:
        dd, vec = args
    else:
        raise TypeError("invalid number of arguments")

    if isinstance(vec, np.ndarray):
        return obj_array_vectorize(
                lambda el: face_mass(discrwb, dd, el), vec)

    return _bound_face_mass(discrwb, dd)(u=vec)

# }}}


# {{{ reductions

@memoize_on_first_arg
def _norm(discrwb, p, dd):
    return bind(discrwb,
            sym.norm(p, sym.var("arg", dd=dd), dd=dd),
            local_only=True)


def norm(discrwb, vec, p, dd=None):
    if dd is None:
        dd = "vol"

    dd = sym.as_dofdesc(dd)

    if isinstance(vec, np.ndarray):
        if p == 2:
            return sum(
                    norm(discrwb, vec[idx], dd=dd)**2
                    for idx in np.ndindex(vec.shape))**0.5
        elif p == np.inf:
            return max(
                    norm(discrwb, vec[idx], np.inf, dd=dd)
                    for idx in np.ndindex(vec.shape))
        else:
            raise ValueError("unsupported norm order")

    return _norm(discrwb, p, dd)(arg=vec)


@memoize_on_first_arg
def _nodal_reduction(discrwb, operator, dd):
    return bind(discrwb, operator(dd)(sym.var("arg")), local_only=True)


def nodal_sum(discrwb, dd, vec):
    return _nodal_reduction(discrwb, sym.NodalSum, dd)(arg=vec)


def nodal_min(discrwb, dd, vec):
    return _nodal_reduction(discrwb, sym.NodalMin, dd)(arg=vec)


def nodal_max(discrwb, dd, vec):
    return _nodal_reduction(discrwb, sym.NodalMax, dd)(arg=vec)

# }}}


@memoize_on_first_arg
def connected_ranks(discrwb):
    from meshmode.distributed import get_connected_partitions
    return get_connected_partitions(discrwb._volume_discr.mesh)


# {{{ interior_trace_pair

def interior_trace_pair(discrwb, vec):
    """Return a :class:`grudge.sym.TracePair` for the interior faces of
    *discrwb*.
    """
    i = project(discrwb, "vol", "int_faces", vec)

    def get_opposite_face(el):
        if isinstance(el, Number):
            return el
        else:
            return discrwb.opposite_face_connection()(el)

    e = obj_array_vectorize(get_opposite_face, i)

    return TracePair("int_faces", interior=i, exterior=e)

# }}}


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
        self.local_dof_array = project(discrwb, "vol", self.remote_btag, vol_field)

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
    if isinstance(vec, Number):
        return [TracePair(BTAG_PARTITION(remote_rank), interior=vec, exterior=vec)
                for remote_rank in connected_ranks(discrwb)]
    else:
        rbcomms = [_RankBoundaryCommunication(discrwb, remote_rank, vec, tag=tag)
                for remote_rank in connected_ranks(discrwb)]
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
            for remote_rank in connected_ranks(discrwb)]
    else:
        return _cross_rank_trace_pairs_scalar_field(discrwb, vec, tag=tag)

# }}}


# vim: foldmethod=marker
