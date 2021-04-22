"""
.. autofunction:: project
.. autofunction:: nodes

.. autofunction:: local_grad
.. autofunction:: local_d_dx
.. autofunction:: local_div

.. autofunction:: weak_local_grad
.. autofunction:: weak_local_d_dx
.. autofunction:: weak_local_div

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
from pytools import memoize_on_first_arg, keyed_memoize_in
from pytools.tag import Tag

import numpy as np  # noqa
from pytools.obj_array import obj_array_vectorize, make_obj_array
import pyopencl.array as cla  # noqa
from grudge import sym, bind

import grudge.dof_desc as dof_desc

from meshmode.mesh import BTAG_ALL, BTAG_NONE, BTAG_PARTITION  # noqa
from meshmode.dof_array import freeze, flatten, unflatten, DOFArray
from meshmode.array_context import FirstAxisIsElementsTag

import loopy as lp

from grudge.symbolic.primitives import TracePair


# {{{ tags

class HasElementwiseMatvecTag(FirstAxisIsElementsTag):
    pass


class MassOperatorTag(HasElementwiseMatvecTag):
    pass

# }}}


# def interp(dcoll, src, tgt, vec):
#     from warnings import warn
#     warn("using 'interp' is deprecated, use 'project' instead.",
#             DeprecationWarning, stacklevel=2)
#
#     return dcoll.project(src, tgt, vec)


def project(dcoll, src, tgt, vec):
    """Project from one discretization to another, e.g. from the
    volume to the boundary, or from the base to the an overintegrated
    quadrature discretization.

    :arg src: a :class:`~grudge.dof_desc.DOFDesc`, or a value convertible to one
    :arg tgt: a :class:`~grudge.dof_desc.DOFDesc`, or a value convertible to one
    :arg vec: a :class:`~meshmode.dof_array.DOFArray`
    """
    src = dof_desc.as_dofdesc(src)
    tgt = dof_desc.as_dofdesc(tgt)
    if src == tgt:
        return vec

    if isinstance(vec, np.ndarray):
        return obj_array_vectorize(
                lambda el: project(dcoll, src, tgt, el), vec)

    if isinstance(vec, Number):
        return vec

    return dcoll.connection_from_dds(src, tgt)(vec)


# {{{ geometric properties

def nodes(dcoll, dd=None):
    r"""Return the nodes of a discretization.

    :arg dd: a :class:`~grudge.dof_desc.DOFDesc`, or a value convertible to one.
        Defaults to the base volume discretization.
    :returns: an object array of :class:`~meshmode.dof_array.DOFArray`\ s
    """
    if dd is None:
        return dcoll._volume_discr.nodes()
    else:
        return dcoll.discr_from_dd(dd).nodes()


@memoize_on_first_arg
def normal(dcoll, dd):
    """Get unit normal to specified surface discretization, *dd*.

    :arg dd: a :class:`~grudge.dof_desc.DOFDesc` as the surface discretization.
    :returns: an object array of :class:`~meshmode.dof_array.DOFArray`.
    """
    surface_discr = dcoll.discr_from_dd(dd)
    actx = surface_discr._setup_actx
    return freeze(
            bind(dcoll,
                sym.normal(dd, surface_discr.ambient_dim, surface_discr.dim),
                local_only=True)
            (array_context=actx))

# }}}


# {{{ derivatives

@memoize_on_first_arg
def _bound_grad(dcoll):
    return bind(dcoll, sym.nabla(dcoll.dim) * sym.Variable("u"), local_only=True)


def local_grad(dcoll, vec):
    r"""Return the element-local gradient of the volume function represented by
    *vec*.

    :arg vec: a :class:`~meshmode.dof_array.DOFArray`
    :returns: an object array of :class:`~meshmode.dof_array.DOFArray`\ s
    """
    return _bound_grad(dcoll)(u=vec)


@memoize_on_first_arg
def _bound_d_dx(dcoll, xyz_axis):
    return bind(dcoll, sym.nabla(dcoll.dim)[xyz_axis] * sym.Variable("u"),
            local_only=True)


def local_d_dx(dcoll, xyz_axis, vec):
    r"""Return the element-local derivative along axis *xyz_axis* of the volume
    function represented by *vec*.

    :arg xyz_axis: an integer indicating the axis along which the derivative
        is taken
    :arg vec: a :class:`~meshmode.dof_array.DOFArray`
    :returns: a :class:`~meshmode.dof_array.DOFArray`\ s
    """
    return _bound_d_dx(dcoll, xyz_axis)(u=vec)


def _div_helper(dcoll, diff_func, vecs):
    if not isinstance(vecs, np.ndarray):
        raise TypeError("argument must be an object array")
    assert vecs.dtype == object

    if vecs.shape[-1] != dcoll.ambient_dim:
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


def local_div(dcoll, vecs):
    r"""Return the element-local divergence of the vector volume function
    represented by *vecs*.

    :arg vec: an object array of
        a :class:`~meshmode.dof_array.DOFArray`\ s,
        where the last axis of the array must have length
        matching the volume dimension.
    :returns: a :class:`~meshmode.dof_array.DOFArray`
    """

    return _div_helper(dcoll,
            lambda i, subvec: local_d_dx(dcoll, i, subvec),
            vecs)


@memoize_on_first_arg
def _bound_weak_grad(dcoll, dd):
    return bind(dcoll,
            sym.stiffness_t(dcoll.dim, dd_in=dd) * sym.Variable("u", dd),
            local_only=True)


def weak_local_grad(dcoll, *args):
    r"""Return the element-local weak gradient of the volume function
    represented by *vec*.

    May be called with ``(vecs)`` or ``(dd, vecs)``.

    :arg dd: a :class:`~grudge.dof_desc.DOFDesc`, or a value convertible to one.
        Defaults to the base volume discretization if not provided.
    :arg vec: a :class:`~meshmode.dof_array.DOFArray`
    :returns: an object array of :class:`~meshmode.dof_array.DOFArray`\ s
    """
    if len(args) == 1:
        vec, = args
        dd = dof_desc.DOFDesc("vol", dof_desc.QTAG_NONE)
    elif len(args) == 2:
        dd, vec = args
    else:
        raise TypeError("invalid number of arguments")

    return _bound_weak_grad(dcoll, dd)(u=vec)


@memoize_on_first_arg
def _bound_weak_d_dx(dcoll, dd, xyz_axis):
    return bind(dcoll,
            sym.stiffness_t(dcoll.dim, dd_in=dd)[xyz_axis]
            * sym.Variable("u", dd),
            local_only=True)


def weak_local_d_dx(dcoll, *args):
    r"""Return the element-local weak derivative along axis *xyz_axis* of the
    volume function represented by *vec*.

    May be called with ``(xyz_axis, vecs)`` or ``(dd, xyz_axis, vecs)``.

    :arg xyz_axis: an integer indicating the axis along which the derivative
        is taken
    :arg vec: a :class:`~meshmode.dof_array.DOFArray`
    :returns: a :class:`~meshmode.dof_array.DOFArray`\ s
    """
    if len(args) == 2:
        xyz_axis, vec = args
        dd = dof_desc.DOFDesc("vol", dof_desc.QTAG_NONE)
    elif len(args) == 3:
        dd, xyz_axis, vec = args
    else:
        raise TypeError("invalid number of arguments")

    return _bound_weak_d_dx(dcoll, dd, xyz_axis)(u=vec)


def weak_local_div(dcoll, *args):
    r"""Return the element-local weak divergence of the vector volume function
    represented by *vecs*.

    May be called with ``(vecs)`` or ``(dd, vecs)``.

    :arg dd: a :class:`~grudge.dof_desc.DOFDesc`, or a value convertible to one.
        Defaults to the base volume discretization if not provided.
    :arg vec: a object array of
        a :class:`~meshmode.dof_array.DOFArray`\ s,
        where the last axis of the array must have length
        matching the volume dimension.
    :returns: a :class:`~meshmode.dof_array.DOFArray`
    """
    if len(args) == 1:
        vecs, = args
        dd = dof_desc.DOFDesc("vol", dof_desc.QTAG_NONE)
    elif len(args) == 2:
        dd, vecs = args
    else:
        raise TypeError("invalid number of arguments")

    return _div_helper(dcoll,
            lambda i, subvec: weak_local_d_dx(dcoll, dd, i, subvec),
            vecs)

# }}}


# {{{ mass-like

@memoize_on_first_arg
def _bound_mass(dcoll, dd):
    return bind(dcoll, sym.MassOperator(dd_in=dd)(sym.Variable("u", dd)),
            local_only=True)


def mass(dcoll, *args):
    if len(args) == 1:
        vec, = args
        dd = dof_desc.DOFDesc("vol", dof_desc.QTAG_NONE)
    elif len(args) == 2:
        dd, vec = args
    else:
        raise TypeError("invalid number of arguments")

    if isinstance(vec, np.ndarray):
        return obj_array_vectorize(
                lambda el: mass(dcoll, dd, el), vec)

    return _bound_mass(dcoll, dd)(u=vec)


@memoize_on_first_arg
def _elwise_linear_loopy_prg(actx):
    result = make_loopy_program(
        """{[iel, idof, j]:
            0<=iel<nelements and
            0<=idof<ndiscr_nodes_out and
            0<=j<ndiscr_nodes_in}""",
        "result[iel, idof] = sum(j, mat[idof, j] * vec[iel, j])",
        name="elwise_linear_op_knl")

    result = lp.tag_array_axes(result, "mat", "stride:auto,stride:auto")
    return result


def reference_mass_matrix(actx, out_element_group, in_element_group):
    @keyed_memoize_in(
        actx, reference_mass_matrix,
        lambda out_grp, in_grp: (out_grp.discretization_key(),
                                 in_grp.discretization_key()))
    def get_ref_mass_mat(out_grp, in_grp):
        if out_grp == in_grp:
            from meshmode.discretization.poly_element import mass_matrix
            return mass_matrix(in_grp)

        from modepy import vandermonde
        basis = out_grp.basis_obj()
        vand = vandermonde(basis.functions, out_grp.unit_nodes)
        o_vand = vandermonde(basis.functions, in_grp.unit_nodes)
        vand_inv_t = np.linalg.inv(vand).T

        weights = in_grp.quadrature_rule().weights
        return actx.freeze(
            actx.from_numpy(
                np.einsum("j,ik,jk->ij", weights, vand_inv_t, o_vand)
            )
        )

    return get_ref_mass_mat(out_element_group, in_element_group)


def _apply_mass_operator(dcoll, dd_out, dd_in, vec):
    in_discr = dcoll.discr_from_dd(dd_in)
    out_discr = dcoll.discr_from_dd(dd_out)

    actx = vec.array_context
    area_elements = in_discr.zeros(actx)  # FIXME *cough*
    return DOFArray(actx,
            tuple(
                actx.einsum("ij,ej,ej->ei",
                    reference_mass_matrix(actx,
                        out_element_group=out_grp,
                        in_element_group=in_grp),
                    ae_i, vec_i,
                    arg_names=("mass_mat", "jac_det", "vec"),
                    tagged=(MassOperatorTag(),))

                for in_grp, out_grp, ae_i, vec_i in zip(
                    in_discr.groups, out_discr.groups,
                    area_elements, vec)))


def mass_operator(dcoll, *args):
    if len(args) == 1:
        vec, = args
        dd = sym.DOFDesc("vol", sym.QTAG_NONE)
    elif len(args) == 2:
        dd, vec = args
    else:
        raise TypeError("invalid number of arguments")

    if isinstance(vec, np.ndarray):
        return obj_array_vectorize(
            lambda el: mass_operator(dcoll, dd, el), vec
        )

    dd_in = dd
    del dd
    from grudge.dof_desc import QTAG_NONE
    dd_out = dd_in.with_qtag(QTAG_NONE)

    return _apply_mass_operator(dcoll, dd_out, dd_in, vec)


@memoize_on_first_arg
def _bound_inverse_mass(dcoll):
    return bind(dcoll, sym.InverseMassOperator()(sym.Variable("u")),
            local_only=True)


def inverse_mass(dcoll, vec):
    if isinstance(vec, np.ndarray):
        return obj_array_vectorize(
                lambda el: inverse_mass(dcoll, el), vec)

    return _bound_inverse_mass(dcoll)(u=vec)


@memoize_on_first_arg
def _bound_face_mass(dcoll, dd):
    u = sym.Variable("u", dd=dd)
    return bind(dcoll, sym.FaceMassOperator(dd_in=dd)(u), local_only=True)


def face_mass(dcoll, *args):
    if len(args) == 1:
        vec, = args
        dd = dof_desc.DOFDesc("all_faces", dof_desc.QTAG_NONE)
    elif len(args) == 2:
        dd, vec = args
    else:
        raise TypeError("invalid number of arguments")

    if isinstance(vec, np.ndarray):
        return obj_array_vectorize(
                lambda el: face_mass(dcoll, dd, el), vec)

    return _bound_face_mass(dcoll, dd)(u=vec)

# }}}


# {{{ reductions

@memoize_on_first_arg
def _norm(dcoll, p, dd):
    return bind(dcoll,
            sym.norm(p, sym.var("arg", dd=dd), dd=dd),
            local_only=True)


def norm(dcoll, vec, p, dd=None):
    if dd is None:
        dd = "vol"

    dd = dof_desc.as_dofdesc(dd)

    if isinstance(vec, np.ndarray):
        if p == 2:
            return sum(
                    norm(dcoll, vec[idx], p, dd=dd)**2
                    for idx in np.ndindex(vec.shape))**0.5
        elif p == np.inf:
            return max(
                    norm(dcoll, vec[idx], np.inf, dd=dd)
                    for idx in np.ndindex(vec.shape))
        else:
            raise ValueError("unsupported norm order")

    return _norm(dcoll, p, dd)(arg=vec)


@memoize_on_first_arg
def _nodal_reduction(dcoll, operator, dd):
    return bind(dcoll, operator(dd)(sym.var("arg")), local_only=True)


def nodal_sum(dcoll, dd, vec):
    return _nodal_reduction(dcoll, sym.NodalSum, dd)(arg=vec)


def nodal_min(dcoll, dd, vec):
    return _nodal_reduction(dcoll, sym.NodalMin, dd)(arg=vec)


def nodal_max(dcoll, dd, vec):
    return _nodal_reduction(dcoll, sym.NodalMax, dd)(arg=vec)

# }}}


@memoize_on_first_arg
def connected_ranks(dcoll):
    from meshmode.distributed import get_connected_partitions
    return get_connected_partitions(dcoll._volume_discr.mesh)


# {{{ interior_trace_pair

def interior_trace_pair(dcoll, vec):
    """Return a :class:`grudge.sym.TracePair` for the interior faces of
    *dcoll*.
    """
    i = project(dcoll, "vol", "int_faces", vec)

    def get_opposite_face(el):
        if isinstance(el, Number):
            return el
        else:
            return dcoll.opposite_face_connection()(el)

    e = obj_array_vectorize(get_opposite_face, i)

    return TracePair("int_faces", interior=i, exterior=e)

# }}}


# {{{ distributed-memory functionality

class _RankBoundaryCommunication:
    base_tag = 1273

    def __init__(self, dcoll, remote_rank, vol_field, tag=None):
        self.tag = self.base_tag
        if tag is not None:
            self.tag += tag

        self.dcoll = dcoll
        self.array_context = vol_field.array_context
        self.remote_btag = BTAG_PARTITION(remote_rank)

        self.bdry_discr = dcoll.discr_from_dd(self.remote_btag)
        self.local_dof_array = project(dcoll, "vol", self.remote_btag, vol_field)

        local_data = self.array_context.to_numpy(flatten(self.local_dof_array))

        comm = self.dcoll.mpi_communicator

        self.send_req = comm.Isend(
                local_data, remote_rank, tag=self.tag)

        self.remote_data_host = np.empty_like(local_data)
        self.recv_req = comm.Irecv(self.remote_data_host, remote_rank, self.tag)

    def finish(self):
        self.recv_req.Wait()

        actx = self.array_context
        remote_dof_array = unflatten(self.array_context, self.bdry_discr,
                actx.from_numpy(self.remote_data_host))

        bdry_conn = self.dcoll.get_distributed_boundary_swap_connection(
                dof_desc.as_dofdesc(dof_desc.DTAG_BOUNDARY(self.remote_btag)))
        swapped_remote_dof_array = bdry_conn(remote_dof_array)

        self.send_req.Wait()

        return TracePair(self.remote_btag,
                interior=self.local_dof_array,
                exterior=swapped_remote_dof_array)


def _cross_rank_trace_pairs_scalar_field(dcoll, vec, tag=None):
    if isinstance(vec, Number):
        return [TracePair(BTAG_PARTITION(remote_rank), interior=vec, exterior=vec)
                for remote_rank in connected_ranks(dcoll)]
    else:
        rbcomms = [_RankBoundaryCommunication(dcoll, remote_rank, vec, tag=tag)
                for remote_rank in connected_ranks(dcoll)]
        return [rbcomm.finish() for rbcomm in rbcomms]


def cross_rank_trace_pairs(dcoll, ary, tag=None):
    r"""Get a list of *ary* trace pairs for each partition boundary.

    For each partition boundary, the field data values in *ary* are
    communicated to/from the neighboring partition. Presumably, this
    communication is MPI (but strictly speaking, may not be, and this
    routine is agnostic to the underlying communication, see e.g.
    _cross_rank_trace_pairs_scalar_field).

    For each face on each partition boundary, a :class:`TracePair` is
    created with the locally, and remotely owned partition boundary face
    data as the `internal`, and `external` components, respectively.
    Each of the TracePair components are structured like *ary*.

    The input field data *ary* may be a single
    :class:`~meshmode.dof_array.DOFArray`, or an object
    array of ``DOFArray``\ s of arbitrary shape.
    """
    if isinstance(ary, np.ndarray):
        oshape = ary.shape
        comm_vec = ary.flatten()

        n, = comm_vec.shape
        result = {}
        # FIXME: Batch this communication rather than
        # doing it in sequence.
        for ivec in range(n):
            for rank_tpair in _cross_rank_trace_pairs_scalar_field(
                    dcoll, comm_vec[ivec]):
                assert isinstance(rank_tpair.dd.domain_tag, dof_desc.DTAG_BOUNDARY)
                assert isinstance(rank_tpair.dd.domain_tag.tag, BTAG_PARTITION)
                result[rank_tpair.dd.domain_tag.tag.part_nr, ivec] = rank_tpair

        return [
            TracePair(
                dd=dof_desc.as_dofdesc(
                    dof_desc.DTAG_BOUNDARY(BTAG_PARTITION(remote_rank))),
                interior=make_obj_array([
                    result[remote_rank, i].int for i in range(n)]).reshape(oshape),
                exterior=make_obj_array([
                    result[remote_rank, i].ext for i in range(n)]).reshape(oshape)
                )
            for remote_rank in connected_ranks(dcoll)]
    else:
        return _cross_rank_trace_pairs_scalar_field(dcoll, ary, tag=tag)

# }}}


# vim: foldmethod=marker
