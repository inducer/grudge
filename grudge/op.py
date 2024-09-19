"""
Core DG routines
^^^^^^^^^^^^^^^^

Elementwise differentiation
---------------------------

.. autofunction:: local_grad
.. autofunction:: local_d_dx
.. autofunction:: local_div

Weak derivative operators
-------------------------

.. autofunction:: weak_local_grad
.. autofunction:: weak_local_d_dx
.. autofunction:: weak_local_div

Mass, inverse mass, and face mass operators
-------------------------------------------

.. autofunction:: mass
.. autofunction:: inverse_mass
.. autofunction:: face_mass

Working around documentation tool awkwardness
---------------------------------------------

.. class:: TracePair

    See :class:`grudge.trace_pair.TracePair`.

Links to canonical locations of external symbols
------------------------------------------------

(This section only exists because Sphinx does not appear able to resolve
these symbols correctly.)

.. class:: ArrayOrContainer

    See :class:`arraycontext.ArrayOrContainer`.
"""

from __future__ import annotations

__copyright__ = """
Copyright (C) 2021 Andreas Kloeckner
Copyright (C) 2021 University of Illinois Board of Trustees
"""

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


from functools import partial

import numpy as np

import modepy as mp
from modepy import (
    multi_vandermonde,
    nodal_quad_mass_matrix,
    vandermonde,
    inverse_mass_matrix,
    mass_matrix
)
from modepy.tools import (
    reshape_array_for_tensor_product_space as fold,
    unreshape_array_for_tensor_product_space as unfold
)

from arraycontext import (
    ArrayContext,
    ArrayOrContainer,
    map_array_container,
    tag_axes
)
from meshmode.dof_array import DOFArray
from meshmode.discretization import (
    InterpolatoryElementGroupBase,
    NodalElementGroupBase
)
from meshmode.discretization.poly_element import (
    TensorProductElementGroupBase,
    SimplexElementGroupBase
)
from meshmode.transform_metadata import (
    DiscretizationAmbientDimAxisTag,
    DiscretizationDOFAxisTag,
    DiscretizationElementAxisTag,
    DiscretizationFaceAxisTag,
    FirstAxisIsElementsTag,
)
from pytools import keyed_memoize_in
from pytools.obj_array import make_obj_array

import grudge.dof_desc as dof_desc
from grudge.discretization import DiscretizationCollection
from grudge.dof_desc import (
    DD_VOLUME_ALL,
    DISCR_TAG_BASE,
    DISCR_TAG_QUAD,
    FACE_RESTR_ALL,
    DOFDesc,
    VolumeDomainTag,
    as_dofdesc,
)
from grudge.interpolation import interp
from grudge.projection import project
from grudge.reductions import (
    elementwise_integral,
    elementwise_max,
    elementwise_min,
    elementwise_sum,
    integral,
    nodal_max,
    nodal_max_loc,
    nodal_min,
    nodal_min_loc,
    nodal_sum,
    nodal_sum_loc,
    norm,
)
from grudge.trace_pair import (
    bdry_trace_pair,
    bv_trace_pair,
    connected_ranks,
    cross_rank_trace_pairs,
    interior_trace_pair,
    interior_trace_pairs,
    local_interior_trace_pair,
    project_tracepair,
    tracepair_with_discr_tag,
)
from grudge.transform.metadata import (
    OutputIsTensorProductDOFArrayOrdered,
    TensorProductDOFAxisTag,
    ReferenceTensorProductMassOperatorTag as MassMatrix1D,
    ReferenceTensorProductMassInverseOperatorTag as InverseMassMatrix1D,
    TensorProductOperatorAxisTag
)


__all__ = (
    "bdry_trace_pair",
    "bv_trace_pair",
    "connected_ranks",
    "cross_rank_trace_pairs",
    "elementwise_integral",
    "elementwise_max",
    "elementwise_min",
    "elementwise_sum",
    "face_mass",
    "integral",
    "interior_trace_pair",
    "interior_trace_pairs",
    "interp",
    "inverse_mass",
    "local_d_dx",
    "local_div",
    "local_grad",
    "local_interior_trace_pair",
    "mass",
    "nodal_max",
    "nodal_max_loc",
    "nodal_min",
    "nodal_min_loc",
    "nodal_sum",
    "nodal_sum_loc",
    "norm",
    "project",
    "project_tracepair",
    "tracepair_with_discr_tag",
    "weak_local_d_dx",
    "weak_local_div",
    "weak_local_grad",
    )


# {{{ common derivative "kernels"

def _single_axis_derivative_kernel(
        actx, out_discr, in_discr, get_diff_mat, inv_jac_mat, xyz_axis, vec,
        *, metric_in_matvec):
    # This gets used from both the strong and the weak derivative. These differ
    # in three ways:
    # - which differentiation matrix gets used,
    # - whether inv_jac_mat is pre-multiplied by a factor that includes the
    #   area element, and
    # - whether the chain rule terms ("inv_jac_mat") sit outside (strong)
    #   or inside (weak) the matrix-vector product that carries out the
    #   derivative, cf. "metric_in_matvec".
    return DOFArray(
        actx,
        data=tuple(
            # r for rst axis
            actx.einsum(
                "rej,rij,ej->ei" if metric_in_matvec else "rei,rij,ej->ei",
                ijm_i[xyz_axis],
                get_diff_mat(
                    actx,
                    out_element_group=out_grp,
                    in_element_group=in_grp),
                vec_i,
                arg_names=("inv_jac_t", "ref_stiffT_mat", "vec", ),
                tagged=(FirstAxisIsElementsTag(),))

            for out_grp, in_grp, vec_i, ijm_i in zip(
                out_discr.groups, in_discr.groups, vec,
                inv_jac_mat)))


def _gradient_kernel(actx, out_discr, in_discr, get_diff_mat, inv_jac_mat, vec,
        *, metric_in_matvec):
    # See _single_axis_derivative_kernel for comments on the usage scenarios
    # (both strong and weak derivative) and their differences.

    def compute_simplicial_gradient(actx, out_grp, in_grp, vec, ijm,
                                    metric_in_matvec):
        return actx.einsum(
            "xrej,rij,ej->xei" if metric_in_matvec else "xrei,rij,ej->xei",
                ijm,
                get_diff_mat(
                    actx,
                    out_element_group=out_grp,
                    in_element_group=in_grp),
                vec,
                arg_names=("inv_jac_t", "ref_stiffT_mat", "vec"),
                tagged=(FirstAxisIsElementsTag(),))

    def compute_tensor_product_gradient(actx, out_grp, in_grp, vec, ijm,
                                        metric_in_matvec):
        vec = actx.einsum(
            "xrej,ej->xrej", ijm, vec,
            tagged=(FirstAxisIsElementsTag(),),
            arg_names=("inv_jac_t", "dofs"))

        # expose tensor product structure
        if in_grp.dim != 1:
            vec = fold(in_grp.space, vec)

        # weak form ref_gradient
        if metric_in_matvec:
            mass_mat, stiff_mat = get_diff_mat(actx, out_grp, in_grp)

            ref_grad = []
            for xyz_axis in range(in_grp.dim):

                ref_grad.append(0.0)
                for rst_axis in range(in_grp.dim):
                    partial = vec[xyz_axis, rst_axis]

                    # apply mass matrix to all axes except current axis
                    for axis in range(in_grp.dim):
                        if axis == rst_axis:
                            continue

                        partial = single_axis_contraction(
                            actx, in_grp.dim, axis, mass_mat, partial,
                            tagged=(FirstAxisIsElementsTag(),
                                OutputIsTensorProductDOFArrayOrdered()),
                            arg_names=("mass_1d", f"dofs_{rst_axis}"))

                    partial = single_axis_contraction(
                        actx, in_grp.dim, rst_axis, stiff_mat, partial,
                        tagged=(FirstAxisIsElementsTag(),
                            OutputIsTensorProductDOFArrayOrdered(),),
                        arg_names=("stiff_mat",
                            f"dofs_with_mass_{rst_axis}"))

                    if in_grp.dim != 1:
                        partial = unfold(out_grp.space, partial)

                    ref_grad[xyz_axis] += partial

        # strong form ref_gradient
        else:
            diff_mat = get_diff_mat(actx, out_grp, in_grp)

            ref_grad = []
            for xyz_axis in range(in_grp.dim):

                ref_grad.append(0.0)
                for rst_axis in range(in_grp.dim):
                    partial = single_axis_contraction(
                        actx, in_grp.dim, rst_axis, diff_mat,
                        vec[xyz_axis, rst_axis],
                        tagged=(FirstAxisIsElementsTag(),
                            OutputIsTensorProductDOFArrayOrdered(),),
                        arg_names=("diff_mat", "dofs"))

                    if out_grp.dim != 1:
                        partial = unfold(out_grp.space, partial)

                    ref_grad[xyz_axis] += partial

        return tag_axes(
            actx,
            {
                0: DiscretizationAmbientDimAxisTag(),
                1: DiscretizationElementAxisTag(),
                2: DiscretizationDOFAxisTag()
            },
            actx.np.stack(ref_grad))

    per_group_grads = []
    for out_grp, in_grp, vec_i, ijm_i in zip(
        out_discr.groups, in_discr.groups, vec, inv_jac_mat):

        if isinstance(in_grp, SimplexElementGroupBase) and \
            isinstance(out_grp, SimplexElementGroupBase):
            per_group_grads.append(
                compute_simplicial_gradient(
                    actx, out_grp, in_grp, vec_i, ijm_i,metric_in_matvec))

        elif isinstance(in_grp, TensorProductElementGroupBase) and \
              isinstance(out_grp, TensorProductElementGroupBase):
            per_group_grads.append(
                compute_tensor_product_gradient(
                    actx, out_grp, in_grp, vec_i, ijm_i, metric_in_matvec))

        else:
            raise TypeError(
                "`in_grp` and `out_grp` must both be either "
                "`SimplexElementGroupBase` or `TensorProductElementGroupBase`. "
                f"Found `in_grp` = {in_grp}, `out_grp` = {out_grp}")

    return make_obj_array([
            DOFArray(actx, data=tuple([  # noqa: C409
                pgg_i[xyz_axis] for pgg_i in per_group_grads
                ]))
            for xyz_axis in range(out_discr.ambient_dim)])


def _divergence_kernel(actx, out_discr, in_discr, get_diff_mat, inv_jac_mat, vec,
        *, metric_in_matvec):
    # See _single_axis_derivative_kernel for comments on the usage scenarios
    # (both strong and weak derivative) and their differences.

    def compute_simplicial_divergence(actx, out_grp, in_grp, vec, ijm,
                                      metric_in_matvec):
        return actx.einsum(
            "xrej,rij,xej->ei" if metric_in_matvec else "xrei,rij,xej->ei",
            ijm,
            get_diff_mat(
                actx,
                out_element_group=out_grp,
                in_element_group=in_grp
            ),
            vec,
            arg_names=("inv_jac_t", "ref_stiffT_mat", "vec"),
            tagged=(FirstAxisIsElementsTag(),))

    def compute_tensor_product_divergence(actx, out_grp, in_grp, vec, ijm,
                                          metric_in_matvec):
        if in_grp.dim != 1:
            vec = fold(in_grp.space, vec)

        div = 0.0
        if metric_in_matvec:
            mass_mat, stiff_mat = get_diff_mat(actx, out_grp, in_grp)

            for func_axis in range(in_grp.dim):
                for rst_axis in range(in_grp.dim):

                    ref_deriv = vec[func_axis]
                    for axis in range(in_grp.dim):
                        if axis == rst_axis:
                            continue
                        ref_deriv = single_axis_contraction(
                            actx, in_grp.dim, axis, mass_mat, ref_deriv,
                            tagged=(FirstAxisIsElementsTag(),
                                OutputIsTensorProductDOFArrayOrdered(),),
                            arg_names=("mass_1d",
                                f"dofs_{func_axis}_{rst_axis}"))

                    ref_deriv = single_axis_contraction(
                            actx, in_grp.dim, rst_axis, stiff_mat, ref_deriv,
                            tagged=(FirstAxisIsElementsTag(),
                                OutputIsTensorProductDOFArrayOrdered(),),
                            arg_names=("stiff_1d",
                                f"dofs_with_mass_{func_axis}_{rst_axis}"))

                    if in_grp.dim != 1:
                        ref_deriv = unfold(out_grp.space, ref_deriv)

                    div += ref_deriv*ijm[func_axis, rst_axis]

        else:
            diff_mat = get_diff_mat(actx, out_grp, in_grp)
            for func_axis in range(vec.shape[0]):
                for rst_axis in range(in_grp.dim):
                    ref_deriv = vec[func_axis]
                    ref_deriv = single_axis_contraction(
                            actx, in_grp.dim, rst_axis, diff_mat, ref_deriv,
                            tagged=(FirstAxisIsElementsTag(),
                                    OutputIsTensorProductDOFArrayOrdered(),),
                            arg_names=("diff_mat", "dofs"))

                    if in_grp.dim != 1:
                        ref_deriv = unfold(out_grp.space, ref_deriv)

                    div += ref_deriv*ijm[func_axis, rst_axis]

        return tag_axes(actx, { 0: DiscretizationElementAxisTag() }, div)


    per_group_divs = []
    for out_grp, in_grp, vec_i, ijm_i in zip(
        out_discr.groups, in_discr.groups, vec, inv_jac_mat):
        if isinstance(in_grp, SimplexElementGroupBase) and \
            isinstance(out_grp, SimplexElementGroupBase):
            per_group_divs.append(compute_simplicial_divergence(
                actx, out_grp, in_grp, vec_i, ijm_i, metric_in_matvec))

        elif isinstance(in_grp, TensorProductElementGroupBase) and \
            isinstance(out_grp, TensorProductElementGroupBase):
            per_group_divs.append(compute_tensor_product_divergence(
                actx, out_grp, in_grp, vec_i, ijm_i, metric_in_matvec))

        else:
            raise TypeError(
                "`in_grp` and `out_grp` must both be either "
                "`SimplexElementGroupBase` or `TensorProductElementGroupBase`. "
                f"Found `in_grp` = {in_grp}, `out_grp` = {out_grp}")

    return DOFArray(actx, data=tuple(per_group_divs))

# }}}


# {{{ Derivative operators

def _reference_derivative_matrices(actx: ArrayContext,
        out_element_group: NodalElementGroupBase,
        in_element_group: InterpolatoryElementGroupBase):

    @keyed_memoize_in(
        actx, _reference_derivative_matrices,
        lambda outgrp, ingrp: (
            outgrp.discretization_key(),
            ingrp.discretization_key()))
    def get_ref_derivative_mats(
                out_grp: NodalElementGroupBase,
                in_grp: InterpolatoryElementGroupBase):
        if isinstance(in_grp, TensorProductElementGroupBase) and \
            isinstance(out_grp, TensorProductElementGroupBase):

            basis_1d = in_grp.basis_obj().bases[0]
            to_nodes_1d = out_grp.unit_nodes[0][:out_grp.order+1].reshape(
                1, out_grp.order+1)
            from_nodes_1d = in_grp.unit_nodes[0][:in_grp.order+1].reshape(
                1, in_grp.order+1)

            diff_mat = mp.diff_matrices(basis_1d, to_nodes_1d,
                                        from_nodes=from_nodes_1d)[0]

            return actx.freeze(
                tag_axes(
                    actx,
                    { i: TensorProductOperatorAxisTag() for i in range(2) },
                    actx.from_numpy(diff_mat)))

        elif isinstance(in_grp, SimplexElementGroupBase) and \
              isinstance(out_grp, SimplexElementGroupBase):
            return actx.freeze(
                    actx.tag_axis(
                        1, DiscretizationDOFAxisTag(),
                        actx.from_numpy(
                            np.asarray(
                                mp.diff_matrices(
                                    in_grp.basis_obj(),
                                    out_grp.unit_nodes,
                                    from_nodes=in_grp.unit_nodes)))))

        else:
            raise TypeError(
                "`in_grp` and `out_grp` must both be either "
                "`SimplexElementGroupBase` or `TensorProductElementGroupBase`. "
                f"Found `in_grp` = {in_grp}, `out_grp` = {out_grp}")

    return get_ref_derivative_mats(out_element_group, in_element_group)


def _strong_scalar_grad(dcoll, dd_in, vec):
    assert isinstance(dd_in.domain_tag, VolumeDomainTag)

    from grudge.geometry import inverse_surface_metric_derivative_mat

    discr = dcoll.discr_from_dd(dd_in)
    actx = vec.array_context

    inverse_jac_mat = inverse_surface_metric_derivative_mat(actx, dcoll, dd=dd_in,
            _use_geoderiv_connection=actx.supports_nonscalar_broadcasting)
    return _gradient_kernel(actx, discr, discr,
            _reference_derivative_matrices, inverse_jac_mat, vec,
            metric_in_matvec=False)


def _strong_scalar_div(dcoll, dd, vecs):
    from arraycontext import get_container_context_recursively, serialize_container

    from grudge.geometry import inverse_surface_metric_derivative_mat

    assert isinstance(vecs, np.ndarray)
    assert vecs.shape == (dcoll.ambient_dim,)

    discr = dcoll.discr_from_dd(dd)

    actx = get_container_context_recursively(vecs)
    vec = actx.np.stack([v for k, v in serialize_container(vecs)])

    inverse_jac_mat = inverse_surface_metric_derivative_mat(actx, dcoll, dd=dd,
            _use_geoderiv_connection=actx.supports_nonscalar_broadcasting)

    return _divergence_kernel(actx, discr, discr,
            _reference_derivative_matrices, inverse_jac_mat, vec,
            metric_in_matvec=False)


def local_grad(
        dcoll: DiscretizationCollection, *args, nested=False) -> ArrayOrContainer:
    r"""Return the element-local gradient of a function :math:`f` represented
    by *vec*:

    .. math::

        \nabla|_E f = \left(
            \partial_x|_E f, \partial_y|_E f, \partial_z|_E f \right)

    May be called with ``(vec)`` or ``(dd_in, vec)``.

    :arg vec: a :class:`~meshmode.dof_array.DOFArray` or an
        :class:`~arraycontext.ArrayContainer` of them.
    :arg dd_in: a :class:`~grudge.dof_desc.DOFDesc`, or a value convertible to one.
        Defaults to the base volume discretization if not provided.
    :arg nested: return nested object arrays instead of a single multidimensional
        array if *vec* is non-scalar.
    :returns: an object array (possibly nested) of
        :class:`~meshmode.dof_array.DOFArray`\ s or
        :class:`~arraycontext.ArrayContainer` of object arrays.
    """
    if len(args) == 1:
        vec, = args
        dd_in = DD_VOLUME_ALL
    elif len(args) == 2:
        dd_in, vec = args
    else:
        raise TypeError("invalid number of arguments")

    from grudge.tools import rec_map_subarrays
    return rec_map_subarrays(
        partial(_strong_scalar_grad, dcoll, dd_in),
        (), (dcoll.ambient_dim,),
        vec, scalar_cls=DOFArray, return_nested=nested,)


def local_d_dx(
        dcoll: DiscretizationCollection, xyz_axis, *args) -> ArrayOrContainer:
    r"""Return the element-local derivative along axis *xyz_axis* of a
    function :math:`f` represented by *vec*:

    .. math::

        \frac{\partial f}{\partial \lbrace x,y,z\rbrace}\Big|_E

    May be called with ``(vec)`` or ``(dd, vec)``.

    :arg xyz_axis: an integer indicating the axis along which the derivative
        is taken.
    :arg dd: a :class:`~grudge.dof_desc.DOFDesc`, or a value convertible to one.
        Defaults to the base volume discretization if not provided.
    :arg vec: a :class:`~meshmode.dof_array.DOFArray` or an
        :class:`~arraycontext.ArrayContainer` of them.
    :returns: a :class:`~meshmode.dof_array.DOFArray` or an
        :class:`~arraycontext.ArrayContainer` of them.
    """
    if len(args) == 1:
        vec, = args
        dd = DD_VOLUME_ALL
    elif len(args) == 2:
        dd, vec = args
    else:
        raise TypeError("invalid number of arguments")

    if not isinstance(vec, DOFArray):
        return map_array_container(partial(local_d_dx, dcoll, xyz_axis, dd), vec)

    discr = dcoll.discr_from_dd(dd)
    actx = vec.array_context

    from grudge.geometry import inverse_surface_metric_derivative_mat
    inverse_jac_mat = inverse_surface_metric_derivative_mat(actx, dcoll, dd=dd,
        _use_geoderiv_connection=actx.supports_nonscalar_broadcasting)

    return _single_axis_derivative_kernel(
        actx, discr, discr,
        _reference_derivative_matrices, inverse_jac_mat, xyz_axis, vec,
        metric_in_matvec=False)


def local_div(dcoll: DiscretizationCollection, *args) -> ArrayOrContainer:
    r"""Return the element-local divergence of the vector function
    :math:`\mathbf{f}` represented by *vecs*:

    .. math::

        \nabla|_E \cdot \mathbf{f} = \sum_{i=1}^d \partial_{x_i}|_E \mathbf{f}_i

    May be called with ``(vecs)`` or ``(dd, vecs)``.

    :arg dd: a :class:`~grudge.dof_desc.DOFDesc`, or a value convertible to one.
        Defaults to the base volume discretization if not provided.
    :arg vecs: an object array of
        :class:`~meshmode.dof_array.DOFArray`\s or an
        :class:`~arraycontext.ArrayContainer` object
        with object array entries. The last axis of the array
        must have length matching the volume dimension.
    :returns: a :class:`~meshmode.dof_array.DOFArray` or an
        :class:`~arraycontext.ArrayContainer` of them.
    """
    if len(args) == 1:
        vecs, = args
        dd = DD_VOLUME_ALL
    elif len(args) == 2:
        dd, vecs = args
    else:
        raise TypeError("invalid number of arguments")

    from grudge.tools import rec_map_subarrays
    return rec_map_subarrays(
        lambda vec: _strong_scalar_div(dcoll, dd, vec),
        (dcoll.ambient_dim,), (),
        vecs, scalar_cls=DOFArray)

# }}}


# {{{ Weak derivative operators

def _reference_stiffness_transpose_matrices(
        actx: ArrayContext, out_element_group, in_element_group):
    @keyed_memoize_in(
        actx, _reference_stiffness_transpose_matrices,
        lambda out_grp, in_grp: (out_grp.discretization_key(),
                                 in_grp.discretization_key()))
    def get_ref_stiffness_transpose_mat(out_grp, in_grp):
        if in_grp == out_grp:
            if isinstance(out_grp, TensorProductElementGroupBase):
                basis_1d = out_grp.basis_obj().bases[0]
                nodes_1d = out_grp.unit_nodes[0][:out_grp.order+1].reshape(
                    1, out_grp.order+1)

                diff_mat_1d = mp.diff_matrices(basis_1d, nodes_1d)[0]
                mass_mat_1d = mp.mass_matrix(basis_1d, nodes_1d)

                stiff_mat_1d = diff_mat_1d.T @ mass_mat_1d.T

                mass_mat_1d = actx.freeze(
                    actx.tag(
                        MassMatrix1D(),
                        tag_axes(
                            actx,
                            {i : TensorProductOperatorAxisTag()
                                for i in range(2) },
                            actx.from_numpy(mass_mat_1d))))

                stiff_mat_1d = actx.freeze(
                    tag_axes(
                        actx,
                        { i: TensorProductOperatorAxisTag() for i in range(2) },
                        actx.from_numpy(stiff_mat_1d)))

                return mass_mat_1d, stiff_mat_1d

            else:
                mass_mat = mp.mass_matrix(out_grp.basis_obj(),
                                          out_grp.unit_nodes)
                diff_matrices = mp.diff_matrices(out_grp.basis_obj(),
                                                 out_grp.unit_nodes)

                return actx.freeze(
                    actx.tag_axis(1, DiscretizationDOFAxisTag(),
                        actx.from_numpy(
                            np.asarray([
                                  dmat.T @ mass_mat.T
                                  for dmat in diff_matrices]))))

        if isinstance(out_grp, TensorProductElementGroupBase) and \
            isinstance(in_grp, TensorProductElementGroupBase):

            basis_1d = out_grp.basis_obj().bases[0]
            nodes_1d_out = out_grp.unit_nodes[0][:out_grp.order+1].reshape(
                1, out_grp.order+1)
            nodes_1d_in = in_grp.unit_nodes[0][:in_grp.order+1].reshape(
                1, in_grp.order+1)

            vand = vandermonde(basis_1d.functions, nodes_1d_out)
            vand_inv_t = np.linalg.inv(vand).T
            grad_vand = multi_vandermonde(basis_1d.gradients, nodes_1d_in)[0]

            weights = in_grp.quadrature_rule().quadratures[0].weights

            stiff_mat_1d = np.einsum("ik,kj,j->ij",
                                     vand_inv_t, grad_vand.T, weights)
            stiff_mat_1d = actx.freeze(
                tag_axes(
                    actx,
                    { i : TensorProductOperatorAxisTag() for i in range(2) },
                    actx.from_numpy(stiff_mat_1d)))

            mass_mat_1d = reference_mass_matrix(
                actx, in_element_group=in_grp, out_element_group=out_grp)

            return mass_mat_1d, stiff_mat_1d

        else:

            basis = out_grp.basis_obj()
            vand = vandermonde(basis.functions, out_grp.unit_nodes)
            grad_vand = multi_vandermonde(basis.gradients, in_grp.unit_nodes)
            vand_inv_t = np.linalg.inv(vand).T

            if not isinstance(grad_vand, tuple):
                # NOTE: special case for 1d
                grad_vand = (grad_vand,)

            weights = in_grp.quadrature_rule().weights
            return actx.freeze(
                actx.from_numpy(
                    np.einsum(
                        "c,bz,acz->abc",
                        weights,
                        vand_inv_t,
                        grad_vand
                    ).copy()  # contigify the array
                ))
    return get_ref_stiffness_transpose_mat(out_element_group, in_element_group)


def _weak_scalar_grad(dcoll, dd_in, vec):
    from grudge.geometry import inverse_surface_metric_derivative_mat

    dd_in = as_dofdesc(dd_in)
    in_discr = dcoll.discr_from_dd(dd_in)
    out_discr = dcoll.discr_from_dd(dd_in.with_discr_tag(DISCR_TAG_BASE))

    actx = vec.array_context
    inverse_jac_mat = inverse_surface_metric_derivative_mat(actx, dcoll,
        dd=dd_in,
        times_area_element=True,
        _use_geoderiv_connection=actx.supports_nonscalar_broadcasting)

    return _gradient_kernel(actx, out_discr, in_discr,
            _reference_stiffness_transpose_matrices, inverse_jac_mat, vec,
            metric_in_matvec=True)


def _weak_scalar_div(dcoll, dd_in, vecs):
    from arraycontext import get_container_context_recursively, serialize_container

    from grudge.geometry import inverse_surface_metric_derivative_mat

    assert isinstance(vecs, np.ndarray)
    assert vecs.shape == (dcoll.ambient_dim,)

    in_discr = dcoll.discr_from_dd(dd_in)
    out_discr = dcoll.discr_from_dd(dd_in.with_discr_tag(DISCR_TAG_BASE))

    actx = get_container_context_recursively(vecs)
    vec = actx.np.stack([v for k, v in serialize_container(vecs)])

    inverse_jac_mat = inverse_surface_metric_derivative_mat(actx, dcoll, dd=dd_in,
            times_area_element=True,
            _use_geoderiv_connection=actx.supports_nonscalar_broadcasting)

    return _divergence_kernel(actx, out_discr, in_discr,
            _reference_stiffness_transpose_matrices, inverse_jac_mat, vec,
            metric_in_matvec=True)


def weak_local_grad(
        dcoll: DiscretizationCollection, *args, nested=False) -> ArrayOrContainer:
    r"""Return the element-local weak gradient of the volume function
    represented by *vec*.

    May be called with ``(vec)`` or ``(dd_in, vec)``.

    Specifically, the function returns an object array where the :math:`i`-th
    component is the weak derivative with respect to the :math:`i`-th coordinate
    of a scalar function :math:`f`. See :func:`weak_local_d_dx` for further
    information. For non-scalar :math:`f`, the function will return a nested object
    array containing the component-wise weak derivatives.

    :arg dd_in: a :class:`~grudge.dof_desc.DOFDesc`, or a value convertible to one.
        Defaults to the base volume discretization if not provided.
    :arg vec: a :class:`~meshmode.dof_array.DOFArray` or an
        :class:`~arraycontext.ArrayContainer` of them.
    :arg nested: return nested object arrays instead of a single multidimensional
        array if *vec* is non-scalar
    :returns: an object array (possibly nested) of
        :class:`~meshmode.dof_array.DOFArray`\ s or
        :class:`~arraycontext.ArrayContainer` of object arrays.
    """
    if len(args) == 1:
        vecs, = args
        dd_in = DD_VOLUME_ALL
    elif len(args) == 2:
        dd_in, vecs = args
    else:
        raise TypeError("invalid number of arguments")

    from grudge.tools import rec_map_subarrays
    return rec_map_subarrays(
        partial(_weak_scalar_grad, dcoll, dd_in),
        (), (dcoll.ambient_dim,),
        vecs, scalar_cls=DOFArray, return_nested=nested)


def weak_local_d_dx(dcoll: DiscretizationCollection, *args) -> ArrayOrContainer:
    r"""Return the element-local weak derivative along axis *xyz_axis* of the
    volume function represented by *vec*.

    May be called with ``(xyz_axis, vec)`` or ``(dd_in, xyz_axis, vec)``.

    Specifically, this function computes the volume contribution of the
    weak derivative in the :math:`i`-th component (specified by *xyz_axis*)
    of a function :math:`f`, in each element :math:`E`, with respect to polynomial
    test functions :math:`\phi`:

    .. math::

        \int_E \partial_i\phi\,f\,\mathrm{d}x \sim
        \mathbf{D}_{E,i}^T \mathbf{M}_{E}^T\mathbf{f}|_E,

    where :math:`\mathbf{D}_{E,i}` is the polynomial differentiation matrix on
    an :math:`E` for the :math:`i`-th spatial coordinate, :math:`\mathbf{M}_E`
    is the elemental mass matrix (see :func:`mass` for more information), and
    :math:`\mathbf{f}|_E` is a vector of coefficients for :math:`f` on :math:`E`.

    :arg dd_in: a :class:`~grudge.dof_desc.DOFDesc`, or a value convertible to one.
        Defaults to the base volume discretization if not provided.
    :arg xyz_axis: an integer indicating the axis along which the derivative
        is taken.
    :arg vec: a :class:`~meshmode.dof_array.DOFArray` or an
        :class:`~arraycontext.ArrayContainer` of them.
    :returns: a :class:`~meshmode.dof_array.DOFArray` or an
        :class:`~arraycontext.ArrayContainer` of them.
    """
    if len(args) == 2:
        xyz_axis, vec = args
        dd_in = dof_desc.DD_VOLUME_ALL
    elif len(args) == 3:
        dd_in, xyz_axis, vec = args
    else:
        raise TypeError("invalid number of arguments")

    if not isinstance(vec, DOFArray):
        return map_array_container(
            partial(weak_local_d_dx, dcoll, dd_in, xyz_axis),
            vec
        )

    from grudge.geometry import inverse_surface_metric_derivative_mat

    dd_in = as_dofdesc(dd_in)
    in_discr = dcoll.discr_from_dd(dd_in)
    out_discr = dcoll.discr_from_dd(dd_in.with_discr_tag(DISCR_TAG_BASE))

    actx = vec.array_context
    inverse_jac_mat = inverse_surface_metric_derivative_mat(actx, dcoll, dd=dd_in,
            times_area_element=True,
            _use_geoderiv_connection=actx.supports_nonscalar_broadcasting)

    return _single_axis_derivative_kernel(
            actx, out_discr, in_discr, _reference_stiffness_transpose_matrices,
            inverse_jac_mat, xyz_axis, vec,
            metric_in_matvec=True)


def weak_local_div(dcoll: DiscretizationCollection, *args) -> ArrayOrContainer:
    r"""Return the element-local weak divergence of the vector volume function
    represented by *vecs*.

    May be called with ``(vecs)`` or ``(dd, vecs)``.

    Specifically, this function computes the volume contribution of the
    weak divergence of a vector function :math:`\mathbf{f}`, in each element
    :math:`E`, with respect to polynomial test functions :math:`\phi`:

    .. math::

        \int_E \nabla \phi \cdot \mathbf{f}\,\mathrm{d}x \sim
        \sum_{i=1}^d \mathbf{D}_{E,i}^T \mathbf{M}_{E}^T\mathbf{f}_i|_E,

    where :math:`\mathbf{D}_{E,i}` is the polynomial differentiation matrix on
    an :math:`E` for the :math:`i`-th spatial coordinate, and :math:`\mathbf{M}_E`
    is the elemental mass matrix (see :func:`mass` for more information).

    :arg dd: a :class:`~grudge.dof_desc.DOFDesc`, or a value convertible to one.
        Defaults to the base volume discretization if not provided.
    :arg vecs: an object array of
        :class:`~meshmode.dof_array.DOFArray`\s or an
        :class:`~arraycontext.ArrayContainer` object
        with object array entries. The last axis of the array
        must have length matching the volume dimension.
    :returns: a :class:`~meshmode.dof_array.DOFArray` or an
        :class:`~arraycontext.ArrayContainer` like *vec*.
    """
    if len(args) == 1:
        vecs, = args
        dd_in = DD_VOLUME_ALL
    elif len(args) == 2:
        dd_in, vecs = args
    else:
        raise TypeError("invalid number of arguments")

    from grudge.tools import rec_map_subarrays
    return rec_map_subarrays(
        lambda vec: _weak_scalar_div(dcoll, dd_in, vec),
        (dcoll.ambient_dim,), (),
        vecs, scalar_cls=DOFArray)

# }}}


# {{{ Mass operator

def reference_mass_matrix(actx: ArrayContext, out_element_group, in_element_group):
    @keyed_memoize_in(
        actx, reference_mass_matrix,
        lambda out_grp, in_grp: (out_grp.discretization_key(),
                                 in_grp.discretization_key()))
    def get_ref_mass_mat(out_grp, in_grp):
        if out_grp == in_grp:
            if isinstance(out_grp, TensorProductElementGroupBase):
                basis_1d = out_grp.basis_obj().bases[0]
                nodes_1d = out_grp.unit_nodes[0][:out_grp.order+1].reshape(
                    1, out_grp.order+1)

                return actx.tag(
                    MassMatrix1D(),
                    tag_axes(
                        actx,
                        { i : TensorProductOperatorAxisTag()
                            for i in range(2) },
                        actx.from_numpy(mp.mass_matrix(basis_1d, nodes_1d))))

            else:
                return actx.freeze(
                    actx.from_numpy(
                        mp.mass_matrix(out_grp.basis_obj(),
                                       out_grp.unit_nodes)))

        if isinstance(in_grp, TensorProductElementGroupBase) and \
            isinstance(out_grp, TensorProductElementGroupBase):

            basis_1d = out_grp.basis_obj().bases[0]
            nodes_1d_out = out_grp.unit_nodes[0][:out_grp.order+1].reshape(
                1, out_grp.order+1)

            mass_matrix = nodal_quad_mass_matrix(
                in_grp.quadrature_rule().quadratures[0], basis_1d.functions,
                nodes_1d_out)

            return actx.freeze(
                actx.tag(
                    MassMatrix1D(),
                    tag_axes(
                        actx,
                        { i : TensorProductOperatorAxisTag()
                            for i in range(2) },
                        actx.from_numpy(mass_matrix))))

        elif isinstance(in_grp, SimplexElementGroupBase) and \
            isinstance(out_grp, SimplexElementGroupBase):
            basis = out_grp.basis_obj()
            vand = vandermonde(basis.functions, out_grp.unit_nodes)
            o_vand = vandermonde(basis.functions, in_grp.unit_nodes)
            vand_inv_t = np.linalg.inv(vand).T

            weights = in_grp.quadrature_rule().weights
            return actx.freeze(
                actx.tag_axis(0, DiscretizationDOFAxisTag(),
                    actx.from_numpy(
                        np.asarray(
                            np.einsum(
                              "j,ik,jk->ij", weights, vand_inv_t, o_vand),
                            order="C"))))

    return get_ref_mass_mat(out_element_group, in_element_group)


def _apply_mass_operator(
        dcoll: DiscretizationCollection, dd_out, dd_in, vec):
    if not isinstance(vec, DOFArray):
        return map_array_container(
            partial(_apply_mass_operator, dcoll, dd_out, dd_in), vec
        )

    def tensor_product_apply_mass(in_grp, out_grp, area_element, vec):
        vec = area_element * vec

        if in_grp.dim != 1:
            vec = fold(in_grp.space, vec)

        ref_mass_1d = reference_mass_matrix(
            actx, out_element_group=out_grp, in_element_group=in_grp)

        for xyz_axis in range(in_grp.dim):
            vec = single_axis_contraction(
                actx, in_grp.dim, xyz_axis, ref_mass_1d, vec,
                tagged=(FirstAxisIsElementsTag(),
                        OutputIsTensorProductDOFArrayOrdered()),
                arg_names=("ref_mass_1d", "dofs"))

        if in_grp.dim != 1:
            return unfold(out_grp.space, vec)

        return vec

    def simplicial_apply_mass(in_grp, out_grp, area_element, vec):
        return actx.einsum("ij,ej,ej->ei",
            reference_mass_matrix(
                actx,
                out_element_group=out_grp,
                in_element_group=in_grp
                ),
            area_element,
            vec,
            arg_names=("mass_mat", "jac", "vec"),
            tagged=(FirstAxisIsElementsTag(),))

    from grudge.geometry import area_element

    in_discr = dcoll.discr_from_dd(dd_in)
    out_discr = dcoll.discr_from_dd(dd_out)

    actx = vec.array_context
    area_elements = area_element(actx, dcoll, dd=dd_in,
            _use_geoderiv_connection=actx.supports_nonscalar_broadcasting)

    group_data = []
    for in_grp, out_grp, ae_i, vec_i in zip(
        in_discr.groups, out_discr.groups, area_elements, vec):
        if isinstance(in_grp, TensorProductElementGroupBase) and \
            isinstance(out_grp, TensorProductElementGroupBase):
            group_data.append(tensor_product_apply_mass(
                in_grp, out_grp, ae_i, vec_i))

        elif isinstance(in_grp, SimplexElementGroupBase) and \
            isinstance(out_grp, SimplexElementGroupBase):
            group_data.append(simplicial_apply_mass(
                in_grp, out_grp, ae_i, vec_i))

        else:
            raise TypeError(
                "`in_grp` and `out_grp` must both be either "
                "`SimplexElementGroupBase` or `TensorProductElementGroupBase`. "
                f"Found `in_grp` = {in_grp}, `out_grp` = {out_grp}")

    return DOFArray(actx, data=tuple(group_data))


def mass(dcoll: DiscretizationCollection, *args) -> ArrayOrContainer:
    r"""Return the action of the DG mass matrix on a vector (or vectors)
    of :class:`~meshmode.dof_array.DOFArray`\ s, *vec*. In the case of
    *vec* being an :class:`~arraycontext.ArrayContainer`,
    the mass operator is applied component-wise.

    May be called with ``(vec)`` or ``(dd_in, vec)``.

    Specifically, this function applies the mass matrix elementwise on a
    vector of coefficients :math:`\mathbf{f}` via:
    :math:`\mathbf{M}_{E}\mathbf{f}|_E`, where

    .. math::

        \left(\mathbf{M}_{E}\right)_{ij} = \int_E \phi_i \cdot \phi_j\,\mathrm{d}x,

    where :math:`\phi_i` are local polynomial basis functions on :math:`E`.

    :arg dd_in: a :class:`~grudge.dof_desc.DOFDesc`, or a value convertible to one.
        Defaults to the base volume discretization if not provided.
    :arg vec: a :class:`~meshmode.dof_array.DOFArray` or an
        :class:`~arraycontext.ArrayContainer` of them.
    :returns: a :class:`~meshmode.dof_array.DOFArray` or an
        :class:`~arraycontext.ArrayContainer` like *vec*.
    """

    if len(args) == 1:
        vec, = args
        dd_in = dof_desc.DD_VOLUME_ALL
    elif len(args) == 2:
        dd_in, vec = args
    else:
        raise TypeError("invalid number of arguments")

    dd_out = dd_in.with_discr_tag(DISCR_TAG_BASE)

    return _apply_mass_operator(dcoll, dd_out, dd_in, vec)

# }}}


# {{{ Mass inverse operator

def reference_inverse_mass_matrix(actx: ArrayContext, element_group):
    @keyed_memoize_in(
        actx, reference_inverse_mass_matrix,
        lambda grp: grp.discretization_key())
    def get_ref_inv_mass_mat(grp):

        if isinstance(grp, TensorProductElementGroupBase):
            basis_1d = grp.basis_obj().bases[0]
            nodes_1d = grp.unit_nodes[0][:grp.order+1].reshape(1, grp.order+1)

            return actx.freeze(
                actx.tag(
                    InverseMassMatrix1D(),
                    tag_axes(
                    actx,
                    { i : TensorProductOperatorAxisTag() for i in range(2) },
                    actx.from_numpy(
                        inverse_mass_matrix(basis_1d, nodes_1d)))))

        else:
            basis = grp.basis_obj()

            return actx.freeze(
                actx.tag_axis(0, DiscretizationDOFAxisTag(),
                    actx.from_numpy(
                        np.asarray(
                            inverse_mass_matrix(basis, grp.unit_nodes),
                            order="C"))))

    return get_ref_inv_mass_mat(element_group)


def _apply_inverse_mass_operator(
        dcoll: DiscretizationCollection, dd_out, dd_in, vec,
        uses_quadrature=False):
    if not isinstance(vec, DOFArray):
        return map_array_container(
            partial(_apply_inverse_mass_operator, dcoll, dd_out, dd_in,
                    uses_quadrature), vec
        )

    from grudge.geometry import area_element

    if dd_out != dd_in and not uses_quadrature:
        raise ValueError(
            "Cannot compute inverse of a mass matrix mapping "
            "between different element groups unless using overintegration; "
            "inverse is not guaranteed to be well-defined")

    def tensor_product_apply_inverse_mass(grp, jac_inv, vec):
        vec = jac_inv * vec

        if grp.dim != 1:
            vec = fold(grp.space, vec)

        for rst_axis in range(grp.dim):
            vec = single_axis_contraction(
                actx, grp.dim, rst_axis,
                reference_inverse_mass_matrix(actx, element_group=grp),
                vec,
                tagged=(FirstAxisIsElementsTag(),
                        OutputIsTensorProductDOFArrayOrdered()),
                arg_names=("ref_inv_mass", "dofs"))

        if grp.dim != 1:
            return unfold(grp.space, vec)

        return vec

    def simplicial_apply_inverse_mass(grp, jac_inv, vec):
        return actx.einsum("ei,ij,ej->ei",
            jac_inv,
            reference_inverse_mass_matrix(actx, element_group=grp),
            vec,
            tagged=(FirstAxisIsElementsTag(),))

    actx = vec.array_context

    # compute on quadrature discretization if used
    inv_area_elements = 1./area_element(actx, dcoll, dd=dd_in,
            _use_geoderiv_connection=actx.supports_nonscalar_broadcasting)

    group_data = []
    if not uses_quadrature: # no overintegration
        # Both element type versions use a weighted approximation to inv mass
        # based on https://arxiv.org/pdf/1608.03836.pdf, i.e.
        # true_Minv ~ ref_Minv * ref_M * (1/jac_det) * ref_Minv

        discr = dcoll.discr_from_dd(dd_in)
        for grp, jac_inv, vec_i in zip(discr.groups, inv_area_elements, vec):
            if isinstance(grp, TensorProductElementGroupBase):
                group_data.append(tensor_product_apply_inverse_mass(
                    grp, jac_inv, vec_i))

            elif isinstance(grp, SimplexElementGroupBase):
                group_data.append(simplicial_apply_inverse_mass(
                    grp, jac_inv, vec_i))

            else:
                raise TypeError(
                    "Expected grp to be either `TensorProductElementGroupBase` "
                    f"or f`SimplexElementGroupBase`. Found grp = {grp}")

    else: # overintegration
        # Weighted approximation as above, but needs a projection to the
        # quadrature domain. The formula above becomes:
        # true_Minv ~ ref_Minv * (ref_M)_qtb * (1/Jac)_quad * P(Minv*vec)
        # P => projection to quadrature, qti => quad-to-base

        discr_base = dcoll.discr_from_dd(dd_out)
        discr_quad = dcoll.discr_from_dd(dd_in)

        # apply base discretization inverse mass to vec
        base_group_data = []
        for grp, vec_i in zip(discr_base.groups, vec):
            if isinstance(grp, TensorProductElementGroupBase):
                if grp.dim != 1:
                    vec_i = fold(grp.space, vec_i)

                for rst_axis in range(grp.dim):
                    vec_i = single_axis_contraction(
                        actx, grp.dim, rst_axis,
                        reference_inverse_mass_matrix(actx, element_group=grp),
                        vec_i,
                        tagged=(FirstAxisIsElementsTag(),
                                OutputIsTensorProductDOFArrayOrdered(),),
                        arg_names=("base_mass_inv", "dofs"))

                if grp.dim != 1:
                    vec_i = unfold(grp.space, vec_i)

            elif isinstance(grp, SimplexElementGroupBase):
                vec_i = actx.einsum(
                    "ij,ej->ei",
                    reference_inverse_mass_matrix(actx, element_group=grp),
                    vec_i,
                    tagged=(FirstAxisIsElementsTag(),))

            else:
                raise TypeError(
                    "Expected grp to be either `TensorProductElementGroupBase` "
                    f"or f`SimplexElementGroupBase`. Found grp = {grp}")

            base_group_data.append(vec_i)

        # apply metric terms to projected result
        projection = inv_area_elements * project(
            dcoll, dd_out, dd_in,
            DOFArray(actx, data=tuple(base_group_data)))

        # apply WADG
        for in_grp, out_grp, vec_i in zip(
            discr_quad.groups, discr_base.groups, projection):
            if isinstance(in_grp, TensorProductElementGroupBase):
                if in_grp.dim != 1:
                    vec_i = fold(in_grp.space, vec_i)

                for rst_axis in range(in_grp.dim):
                    vec_i = single_axis_contraction(
                        actx, in_grp.dim, rst_axis,
                        reference_mass_matrix(actx,
                            in_element_group=in_grp,
                            out_element_group=out_grp),
                        vec_i,
                        tagged=(FirstAxisIsElementsTag(),
                                OutputIsTensorProductDOFArrayOrdered(),),
                        arg_names=("ref_mass_quad", "dofs"))

                for rst_axis in range(in_grp.dim):
                    vec_i = single_axis_contraction(
                        actx, in_grp.dim, rst_axis,
                        reference_inverse_mass_matrix(actx, out_grp),
                        vec_i,
                        tagged=(FirstAxisIsElementsTag(),
                                OutputIsTensorProductDOFArrayOrdered(),),
                        arg_names=("base_mass_inv", "dofs"))

                if out_grp.dim != 1:
                    vec_i = unfold(out_grp.space, vec_i)

            elif isinstance(in_grp, SimplexElementGroupBase):
                vec_i = actx.einsum(
                    "ik,kj,ej->ei",
                    reference_inverse_mass_matrix(actx, out_grp),
                    reference_mass_matrix(actx, out_grp, in_grp),
                    vec_i,
                    tagged=(FirstAxisIsElementsTag(),),
                    arg_names=("base_mass_inv", "quad_mass", "dofs"))

            else:
                raise TypeError(
                    "`in_grp` and `out_grp` must both be either "
                    "`SimplexElementGroupBase` or "
                    "`TensorProductElementGroupBase`. "
                    f"Found `in_grp` = {in_grp}, `out_grp` = {out_grp}")

            group_data.append(vec_i)

    return DOFArray(actx, data=tuple(group_data))


def inverse_mass(dcoll: DiscretizationCollection, *args) -> ArrayOrContainer:
    r"""Return the action of the DG mass matrix inverse on a vector
    (or vectors) of :class:`~meshmode.dof_array.DOFArray`\ s, *vec*.
    In the case of *vec* being an :class:`~arraycontext.ArrayContainer`,
    the inverse mass operator is applied component-wise.

    For affine elements :math:`E`, the element-wise mass inverse
    is computed directly as the inverse of the (physical) mass matrix:

    .. math::

        \left(\mathbf{M}_{J^e}\right)_{ij} =
            \int_{\widehat{E}} \widehat{\phi}_i\cdot\widehat{\phi}_j J^e
            \mathrm{d}\widehat{x},

    where :math:`\widehat{\phi}_i` are basis functions over the reference
    element :math:`\widehat{E}`, and :math:`J^e` is the (constant) Jacobian
    scaling factor (see :func:`grudge.geometry.area_element`).

    For non-affine :math:`E`, :math:`J^e` is not constant. In this case, a
    weight-adjusted approximation is used instead following [Chan_2016]_:

    .. math::

        \mathbf{M}_{J^e}^{-1} \approx
            \widehat{\mathbf{M}}^{-1}\mathbf{M}_{1/J^e}\widehat{\mathbf{M}}^{-1},

    where :math:`\widehat{\mathbf{M}}` is the reference mass matrix on
    :math:`\widehat{E}`.

    May be called with ``(vec)`` or ``(dd, vec)``.

    :arg vec: a :class:`~meshmode.dof_array.DOFArray` or an
        :class:`~arraycontext.ArrayContainer` of them.
    :arg dd: a :class:`~grudge.dof_desc.DOFDesc`, or a value convertible to one.
        Defaults to the base volume discretization if not provided.
    :returns: a :class:`~meshmode.dof_array.DOFArray` or an
        :class:`~arraycontext.ArrayContainer` like *vec*.
    """
    if len(args) == 1:
        vec, = args
        dd = DD_VOLUME_ALL
    elif len(args) == 2:
        dd, vec = args
    else:
        raise TypeError("invalid number of arguments")

    if dd.uses_quadrature():
        dd_in = dd.with_discr_tag(DISCR_TAG_QUAD)
        dd_out = dd.with_discr_tag(DISCR_TAG_BASE)
    else:
        dd_in = dd
        dd_out = dd

    return _apply_inverse_mass_operator(dcoll, dd_out, dd_in, vec,
                                        uses_quadrature=dd.uses_quadrature())

# }}}


# {{{ Face mass operator

def reference_face_mass_matrix(
        actx: ArrayContext, face_element_group, vol_element_group, dtype):
    @keyed_memoize_in(
        actx, reference_mass_matrix,
        lambda face_grp, vol_grp: (face_grp.discretization_key(),
                                   vol_grp.discretization_key()))
    def get_ref_face_mass_mat(face_grp, vol_grp):
        nfaces = vol_grp.mesh_el_group.nfaces
        assert face_grp.nelements == nfaces * vol_grp.nelements

        matrix = np.empty(
            (vol_grp.nunit_dofs,
            nfaces,
            face_grp.nunit_dofs),
            dtype=dtype
        )

        import modepy as mp
        from meshmode.discretization import ElementGroupWithBasis
        from meshmode.discretization.poly_element import \
            QuadratureSimplexElementGroup

        n = vol_grp.order
        m = face_grp.order
        vol_basis = vol_grp.basis_obj()
        faces = mp.faces_for_shape(vol_grp.shape)

        for iface, face in enumerate(faces):
            # If the face group is defined on a higher-order
            # quadrature grid, use the underlying quadrature rule
            if isinstance(face_grp, QuadratureSimplexElementGroup):
                face_quadrature = face_grp.quadrature_rule()
                if face_quadrature.exact_to < m:
                    raise ValueError(
                        "The face quadrature rule is only exact for polynomials "
                        f"of total degree {face_quadrature.exact_to}. Please "
                        "ensure a quadrature rule is used that is at least "
                        f"exact for degree {m}."
                    )
            else:
                # NOTE: This handles the general case where
                # volume and surface quadrature rules may have different
                # integration orders
                face_quadrature = mp.quadrature_for_space(
                    mp.space_for_shape(face, 2*max(n, m)),
                    face
                )

            # If the group has a nodal basis and is unisolvent,
            # we use the basis on the face to compute the face mass matrix
            if (isinstance(face_grp, ElementGroupWithBasis)
                    and face_grp.space.space_dim == face_grp.nunit_dofs):

                face_basis = face_grp.basis_obj()

                # Sanity check for face quadrature accuracy. Not integrating
                # degree N + M polynomials here is asking for a bad time.
                if face_quadrature.exact_to < m + n:
                    raise ValueError(
                        "The face quadrature rule is only exact for polynomials "
                        f"of total degree {face_quadrature.exact_to}. Please "
                        "ensure a quadrature rule is used that is at least "
                        f"exact for degree {n+m}."
                    )

                matrix[:, iface, :] = mp.nodal_mass_matrix_for_face(
                    face, face_quadrature,
                    face_basis.functions, vol_basis.functions,
                    vol_grp.unit_nodes,
                    face_grp.unit_nodes,
                )
            else:
                # Otherwise, we use a routine that is purely quadrature-based
                # (no need for explicit face basis functions)
                matrix[:, iface, :] = mp.nodal_quad_mass_matrix_for_face(
                    face,
                    face_quadrature,
                    vol_basis.functions,
                    vol_grp.unit_nodes,
                )

        return actx.freeze(
                tag_axes(actx, {
                    0: DiscretizationDOFAxisTag(),
                    2: DiscretizationDOFAxisTag()
                    },
                    actx.from_numpy(matrix)))

    return get_ref_face_mass_mat(face_element_group, vol_element_group)


def _apply_face_mass_operator(dcoll: DiscretizationCollection, dd_in, vec):
    if not isinstance(vec, DOFArray):
        return map_array_container(
            partial(_apply_face_mass_operator, dcoll, dd_in), vec
        )

    from grudge.geometry import area_element

    dd_out = DOFDesc(
        VolumeDomainTag(dd_in.domain_tag.volume_tag),
        DISCR_TAG_BASE)

    volm_discr = dcoll.discr_from_dd(dd_out)
    face_discr = dcoll.discr_from_dd(dd_in)
    dtype = vec.entry_dtype
    actx = vec.array_context

    assert len(face_discr.groups) == len(volm_discr.groups)
    surf_area_elements = area_element(actx, dcoll, dd=dd_in,
            _use_geoderiv_connection=actx.supports_nonscalar_broadcasting)

    return DOFArray(
        actx,
        data=tuple(
            actx.einsum("ifj,fej,fej->ei",
                        reference_face_mass_matrix(
                            actx,
                            face_element_group=afgrp,
                            vol_element_group=vgrp,
                            dtype=dtype),
                        actx.tag_axis(1, DiscretizationElementAxisTag(),
                            surf_ae_i.reshape(
                                vgrp.mesh_el_group.nfaces,
                                vgrp.nelements,
                                surf_ae_i.shape[-1])),
                        actx.tag_axis(0, DiscretizationFaceAxisTag(),
                            vec_i.reshape(
                                vgrp.mesh_el_group.nfaces,
                                vgrp.nelements,
                                afgrp.nunit_dofs)),
                        arg_names=("ref_face_mass_mat", "jac_surf", "vec"),
                        tagged=(FirstAxisIsElementsTag(),))

            for vgrp, afgrp, vec_i, surf_ae_i in zip(volm_discr.groups,
                                                     face_discr.groups,
                                                     vec,
                                                     surf_area_elements)))


def face_mass(dcoll: DiscretizationCollection, *args) -> ArrayOrContainer:
    r"""Return the action of the DG face mass matrix on a vector (or vectors)
    of :class:`~meshmode.dof_array.DOFArray`\ s, *vec*. In the case of
    *vec* being an arbitrary :class:`~arraycontext.ArrayContainer`,
    the face mass operator is applied component-wise.

    May be called with ``(vec)`` or ``(dd_in, vec)``.

    Specifically, this function applies the face mass matrix elementwise on a
    vector of coefficients :math:`\mathbf{f}` as the sum of contributions for
    each face :math:`f \subset \partial E`:

    .. math::

        \sum_{f=1}^{N_{\text{faces}} } \mathbf{M}_{f, E}\mathbf{f}|_f,

    where

    .. math::

        \left(\mathbf{M}_{f, E}\right)_{ij} =
            \int_{f \subset \partial E} \phi_i(s)\psi_j(s)\,\mathrm{d}s,

    where :math:`\phi_i` are (volume) polynomial basis functions on :math:`E`
    evaluated on the face :math:`f`, and :math:`\psi_j` are basis functions for
    a polynomial space defined on :math:`f`.

    :arg dd: a :class:`~grudge.dof_desc.DOFDesc`, or a value convertible to one.
        Defaults to the base ``"all_faces"`` discretization if not provided.
    :arg vec: a :class:`~meshmode.dof_array.DOFArray` or an
        :class:`~arraycontext.ArrayContainer` of them.
    :returns: a :class:`~meshmode.dof_array.DOFArray` or an
        :class:`~arraycontext.ArrayContainer` like *vec*.
    """

    if len(args) == 1:
        vec, = args
        dd_in = DD_VOLUME_ALL.trace(FACE_RESTR_ALL)
    elif len(args) == 2:
        dd_in, vec = args
    else:
        raise TypeError("invalid number of arguments")

    return _apply_face_mass_operator(dcoll, dd_in, vec)

# }}}


# {{{ single axis contraction

def single_axis_contraction(actx, dim, axis, operator, data,
                            tagged=None, arg_names=None):
    """
    Used to contract a 1D operator and a tensor over a specified axis.
    """

    data = tag_axes(
        actx,
        {
            i: (DiscretizationElementAxisTag() if i == 0 else
                TensorProductDOFAxisTag(i-1))
            for i in range(dim+1)
        },
        data)

    operator = tag_axes(
        actx,
        { i: TensorProductOperatorAxisTag() for i in range(2) },
        operator)

    # NOTE: shift j into the correct position and contract over j in the einsum
    # example of a 3D gradient using
    #   x-axis (axis = 0): ij,ejop->eiop
    #   y-axis (axis = 1): ij,eajo->eaio
    #   z-axis (axis = 2): ij,eabj->eabi
    operator_spec = "ij"
    data_spec = f"e{"abcdfghklmn"[:axis]}j{"opqrstuvwxy"[:dim-axis-1]}"
    out_spec = f"e{"abcdfghklmn"[:axis]}i{"opqrstuvwxy"[:dim-axis-1]}"
    spec = operator_spec + "," + data_spec + "->" + out_spec

    return tag_axes(
        actx,
        {
            i: (DiscretizationElementAxisTag() if i == 0 else
                TensorProductDOFAxisTag(i-1))
            for i in range(dim+1)
        },
        actx.einsum(spec, operator, data, arg_names=arg_names, tagged=tagged))

# }}}

# vim: foldmethod=marker
