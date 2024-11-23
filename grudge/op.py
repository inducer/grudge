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


from collections.abc import Callable, Iterable
from functools import partial

import numpy as np

import modepy as mp
from arraycontext import ArrayContext, ArrayOrContainer, map_array_container, tag_axes
from meshmode.discretization import (
    Discretization,
    InterpolatoryElementGroupBase,
    NodalElementGroupBase,
)
from meshmode.discretization.poly_element import (
    SimplexElementGroupBase,
    TensorProductElementGroupBase,
)
from meshmode.dof_array import DOFArray
from meshmode.transform_metadata import (
    DiscretizationAmbientDimAxisTag,
    DiscretizationDOFAxisTag,
    DiscretizationElementAxisTag,
    DiscretizationFaceAxisTag,
    FirstAxisIsElementsTag,
)
from modepy import inverse_mass_matrix
from modepy.tools import (
    reshape_array_for_tensor_product_space as fold,
    unreshape_array_for_tensor_product_space as unfold,
)
from pytools import keyed_memoize_in
from pytools.obj_array import make_obj_array

import grudge.dof_desc as dof_desc
from grudge.bilinear_forms import (
    _NonTensorProductBilinearForm,
    _TensorProductBilinearForm,
    apply_bilinear_form,
    single_axis_contraction,
)
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
from grudge.geometry import area_element, inverse_surface_metric_derivative_mat
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
    TensorProductMassOperatorInverseTag,
    TensorProductMassOperatorTag,
    TensorProductOperatorAxisTag,
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


# {{{ Strong derivative operators

def _single_axis_derivative_kernel(
        actx: ArrayContext,
        out_discr: Discretization,
        in_discr: Discretization,
        diff_mat: Callable,
        inv_jac_mat: Iterable,
        xyz_axis: int,
        vec: DOFArray) -> DOFArray:

    def compute_simplicial_derivative(actx, out_grp, in_grp, vec, ijm):
        return actx.einsum(
            "rei,rij,ej->ei",
            ijm[xyz_axis],
            _reference_derivative_matrices(
                actx,
                out_element_group=out_grp,
                in_element_group=in_grp),
            vec,
            arg_names=("inv_jac_t", "ref_stiffT_mat", "vec", ),
            tagged=(FirstAxisIsElementsTag(),))

    def compute_tensor_product_derivative(actx, out_grp, in_grp, vec, ijm):
        vec = vec * ijm[xyz_axis]

        partial = 0.
        for rst_axis in range(in_grp.dim):
            partial += single_axis_contraction(
                actx, in_grp.dim, rst_axis, diff_mat(actx, out_grp, in_grp),
                vec[rst_axis])

        return partial

    group_data = []
    for out_grp, in_grp, vec_i, ijm_i in zip(
        out_discr.groups, in_discr.groups, vec, inv_jac_mat, strict=False):

        if isinstance(in_grp, TensorProductElementGroupBase):
            assert isinstance(out_grp, TensorProductElementGroupBase)
            group_data.append(compute_tensor_product_derivative(
                actx, out_grp, in_grp, vec_i, ijm_i))

        elif isinstance(in_grp, SimplexElementGroupBase):
            assert isinstance(out_grp, SimplexElementGroupBase)
            group_data.append(compute_simplicial_derivative(
                actx, out_grp, in_grp, vec_i, ijm_i))

        else:
            raise TypeError(
                "`in_grp` and `out_grp` must both be either "
                "`SimplexElementGroupBase` or `TensorProductElementGroupBase`. "
                f"Found `in_grp` = {in_grp}, `out_grp` = {out_grp}")

    return DOFArray(actx, data=tuple(group_data))


def _gradient_kernel(actx, out_discr, in_discr, get_diff_mat, inv_jac_mat, vec):
    # See _single_axis_derivative_kernel for comments on the usage scenarios
    # (both strong and weak derivative) and their differences.

    def compute_simplicial_gradient(actx, out_grp, in_grp, vec, ijm):
        return actx.einsum(
            "xrei,rij,ej->xei",
            ijm,
            get_diff_mat(
                actx,
                out_element_group=out_grp,
                in_element_group=in_grp),
            vec,
            arg_names=("inv_jac_t", "ref_stiffT_mat", "vec"),
            tagged=(FirstAxisIsElementsTag(),))

    def compute_tensor_product_gradient(actx, out_grp, in_grp, vec, ijm):
        vec = actx.einsum(
            "xrej,ej->xrej", ijm, vec,
            tagged=(FirstAxisIsElementsTag(),),
            arg_names=("inv_jac_t", "dofs"))

        # expose tensor product structure
        if in_grp.dim != 1:
            vec = fold(in_grp.space, vec)

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
        out_discr.groups, in_discr.groups, vec, inv_jac_mat, strict=False):

        if isinstance(in_grp, SimplexElementGroupBase):
            assert isinstance(out_grp, SimplexElementGroupBase)
            per_group_grads.append(
                compute_simplicial_gradient(
                    actx, out_grp, in_grp, vec_i, ijm_i))
        elif isinstance(in_grp, TensorProductElementGroupBase):
            assert isinstance(out_grp, TensorProductElementGroupBase)
            per_group_grads.append(
                compute_tensor_product_gradient(
                    actx, out_grp, in_grp, vec_i, ijm_i))
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


def _divergence_kernel(actx, out_discr, in_discr, get_diff_mat, inv_jac_mat,
                       vec):
    # See _single_axis_derivative_kernel for comments on the usage scenarios
    # (both strong and weak derivative) and their differences.

    def compute_simplicial_divergence(actx, out_grp, in_grp, vec, ijm):
        return actx.einsum(
            "xrei,rij,xej->ei",
            ijm,
            get_diff_mat(
                actx,
                out_element_group=out_grp,
                in_element_group=in_grp
            ),
            vec,
            arg_names=("inv_jac_t", "ref_stiffT_mat", "vec"),
            tagged=(FirstAxisIsElementsTag(),))

    def compute_tensor_product_divergence(actx, out_grp, in_grp, vec, ijm):
        if in_grp.dim != 1:
            vec = fold(in_grp.space, vec)

        div = 0.0

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

                div += ref_deriv * ijm[func_axis, rst_axis]

        return tag_axes(actx, {0: DiscretizationElementAxisTag()}, div)

    per_group_divs = []
    for out_grp, in_grp, vec_i, ijm_i in zip(
        out_discr.groups, in_discr.groups, vec, inv_jac_mat, strict=False):
        if isinstance(in_grp, SimplexElementGroupBase):
            assert isinstance(out_grp, SimplexElementGroupBase)
            per_group_divs.append(compute_simplicial_divergence(
                actx, out_grp, in_grp, vec_i, ijm_i))

        elif isinstance(in_grp, TensorProductElementGroupBase):
            assert isinstance(out_grp, TensorProductElementGroupBase)
            per_group_divs.append(compute_tensor_product_divergence(
                actx, out_grp, in_grp, vec_i, ijm_i))

        else:
            raise TypeError(
                "`in_grp` and `out_grp` must both be either "
                "`SimplexElementGroupBase` or `TensorProductElementGroupBase`. "
                f"Found `in_grp` = {in_grp}, `out_grp` = {out_grp}")

    return DOFArray(actx, data=tuple(per_group_divs))


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
        if isinstance(in_grp, TensorProductElementGroupBase):
            assert isinstance(out_grp, TensorProductElementGroupBase)

            # FIXME: we make an assumption that the same basis is used for all
            # coordinate directions; we need to enforce this somehow
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
                    {i: TensorProductOperatorAxisTag() for i in range(2)},
                    actx.from_numpy(diff_mat)))

        return actx.freeze(
                actx.tag_axis(
                    1, DiscretizationDOFAxisTag(),
                    actx.from_numpy(
                        np.asarray(
                            mp.diff_matrices(
                                in_grp.basis_obj(),
                                out_grp.unit_nodes,
                                from_nodes=in_grp.unit_nodes)))))

    return get_ref_derivative_mats(out_element_group, in_element_group)


def _strong_scalar_grad(dcoll, dd_in, vec):
    assert isinstance(dd_in.domain_tag, VolumeDomainTag)

    from grudge.geometry import inverse_surface_metric_derivative_mat

    discr = dcoll.discr_from_dd(dd_in)
    actx = vec.array_context

    inverse_jac_mat = inverse_surface_metric_derivative_mat(actx, dcoll, dd=dd_in,
            _use_geoderiv_connection=actx.supports_nonscalar_broadcasting)
    return _gradient_kernel(actx, discr, discr,
            _reference_derivative_matrices, inverse_jac_mat, vec)


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
            _reference_derivative_matrices, inverse_jac_mat, vec)


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
        _reference_derivative_matrices, inverse_jac_mat, xyz_axis, vec)


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

def _weak_scalar_grad(dcoll, dd_in, vec, *args,
                      use_tensor_product_fast_eval=True):

    # {{{ setup and grab discretizations

    actx = vec.array_context

    dd_in = as_dofdesc(dd_in)
    input_discr = dcoll.discr_from_dd(dd_in)
    output_discr = dcoll.discr_from_dd(dd_in.with_discr_tag(DISCR_TAG_BASE))

    # }}}

    # {{{ get metrics, scaling terms

    metrics = inverse_surface_metric_derivative_mat(actx, dcoll, dd=dd_in,
        times_area_element=False,
        _use_geoderiv_connection=actx.supports_nonscalar_broadcasting)

    area_elements = area_element(actx, dcoll, dd=dd_in,
        _use_geoderiv_connection=actx.supports_nonscalar_broadcasting)

    # }}}

    # {{{ build gradient at group granularity

    group_data = []
    for in_group, out_group, vec_i, metric_i, area_elt_i in zip(
        input_discr.groups, output_discr.groups, vec, metrics, area_elements,
        strict=False):

        if isinstance(in_group, TensorProductElementGroupBase):
            assert isinstance(out_group, TensorProductElementGroupBase)

            fast_eval = (use_tensor_product_fast_eval and
                         (in_group == out_group))
            if use_tensor_product_fast_eval and not fast_eval:
                from warnings import warn
                warn("Input and output groups must match to use " +
                     "tensor-product fast operator evaluation. " +
                     "Defaulting to typical operator evaluation.",
                     stacklevel=1)

            if fast_eval:
                bilinear_form = _TensorProductBilinearForm(
                    actx, in_group, out_group, metric_i, area_elt_i,
                    compute_stiffness=True)
            else:
                bilinear_form = _NonTensorProductBilinearForm(
                    actx, in_group, out_group, metric_i, area_elt_i,
                    compute_stiffness=True)
        else:
            bilinear_form = _NonTensorProductBilinearForm(
                actx, in_group, out_group, metric_i, area_elt_i,
                compute_stiffness=True)

        group_data.append(bilinear_form.gradient_operator(vec_i))

    # }}}

    return make_obj_array([
        DOFArray(actx, data=tuple(
            group_grad_i[xyz_axis]
            for group_grad_i in group_data
        ))
        for xyz_axis in range(output_discr.ambient_dim)
    ])


def _weak_scalar_div(dcoll, dd_in, vecs, *args,
                     use_tensor_product_fast_eval=True):
    from arraycontext import get_container_context_recursively, serialize_container

    from grudge.geometry import inverse_surface_metric_derivative_mat

    assert isinstance(vecs, np.ndarray)
    assert vecs.shape == (dcoll.ambient_dim,)

    # {{{ setup and grab discretizations

    actx = get_container_context_recursively(vecs)
    assert actx is not None
    vec = actx.np.stack([v for k, v in serialize_container(vecs)])

    dd_in = as_dofdesc(dd_in)
    input_discr = dcoll.discr_from_dd(dd_in)
    output_discr = dcoll.discr_from_dd(dd_in.with_discr_tag(DISCR_TAG_BASE))

    # }}}

    # {{{ get metrics, scaling terms

    metrics = inverse_surface_metric_derivative_mat(actx, dcoll, dd=dd_in,
        times_area_element=False,
        _use_geoderiv_connection=actx.supports_nonscalar_broadcasting)

    area_elements = area_element(actx, dcoll, dd=dd_in,
        _use_geoderiv_connection=actx.supports_nonscalar_broadcasting)

    # }}}

    # {{{ compute divergence at group granularity

    group_divs = []
    for in_group, out_group, vec_i, metric_i, area_elt_i in zip(
        input_discr.groups, output_discr.groups, vec, metrics, area_elements,
        strict=False):

        if isinstance(in_group, TensorProductElementGroupBase):
            assert isinstance(out_group, TensorProductElementGroupBase)

            fast_eval = (use_tensor_product_fast_eval and
                         (in_group == out_group))

            if use_tensor_product_fast_eval and not fast_eval:
                from warnings import warn
                warn("Input and output groups must match to use " +
                     "tensor-product fast operator evaluation. " +
                     "Defaulting to typical operator evaluation.",
                     stacklevel=1)

            if fast_eval:
                bilinear_form = _TensorProductBilinearForm(
                    actx, in_group, out_group, metric_i, area_elt_i,
                    compute_stiffness=True)
            else:
                bilinear_form = _NonTensorProductBilinearForm(
                    actx, in_group, out_group, metric_i, area_elt_i,
                    compute_stiffness=True)
        else:
            bilinear_form = _NonTensorProductBilinearForm(
                actx, in_group, out_group, metric_i, area_elt_i,
                compute_stiffness=True)

        group_divs.append(bilinear_form.divergence_operator(vec_i))

    # }}}

    return DOFArray(actx, data=tuple(group_divs))


def weak_local_d_dx(dcoll: DiscretizationCollection, *args,
                    use_tensor_product_fast_eval=True) -> ArrayOrContainer:
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

    dd_in = as_dofdesc(dd_in)

    return apply_bilinear_form(dcoll, vec, dd_in,
        use_tensor_product_fast_eval=use_tensor_product_fast_eval,
        test_derivative=xyz_axis, in_shape=(), out_shape=(dcoll.ambient_dim,))


def weak_local_grad(
        dcoll: DiscretizationCollection, *args, nested=False,
        use_tensor_product_fast_eval=True) -> ArrayOrContainer:
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

    return apply_bilinear_form(dcoll, vecs, dd_in,
        use_tensor_product_fast_eval=use_tensor_product_fast_eval,
        dispatcher=_weak_scalar_grad, return_nested=nested,
        in_shape=(), out_shape=(dcoll.ambient_dim,))


def weak_local_div(dcoll: DiscretizationCollection, *args,
                   use_tensor_product_fast_eval=True) -> ArrayOrContainer:
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

    return apply_bilinear_form(dcoll, vecs, dd_in,
        use_tensor_product_fast_eval=use_tensor_product_fast_eval,
        dispatcher=_weak_scalar_div, in_shape=(dcoll.ambient_dim,),
        out_shape=())

# }}}


# {{{ Mass operator

def mass(dcoll: DiscretizationCollection, *args,
         use_tensor_product_fast_eval=True) -> ArrayOrContainer:
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

    return apply_bilinear_form(dcoll, vec, dd_in,
        use_tensor_product_fast_eval=use_tensor_product_fast_eval)

# }}}


# {{{ Mass inverse operator

def reference_inverse_mass_matrix(actx: ArrayContext, element_group,
                                  use_tensor_product_fast_eval=True):
    @keyed_memoize_in(
        actx, reference_inverse_mass_matrix,
        lambda grp, use_tensor_product_fast_eval:
            (grp.discretization_key(), use_tensor_product_fast_eval))
    def get_ref_inv_mass_mat(grp, use_tensor_product_fast_eval):

        if isinstance(grp, TensorProductElementGroupBase) and \
            use_tensor_product_fast_eval:

            basis_1d = grp.basis_obj().bases[0]
            nodes_1d = grp.unit_nodes_1d

            return actx.freeze(
                actx.tag(
                    TensorProductMassOperatorInverseTag(),
                    tag_axes(
                    actx,
                    {i: TensorProductOperatorAxisTag() for i in range(2)},
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

    return get_ref_inv_mass_mat(element_group, use_tensor_product_fast_eval)


def _simplicial_apply_inverse_mass_operator(actx, grp, vec):
    return actx.einsum(
        "ij,ej->ei",
        reference_inverse_mass_matrix(actx, element_group=grp,
                                      use_tensor_product_fast_eval=False),
        vec,
        tagged=(FirstAxisIsElementsTag(),))


def _tensor_product_apply_inverse_mass_operator(actx, grp, vec):
    if grp.dim != 1:
        vec = fold(grp.space, vec)

    for rst_axis in range(grp.dim):
        vec = single_axis_contraction(
            actx, grp.dim, rst_axis,
            reference_inverse_mass_matrix(actx, element_group=grp,
                                          use_tensor_product_fast_eval=True),
            vec,
            tagged=(FirstAxisIsElementsTag(),
                    OutputIsTensorProductDOFArrayOrdered(),),
            arg_names=("base_mass_inv", "dofs"))

    if grp.dim != 1:
        vec = unfold(grp.space, vec)

    return vec


def _dispatch_inverse_mass_applier(
        dcoll: DiscretizationCollection, dd_out, dd_in, vec,
        uses_quadrature=False,
        use_tensor_product_fast_eval=True):
    if not isinstance(vec, DOFArray):
        return map_array_container(
            partial(_dispatch_inverse_mass_applier, dcoll, dd_out, dd_in,
                    uses_quadrature), vec
        )

    if dd_out != dd_in and not uses_quadrature:
        raise ValueError(
            "Cannot compute inverse of a mass matrix mapping "
            "between different element groups unless using overintegration; "
            "inverse is not guaranteed to be well-defined")

    actx = vec.array_context
    discr_base = dcoll.discr_from_dd(dd_out)
    discr_quad = dcoll.discr_from_dd(dd_in)

    inv_area_elements = 1./area_element(actx, dcoll, dd=dd_in,
        _use_geoderiv_connection=actx.supports_nonscalar_broadcasting)

    if discr_base == discr_quad:
        group_data = []
        for in_grp, out_grp, vec_i, inv_ae_i in zip(discr_quad.groups,
                discr_base.groups, vec, inv_area_elements, strict=False):

            if isinstance(in_grp, TensorProductElementGroupBase) and \
                    use_tensor_product_fast_eval:
                assert isinstance(out_grp, TensorProductElementGroupBase)
                group_data.append(_tensor_product_apply_inverse_mass_operator(
                    actx, out_grp, vec_i) * inv_ae_i)
            else:
                group_data.append(_simplicial_apply_inverse_mass_operator(
                    actx, out_grp, vec_i) * inv_ae_i)

        return DOFArray(actx, data=tuple(group_data))

    # see WADG: https://arxiv.org/pdf/1608.03836

    # apply reference inverse mass
    group_data = []
    for in_grp, out_grp, vec_i in zip(discr_quad.groups, discr_base.groups, vec,
                          strict=False):
        if isinstance(out_grp, TensorProductElementGroupBase):
            assert isinstance(out_grp, TensorProductElementGroupBase)

            fast_eval = (use_tensor_product_fast_eval and (in_grp == out_grp))
            if use_tensor_product_fast_eval and not fast_eval:
                from warnings import warn
                warn("Input and output groups must match to use " +
                     "tensor-product fast operator evaluation. " +
                     "Defaulting to typical operator evaluation.",
                     stacklevel=1)

            if fast_eval:
                group_data.append(_tensor_product_apply_inverse_mass_operator(
                    actx, out_grp, vec_i))
            else:
                group_data.append(_simplicial_apply_inverse_mass_operator(
                    actx, out_grp, vec_i))
        else:
            group_data.append(_simplicial_apply_inverse_mass_operator(
                actx, out_grp, vec_i))

    # project to quadrature discretization
    vec = inv_area_elements * project(
        dcoll, dd_out, dd_in, DOFArray(actx, data=tuple(group_data)))

    # finish applying WADG
    group_data = []
    for in_grp, out_grp, vec_i in zip(
            discr_quad.groups, discr_base.groups, vec, strict=False):
        if isinstance(in_grp, TensorProductElementGroupBase):
            assert isinstance(out_grp, TensorProductElementGroupBase)

            fast_eval = (use_tensor_product_fast_eval and (in_grp == out_grp))
            if use_tensor_product_fast_eval and not fast_eval:
                from warnings import warn
                warn("Input and output groups must match to use " +
                     "tensor-product fast operator evaluation. " +
                     "Defaulting to typical operator evaluation.",
                     stacklevel=1)

            if fast_eval:
                bilinear_form = _TensorProductBilinearForm(
                    actx, in_grp, out_grp, compute_stiffness=False)
                vec_i = bilinear_form.mass_operator(vec_i, exclude_metric=True)
                group_data.append(_tensor_product_apply_inverse_mass_operator(
                    actx, out_grp, vec_i))
            else:
                bilinear_form = _NonTensorProductBilinearForm(
                    actx, in_grp, out_grp, compute_stiffness=False)
                vec_i = bilinear_form.mass_operator(vec_i, exclude_metric=True)
                group_data.append(_simplicial_apply_inverse_mass_operator(
                    actx, out_grp, vec_i))

        else:
            bilinear_form = _NonTensorProductBilinearForm(
                actx, in_grp, out_grp, compute_stiffness=False)
            vec_i = bilinear_form.mass_operator(vec_i, exclude_metric=True)
            group_data.append(_simplicial_apply_inverse_mass_operator(
                actx, out_grp, vec_i))

    return DOFArray(actx, data=tuple(group_data))


def inverse_mass(dcoll: DiscretizationCollection, *args,
                 use_tensor_product_fast_eval=True) -> ArrayOrContainer:
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

    return _dispatch_inverse_mass_applier(
        dcoll, dd_out, dd_in, vec,
        uses_quadrature=dd.uses_quadrature(),
        use_tensor_product_fast_eval=use_tensor_product_fast_eval)

# }}}


# {{{ Face mass operator

def reference_face_mass_matrix(
        actx: ArrayContext, face_element_group, vol_element_group, dtype):
    @keyed_memoize_in(
        actx, reference_face_mass_matrix,
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
        from meshmode.discretization.poly_element import QuadratureSimplexElementGroup

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
                                                     surf_area_elements,
                                                     strict=True)))


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


# {{{ deprecated functionality (gone in 2025)

def reference_mass_matrix(actx: ArrayContext, out_element_group, in_element_group):
    @keyed_memoize_in(
        actx, reference_mass_matrix,
        lambda out_grp, in_grp: (out_grp.discretization_key(),
                                 in_grp.discretization_key()))
    def get_ref_mass_mat(out_grp, in_grp):
        import modepy as mp

        if out_grp == in_grp:
            if isinstance(out_grp, TensorProductElementGroupBase):
                basis_1d = out_grp.basis_obj().bases[0]
                nodes_1d = out_grp.unit_nodes[0][:out_grp.order+1].reshape(
                    1, out_grp.order+1)

                return actx.tag(
                    TensorProductMassOperatorTag(),
                    tag_axes(
                        actx,
                        {i: TensorProductOperatorAxisTag()
                            for i in range(2)},
                        actx.from_numpy(mp.mass_matrix(basis_1d, nodes_1d))))

            else:
                return actx.freeze(
                    actx.from_numpy(
                        mp.mass_matrix(out_grp.basis_obj(),
                                       out_grp.unit_nodes)))

        if isinstance(in_grp, TensorProductElementGroupBase):

            basis_1d = out_grp.basis_obj().bases[0]
            nodes_1d_out = out_grp.unit_nodes[0][:out_grp.order+1].reshape(
                1, out_grp.order+1)

            mass_matrix = mp.nodal_quad_mass_matrix(
                in_grp.quadrature_rule().quadratures[0], basis_1d.functions,
                nodes_1d_out)

            return actx.freeze(
                actx.tag(
                    TensorProductMassOperatorTag(),
                    tag_axes(
                        actx,
                        {i: TensorProductOperatorAxisTag()
                            for i in range(2)},
                        actx.from_numpy(mass_matrix))))

        elif isinstance(out_grp, SimplexElementGroupBase):

            basis = out_grp.basis_obj()
            vand = mp.vandermonde(basis.functions, out_grp.unit_nodes)
            o_vand = mp.vandermonde(basis.functions, in_grp.unit_nodes)
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
        in_discr.groups, out_discr.groups, area_elements, vec, strict=False):
        if isinstance(in_grp, TensorProductElementGroupBase):
            assert isinstance(out_grp, TensorProductElementGroupBase)
            group_data.append(tensor_product_apply_mass(
                in_grp, out_grp, ae_i, vec_i))

        elif isinstance(in_grp, SimplexElementGroupBase):
            assert isinstance(out_grp, SimplexElementGroupBase)
            group_data.append(simplicial_apply_mass(
                in_grp, out_grp, ae_i, vec_i))

        else:
            raise TypeError(
                "`in_grp` and `out_grp` must both be either "
                "`SimplexElementGroupBase` or `TensorProductElementGroupBase`. "
                f"Found `in_grp` = {in_grp}, `out_grp` = {out_grp}")

    return DOFArray(actx, data=tuple(group_data))

# }}}

# vim: foldmethod=marker
