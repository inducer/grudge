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

from arraycontext import (
    ArrayContext,
    ArrayOrContainer,
    map_array_container,
    tag_axes,
)
from meshmode.discretization import (
    Discretization,
    ElementGroupBase,
    InterpolatoryElementGroupBase,
    NodalElementGroupBase,
)
from meshmode.discretization.poly_element import (
    TensorProductElementGroupBase,
)
from meshmode.dof_array import DOFArray
from meshmode.transform_metadata import (
    DiscretizationDOFAxisTag,
    DiscretizationElementAxisTag,
    DiscretizationFaceAxisTag,
    FirstAxisIsElementsTag,
)
from modepy.tools import (
    reshape_array_for_tensor_product_space as fold,
    unreshape_array_for_tensor_product_space as unfold,
)
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
from grudge.geometry import area_element, inverse_surface_metric_derivative_mat
from grudge.interpolation import interp
from grudge.matrices import (
    reference_derivative_matrices,
    reference_face_mass_matrix,
    reference_inverse_mass_matrix,
    reference_mass_matrix,
    reference_stiffness_transpose_matrices,
)
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
from grudge.tools import rec_map_subarrays
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
    TensorProductDOFAxisTag,
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

# TODO:
# 1. implement proper axis and array tags


# {{{ tensor product operator application helper

def _single_axis_contraction(
        actx: ArrayContext,
        dim: int,
        axis: int,
        operator: ArrayOrContainer,
        data: DOFArray,
        tagged=None,
        arg_names=None,
    ) -> ArrayOrContainer:
    """
    Generic routine to apply a 1D operator to a particular axis of *data*.

    The einsum specification is constructed based on the dimension of the
    problem and can support up to 1 reduction axis and 22 non-reduction
    (DOF) axes. The element axis is not counted since it is reserved.
    """
    data = tag_axes(
        actx,
        {
            i: (DiscretizationElementAxisTag() if i == 0 else
                TensorProductDOFAxisTag(i-1))
            for i in range(dim+1)
        },
        data)

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
        actx.einsum(spec, operator, data, arg_names=arg_names,
                    tagged=tagged))


# }}}


# {{{ Derivative operator kernels

def _single_axis_derivative_kernel(
        actx: ArrayContext,
        output_discr: Discretization,
        input_discr: Discretization,
        inv_jac_mat: Iterable,
        xyz_axis: int,
        vec: DOFArray,
        compute_single_axis_derivative: Callable,
        use_tensor_product_fast_eval: bool = True
    ) -> DOFArray:
    """
    Computes the *xyz_axis*th derivative of *vec* using
    *compute_single_axis_derivative*.

    Strong and weak dispatch routines pass the corresponding
    *compute_single_axis_derivative* routine so that the correct form is
    computed.
    """

    group_data = []
    for output_group, input_group, vec_i, ijm_i in zip(
            output_discr.groups, input_discr.groups, vec, inv_jac_mat,
            strict=False):

        group_data.append(compute_single_axis_derivative(
            actx, xyz_axis, output_group, input_group, vec_i, ijm_i,
            use_tensor_product_fast_eval=use_tensor_product_fast_eval))

    return DOFArray(actx, data=tuple(group_data))


def _gradient_kernel(
        actx: ArrayContext,
        output_discr: Discretization,
        input_discr: Discretization,
        inv_jac_mat: Iterable,
        vec: DOFArray,
        compute_single_axis_derivative: Callable,
        use_tensor_product_fast_eval: bool = True
    ) -> DOFArray:
    """
    Computes gradient of *vec* using *compute_single_axis_derivative* for each
    entry in the gradient.

    Strong and weak dispatch routines provide the proper
    *compute_single_axis_derivative* routine.
    """

    per_group_grads = []
    for output_group, input_group, vec_i, ijm_i in zip(
            output_discr.groups, input_discr.groups, vec, inv_jac_mat,
            strict=False):

        per_group_grads.append([compute_single_axis_derivative(
            actx, xyz_axis, output_group, input_group, vec_i, ijm_i,
            use_tensor_product_fast_eval=use_tensor_product_fast_eval)
            for xyz_axis in range(input_group.dim)
        ])

    return make_obj_array([
        DOFArray(actx, data=tuple([  # noqa: C409
            pgg_i[xyz_axis] for pgg_i in per_group_grads
        ]))
        for xyz_axis in range(output_discr.ambient_dim)
    ])


def _divergence_kernel(
        actx: ArrayContext,
        output_discr: Discretization,
        input_discr: Discretization,
        inv_jac_mat: Iterable,
        vec: DOFArray,
        compute_single_axis_derivative: Callable,
        use_tensor_product_fast_eval: bool = True
    ) -> DOFArray:
    """
    Computes divergence of *vec* by summing over each spatial derivative
    computed by *compute_single_axis_derivative*.

    Strong and weak dispatch routines provide the proper
    *compute_single_axis_derivative* routine.
    """

    per_group_divs = []
    for output_group, input_group, vec_i, ijm_i in zip(
            output_discr.groups, input_discr.groups, vec, inv_jac_mat,
            strict=False):

        per_group_divs.append(sum(compute_single_axis_derivative(
            actx, xyz_axis, output_group, input_group, vec_ij, ijm_i,
            use_tensor_product_fast_eval=use_tensor_product_fast_eval)
            for xyz_axis, vec_ij in enumerate(vec_i)))

    return DOFArray(actx, data=tuple(per_group_divs))

# }}}


# {{{ Strong derivative operators

def _strong_tensor_product_single_axis_derivative(
        actx: ArrayContext,
        xyz_axis: int,
        output_group: NodalElementGroupBase,
        input_group: InterpolatoryElementGroupBase,
        vec: ArrayOrContainer,
        metrics: ArrayOrContainer
    ) -> ArrayOrContainer:

    return sum(metrics[xyz_axis, rst_axis] * unfold(
            output_group.space,
            _single_axis_contraction(
                actx,
                input_group.dim,
                rst_axis,
                reference_derivative_matrices(
                    actx, output_group, input_group,
                    use_tensor_product_fast_eval=True)[0],
                fold(input_group.space, vec),
                arg_names=("diff_op_1d", "vec")
            )
        )
        for rst_axis in range(input_group.dim)
    )


def _strong_simplicial_single_axis_derivative(
        actx: ArrayContext,
        xyz_axis: int,
        output_group: NodalElementGroupBase,
        input_group: InterpolatoryElementGroupBase,
        vec: ArrayOrContainer,
        metrics: ArrayOrContainer
    ) -> ArrayOrContainer:

    return sum(
        metrics[xyz_axis, rst_axis] * actx.einsum(
            "ij,ej->ei",
            reference_derivative_matrices(
                actx, output_group, input_group,
                use_tensor_product_fast_eval=False)[rst_axis],
            vec,
            arg_names=("diff_op", "vec"))
        for rst_axis in range(input_group.dim)
    )


def _strong_scalar_d_dx(
        actx: ArrayContext,
        xyz_axis: int,
        output_group: NodalElementGroupBase,
        input_group: InterpolatoryElementGroupBase,
        vec: ArrayOrContainer,
        metrics: ArrayOrContainer,
        use_tensor_product_fast_eval: bool = True
    ) -> ArrayOrContainer:

    if isinstance(input_group, TensorProductElementGroupBase) and \
            use_tensor_product_fast_eval:
        return _strong_tensor_product_single_axis_derivative(
            actx, xyz_axis, output_group, input_group, vec, metrics
        )

    return _strong_simplicial_single_axis_derivative(
        actx, xyz_axis, output_group, input_group, vec, metrics
    )


def _strong_scalar_grad(
        dcoll: DiscretizationCollection,
        dd_in: DOFDesc,
        vec: ArrayOrContainer,
        *args,
        use_tensor_product_fast_eval: bool = True
    ) -> ArrayOrContainer:

    from grudge.geometry import inverse_surface_metric_derivative_mat

    assert isinstance(dd_in.domain_tag, VolumeDomainTag)

    discr = dcoll.discr_from_dd(dd_in)
    actx = vec.array_context
    metrics = inverse_surface_metric_derivative_mat(actx, dcoll,
        dd=dd_in, _use_geoderiv_connection=actx.supports_nonscalar_broadcasting)

    return _gradient_kernel(
        actx, discr, discr, metrics, vec, _strong_scalar_d_dx,
        use_tensor_product_fast_eval=use_tensor_product_fast_eval)


def _strong_scalar_div(
        dcoll: DiscretizationCollection,
        dd: DOFDesc,
        vecs: ArrayOrContainer,
        *args,
        use_tensor_product_fast_eval: bool = True) -> ArrayOrContainer:
    from arraycontext import (
        get_container_context_recursively,
        serialize_container,
    )

    from grudge.geometry import inverse_surface_metric_derivative_mat

    assert isinstance(vecs, np.ndarray)
    assert vecs.shape == (dcoll.ambient_dim,)

    actx = get_container_context_recursively(vecs)
    assert actx is not None

    discr = dcoll.discr_from_dd(dd)
    vec = actx.np.stack([v for _, v in serialize_container(vecs)])
    metrics = inverse_surface_metric_derivative_mat(actx, dcoll, dd=dd,
            _use_geoderiv_connection=actx.supports_nonscalar_broadcasting)

    return _divergence_kernel(
        actx, discr, discr, metrics, vec, _strong_scalar_d_dx,
        use_tensor_product_fast_eval=use_tensor_product_fast_eval)


def local_grad(
        dcoll: DiscretizationCollection,
        *args,
        nested=False,
        use_tensor_product_fast_eval: bool = True
    ) -> ArrayOrContainer:
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
        partial(_strong_scalar_grad, dcoll, dd_in,
                use_tensor_product_fast_eval=use_tensor_product_fast_eval),
        (), (dcoll.ambient_dim,),
        vec, scalar_cls=DOFArray, return_nested=nested,)


def local_d_dx(
        dcoll: DiscretizationCollection,
        xyz_axis,
        *args,
        use_tensor_product_fast_eval: bool = True
    ) -> ArrayOrContainer:
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
        return map_array_container(
            partial(local_d_dx, dcoll, xyz_axis, dd,
                    use_tensor_product_fast_eval=use_tensor_product_fast_eval),
            vec)

    from grudge.geometry import inverse_surface_metric_derivative_mat

    discr = dcoll.discr_from_dd(dd)
    actx = vec.array_context
    metrics = inverse_surface_metric_derivative_mat(actx, dcoll, dd=dd,
        _use_geoderiv_connection=actx.supports_nonscalar_broadcasting)

    return _single_axis_derivative_kernel(
        actx, discr, discr, metrics, xyz_axis, vec, _strong_scalar_d_dx,
        use_tensor_product_fast_eval=use_tensor_product_fast_eval)


def local_div(
        dcoll: DiscretizationCollection,
        *args,
        use_tensor_product_fast_eval: bool = True
    ) -> ArrayOrContainer:
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
        lambda vec: _strong_scalar_div(
            dcoll, dd, vec,
            use_tensor_product_fast_eval=use_tensor_product_fast_eval),
        (dcoll.ambient_dim,), (),
        vecs, scalar_cls=DOFArray)

# }}}


# {{{ Weak derivative operators

def _weak_tensor_product_single_axis_derivative(
        actx: ArrayContext,
        xyz_axis: int,
        output_group: NodalElementGroupBase,
        input_group: InterpolatoryElementGroupBase,
        vec: ArrayOrContainer,
        metrics: ArrayOrContainer
    ) -> ArrayOrContainer:

    vec_with_metrics = vec * metrics[xyz_axis]
    weak_derivative = 0.0

    for rst_axis in range(input_group.dim):
        apply_mass_axes = set(range(input_group.dim)) - {rst_axis}
        weak_ref_derivative = fold(input_group.space,
                                   vec_with_metrics[rst_axis])

        for ax in apply_mass_axes:
            weak_ref_derivative = _single_axis_contraction(
                actx,
                input_group.dim,
                ax,
                reference_mass_matrix(actx,
                                      output_group=output_group,
                                      input_group=input_group,
                                      use_tensor_product_fast_eval=True),
                weak_ref_derivative,
                arg_names=("mass_1d", "vec")
            )

        weak_derivative += unfold(
            output_group.space,
            _single_axis_contraction(
                actx,
                input_group.dim,
                rst_axis,
                reference_stiffness_transpose_matrices(
                    actx, input_group, output_group,
                    use_tensor_product_fast_eval=True)[0],
                weak_ref_derivative,
                arg_names=("mass_1d", "vec")
            )
        )

    return weak_derivative


def _weak_simplicial_single_axis_derivative(
        actx: ArrayContext,
        xyz_axis: int,
        output_group: NodalElementGroupBase,
        input_group: InterpolatoryElementGroupBase,
        vec: ArrayOrContainer,
        metrics: ArrayOrContainer
    ) -> ArrayOrContainer:

    vec_with_metrics = vec * metrics[xyz_axis]

    return sum(
        actx.einsum(
            "ij,ej->ei",
            reference_stiffness_transpose_matrices(
                actx, input_group, output_group,
                use_tensor_product_fast_eval=False)[rst_axis],
            vec_with_metrics[rst_axis],
            arg_names=(f"stiffness_t_{rst_axis}", "vec_scaled"))
        for rst_axis in range(input_group.dim)
    )


def _weak_scalar_d_dx(
        actx: ArrayContext,
        xyz_axis: int,
        output_group: NodalElementGroupBase,
        input_group: InterpolatoryElementGroupBase,
        vec: ArrayOrContainer,
        metrics: ArrayOrContainer,
        use_tensor_product_fast_eval: bool = True
    ) -> ArrayOrContainer:

    if isinstance(input_group, TensorProductElementGroupBase) and \
            use_tensor_product_fast_eval:
        return _weak_tensor_product_single_axis_derivative(
            actx, xyz_axis, output_group, input_group, vec, metrics
        )

    return _weak_simplicial_single_axis_derivative(
        actx, xyz_axis, output_group, input_group, vec, metrics
    )


def _weak_scalar_grad(dcoll, dd_in, vec, *args,
                      use_tensor_product_fast_eval=True):

    # {{{ setup and grab discretizations

    actx = vec.array_context

    dd_in = as_dofdesc(dd_in)
    input_discr = dcoll.discr_from_dd(dd_in)
    output_discr = dcoll.discr_from_dd(dd_in.with_discr_tag(DISCR_TAG_BASE))

    # }}}

    # {{{ get metrics and apply scaling terms

    metrics = inverse_surface_metric_derivative_mat(actx, dcoll, dd=dd_in,
        times_area_element=False,
        _use_geoderiv_connection=actx.supports_nonscalar_broadcasting)

    vec_scaled = vec * area_element(actx, dcoll, dd=dd_in,
        _use_geoderiv_connection=actx.supports_nonscalar_broadcasting)

    # }}}

    return _gradient_kernel(
        actx, output_discr, input_discr, metrics, vec_scaled, _weak_scalar_d_dx,
        use_tensor_product_fast_eval=use_tensor_product_fast_eval
    )


def _weak_scalar_div(dcoll, dd_in, vecs, *args,
                     use_tensor_product_fast_eval=True):
    from arraycontext import (
        get_container_context_recursively,
        serialize_container,
    )

    # {{{ setup and grab discretizations

    actx = get_container_context_recursively(vecs)
    assert actx is not None

    vec = actx.np.stack([v for _, v in serialize_container(vecs)])

    dd_in = as_dofdesc(dd_in)
    input_discr = dcoll.discr_from_dd(dd_in)
    output_discr = dcoll.discr_from_dd(dd_in.with_discr_tag(DISCR_TAG_BASE))

    # }}}

    # {{{ get metrics and apply scaling terms

    metrics = inverse_surface_metric_derivative_mat(actx, dcoll, dd=dd_in,
        times_area_element=False,
        _use_geoderiv_connection=actx.supports_nonscalar_broadcasting)

    vec_scaled = vec * area_element(actx, dcoll, dd=dd_in,
        _use_geoderiv_connection=actx.supports_nonscalar_broadcasting)

    # }}}

    return _divergence_kernel(
        actx, output_discr, input_discr, metrics, vec_scaled, _weak_scalar_d_dx,
        use_tensor_product_fast_eval=use_tensor_product_fast_eval
    )


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

    discr = dcoll.discr_from_dd(dd)
    actx = vec.array_context
    metrics = inverse_surface_metric_derivative_mat(actx, dcoll, dd=dd,
        _use_geoderiv_connection=actx.supports_nonscalar_broadcasting)
    vec_scaled = vec * area_element(actx, dcoll, dd=dd_in,
        _use_geoderiv_connection=actx.supports_nonscalar_broadcasting)

    return _single_axis_derivative_kernel(
        actx, discr, discr, metrics, xyz_axis, vec_scaled, _weak_scalar_d_dx,
        use_tensor_product_fast_eval=use_tensor_product_fast_eval)


def weak_local_grad(
        dcoll: DiscretizationCollection,
        *args,
        nested=False,
        use_tensor_product_fast_eval=True
    ) -> ArrayOrContainer:
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

    return rec_map_subarrays(
        partial(_weak_scalar_grad, dcoll, dd_in,
                use_tensor_product_fast_eval=use_tensor_product_fast_eval),
        (), (dcoll.ambient_dim,),
        vecs, scalar_cls=DOFArray, return_nested=nested,
    )


def weak_local_div(dcoll: DiscretizationCollection,
                   *args,
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

    return rec_map_subarrays(
        lambda vec: _weak_scalar_div(
            dcoll, dd_in, vec,
            use_tensor_product_fast_eval=use_tensor_product_fast_eval),
        (dcoll.ambient_dim,), (), vecs, scalar_cls=DOFArray
    )

# }}}


# {{{ Mass operator

def _apply_mass_tensor_product(
        actx: ArrayContext,
        input_group: NodalElementGroupBase,
        output_group: InterpolatoryElementGroupBase,
        vec: ArrayOrContainer
    ) -> ArrayOrContainer:
    for xyz_axis in range(input_group.dim):
        vec = _single_axis_contraction(
            actx,
            input_group.dim,
            xyz_axis,
            reference_mass_matrix(
                actx,
                output_group=output_group,
                input_group=input_group,
                use_tensor_product_fast_eval=True),
            vec,
            arg_names=("ref_mass_1d", "dofs"))

    return vec


def _apply_mass_simplicial(
        actx: ArrayContext,
        input_group: NodalElementGroupBase,
        output_group: InterpolatoryElementGroupBase,
        vec: ArrayOrContainer
    ) -> ArrayOrContainer:
    return actx.einsum("ij,ej->ei",
        reference_mass_matrix(
            actx,
            output_group=output_group,
            input_group=input_group,
            use_tensor_product_fast_eval=False),
        vec,
        arg_names=("mass_mat", "vec"))


# FIXME: start here 12/17/2024
def _apply_mass_operator(
        dcoll: DiscretizationCollection,
        dd: DOFDesc,
        vec: ArrayOrContainer,
        use_tensor_product_fast_eval: bool = True
    ) -> DOFArray:

    if not isinstance(vec, DOFArray):
        return map_array_container(
            partial(_apply_mass_operator, dcoll, dd), vec
        )

    from grudge.geometry import area_element

    dd_in = as_dofdesc(dd)

    input_discr = dcoll.discr_from_dd(dd_in)
    output_discr = dcoll.discr_from_dd(dd_in.with_discr_tag(DISCR_TAG_BASE))

    actx = vec.array_context
    vec = vec * area_element(actx, dcoll, dd=dd,
            _use_geoderiv_connection=actx.supports_nonscalar_broadcasting)

    group_data = []
    for input_group, output_group, vec_i in zip(
        input_discr.groups, output_discr.groups, vec, strict=False):
        if isinstance(input_group, TensorProductElementGroupBase) and \
                use_tensor_product_fast_eval:
            group_data.append(_apply_mass_tensor_product(
                actx, input_group, output_group, vec_i))

        else:
            group_data.append(_apply_mass_simplicial(
                actx, input_group, output_group, vec_i))

    return DOFArray(actx, data=tuple(group_data))


def mass(dcoll: DiscretizationCollection,
         *args,
         use_tensor_product_fast_eval: bool = True) -> ArrayOrContainer:
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

    return _apply_mass_operator(
        dcoll, dd_in, vec,
        use_tensor_product_fast_eval=use_tensor_product_fast_eval
    )

# }}}


# {{{ Face mass operator

def _apply_face_mass_tensor_product(
        actx: ArrayContext,
        face_group: ElementGroupBase,
        volume_group: NodalElementGroupBase,
        vec: ArrayOrContainer,
    ) -> ArrayOrContainer:

    pass


def _apply_face_mass_simplicial(
        actx: ArrayContext,
        face_group: ElementGroupBase,
        volume_group: NodalElementGroupBase,
        vec: ArrayOrContainer
    ) -> ArrayOrContainer:

    return actx.einsum(
        "ifj,fej->ei",
        reference_face_mass_matrix(
            actx,
            face_group=face_group,
            vol_group=volume_group,
            dtype=vec.dtype),
        vec.reshape(
                volume_group.mesh_el_group.nfaces,
                volume_group.nelements,
                face_group.nunit_dofs),
        arg_names=("ref_face_mass_mat", "jac_surf", "vec")
    )


def _apply_face_mass_operator(
        dcoll: DiscretizationCollection,
        dd_in: DOFDesc,
        vec: ArrayOrContainer,
        use_tensor_product_fast_eval: bool = True
    ) -> DOFArray:
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
    actx = vec.array_context

    assert len(face_discr.groups) == len(volm_discr.groups)
    surf_area_elements = area_element(actx, dcoll, dd=dd_in,
            _use_geoderiv_connection=actx.supports_nonscalar_broadcasting)

    # FIXME: enable fast operator evaluation at some point
    use_tensor_product_fast_eval = False

    group_data = []
    for vgroup, afgroup, vec_i, surf_ae_i in zip(
            volm_discr.groups, face_discr.groups, vec, surf_area_elements,
            strict=True):
        if isinstance(vgroup, TensorProductElementGroupBase) and \
                use_tensor_product_fast_eval:
            group_data.append(
                _apply_face_mass_tensor_product(
                    actx, afgroup, vgroup, vec_i * surf_ae_i
                )
              )
        else:
            group_data.append(
                _apply_face_mass_simplicial(
                    actx, afgroup, vgroup, vec_i * surf_ae_i
                )
              )

    return DOFArray(actx, data=tuple(group_data))


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


# {{{ Mass inverse operator

def _apply_inverse_mass_tensor_product(
        actx: ArrayContext,
        group: InterpolatoryElementGroupBase,
        vec: ArrayOrContainer
    ) -> ArrayOrContainer:

    vec = fold(group.space, vec)

    for rst_axis in range(group.dim):
        vec = _single_axis_contraction(
            actx, group.dim, rst_axis,
            reference_inverse_mass_matrix(
                actx, group,
                use_tensor_product_fast_eval=True),
            vec,
            arg_names=("base_mass_inv", "dofs"))

    return unfold(group.space, vec)


def _apply_inverse_mass_simplicial(
        actx: ArrayContext,
        group: InterpolatoryElementGroupBase,
        vec: ArrayOrContainer
    ) -> ArrayOrContainer:

    return actx.einsum(
        "ij,ej->ei",
        reference_inverse_mass_matrix(
            actx, group,
            use_tensor_product_fast_eval=False),
        vec,
        arg_names=("ref_inv_mass", "vec")
    )


# {{{ non-WADG

def _apply_quad_inverse_mass(
        actx: ArrayContext,
        discr: Discretization,
        vec: ArrayOrContainer,
        inv_area_elements: ArrayOrContainer,
        use_tensor_product_fast_eval: bool = True
    ) -> DOFArray:

    group_data = []
    for input_group, output_group, vec_i, inv_ae_i in zip(
            discr.groups, discr.groups, vec, inv_area_elements, strict=False):
        if isinstance(input_group, TensorProductElementGroupBase) and \
                use_tensor_product_fast_eval:
            group_data.append(_apply_inverse_mass_tensor_product(
                actx, output_group, vec_i) * inv_ae_i)
        else:
            group_data.append(_apply_inverse_mass_simplicial(
                actx, output_group, vec_i) * inv_ae_i)

    return DOFArray(actx, data=tuple(group_data))

# }}}


# {{{ WADG

def _apply_wadg_inverse_mass(
        actx: ArrayContext,
        dcoll: DiscretizationCollection,
        dd_out: DOFDesc,
        dd_in: DOFDesc,
        vec: ArrayOrContainer,
        inv_area_elements: ArrayOrContainer,
        use_tensor_product_fast_eval: bool = True
    ) -> DOFArray:

    discr_base = dcoll.discr_from_dd(dd_out)
    discr_quad = dcoll.discr_from_dd(dd_in)

    group_data = []
    for input_group, output_group, vec_i in zip(
            discr_quad.groups, discr_base.groups, vec, strict=False):

        if isinstance(input_group, TensorProductElementGroupBase) and \
                use_tensor_product_fast_eval:
            group_data.append(_apply_inverse_mass_tensor_product(
                actx, output_group, vec_i))

        else:
            group_data.append(_apply_inverse_mass_simplicial(
                actx, output_group, vec_i))

    vec = inv_area_elements * project(
        dcoll, dd_out, dd_in, DOFArray(actx, data=tuple(group_data)))

    group_data = []
    for input_group, output_group, vec_i in zip(
            discr_quad.groups, discr_base.groups, vec, strict=False):

        if isinstance(input_group, TensorProductElementGroupBase) and \
                use_tensor_product_fast_eval:
            group_data.append(_apply_inverse_mass_tensor_product(
                actx, output_group, _apply_mass_tensor_product(
                    actx, input_group, output_group, vec_i)))

        else:
            group_data.append(_apply_inverse_mass_simplicial(
                actx, output_group, _apply_mass_simplicial(
                    actx, input_group, output_group, vec_i)))

    return DOFArray(actx, data=tuple(group_data))

# }}}


def _apply_inverse_mass(
        dcoll: DiscretizationCollection,
        dd_out: DOFDesc,
        dd_in: DOFDesc,
        vec: ArrayOrContainer,
        uses_quadrature: bool = False,
        use_tensor_product_fast_eval: bool = True):
    if not isinstance(vec, DOFArray):
        return map_array_container(
            partial(_apply_inverse_mass, dcoll, dd_out, dd_in,
                    uses_quadrature, use_tensor_product_fast_eval), vec
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
        return _apply_quad_inverse_mass(
            actx, discr_base, vec, inv_area_elements,
            use_tensor_product_fast_eval=use_tensor_product_fast_eval
        )

    else:
        # FIXME: probably need to add a link to more details about this in docs
        if use_tensor_product_fast_eval:
            from warnings import warn
            warn("Fast operator evaluation + overintegration is not supported. "
                 "Defaulting to typical (full) operator evaluation.",
                 stacklevel=1)

        # see WADG: https://arxiv.org/pdf/1608.03836
        return _apply_wadg_inverse_mass(
            actx, dcoll, dd_out, dd_in, vec, inv_area_elements,
            use_tensor_product_fast_eval=False
        )


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

    return _apply_inverse_mass(
        dcoll, dd_out, dd_in, vec,
        uses_quadrature=dd.uses_quadrature(),
        use_tensor_product_fast_eval=use_tensor_product_fast_eval)

# }}}


# vim: foldmethod=marker
