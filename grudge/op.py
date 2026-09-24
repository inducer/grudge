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
from typing import TYPE_CHECKING, Any, cast, overload

import numpy as np

import modepy as mp
from arraycontext import (
    Array,
    ArrayContainer,
    ArrayContainerT,
    ArrayContext,
    ArrayOrContainerOrScalar,
    is_array_container,
    map_array_container,
    tag_axes,
)
from arraycontext.typing import is_scalar_like
from meshmode.discretization import (
    InterpolatoryElementGroupBase,
    NodalElementGroupBase,
)
from meshmode.discretization.poly_element import TensorProductElementGroupBase
from meshmode.dof_array import DOFArray
from meshmode.transform_metadata import (
    DiscretizationDOFAxisTag,
    DiscretizationFaceAxisTag,
    FirstAxisIsElementsTag,
)
from pytools import keyed_memoize_in, obj_array

from grudge.bilinear_forms import (
    make_face_mass_operator,
    make_inverse_mass_operator,
    make_mass_operator,
    make_stiffness_t_operator,
    make_strong_differentiation_operator,
)
from grudge.dof_desc import (
    DD_VOLUME_ALL,
    DISCR_TAG_BASE,
    FACE_RESTR_ALL,  # pyright: ignore[reportPrivateLocalImportUsage]
    BoundaryDomainTag,
    DOFDesc,
    ToDOFDescConvertible,
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
    connected_parts,
    cross_rank_trace_pairs,
    interior_trace_pair,
    interior_trace_pairs,
    local_interior_trace_pair,
    project_tracepair,
    tracepair_with_discr_tag,
)


if TYPE_CHECKING:
    from collections.abc import Callable, Hashable

    # NOTE: do not be tempted to move this import out of TYPE_CHECKING: doing so
    # makes sphinx *sometimes* expand the type alias and fail due to some types
    # that are not actually documented (e.g. _UserDefinedArrayContainer)
    from arraycontext import ArrayOrContainer
    from meshmode.discretization import (
        ElementGroupBase,
    )

    from grudge.discretization import DiscretizationCollection


__all__ = (
    "bdry_trace_pair",
    "bv_trace_pair",
    "connected_parts",
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


# {{{ general operator application routines


def _apply_operator_to_group(
    actx: ArrayContext,
    in_group: ElementGroupBase,
    out_group: ElementGroupBase,
    operator: Array | tuple[Array | None, ...],
    vec: Array,
    operator_name: str,
    *,
    enable_sum_factorization: bool = True,
) -> Array:

    if (enable_sum_factorization
            and isinstance(in_group, TensorProductElementGroupBase)
            and isinstance(out_group, TensorProductElementGroupBase)):
        if not isinstance(operator, tuple):
            raise TypeError(
                "tensor-product application requires a tuple of factors"
            )

        if len(operator) != out_group.dim:
            raise ValueError("expected one factor per reference direction")

        from string import ascii_lowercase

        from modepy.tools import (
            reshape_array_for_tensor_product_space,
            unreshape_array_for_tensor_product_space,
        )

        from grudge.transform.metadata import (
            OutputIsTensorProductDOFArrayOrdered,
        )

        vec_tp = vec
        if len(vec.shape) == 2:
            vec_tp = reshape_array_for_tensor_product_space(
                in_group.space,
                vec,  # pyright: ignore[reportArgumentType]
            )

        ndim = len(vec_tp.shape)
        indices = ascii_lowercase[:ndim]
        output_index = ascii_lowercase[ndim]
        for reference_axis, factor in enumerate(operator):
            # strong differentiation applies identity, so skip rather than carry
            # it out
            if factor is None:
                continue

            # skip element axis
            axis = reference_axis + 1

            reduction_index = indices[axis]
            output_indices = indices[:axis] + output_index + indices[axis + 1 :]

            vec_tp = actx.einsum(
                f"{output_index}{reduction_index},{indices}->{output_indices}",
                factor,
                vec_tp,
                arg_names=(operator_name, "vec_tp"),
                tagged=(OutputIsTensorProductDOFArrayOrdered(),),
            )

        return unreshape_array_for_tensor_product_space(out_group.space, vec_tp)  # pyright: ignore[reportArgumentType]

    else:
        if isinstance(operator, tuple):
            if len(operator) != 1:
                raise ValueError(
                    "only a single dense operator can be supplied."
                )
            assert operator[0] is not None
            operator = operator[0]

        return actx.einsum(
            "ij,ej->ei",
            operator,
            vec,
            arg_names=(operator_name, "vec"),
            tagged=(FirstAxisIsElementsTag(),),
        )


# }}}


# {{{ Derivative operators

def _strong_scalar_grad(
    dcoll: DiscretizationCollection,
    dd_in: DOFDesc,
    vec: ArrayOrContainer,
    *,
    enable_sum_factorization: bool = True,
) -> obj_array.ObjectArray1D[DOFArray]:
    assert isinstance(dd_in.domain_tag, VolumeDomainTag)
    assert isinstance(vec, DOFArray)

    from grudge.geometry import inverse_surface_metric_derivative_mat

    discr = dcoll.discr_from_dd(dd_in)
    actx = vec.array_context
    assert actx is not None

    inverse_jac_mat = inverse_surface_metric_derivative_mat(
        actx,
        dcoll,
        dd=dd_in,
        _use_geoderiv_connection=actx.supports_nonscalar_broadcasting,
    )

    # NOTE: we explicitly write this without using the single axis derivative
    # kernel to avoid recomputation of reference derivatives
    per_group_gradient = []
    for in_group, out_group, vec_i, ijm_i in zip(
        discr.groups, discr.groups, vec, inverse_jac_mat, strict=True
    ):
        operators = make_strong_differentiation_operator(
            actx,
            in_group,
            out_group,
            enable_sum_factorization=enable_sum_factorization,
        )
        ref_axes = "rst"

        reference_derivatives = actx.np.stack([
            _apply_operator_to_group(
                actx,
                in_group,
                out_group,
                operators[rst_axis],
                vec_i,
                operator_name=f"strong_ref_deriv_{ref_axes[rst_axis]}",
                enable_sum_factorization=enable_sum_factorization,
            )
            for rst_axis in range(in_group.dim)
        ])

        per_group_gradient.append(
            actx.einsum(
                "xrej,rej->xej",
                ijm_i,
                reference_derivatives,
                arg_names=("inv_jac_mat", "ref_derivatives"),
            )
        )

    return obj_array.new_1d([
        DOFArray(
            actx, data=tuple([pgg_i[xyz_axis] for pgg_i in per_group_gradient])
        )
        for xyz_axis in range(discr.ambient_dim)
    ])


def _strong_scalar_div(
    dcoll: DiscretizationCollection,
    dd: DOFDesc,
    vecs: obj_array.ObjectArray1D[DOFArray],
    *,
    enable_sum_factorization: bool = True,
) -> DOFArray:
    assert isinstance(vecs, np.ndarray)
    assert vecs.shape == (dcoll.ambient_dim,)

    return sum(
        local_d_dx(
            dcoll,
            xyz_axis,
            dd,
            vecs[xyz_axis],
            enable_sum_factorization=enable_sum_factorization,
        )
        for xyz_axis in range(dcoll.ambient_dim)  # pyright: ignore[reportArgumentType,reportCallIssue]
    )  # pyright: ignore[reportCallIssue]


@overload
def local_grad(
        dcoll: DiscretizationCollection, vec: ArrayOrContainer, /, *,
        nested: bool = False,
        enable_sum_factorization: bool = True,
) -> ArrayOrContainer: ...


@overload
def local_grad(
        dcoll: DiscretizationCollection, dd_in: DOFDesc, vec: ArrayOrContainer, /, *,
        nested: bool = False,
        enable_sum_factorization: bool = True,
) -> ArrayOrContainer: ...


def local_grad(
        dcoll: DiscretizationCollection,
        *args: Any,
        nested: bool = False,
        enable_sum_factorization: bool = True) -> ArrayOrContainer:
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
    :arg enable_sum_factorization: use tensor-product factors where supported.
        If *False*, construct and apply full dense reference matrices instead.
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
                enable_sum_factorization=enable_sum_factorization),
        (), (dcoll.ambient_dim,),
        vec, scalar_cls=DOFArray, return_nested=nested,)


@overload
def local_d_dx(
        dcoll: DiscretizationCollection, xyz_axis: int,
        vec: ArrayOrContainer, /, *,
        enable_sum_factorization: bool = True,
) -> ArrayOrContainer: ...


@overload
def local_d_dx(
        dcoll: DiscretizationCollection, xyz_axis: int,
        dd: DOFDesc, vec: ArrayOrContainer, /, *,
        enable_sum_factorization: bool = True,
) -> ArrayOrContainer: ...


def local_d_dx(
    dcoll: DiscretizationCollection,
    xyz_axis: int,
    *args: Any,
    enable_sum_factorization: bool = True,
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
        (vec,) = args
        dd = DD_VOLUME_ALL
    elif len(args) == 2:
        dd, vec = args
    else:
        raise TypeError(
            f"invalid number of arguments to 'local_d_dx': {len(args)}"
        )

    if is_scalar_like(vec):
        raise TypeError(f"scalars not allowed: {vec}")

    if not isinstance(vec, DOFArray):
        return map_array_container(
            cast(
                "Callable[[ArrayOrContainerOrScalar], ArrayOrContainer]",
                partial(
                    local_d_dx,
                    dcoll,
                    xyz_axis,
                    dd,
                    enable_sum_factorization=enable_sum_factorization,
                ),
            ),
            vec,
        )

    discr = dcoll.discr_from_dd(dd)
    actx = vec.array_context
    assert actx is not None

    from grudge.geometry import inverse_surface_metric_derivative_mat

    inverse_jac_mat = inverse_surface_metric_derivative_mat(
        actx,
        dcoll,
        dd=dd,
        _use_geoderiv_connection=actx.supports_nonscalar_broadcasting,
    )

    per_group_derivative = []
    for in_group, out_group, vec_i, ijm_i in zip(
        discr.groups, discr.groups, vec, inverse_jac_mat, strict=True
    ):
        operators = make_strong_differentiation_operator(
            actx,
            in_group,
            out_group,
            enable_sum_factorization=enable_sum_factorization,
        )
        ref_axes = "rst"

        reference_derivatives = actx.np.stack([
            _apply_operator_to_group(
                actx,
                in_group,
                out_group,
                operators[rst_axis],
                vec_i,
                operator_name=f"strong_ref_deriv_{ref_axes[rst_axis]}",
                enable_sum_factorization=enable_sum_factorization,
            )
            for rst_axis in range(in_group.dim)
        ])

        per_group_derivative.append(
            actx.einsum(
                "rej,rej->ej",
                ijm_i[xyz_axis],
                reference_derivatives,
                arg_names=("inv_jac_t", "vec"),
            )
        )

    return DOFArray(actx, data=tuple(per_group_derivative))


@overload
def local_div(
        dcoll: DiscretizationCollection, vecs: ArrayOrContainer, /, *,
        enable_sum_factorization: bool = True,
) -> ArrayOrContainer: ...


@overload
def local_div(
        dcoll: DiscretizationCollection, dd: DOFDesc, vecs: ArrayOrContainer, /, *,
        enable_sum_factorization: bool = True,
) -> ArrayOrContainer: ...


def local_div(dcoll: DiscretizationCollection, *args: Any,
              enable_sum_factorization: bool = True) -> ArrayOrContainer:
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
        lambda vec: _strong_scalar_div(dcoll, dd, vec,
            enable_sum_factorization=enable_sum_factorization),
        (dcoll.ambient_dim,), (),
        vecs, scalar_cls=DOFArray)

# }}}


# {{{ Weak derivative operators

def _weak_scalar_grad(
    dcoll: DiscretizationCollection,
    dd_in: DOFDesc,
    vec: ArrayOrContainer,
    *,
    enable_sum_factorization: bool = True,
) -> obj_array.ObjectArray1D[ArrayOrContainer]:
    assert isinstance(vec, DOFArray)

    dd_in = as_dofdesc(dd_in)
    out_discr = dcoll.discr_from_dd(dd_in.with_discr_tag(DISCR_TAG_BASE))

    return obj_array.new_1d([
        weak_local_d_dx(
            dcoll,
            dd_in,
            xyz_axis,
            vec,
            enable_sum_factorization=enable_sum_factorization,
        )
        for xyz_axis in range(out_discr.ambient_dim)
    ])


def _weak_scalar_div(
    dcoll: DiscretizationCollection,
    dd_in: DOFDesc,
    vecs: obj_array.ObjectArray1D[DOFArray],
    *,
    enable_sum_factorization: bool = True,
) -> DOFArray:
    assert isinstance(vecs, np.ndarray)
    assert vecs.shape == (dcoll.ambient_dim,)

    return sum(
        weak_local_d_dx(
            dcoll,
            dd_in,
            xyz_axis,
            vecs[xyz_axis],
            enable_sum_factorization=enable_sum_factorization,
        )
        for xyz_axis in range(dcoll.ambient_dim)  # pyright: ignore[reportArgumentType,reportCallIssue]
    )  # pyright: ignore[reportCallIssue]


@overload
def weak_local_grad(
        dcoll: DiscretizationCollection, vec: ArrayOrContainer, /, *,
        nested: bool = False,
        enable_sum_factorization: bool = True,
) -> ArrayOrContainer: ...


@overload
def weak_local_grad(
        dcoll: DiscretizationCollection, dd_in: DOFDesc, vec: ArrayOrContainer, /, *,
        nested: bool = False,
        enable_sum_factorization: bool = True,
) -> ArrayOrContainer: ...


def weak_local_grad(
        dcoll: DiscretizationCollection,
        *args: Any, nested: bool = False,
        enable_sum_factorization: bool = True) -> ArrayOrContainer:
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
        partial(_weak_scalar_grad, dcoll, dd_in,
                enable_sum_factorization=enable_sum_factorization),
        (), (dcoll.ambient_dim,),
        vecs, scalar_cls=DOFArray, return_nested=nested)


@overload
def weak_local_d_dx(
        dcoll: DiscretizationCollection,
        xyz_axis: int, vec: ArrayOrContainer, /, *,
        enable_sum_factorization: bool = True,
) -> ArrayOrContainer: ...


@overload
def weak_local_d_dx(
        dcoll: DiscretizationCollection,
        dd_in: DOFDesc, xyz_axis: int, vec: ArrayOrContainer, /, *,
        enable_sum_factorization: bool = True,
) -> ArrayOrContainer: ...


def weak_local_d_dx(
    dcoll: DiscretizationCollection,
    *args: Any,
    enable_sum_factorization: bool = True,
) -> ArrayOrContainer:
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
        dd_in = DD_VOLUME_ALL
    elif len(args) == 3:
        dd_in, xyz_axis, vec = args
    else:
        raise TypeError("invalid number of arguments")

    if is_scalar_like(vec):
        raise TypeError(f"scalars not allowed: {vec}")

    if not isinstance(vec, DOFArray):
        return map_array_container(
            cast(
                "Callable[[ArrayOrContainerOrScalar], ArrayOrContainer]",
                partial(
                    weak_local_d_dx,
                    dcoll,
                    dd_in,
                    xyz_axis,
                    enable_sum_factorization=enable_sum_factorization,
                ),
            ),
            vec,
        )

    from grudge.geometry import inverse_surface_metric_derivative_mat

    dd_in = as_dofdesc(dd_in)
    in_discr = dcoll.discr_from_dd(dd_in)
    out_discr = dcoll.discr_from_dd(dd_in.with_discr_tag(DISCR_TAG_BASE))

    actx = vec.array_context
    assert actx is not None
    inverse_jac_mat = inverse_surface_metric_derivative_mat(
        actx,
        dcoll,
        dd=dd_in,
        times_area_element=True,
        _use_geoderiv_connection=actx.supports_nonscalar_broadcasting,
    )

    per_group_weak_derivative = []
    for in_group, out_group, vec_i, ijm_i in zip(
        in_discr.groups, out_discr.groups, vec, inverse_jac_mat, strict=True
    ):
        operators = make_stiffness_t_operator(
            actx,
            in_group,
            out_group,
            enable_sum_factorization=enable_sum_factorization,
        )
        ref_axes = "rst"

        ijm_applied = actx.einsum(
            "rej,ej->rej",
            ijm_i[xyz_axis],
            vec_i,
            arg_names=(f"inv_jac_t_{xyz_axis}", "vec"),
        )

        per_group_weak_derivative.append(
            sum(
                _apply_operator_to_group(
                    actx,
                    in_group,
                    out_group,
                    operators[rst_axis],
                    ijm_applied[rst_axis],
                    operator_name=f"weak_local_ref_d_dx_{ref_axes[rst_axis]}",
                    enable_sum_factorization=enable_sum_factorization,
                )
                for rst_axis in range(out_group.dim)
            )
        )

    return DOFArray(actx, data=tuple(per_group_weak_derivative))


@overload
def weak_local_div(
        dcoll: DiscretizationCollection,
        vecs: ArrayOrContainer, /, *,
        enable_sum_factorization: bool = True,
) -> ArrayOrContainer: ...


@overload
def weak_local_div(
        dcoll: DiscretizationCollection,
        dd: DOFDesc, vecs: ArrayOrContainer, /, *,
        enable_sum_factorization: bool = True,
) -> ArrayOrContainer: ...


def weak_local_div(dcoll: DiscretizationCollection, *args: Any,
                   enable_sum_factorization: bool = True) -> ArrayOrContainer:
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
        lambda vec: _weak_scalar_div(dcoll, dd_in, vec,
            enable_sum_factorization=enable_sum_factorization),
        (dcoll.ambient_dim,), (),
        vecs, scalar_cls=DOFArray)

# }}}


# {{{ Mass operator

def reference_mass_matrix(
        actx: ArrayContext,
        out_element_group: ElementGroupBase,
        in_element_group: ElementGroupBase) -> Array:
    from warnings import warn

    warn(
        "'reference_mass_matrix' is deprecated and will become unavailable in 2027. "
        "Use grudge.bilinear_forms.make_mass_operator instead.",
        DeprecationWarning,
        stacklevel=2,
    )

    if not isinstance(out_element_group, InterpolatoryElementGroupBase):
        raise TypeError(
            f"'out_element_group' must be interpolatory: {type(out_element_group)}")

    if not isinstance(in_element_group, NodalElementGroupBase):
        raise TypeError(f"'in_element_group' must be nodal: {type(in_element_group)}")

    def memoize_key(out_grp: InterpolatoryElementGroupBase,
                    in_grp: NodalElementGroupBase) -> Hashable:
        return out_grp.discretization_key(), in_grp.discretization_key()

    @keyed_memoize_in(
        actx, reference_mass_matrix,
        memoize_key)
    def get_ref_mass_mat(out_grp: InterpolatoryElementGroupBase,
                         in_grp: NodalElementGroupBase) -> Array:
        if out_grp == in_grp:
            return tag_axes(actx, {
                    0: DiscretizationDOFAxisTag(),
                    1: DiscretizationDOFAxisTag(),
                    }, actx.freeze(actx.from_numpy(
                        mp.mass_matrix(out_grp.basis_obj(), out_grp.unit_nodes))))

        from modepy import vandermonde
        basis = out_grp.basis_obj()
        vand = vandermonde(basis.functions, out_grp.unit_nodes)
        o_vand = vandermonde(basis.functions, in_grp.unit_nodes)
        vand_inv_t = np.linalg.inv(vand).T

        weights = in_grp.quadrature_rule().weights
        return tag_axes(actx, {
                    0: DiscretizationDOFAxisTag(),
                    1: DiscretizationDOFAxisTag(),
                    },
                    actx.freeze(actx.from_numpy(
                        np.asarray(
                            np.einsum("j,ik,jk->ij", weights, vand_inv_t, o_vand),
                            order="C"))))

    return get_ref_mass_mat(out_element_group, in_element_group)


def _apply_mass_operator(
    dcoll: DiscretizationCollection,
    dd_out: ToDOFDescConvertible,
    dd_in: ToDOFDescConvertible,
    vec: ArrayContainerT,
    *,
    enable_sum_factorization: bool = True,
) -> ArrayContainerT:
    if is_scalar_like(vec):
        raise TypeError(f"scalars not allowed: {vec}")

    if not isinstance(vec, DOFArray):
        result = map_array_container(
            cast(
                "Callable[[ArrayOrContainerOrScalar], ArrayContainer]",
                partial(
                    _apply_mass_operator,
                    dcoll,
                    dd_out,
                    dd_in,
                    enable_sum_factorization=enable_sum_factorization,
                ),
            ),
            vec,
        )
        assert is_array_container(result)
        return cast("ArrayContainerT", result)

    from grudge.geometry import area_element

    in_discr = dcoll.discr_from_dd(dd_in)
    out_discr = dcoll.discr_from_dd(dd_out)

    actx = vec.array_context
    assert actx is not None

    area_elements = area_element(
        actx,
        dcoll,
        dd=dd_in,
        _use_geoderiv_connection=actx.supports_nonscalar_broadcasting,
    )
    assert isinstance(area_elements, DOFArray)

    return type(vec)(
        actx,
        data=tuple(
            _apply_operator_to_group(
                actx,
                in_grp,
                out_grp,
                make_mass_operator(
                    actx,
                    in_grp,
                    out_grp,
                    enable_sum_factorization=enable_sum_factorization,
                ),
                ae_i * vec_i,
                "mass_op",
                enable_sum_factorization=enable_sum_factorization,
            )
            for in_grp, out_grp, vec_i, ae_i in zip(
                in_discr.groups,
                out_discr.groups,
                vec,
                area_elements,
                strict=True,
            )
        ),
    )


@overload
def mass(
        dcoll: DiscretizationCollection,
    vec: ArrayOrContainer, /, *,
    enable_sum_factorization: bool = True,
) -> ArrayOrContainer: ...


@overload
def mass(
        dcoll: DiscretizationCollection,
        dd_in: DOFDesc, vec: ArrayOrContainer, /, *,
        enable_sum_factorization: bool = True,
) -> ArrayOrContainer: ...


def mass(dcoll: DiscretizationCollection, *args: Any,
         enable_sum_factorization: bool = True) -> ArrayOrContainer:
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

    :arg enable_sum_factorization: use tensor-product factors where supported.
        If *False*, construct and apply full dense reference matrices instead.
    :arg dd_in: a :class:`~grudge.dof_desc.DOFDesc`, or a value convertible to one.
        Defaults to the base volume discretization if not provided.
    :arg vec: a :class:`~meshmode.dof_array.DOFArray` or an
        :class:`~arraycontext.ArrayContainer` of them.
    :returns: a :class:`~meshmode.dof_array.DOFArray` or an
        :class:`~arraycontext.ArrayContainer` like *vec*.
    """

    if len(args) == 1:
        vec, = args
        dd_in = DD_VOLUME_ALL
    elif len(args) == 2:
        dd_in, vec = args
    else:
        raise TypeError("invalid number of arguments")

    dd_out = dd_in.with_discr_tag(DISCR_TAG_BASE)

    return _apply_mass_operator(dcoll, dd_out, dd_in, vec,
            enable_sum_factorization=enable_sum_factorization)

# }}}


# {{{ Mass inverse operator

def reference_inverse_mass_matrix(
        actx: ArrayContext, element_group: ElementGroupBase
    ) -> Array:
    from warnings import warn

    warn(
        "'reference_inverse_mass_matrix' is deprecated and will become "
        "unavailable in 2027. Use grudge.bilinear_forms.make_inverse_mass_operator "
        "instead.",
        DeprecationWarning,
        stacklevel=2,
    )

    if not isinstance(element_group, InterpolatoryElementGroupBase):
        raise TypeError(f"'element_group' must be interpolatory: {type(element_group)}")

    def memoize_key(grp: InterpolatoryElementGroupBase) -> Hashable:
        return grp.discretization_key()

    @keyed_memoize_in(
        actx, reference_inverse_mass_matrix,
        memoize_key)
    def get_ref_inv_mass_mat(grp: InterpolatoryElementGroupBase) -> Array:
        from modepy import inverse_mass_matrix
        basis = grp.basis_obj()

        return tag_axes(actx, {
                0: DiscretizationDOFAxisTag(),
                1: DiscretizationDOFAxisTag(),
                },
                actx.freeze(actx.from_numpy(
                    np.asarray(
                        inverse_mass_matrix(basis, grp.unit_nodes),
                        order="C"))))

    return get_ref_inv_mass_mat(element_group)


def _apply_inverse_mass_operator(
    dcoll: DiscretizationCollection,
    dd_out: ToDOFDescConvertible,
    dd_in: ToDOFDescConvertible,
    vec: ArrayContainer,
    *,
    enable_sum_factorization: bool = True,
) -> ArrayContainer:
    if not isinstance(vec, DOFArray):
        return map_array_container(
            cast(
                "Callable[[ArrayOrContainerOrScalar], ArrayContainer]",
                partial(
                    _apply_inverse_mass_operator,
                    dcoll,
                    dd_out,
                    dd_in,
                    enable_sum_factorization=enable_sum_factorization,
                ),
            ),
            vec,
        )

    from grudge.geometry import area_element

    if dd_out != dd_in:
        raise ValueError(
            "Cannot compute inverse of a mass matrix mapping "
            "between different element groups; inverse is not "
            "guaranteed to be well-defined"
        )

    actx = vec.array_context
    assert actx is not None

    discr = dcoll.discr_from_dd(dd_in)
    inv_area_elements = 1.0 / area_element(
        actx,
        dcoll,
        dd=dd_in,
        _use_geoderiv_connection=actx.supports_nonscalar_broadcasting,
    )

    group_data = [
        # Based on https://arxiv.org/pdf/1608.03836.pdf
        # true_Minv ~ ref_Minv * ref_M * (1/jac_det) * ref_Minv
        _apply_operator_to_group(
            actx,
            grp,
            grp,
            make_inverse_mass_operator(
                actx, grp, enable_sum_factorization=enable_sum_factorization
            ),
            vec_i,
            "inv_mass_op",
            enable_sum_factorization=enable_sum_factorization,
        )
        * jac_inv
        for grp, jac_inv, vec_i in zip(
            discr.groups, inv_area_elements, vec, strict=True
        )
    ]

    return DOFArray(actx, data=tuple(group_data))


@overload
def inverse_mass(
        dcoll: DiscretizationCollection,
    vec: ArrayOrContainer, /, *,
    enable_sum_factorization: bool = True,
) -> ArrayOrContainer: ...


@overload
def inverse_mass(
        dcoll: DiscretizationCollection,
        dd: DOFDesc, vec: ArrayOrContainer, /, *,
        enable_sum_factorization: bool = True,
) -> ArrayOrContainer: ...


def inverse_mass(dcoll: DiscretizationCollection, *args: Any,
                 enable_sum_factorization: bool = True) -> ArrayOrContainer:
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

    :arg enable_sum_factorization: use tensor-product factors where supported.
        If *False*, construct and apply full dense reference matrices instead.
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

    return _apply_inverse_mass_operator(dcoll, dd, dd, vec,
            enable_sum_factorization=enable_sum_factorization)

# }}}


# {{{ Face mass operator

def reference_face_mass_matrix(
            actx: ArrayContext,
            face_element_group: ElementGroupBase,
            vol_element_group: ElementGroupBase,
            dtype: np.dtype[Any]) -> Array:
    from warnings import warn

    warn(
        "'reference_face_mass_matrix' is deprecated and will become unavailable "
        "in 2027. Use grudge.bilinear_forms.make_face_mass_operator instead.",
        DeprecationWarning,
        stacklevel=2,
    )

    if not isinstance(vol_element_group, InterpolatoryElementGroupBase):
        raise TypeError(
            f"'vol_element_group' must be interpolatory: {type(vol_element_group)}")

    def memoize_key(face_grp: ElementGroupBase,
                    vol_grp: InterpolatoryElementGroupBase) -> Hashable:
        return face_grp.discretization_key(), vol_grp.discretization_key(), dtype

    @keyed_memoize_in(
        actx, reference_face_mass_matrix,
        memoize_key)
    def get_ref_face_mass_mat(
                face_grp: ElementGroupBase,
                vol_grp: InterpolatoryElementGroupBase):
        nfaces = vol_grp.mesh_el_group.nfaces
        assert face_grp.nelements == nfaces * vol_grp.nelements

        matrix = np.empty(
            (vol_grp.nunit_dofs,
            nfaces,
            face_grp.nunit_dofs),
            dtype=dtype
        )

        import modepy as mp
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
            if (isinstance(face_grp, InterpolatoryElementGroupBase)
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

        return tag_axes(actx, {
                    0: DiscretizationDOFAxisTag(),
                    1: DiscretizationFaceAxisTag(),
                    2: DiscretizationDOFAxisTag()
                    },
                    actx.freeze(actx.from_numpy(matrix)))

    return get_ref_face_mass_mat(face_element_group, vol_element_group)


def _apply_face_mass_operator(
    dcoll: DiscretizationCollection,
    dd_in: DOFDesc,
    vec: ArrayOrContainer,
    *,
    enable_sum_factorization: bool = True,
) -> ArrayOrContainer:
    if is_scalar_like(vec):
        raise TypeError(f"scalars not allowed: {vec}")

    if not isinstance(vec, DOFArray):
        result = map_array_container(
            cast(
                "Callable[[ArrayOrContainerOrScalar], ArrayOrContainer]",
                partial(
                    _apply_face_mass_operator,
                    dcoll,
                    dd_in,
                    enable_sum_factorization=enable_sum_factorization,
                ),
            ),
            vec,
        )
        return cast("ArrayOrContainer", result)

    from grudge.geometry import area_element

    dd_in = as_dofdesc(dd_in)
    assert isinstance(dd_in.domain_tag, BoundaryDomainTag)

    dd_out = DOFDesc(
        VolumeDomainTag(dd_in.domain_tag.volume_tag), DISCR_TAG_BASE
    )

    volm_discr = dcoll.discr_from_dd(dd_out)
    face_discr = dcoll.discr_from_dd(dd_in)
    actx = vec.array_context
    assert actx is not None

    assert len(face_discr.groups) == len(volm_discr.groups)
    surf_area_elements = area_element(
        actx,
        dcoll,
        dd=dd_in,
        _use_geoderiv_connection=actx.supports_nonscalar_broadcasting,
    )

    group_data = []
    for vol_group, face_group, vec_i, surf_ae_i in zip(
        volm_discr.groups,
        face_discr.groups,
        vec,
        surf_area_elements,
        strict=True,
    ):
        nfaces = vol_group.mesh_el_group.nfaces
        if face_group.nelements != nfaces * vol_group.nelements:
            raise ValueError("face mass requires data on all element faces")
        operators = make_face_mass_operator(
            actx,
            face_group,
            vol_group,
            enable_sum_factorization=enable_sum_factorization,
        )
        if len(operators) != nfaces:
            raise ValueError("expected one face mass operator per face")
        weighted_faces = (surf_ae_i * vec_i).reshape((
            nfaces,
            vol_group.nelements,
            face_group.nunit_dofs,
        ))
        face_results = []
        for iface, operator in enumerate(operators):
            face_vec = weighted_faces[iface]
            factorized = isinstance(operator, tuple)
            if factorized:
                from modepy.tools import reshape_array_for_tensor_product_space

                face = mp.faces_for_shape(vol_group.shape)[iface]
                mapped = face.map_to_volume(
                    np.column_stack((
                        np.zeros(face_group.dim),
                        np.eye(face_group.dim),
                    ))
                )
                directions = mapped[:, 1:] - mapped[:, :1]
                volume_axes = np.argmax(np.abs(directions), axis=0)
                # Reversals are in the factors; align tangential axes here.
                permutation = (
                    0,
                    *(1 + int(axis) for axis in np.argsort(volume_axes)),
                )
                face_vec = reshape_array_for_tensor_product_space(
                    face_group.space,
                    face_vec,  # pyright: ignore[reportArgumentType]
                ).transpose(permutation).reshape(
                    (vol_group.nelements,
                     *(cast("int", factor.shape[1]) for factor in operator)),
                    order="F",
                )

            face_results.append(
                _apply_operator_to_group(
                    actx,
                    face_group,
                    vol_group,
                    operator,
                    face_vec,
                    "face_mass_op",
                    enable_sum_factorization=factorized,
                )
            )
        group_data.append(sum(face_results))

    return DOFArray(actx, data=tuple(group_data))


@overload
def face_mass(
        dcoll: DiscretizationCollection,
        vec: ArrayOrContainer, /, *,
        enable_sum_factorization: bool = True,
) -> ArrayOrContainer: ...


@overload
def face_mass(
        dcoll: DiscretizationCollection,
        dd: DOFDesc, vec: ArrayOrContainer, /, *,
        enable_sum_factorization: bool = True,
) -> ArrayOrContainer: ...


def face_mass(dcoll: DiscretizationCollection, *args: Any,
              enable_sum_factorization: bool = True) -> ArrayOrContainer:
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
    :arg enable_sum_factorization: use tensor-product face factors where
        supported. If *False*, construct and apply dense reference matrices.
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

    return _apply_face_mass_operator(dcoll, dd_in, vec,
            enable_sum_factorization=enable_sum_factorization)

# }}}


# vim: foldmethod=marker
