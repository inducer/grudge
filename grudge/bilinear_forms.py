from __future__ import annotations

from typing import TYPE_CHECKING

from constantdict import constantdict

import modepy as mp
from meshmode.discretization import (
    ElementGroupBase,
    InterpolatoryElementGroupBase,
    NodalElementGroupBase,
)
from meshmode.discretization.poly_element import TensorProductElementGroupBase
from meshmode.transform_metadata import DiscretizationDOFAxisTag
from modepy.quadrature import TensorProductQuadrature
from pytools import keyed_memoize_on_first_arg


if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from arraycontext import Array, ArrayContext
    from modepy.typing import ArrayF
    from pytools.tag import Tag


def _make_operator_key(
    matrix: ArrayF,
    array_tags: Sequence[Tag],
    axis_tags: Mapping[int, Sequence[Tag]],
) -> tuple[
    frozenset[Tag], Mapping[int, frozenset[Tag]], tuple[int, ...], str, bytes
]:
    return (
        frozenset(array_tags),
        constantdict(
            (axis, frozenset(tags)) for axis, tags in axis_tags.items()
        ),
        matrix.shape,
        matrix.dtype.str,
        matrix.tobytes(),
    )


@keyed_memoize_on_first_arg(_make_operator_key)
def _as_cached_operator(
    actx: ArrayContext,
    matrix: ArrayF,
    array_tags: Sequence[Tag],
    axis_tags: Mapping[int, Sequence[Tag]],
) -> Array:

    matrix_actx = actx.tag(array_tags, actx.from_numpy(matrix))
    for axis, tag in axis_tags.items():
        matrix_actx = actx.tag_axis(axis, tag, matrix_actx)

    return actx.freeze(matrix_actx)


def _make_dense_mass_operator(
    actx: ArrayContext,
    in_element_group: NodalElementGroupBase,
    out_element_group: InterpolatoryElementGroupBase,
) -> ArrayF:

    in_nodes = in_element_group.unit_nodes
    out_nodes = out_element_group.unit_nodes
    test_basis = out_element_group.basis_obj()

    if in_element_group == out_element_group:
        trial_basis = test_basis
        quadrature = mp.quadrature_for_space(
            mp.space_for_shape(
                out_element_group.shape, 2 * out_element_group.order
            ),
            out_element_group.shape,
        )
        matrix = mp.nodal_quadrature_bilinear_form_matrix(
            quadrature=quadrature,
            test_functions=test_basis.functions,
            trial_functions=trial_basis.functions,
            nodal_interp_functions_test=test_basis.functions,
            nodal_interp_functions_trial=trial_basis.functions,
            input_nodes=in_nodes,
            output_nodes=out_nodes,
        )
    else:
        quadrature = in_element_group.quadrature_rule()
        matrix = mp.nodal_quadrature_test_matrix(
            quadrature=quadrature,
            test_functions=test_basis.functions,
            nodal_interp_functions=test_basis.functions,
            nodes=out_nodes,
        )

    return matrix


def _make_tensor_product_mass_operator(
    actx: ArrayContext,
    in_element_group: TensorProductElementGroupBase,
    out_element_group: TensorProductElementGroupBase,
) -> tuple[ArrayF, ...]:

    in_nodes = in_element_group.unit_nodes_1d
    out_nodes = out_element_group.unit_nodes_1d
    test_basis = out_element_group.basis_obj()
    assert isinstance(test_basis, mp.TensorProductBasis)

    # WARNING: as of 9/23/2026 meshmode only constructs isotropic tensor product
    # elements. if this is ever expanded, then we will need to generate a
    # sequence of operators for each coordinate direction
    test_basis = test_basis.bases[0]

    if in_element_group == out_element_group:
        trial_basis = test_basis
        quadrature = mp.LegendreGaussQuadrature(
            out_element_group.order, force_dim_axis=True
        )
        matrix = mp.nodal_quadrature_bilinear_form_matrix(
            quadrature=quadrature,
            test_functions=test_basis.functions,
            trial_functions=trial_basis.functions,
            nodal_interp_functions_test=test_basis.functions,
            nodal_interp_functions_trial=trial_basis.functions,
            input_nodes=in_nodes,
            output_nodes=out_nodes,
        )
    else:
        quadrature = in_element_group.quadrature_rule()
        assert isinstance(quadrature, TensorProductQuadrature)
        quadrature = quadrature.quadratures[0]

        matrix = mp.nodal_quadrature_test_matrix(
            quadrature=quadrature,
            test_functions=test_basis.functions,
            nodal_interp_functions=test_basis.functions,
            nodes=out_nodes,
        )

    return (matrix,) * out_element_group.dim


@keyed_memoize_on_first_arg(
    lambda in_element_group, out_element_group: (
        in_element_group.discretization_key(),
        out_element_group.discretization_key(),
    )
)
def make_mass_operator(
    actx: ArrayContext,
    in_element_group: ElementGroupBase,
    out_element_group: ElementGroupBase,
) -> Array | tuple[Array, ...]:

    if not isinstance(in_element_group, NodalElementGroupBase):
        raise TypeError(
            f"'in_element_group' must be nodal: {type(in_element_group)}"
        )
    if not isinstance(out_element_group, InterpolatoryElementGroupBase):
        raise TypeError(
            f"'out_element_group' must be interpolatory: {type(out_element_group)}"
        )

    if isinstance(
        in_element_group, TensorProductElementGroupBase
    ) and isinstance(out_element_group, TensorProductElementGroupBase):
        matrices = _make_tensor_product_mass_operator(
            actx, in_element_group, out_element_group
        )

        array_tags = ()
        axis_tags = {0: (DiscretizationDOFAxisTag(),)}

        return tuple(
            _as_cached_operator(actx, matrix, array_tags, axis_tags)
            for matrix in matrices
        )

    matrix = _make_dense_mass_operator(
        actx, in_element_group, out_element_group
    )

    # FIXME: incorrect for TP
    array_tags = ()
    axis_tags = {0: (DiscretizationDOFAxisTag(),)}

    return _as_cached_operator(actx, matrix, array_tags, axis_tags)


@keyed_memoize_on_first_arg(
    lambda element_group: element_group.discretization_key()
)
def make_inverse_mass_operator(
    actx: ArrayContext,
    element_group: ElementGroupBase,
) -> Array | tuple[Array, ...]:

    if not isinstance(element_group, InterpolatoryElementGroupBase):
        raise TypeError(
            f"'element_group' must be interpolatory: {type(element_group)}"
        )

    import numpy.linalg as la

    if isinstance(element_group, TensorProductElementGroupBase):
        matrices = _make_tensor_product_mass_operator(
            actx, element_group, element_group
        )

        # FIXME: incorrect for TP
        array_tags = ()
        axis_tags = {0: (DiscretizationDOFAxisTag(),)}

        return tuple(
            _as_cached_operator(actx, la.inv(matrix), array_tags, axis_tags)
            for matrix in matrices
        )

    matrix = _make_dense_mass_operator(actx, element_group, element_group)

    array_tags = ()
    axis_tags = {0: (DiscretizationDOFAxisTag(),)}

    return _as_cached_operator(actx, la.inv(matrix), array_tags, axis_tags)
