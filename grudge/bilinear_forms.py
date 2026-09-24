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
    from collections.abc import Hashable, Mapping, Sequence

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

# {{{ mass / inverse mass

def _make_dense_mass_operator(
    actx: ArrayContext,
    in_element_group: NodalElementGroupBase,
    out_element_group: InterpolatoryElementGroupBase,
    *,
    enable_sum_factorization: bool = True,
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
    *,
    enable_sum_factorization: bool = True,
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


def _operator_discretization_key(
    in_element_group: ElementGroupBase,
    out_element_group: ElementGroupBase,
    *,
    enable_sum_factorization: bool = True,
) -> Hashable:
    return (
        in_element_group.discretization_key(),
        out_element_group.discretization_key(),
        enable_sum_factorization,
    )


@keyed_memoize_on_first_arg(_operator_discretization_key)
def make_mass_operator(
    actx: ArrayContext,
    in_element_group: ElementGroupBase,
    out_element_group: ElementGroupBase,
    *,
    enable_sum_factorization: bool = True,
) -> Array | tuple[Array, ...]:

    if not isinstance(in_element_group, NodalElementGroupBase):
        raise TypeError(
            f"'in_element_group' must be nodal: {type(in_element_group)}"
        )
    if not isinstance(out_element_group, InterpolatoryElementGroupBase):
        raise TypeError(
            f"'out_element_group' must be interpolatory: {type(out_element_group)}"
        )

    if (enable_sum_factorization
            and isinstance(in_element_group, TensorProductElementGroupBase)
            and isinstance(out_element_group, TensorProductElementGroupBase)):
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


def _inverse_mass_discretization_key(
    element_group: ElementGroupBase,
    *,
    enable_sum_factorization: bool = True,
) -> Hashable:
    return element_group.discretization_key(), enable_sum_factorization


@keyed_memoize_on_first_arg(_inverse_mass_discretization_key)
def make_inverse_mass_operator(
    actx: ArrayContext,
    element_group: ElementGroupBase,
    *,
    enable_sum_factorization: bool = True,
) -> Array | tuple[Array, ...]:

    if not isinstance(element_group, InterpolatoryElementGroupBase):
        raise TypeError(
            f"'element_group' must be interpolatory: {type(element_group)}"
        )

    import numpy.linalg as la

    if (enable_sum_factorization
            and isinstance(element_group, TensorProductElementGroupBase)):
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

# }}}

# {{{ strong differentiation operators


@keyed_memoize_on_first_arg(_operator_discretization_key)
def make_strong_differentiation_operator(
    actx: ArrayContext,
    in_element_group: ElementGroupBase,
    out_element_group: ElementGroupBase,
    *,
    enable_sum_factorization: bool = True,
) -> tuple[Array | tuple[Array | None, ...], ...]:

    if not isinstance(in_element_group, InterpolatoryElementGroupBase):
        raise TypeError(
            f"'in_element_group' must be interpolatory: {type(out_element_group)}"
        )
    if not isinstance(out_element_group, NodalElementGroupBase):
        raise TypeError(
            f"'out_element_group' must be nodal: {type(in_element_group)}"
        )

    if (enable_sum_factorization
            and isinstance(in_element_group, TensorProductElementGroupBase)
            and isinstance(out_element_group, TensorProductElementGroupBase)):
        if in_element_group != out_element_group:
            raise NotImplementedError(
                "sum factorized strong differentiation between different "
                "element groups is not yet supported"
            )

        # WARNING: as of 9/23/2026 meshmode only constructs isotropic tensor product
        # elements. if this is ever expanded, then we will need to generate a
        # sequence of operators for each coordinate direction
        basis = in_element_group.basis_obj()
        assert isinstance(basis, mp.TensorProductBasis)

        basis = basis.bases[0]

        array_tags = ()
        axis_tags = {0: (DiscretizationDOFAxisTag(),)}

        D = mp.diff_matrices(
            basis,
            out_element_group.unit_nodes_1d,
            from_nodes=in_element_group.unit_nodes_1d,
        )[0]

        matrix: Array = _as_cached_operator(actx, D, array_tags, axis_tags)

        # None => identity matrix. Skip rather than construct + apply identity
        return tuple(
            tuple(matrix if axis == derivative_axis else None
                  for axis in range(in_element_group.dim))
            for derivative_axis in range(in_element_group.dim)
        )

    else:
        array_tags = ()
        axis_tags = {0: (DiscretizationDOFAxisTag(),)}

        matrices = mp.diff_matrices(
            in_element_group.basis_obj(),
            out_element_group.unit_nodes,
            from_nodes=in_element_group.unit_nodes,
        )

        return tuple(
            _as_cached_operator(actx, matrix, array_tags, axis_tags)
            for matrix in matrices
        )

# }}}

# {{{ weak differentiation operators
# }}}
