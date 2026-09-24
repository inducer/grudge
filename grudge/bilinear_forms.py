from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from constantdict import constantdict

import modepy as mp
from meshmode.discretization import (
    ElementGroupBase,
    InterpolatoryElementGroupBase,
    NodalElementGroupBase,
)
from meshmode.discretization.poly_element import TensorProductElementGroupBase
from meshmode.transform_metadata import (
    DiscretizationDOFAxisTag,
    DiscretizationTopologicalDimAxisTag,
)
from modepy.quadrature import TensorProductQuadrature
from pytools import keyed_memoize_on_first_arg


if TYPE_CHECKING:
    from collections.abc import Hashable, Mapping, Sequence

    from arraycontext import Array, ArrayContext
    from modepy.typing import ArrayF
    from pytools.tag import Tag


def _make_dense_operator_key(
    in_element_group: NodalElementGroupBase,
    out_element_group: InterpolatoryElementGroupBase,
    *,
    array_tags: Sequence[Tag],
    axis_tags: Mapping[int, Sequence[Tag]],
) -> Hashable:
    return (
        in_element_group.discretization_key(),
        out_element_group.discretization_key(),
        in_element_group == out_element_group,
        frozenset(array_tags),
        constantdict(
            (axis, frozenset(tags)) for axis, tags in axis_tags.items()
        ),
    )


def _dense_mass_operator_key(
    in_element_group: NodalElementGroupBase,
    out_element_group: InterpolatoryElementGroupBase,
    *,
    array_tags: Sequence[Tag],
    axis_tags: Mapping[int, Sequence[Tag]],
    inverse: bool = False,
) -> Hashable:
    return (
        _make_dense_operator_key(
            in_element_group, out_element_group,
            array_tags=array_tags, axis_tags=axis_tags,
        ),
        inverse,
    )


def _tensor_product_operator_key(
    in_element_group: TensorProductElementGroupBase,
    out_element_group: TensorProductElementGroupBase,
    *,
    array_tags: Sequence[Tag],
    axis_tags: Mapping[int, Sequence[Tag]],
) -> Hashable:
    return _make_dense_operator_key(
        in_element_group, out_element_group,
        array_tags=array_tags, axis_tags=axis_tags,
    )


def _tensor_product_mass_operator_key(
    in_element_group: TensorProductElementGroupBase,
    out_element_group: TensorProductElementGroupBase,
    *,
    array_tags: Sequence[Tag],
    axis_tags: Mapping[int, Sequence[Tag]],
    inverse: bool = False,
) -> Hashable:
    return _dense_mass_operator_key(
        in_element_group, out_element_group,
        array_tags=array_tags, axis_tags=axis_tags, inverse=inverse,
    )


def _tag_and_freeze_operator(
    actx: ArrayContext,
    matrix: ArrayF,
    array_tags: Sequence[Tag],
    axis_tags: Mapping[int, Sequence[Tag]],
) -> Array:

    tagged_matrix = actx.tag(array_tags, actx.from_numpy(matrix))
    for axis, tag in axis_tags.items():
        tagged_matrix = actx.tag_axis(axis, tag, tagged_matrix)

    return actx.freeze(tagged_matrix)

# {{{ mass / inverse mass


@keyed_memoize_on_first_arg(_dense_mass_operator_key)
def _make_dense_mass_operator(
    actx: ArrayContext,
    in_element_group: NodalElementGroupBase,
    out_element_group: InterpolatoryElementGroupBase,
    *,
    array_tags: Sequence[Tag],
    axis_tags: Mapping[int, Sequence[Tag]],
    inverse: bool = False,
) -> Array:

    if inverse and in_element_group != out_element_group:
        raise ValueError("inverse mass requires matching element groups")

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

    if inverse:
        matrix = np.linalg.inv(matrix)
    return _tag_and_freeze_operator(actx, matrix, array_tags, axis_tags)


@keyed_memoize_on_first_arg(_tensor_product_mass_operator_key)
def _make_tensor_product_mass_operator(
    actx: ArrayContext,
    in_element_group: TensorProductElementGroupBase,
    out_element_group: TensorProductElementGroupBase,
    *,
    array_tags: Sequence[Tag],
    axis_tags: Mapping[int, Sequence[Tag]],
    inverse: bool = False,
) -> Array:

    if inverse and in_element_group != out_element_group:
        raise ValueError("inverse mass requires matching element groups")

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

    if inverse:
        matrix = np.linalg.inv(matrix)
    return _tag_and_freeze_operator(actx, matrix, array_tags, axis_tags)


def _operator_discretization_key(
    in_element_group: ElementGroupBase,
    out_element_group: ElementGroupBase,
    *,
    enable_sum_factorization: bool = True,
) -> Hashable:
    return (
        in_element_group.discretization_key(),
        out_element_group.discretization_key(),
        in_element_group == out_element_group,
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

    array_tags = ()
    # Reference matrices have output/input DOF axes, but no element axis.
    axis_tags = {
        0: (DiscretizationDOFAxisTag(),),
        1: (DiscretizationDOFAxisTag(),),
    }

    if (
        enable_sum_factorization
        and isinstance(in_element_group, TensorProductElementGroupBase)
        and isinstance(out_element_group, TensorProductElementGroupBase)
    ):
        matrix = _make_tensor_product_mass_operator(
            actx, in_element_group, out_element_group,
            array_tags=array_tags, axis_tags=axis_tags,
        )

        return (matrix,) * out_element_group.dim

    return _make_dense_mass_operator(
        actx, in_element_group, out_element_group,
        array_tags=array_tags, axis_tags=axis_tags,
    )


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

    array_tags = ()
    axis_tags = {
        0: (DiscretizationDOFAxisTag(),),
        1: (DiscretizationDOFAxisTag(),),
    }

    if enable_sum_factorization and isinstance(
        element_group, TensorProductElementGroupBase
    ):
        matrix = _make_tensor_product_mass_operator(
            actx, element_group, element_group,
            array_tags=array_tags, axis_tags=axis_tags,
            inverse=True,
        )

        return (matrix,) * element_group.dim

    return _make_dense_mass_operator(
        actx, element_group, element_group,
        array_tags=array_tags, axis_tags=axis_tags,
        inverse=True,
    )

# }}}

# {{{ face mass operators


def _dense_face_mass_operator_key(
    in_element_group: NodalElementGroupBase,
    out_element_group: InterpolatoryElementGroupBase,
    *,
    array_tags: Sequence[Tag],
    axis_tags: Mapping[int, Sequence[Tag]],
) -> Hashable:
    return (
        _make_dense_operator_key(
            in_element_group,
            out_element_group,
            array_tags=array_tags,
            axis_tags=axis_tags,
        ),
    )


@keyed_memoize_on_first_arg(_dense_face_mass_operator_key)
def _make_dense_face_mass_operator(
    actx: ArrayContext,
    in_element_group: NodalElementGroupBase,
    out_element_group: InterpolatoryElementGroupBase,
    *,
    array_tags: Sequence[Tag],
    axis_tags: Mapping[int, Sequence[Tag]],
) -> tuple[Array, ...]:
    test_basis = out_element_group.basis_obj()
    faces = mp.faces_for_shape(out_element_group.shape)

    operators = []
    for face in faces:
        if isinstance(in_element_group, InterpolatoryElementGroupBase):
            trial_basis = in_element_group.basis_obj()
            quadrature = mp.quadrature_for_space(
                mp.space_for_shape(
                    face,
                    2 * max(in_element_group.order, out_element_group.order),
                ),
                face,
            )
            matrix = mp.nodal_quadrature_bilinear_form_matrix(
                quadrature=quadrature,
                test_functions=test_basis.functions,
                trial_functions=trial_basis.functions,
                nodal_interp_functions_test=test_basis.functions,
                nodal_interp_functions_trial=trial_basis.functions,
                input_nodes=in_element_group.unit_nodes,
                output_nodes=out_element_group.unit_nodes,
                test_function_node_map=face.map_to_volume,
            )
        else:
            quadrature = in_element_group.quadrature_rule()
            if quadrature.exact_to < in_element_group.order:
                raise ValueError(
                    "face quadrature does not meet the input group order"
                )
            matrix = mp.nodal_quadrature_test_matrix(
                quadrature=quadrature,
                test_functions=test_basis.functions,
                nodal_interp_functions=test_basis.functions,
                nodes=out_element_group.unit_nodes,
                test_function_node_map=face.map_to_volume,
            )

        operators.append(
            _tag_and_freeze_operator(actx, matrix, array_tags, axis_tags)
        )

    return tuple(operators)


@keyed_memoize_on_first_arg(_tensor_product_operator_key)
def _make_tensor_product_face_mass_operator(
    actx: ArrayContext,
    in_element_group: TensorProductElementGroupBase,
    out_element_group: TensorProductElementGroupBase,
    *,
    array_tags: Sequence[Tag],
    axis_tags: Mapping[int, Sequence[Tag]],
) -> tuple[tuple[Array, ...], ...]:
    test_basis = out_element_group.basis_obj()
    trial_basis = in_element_group.basis_obj()
    assert isinstance(test_basis, mp.TensorProductBasis)
    assert isinstance(trial_basis, mp.TensorProductBasis)
    # Meshmode's supported tensor-product groups are isotropic.
    test_basis = test_basis.bases[0]
    trial_basis = trial_basis.bases[0]
    quadrature = mp.LegendreGaussQuadrature(
        max(in_element_group.order, out_element_group.order), force_dim_axis=True
    )
    tangential = {}
    boundary = {}
    for sign in (-1, 1):
        matrix = mp.nodal_quadrature_bilinear_form_matrix(
            quadrature=quadrature,
            test_functions=test_basis.functions,
            trial_functions=trial_basis.functions,
            nodal_interp_functions_test=test_basis.functions,
            nodal_interp_functions_trial=trial_basis.functions,
            input_nodes=in_element_group.unit_nodes_1d,
            output_nodes=out_element_group.unit_nodes_1d,
            test_function_node_map=lambda nodes, sign=sign: sign * nodes,
        )
        tangential[sign] = _tag_and_freeze_operator(
            actx, np.ascontiguousarray(matrix), array_tags, axis_tags
        )
        matrix = mp.resampling_matrix(
            test_basis.functions, np.array([[float(sign)]]),
            out_element_group.unit_nodes_1d,
        ).T
        boundary[sign] = _tag_and_freeze_operator(
            actx, np.ascontiguousarray(matrix), array_tags, axis_tags
        )

    operators = []
    for face in mp.faces_for_shape(out_element_group.shape):
        mapped = face.map_to_volume(np.column_stack((
            np.zeros(in_element_group.dim), np.eye(in_element_group.dim),
        )))
        origin = mapped[:, 0]
        directions = mapped[:, 1:] - origin[:, np.newaxis]
        factors = []
        for axis in range(out_element_group.dim):
            face_axes = np.flatnonzero(directions[axis])
            if len(face_axes) == 0:
                factors.append(boundary[int(origin[axis])])
            else:
                factors.append(tangential[int(directions[axis, face_axes[0]])])
        operators.append(tuple(factors))
    return tuple(operators)


def _face_mass_discretization_key(
    in_element_group: ElementGroupBase,
    out_element_group: ElementGroupBase,
    *,
    enable_sum_factorization: bool = True,
) -> Hashable:
    return (
        _operator_discretization_key(
            in_element_group, out_element_group,
            enable_sum_factorization=enable_sum_factorization,
        ),
    )


@keyed_memoize_on_first_arg(_face_mass_discretization_key)
def make_face_mass_operator(
    actx: ArrayContext,
    in_element_group: ElementGroupBase,
    out_element_group: ElementGroupBase,
    *,
    enable_sum_factorization: bool = True,
) -> tuple[Array, ...] | tuple[tuple[Array, ...], ...]:
    """Return one dense matrix or tuple of 1D factors for each face.

    Dense matrices have shape ``(volume_dofs, face_dofs)``. Tensor-product
    factors are in volume-axis order, including a normal endpoint column.
    Surface metrics, input-axis alignment, and face accumulation belong to
    the caller.
    """
    if not isinstance(in_element_group, NodalElementGroupBase):
        raise TypeError(
            f"'in_element_group' must be nodal: {type(in_element_group)}"
        )
    if not isinstance(out_element_group, InterpolatoryElementGroupBase):
        raise TypeError(
            f"'out_element_group' must be interpolatory: {type(out_element_group)}"
        )
    if in_element_group.dim + 1 != out_element_group.dim:
        raise ValueError("face mass requires a codimension-one input group")

    if (enable_sum_factorization
            and isinstance(in_element_group, TensorProductElementGroupBase)
            and isinstance(out_element_group, TensorProductElementGroupBase)
            and in_element_group.dim > 0):
        return _make_tensor_product_face_mass_operator(
            actx, in_element_group, out_element_group,
            array_tags=(),
            axis_tags={0: (DiscretizationDOFAxisTag(),),
                       1: (DiscretizationDOFAxisTag(),)},
        )

    return _make_dense_face_mass_operator(
        actx,
        in_element_group,
        out_element_group,
        array_tags=(),
        axis_tags={
            0: (DiscretizationDOFAxisTag(),),
            1: (DiscretizationDOFAxisTag(),),
        },
    )

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

    if (
        enable_sum_factorization
        and isinstance(in_element_group, TensorProductElementGroupBase)
        and isinstance(out_element_group, TensorProductElementGroupBase)
    ):
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
        axis_tags = {
            0: (DiscretizationDOFAxisTag(),),
            1: (DiscretizationDOFAxisTag(),),
        }

        diff_mat = mp.diff_matrices(
            basis,
            out_element_group.unit_nodes_1d,
            from_nodes=in_element_group.unit_nodes_1d,
        )[0]

        matrix = _tag_and_freeze_operator(actx, diff_mat, array_tags, axis_tags)

        # None => identity matrix. Skip rather than construct + apply identity
        return tuple(
            tuple(
                matrix if axis == derivative_axis else None
                for axis in range(in_element_group.dim)
            )
            for derivative_axis in range(in_element_group.dim)
        )

    else:
        array_tags = ()
        axis_tags = {
            0: (DiscretizationDOFAxisTag(),),
            1: (DiscretizationDOFAxisTag(),),
        }

        matrices = mp.diff_matrices(
            in_element_group.basis_obj(),
            out_element_group.unit_nodes,
            from_nodes=in_element_group.unit_nodes,
        )

        return tuple(
            _tag_and_freeze_operator(actx, matrix, array_tags, axis_tags)
            for matrix in matrices
        )

# }}}

# {{{ weak differentiation operators


@keyed_memoize_on_first_arg(_tensor_product_operator_key)
def _make_tensor_product_stiffness_t_operator(
    actx: ArrayContext,
    in_element_group: TensorProductElementGroupBase,
    out_element_group: TensorProductElementGroupBase,
    *,
    array_tags: Sequence[Tag],
    axis_tags: Mapping[int, Sequence[Tag]],
) -> Array:

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
        stiffness_t = mp.nodal_quadrature_bilinear_form_matrix(
            quadrature=quadrature,
            test_functions=test_basis.derivatives(0),
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

        stiffness_t = mp.nodal_quadrature_test_matrix(
            quadrature=quadrature,
            test_functions=test_basis.derivatives(0),
            nodal_interp_functions=test_basis.functions,
            nodes=out_nodes,
        )

    return _tag_and_freeze_operator(actx, stiffness_t, array_tags, axis_tags)


@keyed_memoize_on_first_arg(_make_dense_operator_key)
def _make_dense_stiffness_t_operator(
    actx: ArrayContext,
    in_element_group: NodalElementGroupBase,
    out_element_group: InterpolatoryElementGroupBase,
    *,
    array_tags: Sequence[Tag],
    axis_tags: Mapping[int, Sequence[Tag]],
) -> Array:

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
        matrices = [
            mp.nodal_quadrature_bilinear_form_matrix(
                quadrature=quadrature,
                test_functions=test_basis.derivatives(rst_axis),
                trial_functions=trial_basis.functions,
                nodal_interp_functions_test=test_basis.functions,
                nodal_interp_functions_trial=trial_basis.functions,
                input_nodes=in_nodes,
                output_nodes=out_nodes,
            )
            for rst_axis in range(out_element_group.dim)
        ]
    else:
        quadrature = in_element_group.quadrature_rule()
        matrices = [
            mp.nodal_quadrature_test_matrix(
                quadrature=quadrature,
                test_functions=test_basis.derivatives(rst_axis),
                nodal_interp_functions=test_basis.functions,
                nodes=out_nodes,
            )
            for rst_axis in range(out_element_group.dim)
        ]

    return _tag_and_freeze_operator(actx, np.asarray(matrices), array_tags, axis_tags)


@keyed_memoize_on_first_arg(_operator_discretization_key)
def make_stiffness_t_operator(
    actx: ArrayContext,
    in_element_group: ElementGroupBase,
    out_element_group: ElementGroupBase,
    *,
    enable_sum_factorization: bool = True,
) -> Array | tuple[tuple[Array, ...], ...]:

    if not isinstance(in_element_group, NodalElementGroupBase):
        raise TypeError(
            f"'in_element_group' must be nodal: {type(in_element_group)}"
        )
    if not isinstance(out_element_group, InterpolatoryElementGroupBase):
        raise TypeError(
            f"'out_element_group' must be interpolatory: {type(out_element_group)}"
        )

    if (
        enable_sum_factorization
        and isinstance(in_element_group, TensorProductElementGroupBase)
        and isinstance(out_element_group, TensorProductElementGroupBase)
    ):
        array_tags_st = ()
        axis_tags_st = {
            0: (DiscretizationDOFAxisTag(),),
            1: (DiscretizationDOFAxisTag(),),
        }

        stiffness_t = _make_tensor_product_stiffness_t_operator(
            actx, in_element_group, out_element_group,
            array_tags=array_tags_st, axis_tags=axis_tags_st,
        )

        array_tags_mass = ()
        axis_tags_mass = {
            0: (DiscretizationDOFAxisTag(),),
            1: (DiscretizationDOFAxisTag(),),
        }

        mass = _make_tensor_product_mass_operator(
            actx, in_element_group, out_element_group,
            array_tags=array_tags_mass, axis_tags=axis_tags_mass,
        )

        return tuple(
            tuple(stiffness_t if axis == derivative_axis else mass
                  for axis in range(out_element_group.dim))
            for derivative_axis in range(out_element_group.dim)
        )

    array_tags = ()
    # The leading axis selects a reference derivative, not an element or DOF.
    axis_tags = {
        0: (DiscretizationTopologicalDimAxisTag(),),
        1: (DiscretizationDOFAxisTag(),),
        2: (DiscretizationDOFAxisTag(),),
    }

    return _make_dense_stiffness_t_operator(
        actx, in_element_group, out_element_group,
        array_tags=array_tags, axis_tags=axis_tags,
    )

# }}}
