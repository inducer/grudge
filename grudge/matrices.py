"""
Core DG Matrices
^^^^^^^^^^^^^^^^

Strong differentiation
----------------------

.. autofunction:: reference_derivative_matrices

Weak differentiation
--------------------

.. autofunction:: reference_stiffness_transpose_matrices

Mass, inverse mass, and face mass matrices
------------------------------------------

.. autofunction:: reference_mass_matrix
.. autofunction:: reference_inverse_mass_matrix
.. autofunction:: reference_face_mass_matrix
"""

from __future__ import annotations


__copyright__ = """
Copyright (C) 2024 Addison Alvey-Blanco
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

from collections.abc import Mapping

import modepy as mp

import numpy as np
import numpy.linalg as la

from arraycontext import ArrayContext, ArrayOrContainer, tag_axes
from grudge.tools import (
    get_basis_for_face_group,
    get_accurate_quadrature_rule,
    get_element_group_basis,
    get_element_group_nodes,
    get_quadrature_for_face
)
from meshmode.discretization import (
    InterpolatoryElementGroupBase,
    NodalElementGroupBase,
)
from pytools.tag import Tag
from pytools import keyed_memoize_on_first_arg


@keyed_memoize_on_first_arg(
    lambda output_group, input_group, use_tensor_product_fast_eval: (
        input_group.discretization_key(),
        output_group.discretization_key(),
        use_tensor_product_fast_eval
    )
)
def reference_derivative_matrices(
        actx: ArrayContext,
        input_group: InterpolatoryElementGroupBase,
        output_group: NodalElementGroupBase,
        use_tensor_product_fast_eval: bool = True,
        axis_tags: Mapping[int, tuple[Tag]] | None = None,
        ary_tags: tuple[Tag] | None = None,
    ) -> ArrayOrContainer:
    """
    Computes all reference derivative matrices. See
    :func:`~grudge.matrices.reference_derivative_matrix` for more information.
    """

    if axis_tags is None:
        axis_tags = {}  # type: ignore
    if ary_tags is None:
        ary_tags = ()  # type: ignore

    basis = get_element_group_basis(
        input_group, use_tensor_product_fast_eval=use_tensor_product_fast_eval)

    input_nodes = get_element_group_nodes(
        input_group, use_tensor_product_fast_eval=use_tensor_product_fast_eval)

    output_nodes = get_element_group_nodes(
        output_group, use_tensor_product_fast_eval=use_tensor_product_fast_eval)

    return actx.tag(
        ary_tags,
        tag_axes(
            actx,
            axis_tags,
            actx.from_numpy(
                np.asarray(
                    mp.diff_matrices(basis, output_nodes,
                                     from_nodes=input_nodes),
                order="C")
            )
        )
   )


@keyed_memoize_on_first_arg(
    lambda output_group, input_group, use_tensor_product_fast_eval: (
        input_group.discretization_key(),
        output_group.discretization_key(),
        use_tensor_product_fast_eval
    )
)
def reference_stiffness_transpose_matrices(
        actx: ArrayContext,
        input_group: NodalElementGroupBase,
        output_group: InterpolatoryElementGroupBase,
        use_tensor_product_fast_eval: bool = True,
        axis_tags: Mapping[int, tuple[Tag]] | None = None,
        ary_tags: tuple[Tag] | None = None,
    ) -> ArrayOrContainer:

    if axis_tags is None:
        axis_tags = {}  # type: ignore
    if ary_tags is None:
        ary_tags = ()  # type: ignore

    basis = get_element_group_basis(
        output_group, use_tensor_product_fast_eval=use_tensor_product_fast_eval)

    output_nodes = get_element_group_nodes(
        output_group, use_tensor_product_fast_eval=use_tensor_product_fast_eval)

    quadrature = get_accurate_quadrature_rule(
        input_group,
        required_exactness=2*output_group.order,
        use_tensor_product_fast_eval=use_tensor_product_fast_eval
    )

    if use_tensor_product_fast_eval:
        num_matrices = 1
    else:
        num_matrices = input_group.dim

    if input_group == output_group:
        return actx.tag(
            ary_tags,
            tag_axes(
                actx,
                axis_tags,
                actx.from_numpy(
                    np.asarray([
                        mp.nodal_quad_bilinear_form(
                            quadrature=quadrature,
                            test_functions=basis.derivatives(rst_axis),
                            trial_functions=basis.functions,
                            proj_functions=basis.functions,
                            nodes=output_nodes
                        )
                        for rst_axis in range(num_matrices)
                    ], order="C")
                )
            )
        )

    return actx.tag(
        ary_tags,
        tag_axes(
            actx,
            axis_tags,
            actx.from_numpy(
                np.asarray([
                    mp.nodal_quad_operator(
                        quadrature=quadrature,
                        test_functions=basis.derivatives(rst_axis),
                        proj_functions=basis.functions,
                        nodes=output_nodes
                    )
                    for rst_axis in range(num_matrices)
                ], order="C")
            )
        )
    )


@keyed_memoize_on_first_arg(
    lambda output_group, input_group, use_tensor_product_fast_eval: (
        input_group.discretization_key(),
        output_group.discretization_key(),
        use_tensor_product_fast_eval
    )
)
def reference_mass_matrix(
        actx: ArrayContext,
        input_group: NodalElementGroupBase,
        output_group: InterpolatoryElementGroupBase,
        use_tensor_product_fast_eval: bool = False,
        axis_tags: Mapping[int, tuple[Tag]] | None = None,
        ary_tags: tuple[Tag] | None = None
    ) -> ArrayOrContainer:

    if axis_tags is None:
        axis_tags = {}  # type: ignore
    if ary_tags is None:
        ary_tags = ()  # type: ignore

    basis = get_element_group_basis(
        output_group, use_tensor_product_fast_eval=use_tensor_product_fast_eval)

    output_nodes = get_element_group_nodes(
        output_group, use_tensor_product_fast_eval=use_tensor_product_fast_eval)

    quadrature = get_accurate_quadrature_rule(
        input_group,
        required_exactness=2*(output_group.order + 1),
        use_tensor_product_fast_eval=use_tensor_product_fast_eval
    )

    if input_group == output_group:
        return actx.tag(
            ary_tags,
            tag_axes(
                actx,
                axis_tags,
                actx.from_numpy(
                    mp.nodal_quad_bilinear_form(
                        quadrature=quadrature,
                        test_functions=basis.functions,
                        trial_functions=basis.functions,
                        proj_functions=basis.functions,
                        nodes=output_nodes
                    )
                )
            )
        )

    return actx.tag(
        ary_tags,
        tag_axes(
            actx,
            axis_tags,
            actx.from_numpy(
                mp.nodal_quad_operator(
                    quadrature=quadrature,
                    test_functions=basis.functions,
                    proj_functions=basis.functions,
                    nodes=output_nodes
                )
            )
        )
    )


@keyed_memoize_on_first_arg(
    lambda group, use_tensor_product_fast_eval: (
        group.discretization_key(),
        use_tensor_product_fast_eval
    )
)
def reference_inverse_mass_matrix(
        actx: ArrayContext,
        group: InterpolatoryElementGroupBase,
        use_tensor_product_fast_eval: bool = False,
        axis_tags: Mapping[int, tuple[Tag]] | None = None,
        ary_tags: tuple[Tag] | None = None
    ) -> ArrayOrContainer:

    if axis_tags is None:
        axis_tags = {}  # type: ignore
    if ary_tags is None:
        ary_tags = ()  # type: ignore

    basis = get_element_group_basis(
        group, use_tensor_product_fast_eval=use_tensor_product_fast_eval)

    output_nodes = get_element_group_nodes(
        group, use_tensor_product_fast_eval=use_tensor_product_fast_eval)

    quadrature = get_accurate_quadrature_rule(
        group,
        required_exactness=2*(group.order + 1),
        use_tensor_product_fast_eval=use_tensor_product_fast_eval
    )

    return actx.tag(
        ary_tags,
        tag_axes(
            actx,
            axis_tags,
            actx.from_numpy(
                la.inv(
                    mp.nodal_quad_bilinear_form(
                        quadrature=quadrature,
                        test_functions=basis.functions,
                        trial_functions=basis.functions,
                        proj_functions=basis.functions,
                        nodes=output_nodes
                    )
                )
            )
        )
    )


@keyed_memoize_on_first_arg(
    lambda face_group, vol_group, dtype: (
        face_group.discretization_key(),
        vol_group.discretization_key(),
        dtype
    )
)
def reference_face_mass_matrix(
        actx: ArrayContext,
        face_group: NodalElementGroupBase,
        vol_group: InterpolatoryElementGroupBase,
        dtype,
        ary_tags: tuple[Tag] | None = None,
        axis_tags: Mapping[int, Tag] | None = None,
        use_tensor_product_fast_eval: bool = True,
    ) -> ArrayOrContainer:

    if ary_tags is None:
        ary_tags = ()  # type: ignore
    if axis_tags is None:
        axis_tags = {}  # type: ignore

    # FIXME: turn on fast evaluation eventually
    use_tensor_product_fast_eval = False

    face_mass = np.empty(
        (vol_group.nunit_dofs,
        vol_group.mesh_el_group.nfaces,
        face_group.nunit_dofs),
        dtype=dtype
    )

    faces = mp.faces_for_shape(vol_group.shape)

    vol_nodes = get_element_group_nodes(
        vol_group,
        use_tensor_product_fast_eval=use_tensor_product_fast_eval
    )

    face_nodes = get_element_group_nodes(
        face_group,
        use_tensor_product_fast_eval=use_tensor_product_fast_eval
    )

    face_basis = get_basis_for_face_group(
        face_group,
        use_tensor_product_fast_eval=use_tensor_product_fast_eval
    )

    vol_basis = get_element_group_basis(
        vol_group,
        use_tensor_product_fast_eval=use_tensor_product_fast_eval
    )

    for iface, face in enumerate(faces):
        face_quadrature = get_quadrature_for_face(
            face_group,
            face,
            required_exactness=(vol_group.order + face_group.order),
            use_tensor_product_fast_eval=use_tensor_product_fast_eval
        )

        if face_basis is not None:
            face_mass[:, iface, :] = mp.nodal_mass_matrix_for_face(
                face,
                face_quadrature,
                face_basis.functions,
                vol_basis.functions,
                vol_nodes,
                face_nodes
            )

        else:
            face_mass[:, iface, :] = mp.nodal_quad_mass_matrix_for_face(
                face,
                face_quadrature,
                vol_basis.functions,
                vol_nodes
            )

    return actx.tag(
        ary_tags,
        tag_axes(
            actx,
            axis_tags,
            actx.from_numpy(face_mass)
        )
    )

# vim: foldmethod=marker
