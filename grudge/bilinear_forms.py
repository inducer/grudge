from __future__ import annotations

__copyright__ = """
Copyright (C) 2024 Addison Alvey-Blanco
Copyright (C) 2024 University of Illinois Board of Trustees
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


from arraycontext import Array, ArrayContext, ArrayOrContainer, tag_axes

from collections.abc import Callable

from dataclasses import dataclass

from grudge.discretization import DiscretizationCollection
from grudge.dof_desc import (
    dof_desc,
    DOFDesc
)
from grudge.geometry import (
    area_element,
    inverse_surface_metric_derivative_mat
)
from grudge.tools import rec_map_subarrays
from grudge.transform.metadata import (
    ReferenceTensorProductMassOperatorTag,
    TensorProductDOFAxisTag,
    TensorProductOperatorAxisTag
)

from meshmode.discretization import (
    ElementGroupBase,
    InterpolatoryElementGroupBase,
    NodalElementGroupBase
)
from meshmode.discretization.poly_element import (
    PolynomialSimplexElementGroupBase,
    TensorProductElementGroupBase
)
from meshmode.dof_array import DOFArray
from meshmode.transform_metadata import (
    DiscretizationDOFAxisTag,
    DiscretizationElementAxisTag,
)

import modepy as mp
from modepy.spaces import TensorProductSpace

import numpy as np
import numpy.linalg as la

from typing import Optional


@dataclass
class _BilinearForm:
    """
    An abstract representation of a bilinear form.

    .. autoattribute:: actx

    .. autoattribute:: input_group

    .. autoattribute:: output_group

    .. autoattribute:: area_element

    .. autoattribute:: metric_terms

    .. autoattribute:: test_derivative

    """

    actx: ArrayContext
    """
    An :class:`~arraycontext.ArrayContext` that should be used
    """

    input_group: ElementGroupBase
    """
    An :class:`~meshmode.poly_element.ElementGroupBase` on which the input
    data is expected to live.
    """

    output_group: InterpolatoryElementGroupBase
    """
    An :class:`~meshmode.poly_element.InterpolatoryElementGroupBase` on which
    the result of applying the operator will live.
    """

    area_element: DOFArray
    """
    Area scaling term found by computing the determinant of the inverse of the
    Jacobian of the mapping from the reference to physical space.
    """

    metric_terms: DOFArray
    """
    Chain rule terms arising from taking derivatives of functions whose
    arguments are the inverse mapping from the reference space to physical
    space.
    """

    test_derivative: int | None
    """
    An integer signifying the physical axis to differentiate the test functions
    with respect to or None signifying that no derivatives should be taken.
    """


    def _build_discrete_reference_operator(self):
        """
        Use quadrature evaluate the discrete reference operator.
        """
        raise NotImplementedError("Subclasses should implement how discrete "
                                  "operators are built")

    def __call__(self, vec: DOFArray) -> DOFArray:
        """
        Apply the discrete operator elementwise to the data in *vec*.
        """
        raise NotImplementedError("Subclasses should implement how discrete "
                                  "operators are applied")


class _NonTensorProductBilinearForm(_BilinearForm):
    """
    Represents a bilinear form whose corresponding discrete operator cannot (or
    should not) be applied as a sequence of tensor contractions using a 1D
    operator. This is the default for discretizations using simplicial elements.

    If it is requested to *not* use fast operator evaluation with a
    tensor-product discretization, then this is used instead.
    """

    operator: Array

    def __init__(self,
                 actx: ArrayContext,
                 input_group: NodalElementGroupBase,
                 output_group: InterpolatoryElementGroupBase,
                 area_element: DOFArray,
                 metric_terms: DOFArray,
                 test_derivative: int | None = None):

        super().__init__(actx, input_group, output_group, area_element,
                         metric_terms, test_derivative=test_derivative)

        self._build_discrete_reference_operator()

    def _build_discrete_reference_operator(self):
        assert isinstance(self.input_group, NodalElementGroupBase)

        # {{{ get basis, nodes, quadrature

        test_basis = self.output_group.basis_obj()
        out_nodes = self.output_group.unit_nodes

        quad_rule = self.input_group.quadrature_rule()

        # }}}

        # {{{ compute the operator using quadrature

        vdm_out = mp.vandermonde(test_basis.functions, out_nodes)

        # construct stiffness operator
        if self.test_derivative is not None:
            vdm_in = mp.multi_vandermonde(test_basis.gradients, quad_rule.nodes)

            # r is reference partial derivative direction
            stiffness_operator = np.einsum(
                "ki,rjk,j->rij",
                la.inv(vdm_out),
                vdm_in,
                quad_rule.weights)

            self.operator = self.actx.freeze(
                self.actx.tag_axis(
                    1,
                    DiscretizationDOFAxisTag(),
                    self.actx.from_numpy(stiffness_operator)))

        # construct mass operator
        else:
            vdm_in = mp.vandermonde(test_basis.functions, quad_rule.nodes)
            mass_operator = np.einsum(
                "ki,jk,j->ij",
                la.inv(vdm_out),
                vdm_in,
                quad_rule.weights)

            self.operator = self.actx.freeze(
                self.actx.tag_axis(
                    0,
                    DiscretizationDOFAxisTag(),
                    self.actx.from_numpy(mass_operator)))

        # }}}

    def __call__(self, vec: DOFArray) -> DOFArray:
        """
        Implements operator application as a matvec. Pointwise multiplication by
        area scaling and metric terms happens before operator application.
        """

        # compute partial derivative
        if self.test_derivative is not None:
            return self.actx.einsum(
                "rij,rej->ei",
                self.operator,
                vec*self.metric_terms[self.test_derivative]*self.area_element,
                arg_names=(f"stiffness_T_{self.test_derivative}", "vec"))

        # apply mass operator
        else:
            return self.actx.einsum(
                "ij,ej->ei",
                self.operator,
                vec * self.area_element,
                arg_names=("mass_operator", "vec"))


class _TensorProductBilinearForm(_BilinearForm):
    r"""
    Represents a bilinear form whose corresponding discrete operator can be
    applied as a sequence of tensor contractions using a 1D operator. This is
    the default for discretizations using tensor-product elements.

    .. autoattribute:: mass_operator

    .. autoattribute:: differentiation_operator
    """

    mass_operator: Array
    """
    An :class:`~arraycontext.Array` representing a 1D mass operator. This
    operator is always constructed.
    """

    differentiation_operator: Optional[Array]
    """
    An :class:`~arraycontext.Array` representing a 1D stiffness operator. This
    operator is only constructed if derivatives are requested.
    """

    def __init__(self,
                 actx: ArrayContext,
                 input_group: TensorProductElementGroupBase,
                 output_group: TensorProductElementGroupBase,
                 area_element: DOFArray,
                 metric_terms: DOFArray,
                 test_derivative: int | None = None):

        super().__init__(actx, input_group, output_group, area_element,
                         metric_terms, test_derivative=test_derivative)

        self._build_discrete_reference_operator()

    def _build_discrete_reference_operator(self):
        assert isinstance(self.input_group, TensorProductElementGroupBase)
        assert isinstance(self.output_group, TensorProductElementGroupBase)

        # {{{ get 1D nodes, weights

        test_basis_1d = self.output_group.basis_obj().bases[0]

        out_nodes_1d = self.output_group.unit_nodes[:self.output_group.order+1]
        out_nodes_1d = out_nodes_1d.reshape(-1, 1)

        quad_rule_1d = self.input_group.quadrature_rule().quadratures[0]
        in_nodes_1d = quad_rule_1d.nodes
        weights_1d = quad_rule_1d.weights

        # }}}

        # {{{ build interpolation operators

        vdm_out = mp.vandermonde(test_basis_1d.functions, out_nodes_1d)
        vdm_in = mp.vandermonde(test_basis_1d.functions, in_nodes_1d)

        # }}}

        # {{{ build 1D mass, optionally 1D stiffness

        axes_to_tags = {
            0: TensorProductOperatorAxisTag(),
            1: TensorProductOperatorAxisTag()
        }

        mass_1d = np.einsum(
            "ki,jk,j->ij",
            la.inv(vdm_out),
            vdm_in,
            weights_1d)

        self.mass_operator = self.actx.freeze(
            self.actx.tag(
                ReferenceTensorProductMassOperatorTag(),
                tag_axes(
                    self.actx,
                    axes_to_tags,
                    self.actx.from_numpy(mass_1d))))

        if self.test_derivative is not None:
            vdm_p_in = mp.multi_vandermonde(test_basis_1d.gradients,
                                            in_nodes_1d)[0]

            stiff_1d = np.einsum(
                "ki,jk,j->ij",
                la.inv(vdm_out),
                vdm_p_in,
                weights_1d)

            self.differentiation_operator = self.actx.freeze(
                tag_axes(
                    self.actx,
                    axes_to_tags,
                    self.actx.from_numpy(stiff_1d)))

        # }}}

    def _single_axis_contraction(self,
                                 dim: int,
                                 axis: int,
                                 data: DOFArray,
                                 operator: Array,
                                 tagged=None, arg_names=None):
        """
        Generic routine to apply a 1D operator to a particular axis of *data*.

        The einsum specification is constructed based on the dimension of the
        problem and can support up to 1 reduction axis and 22 non-reduction
        (DOF) axes. The element axis is not counted since it is reserved.
        """

        data = tag_axes(
            self.actx,
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
            self.actx,
            {
                i: (DiscretizationElementAxisTag() if i == 0 else
                    TensorProductDOFAxisTag(i-1))
                for i in range(dim+1)
            },
            self.actx.einsum(spec, operator, data, arg_names=arg_names,
                        tagged=tagged))

    def _apply_mass_operator(self, vec: DOFArray, exclude_axis=None) -> DOFArray:
        """
        Apply a mass operator to all axes other than *exclude_axis*.

        Since a mass operator is applied even in the case of differentiation, we
        exclude the axis that the differentiation operator is being applied to.
        """
        apply_mass_axes = set(np.arange(self.input_group.dim)) - {exclude_axis}
        for ax in apply_mass_axes:
            vec = self._single_axis_contraction(
                self.input_group.dim, ax, vec, self.mass_operator,
                arg_names=("mass_1d", "vec"))

        return vec

    def _compute_partial_derivative(self, vec: DOFArray) -> DOFArray:
        """
        Compute the physical space partial derivative using a 1D differentiation
        operator. This is done by computing n-dimensions worth of reference
        partial derivatives and summing over the results.

        :arg vec: a :class:`~meshmode.dof_array.DOFArray` assumed to have
            already been pointwise multiplied with chain rule and area scaling
            terms. It is also assumed *vec* has been properly reshaped.
        """
        partial_derivative = self.actx.zeros_like(vec[0])
        for rst_axis in range(self.input_group.dim):
            reference_partial = vec[rst_axis]
            reference_partial = self._apply_mass_operator(reference_partial,
                                                          exclude_axis=rst_axis)
            reference_partial = self._single_axis_contraction(
                self.input_group.dim, rst_axis, reference_partial,
                self.mass_operator,
                arg_names=("stiff_1d", f"ref_partial_{rst_axis}"))

            partial_derivative += reference_partial

        return partial_derivative

    def __call__(self, vec):
        """
        Applies the action of the requested operator to *vec* via tensor
        contractions with a 1D operator.

        If a derivative is to be taken, then a 1D mass operator and
        differentiation operator will be applied.
        """

        from modepy.tools import (
            reshape_array_for_tensor_product_space as fold,
            unreshape_array_for_tensor_product_space as unfold
        )

        # apply area scaling
        vec = vec * self.area_element

        # reveal tensor product structure
        if self.input_group.dim != 1:
            assert isinstance(self.input_group.space, TensorProductSpace)
            vec = fold(self.input_group.space, vec)

        # apply the bilinear form
        if self.test_derivative is not None:
            vec = vec * self.metric_terms[self.test_derivative]
            vec = self._compute_partial_derivative(vec)
        else:
            vec = self._apply_mass_operator(vec)

        # hide tensor product structure
        if self.output_group.dim != 1:
            assert isinstance(self.output_group.space, TensorProductSpace)
            return unfold(self.output_group.space, vec)

        return vec


def _dispatch_bilinear_form(
        dcoll: DiscretizationCollection,
        vec: DOFArray,
        dd_in: DOFDesc,
        test_derivative: int | None = None,
        use_tensor_product_fast_eval: bool = True) -> DOFArray:
    """
    An general intermediate routine that dispatches arguments to the correct
    bilinear form appliers. The routine that is chosen is based on desired
    discretization choices and desired optimizations (in the case of
    tensor-product elements).

    This routine serves as a template for special-case routines for common DG
    operators like the mass and stiffness operators.
    """

    # {{{ grab discretizations

    actx = vec.array_context

    input_discr = dcoll.discr_from_dd(dd_in)
    output_discr = dcoll.discr_from_dd(
        dd_in.with_discr_tag(dof_desc.DISCR_TAG_BASE))

    # }}}

    # {{{ get metrics, scaling terms

    metrics = inverse_surface_metric_derivative_mat(actx, dcoll,
        dd=dd_in, times_area_element=False,
        _use_geoderiv_connection=actx.supports_nonscalar_broadcasting)

    area_elements = area_element(actx, dcoll, dd=dd_in,
        _use_geoderiv_connection=actx.supports_nonscalar_broadcasting)

    # }}}

    # {{{ build and apply operators at group granularity

    group_data = []
    for in_group, out_group, vec_i, metric_i, area_elt_i in zip(
        input_discr.groups, output_discr.groups, vec, metrics, area_elements):

        # FIXME: separate out metric terms from area/volume scaling
        if isinstance(in_group, TensorProductElementGroupBase):
            assert isinstance(out_group, TensorProductElementGroupBase)
            if use_tensor_product_fast_eval:
                bilinear_form = _TensorProductBilinearForm(
                    actx, in_group, out_group, metric_i, area_elt_i,
                    test_derivative=test_derivative)
            else:
                bilinear_form = _NonTensorProductBilinearForm(
                    actx, in_group, out_group, metric_i, area_elt_i,
                    test_derivative=test_derivative)

        elif isinstance(in_group, PolynomialSimplexElementGroupBase):
            assert isinstance(out_group, PolynomialSimplexElementGroupBase)
            bilinear_form = _NonTensorProductBilinearForm(
                actx, in_group, out_group, metric_i, area_elt_i,
                test_derivative=test_derivative)

        else:
            raise ValueError(
                "Element groups must either be subclasses of "
                "`TensorProductElemenGroupBase` or `SimplexElemenGroupBase`")

        group_data.append(bilinear_form(vec_i))

    # }}}

    return DOFArray(actx, data=tuple(group_data))


def apply_bilinear_form(
        dcoll: DiscretizationCollection,
        vecs: ArrayOrContainer,
        dd_in: DOFDesc | None = None,
        test_derivative: int | None = None,
        use_tensor_product_fast_eval: bool = True,
        _dispatcher: Callable = _dispatch_bilinear_form) -> ArrayOrContainer:
    r"""
    Applies an element-local operator built by evaluating a bilinear form using
    quadrature to all element-local data contained in *vecs*.

    As an example, the element-local mass operator is given as the bilinear form

    .. math::

        M_{ij}^k = (\phi_j, \psi_i)_{\Omega^k} =
        \int_{\hat{\Omega}} \phi_j \psi_i |J^k| d\hat{\Omega}

    where $\Omega^k$ is the $k$-th element, $phi_j$ are trial functions
    over the reference element, $\psi_i$ are test functions over the reference
    element, and $|J^k|$ is the determinant of the Jacobian of the mapping from
    the reference element to the $k$-th element.

    Applying $M$ to a vector of nodal coefficients $\mathbf{u}$, we have

    .. math::

        (M^{k}\mathbf{u}|_{k})_i = (u|_{k}, \psi_i)_{\Omega^k}

    The discrete inner product is computed using quadrature. If a quadrature
    discretization is available, then quadrature will be computed on that
    discretization and the result will be mapped back to the base
    discretization. Otherwise, a default quadrature rule will be used (as
    determined by :func:`~modepy.quadrature_for_space`. If a quadrature rule
    other than the default is to be used, then *dd_in* must be specified as the
    quadrature :class:`grudge.dof_desc.DOFDesc`.

    This is a top-level routine that dispatches to routines that are specific to
    an element type. Since bilinear forms arising from a tensor-product
    discretization are separable, operators are applied as a sequence of tensor
    contractions. This is not the case for simplices, so operators are applied
    as a sequence of matvecs.

    If derivatives are required on the trial basis, then that must be done
    before passing the DOFs as an argument to this routine.

    :arg dcoll: a :class:`DiscretizationCollection` containing at least a base
        discretization on which the result of applying the operator should live.

    :arg vecs: a :class:`~meshmode.dof_array.DOFArray` or an
        :class:`~arraycontext.ArrayContainer` of them.

    :arg test_derivative: an integer representing the coordinate direction to
        differentiate the test functions against (x = 0, y = 1, ...).

    :arg dd_in: a :class:`~grudge.dof_desc.DOFDesc`, or a value convertible to
        one. Defaults to the base discretization if not provided.

    :arg use_tensor_product_fast_eval: a flag used to turn off fast operator
        evaluation with tensor-product elements. Default is set to True.

    :returns: an object array of :class:`~meshmode.dof_array.DOFArray`\ s or an
        :class:`~arraycontext.ArrayContainer` of object arrays.
    """

    if dd_in is None:
        dd_in = dof_desc.DD_VOLUME_ALL

    return rec_map_subarrays(
        lambda vec: _dispatcher(
            dcoll, dd_in, vec, test_derivative, use_tensor_product_fast_eval),
        in_shape=(), out_shape=(), ary=vecs, scalar_cls=DOFArray)
