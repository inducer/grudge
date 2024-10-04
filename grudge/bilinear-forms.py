from __future__ import annotations
from collections.abc import Callable

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

from meshmode.discretization import InterpolatoryElementGroupBase
from meshmode.discretization.poly_element import (
    PolynomialSimplexElementGroupBase,
    TensorProductElementGroupBase
)
from meshmode.dof_array import DOFArray

from meshmode.transform_metadata import (
    DiscretizationDOFAxisTag,
    DiscretizationTopologicalDimAxisTag
)

import modepy as mp

import numpy as np
import numpy.linalg as la


class BilinearForm:
    """
    An abstract representation of a bilinear form.

    :arg actx: an :class:`~arraycontext.ArrayContext` that should be used

    :arg input_discretization: an
        :class:`~meshmode.discretization.Discretization` on which the input data
        is expected to live. A case where the input and output discretization
        will differ is in the case that a particular quadrature domain is used.

    :arg output_discretization: an
        :class:`~meshmode.discretization.Discretization` on which the result of
        applying the operator will live.

    :arg trial_basis: a :class:`~modepy.Basis` that the trial solution is in.

    :arg test_basis: a :class:`~modepy.Basis` that the test solution is in. A
        common place for the test and trial basis to differ is in the
        application of the face mass operator.

    :arg test_derivative: an integer signifying the axis to differentiate the
        test functions with respect to or None signifying that no derivatives
        should be taken

    :arg trial_derivative: an integer signifying the axis to differentiate the
        trial functions with respect to or None signifying that no derivatives
        should be taken
    """
    actx: ArrayContext
    input_group: InterpolatoryElementGroupBase
    output_group: InterpolatoryElementGroupBase
    metrics: DOFArray
    test_derivative: int | None
    trial_derivative: int | None

    operator: Array

    def __init__(self,
                 actx: ArrayContext,
                 input_group: InterpolatoryElementGroupBase,
                 output_group: InterpolatoryElementGroupBase,
                 metrics: DOFArray,
                 trial_derivative: int | None = None,
                 test_derivative: int | None = None):

        self.actx = actx
        self.input_group = input_group
        self.output_group = output_group
        self.trial_derivative = trial_derivative
        self.test_derivative = test_derivative

        self._metrics = metrics

    def _build_discrete_reference_operator(self):
        """
        Use quadrature evaluate the discrete reference operator.
        """
        raise NotImplementedError("Subclasses should implement how discrete "
                                  "operators are built")

    def apply_to(self, vec: DOFArray) -> DOFArray:
        """
        Apply the discrete operator elementwise to the data in *vec*.
        """
        raise NotImplementedError("Subclasses should implement how discrete "
                                  "operators are applied")

    def __call__(self, vec: DOFArray) -> DOFArray:
        return self.apply_to(vec)


class NonseparableBilinearForm(BilinearForm):
    """
    Represents a bilinear form whose corresponding discrete operator cannot be
    applied as a sequence of tensor contractions using a 1D operator. Typically,
    this is the case with bilinear forms arising from a simplicial
    discretization.

    If it is requested to *not* use fast operator evaluation with a
    tensor-product discretization, then this is used instead.
    """

    def __init__(self,
                 actx: ArrayContext,
                 input_group: InterpolatoryElementGroupBase,
                 output_group: InterpolatoryElementGroupBase,
                 metrics: DOFArray,
                 trial_derivative: int | None = None,
                 test_derivative: int | None = None):

        super().__init__(actx, input_group, output_group, metrics,
                         trial_derivative=trial_derivative,
                         test_derivative=test_derivative)

        # {{{ build the discrete operator

        self._build_discrete_reference_operator()

        # }}}

    def _build_discrete_reference_operator(self):

        # {{{ vandermonde matrices

        if self.test_derivative is not None:
            vdm_out = mp.multi_vandermonde(
                self.output_group.basis_obj().gradients,
                self.output_group.unit_nodes)
        else:
            vdm_out = mp.vandermonde(self.output_group.basis_obj().functions,
                                     self.output_group.unit_nodes)

        if self.trial_derivative is not None:
            vdm_in = mp.multi_vandermonde(
                self.input_group.basis_obj().gradients,
                self.input_group.unit_nodes)
        else:
            vdm_in = mp.vandermonde(self.input_group.basis_obj().functions,
                                    self.input_group.unit_nodes)

        # }}}

        # {{{ apply weights and form the operator

        weights = self.input_group.quadrature_rule().weights

        if self.trial_derivative is not None:
            operator = np.einsum(
                "kj,rij,i->rki",
                la.inv(vdm_out).T,
                vdm_in,
                weights)

            axes_to_tags = {
                0: DiscretizationTopologicalDimAxisTag(),
                1: DiscretizationDOFAxisTag(),
                2: DiscretizationDOFAxisTag()
            }

        else:
            operator = np.einsum(
                "kj,ij,i->ki",
                la.inv(vdm_out).T,
                vdm_in,
                weights)

            axes_to_tags = {
                0: DiscretizationDOFAxisTag(),
                1: DiscretizationDOFAxisTag()
            }

        # }}}

        self.operator = self.actx.freeze(
            tag_axes(self.actx, axes_to_tags, self.actx.from_numpy(operator)))

    def apply_to(self, vec: DOFArray) -> DOFArray:
        """
        Implements operator application as a matvec (plus metric term
        application).
        """

        if self.trial_derivative is not None:
            return self.actx.einsum(
                "rij,rej->ei",
                self.operator,
                vec * self.metrics[self.trial_derivative],
                arg_names=("operator", "vec"))
        else:
            return self.actx.einsum(
                "ij,ej->ei",
                self.operator,
                vec * self.metrics,
                arg_names=("operator", "vec"))


class SeparableBilinearForm(BilinearForm):
    """
    Represents a bilinear form whose corresponding discrete operator can be
    applied as a sequence of tensor contractions using a 1D operator. Typically,
    this is the case with bilinear forms arising from a tensor-product
    discretization.
    """

    def __init__(self,
                 actx: ArrayContext,
                 input_group: TensorProductElementGroupBase,
                 output_group: TensorProductElementGroupBase,
                 metrics: DOFArray,
                 trial_derivative: int | None = None,
                 test_derivative: int | None = None):

        super().__init__(actx, input_group, output_group, metrics,
                         trial_derivative=trial_derivative,
                         test_derivative=test_derivative)


def _dispatch_bilinear_form(
        dcoll: DiscretizationCollection,
        vec: DOFArray,
        dd_in: DOFDesc,
        trial_derivative: int | None = None,
        test_derivative: int | None = None,
        use_tensor_product_fast_eval: bool = True) -> ArrayOrContainer:
    """
    An general intermediate routine that dispatches arguments to the correct
    bilinear form appliers. The routine that is chosen is based on desired
    discretization choices and desired optimizations (in the case of
    tensor-product elements).

    This routine serves as a template for special-case routines for common DG
    operators like the mass and stiffness operators.
    """
    actx = vec.array_context

    input_discr = dcoll.discr_from_dd(dd_in)
    output_discr = dcoll.discr_from_dd(
        dd_in.with_discr_tag(dof_desc.DISCR_TAG_BASE))

    # {{{ get metrics

    # FIXME: for now, always assume weak form
    if test_derivative and trial_derivative is None:
        metrics = area_element(actx, dcoll, dd=dd_in,
            _use_geoderiv_connection=actx.supports_nonscalar_broadcasting)
    else:
        metrics = inverse_surface_metric_derivative_mat(actx, dcoll,
            dd=dd_in,
            times_area_element=True,
            _use_geoderiv_connection=actx.supports_nonscalar_broadcasting)

    # }}}

    # {{{ build and apply operators at group granularity

    group_data = []
    for in_group, out_group, vec_i, metric_i in zip(
        input_discr.groups, output_discr.groups, vec, metrics):

        if isinstance(in_group, TensorProductElementGroupBase):
            assert isinstance(out_group, TensorProductElementGroupBase)
            if use_tensor_product_fast_eval:
                bilinear_form = SeparableBilinearForm(
                    actx, in_group, out_group, metric_i,
                    test_derivative=test_derivative,
                    trial_derivative=trial_derivative)
            else:
                bilinear_form = NonseparableBilinearForm(
                    actx, in_group, out_group, metric_i,
                    test_derivative=test_derivative,
                    trial_derivative=trial_derivative)

        elif isinstance(in_group, PolynomialSimplexElementGroupBase):
            assert isinstance(out_group, PolynomialSimplexElementGroupBase)
            bilinear_form = NonseparableBilinearForm(
                actx, in_group, out_group, metric_i,
                test_derivative=test_derivative,
                trial_derivative=trial_derivative)

        else:
            raise ValueError(
                "Element groups must either be subclasses of "
                "`TensorProductElemenGroupBase` or `SimplexElemenGroupBase`")

        group_data.append(bilinear_form.apply_to(vec_i))

    # }}}

    return DOFArray(actx, data=tuple(group_data))


def apply_bilinear_form(
        dcoll: DiscretizationCollection,
        vecs: ArrayOrContainer,
        dd_in: DOFDesc | None = None,
        trial_derivative: int | None = None,
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

    :arg dcoll: a :class:`DiscretizationCollection` containing at least a base
        discretization on which the result of applying the operator should live.

    :arg vecs: a :class:`~meshmode.dof_array.DOFArray` or an
        :class:`~arraycontext.ArrayContainer` of them.

    :arg trial_derivative: an integer representing the coordinate direction to
        differentiate the trial functions against (x = 0, y = 1, ...).

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
            dcoll, dd_in, vec, test_derivative, trial_derivative,
            use_tensor_product_fast_eval),
        in_shape=(), out_shape=(), ary=vecs, scalar_cls=DOFArray)
