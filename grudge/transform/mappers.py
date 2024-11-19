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


import pytato as pt
from pytato.array import Einsum
from pytato.transform import CombineMapper, CopyMapperWithExtraArgs

from grudge.transform.metadata import (
    TensorProductMassOperatorInverseTag,
    TensorProductMassOperatorTag,
    TensorProductStiffnessOperatorTag,
)


# {{{ utilities


class MassCounter(CombineMapper):
    def combine(self, *n_list):
        return sum(n_list)

    def map_einsum(self, expr):
        acc = 0
        for arg in expr.args:
            if arg.tags_of_type(TensorProductMassOperatorTag):
                acc += 1
            acc += self.rec(arg)

        return acc


class MassInverseCounter(CombineMapper):
    def combine(self, *n_list):
        return sum(n_list)

    def map_einsum(self, expr):
        acc = 0
        for arg in expr.args:
            if arg.tags_of_type(TensorProductMassOperatorInverseTag):
                acc += 1
            acc += self.rec(arg)

        return acc

# }}}


# {{{ algebraic dag rewrites

class MassInverseTimesStiffnessSimplifier(CopyMapperWithExtraArgs):
    """
    This is part three of a three part transformation pipeline.

    Creates a new operator that is the result of a mass inverse times weak
    derivative operator and replaces all instances of these two operators in
    each einsum with the new operator.

    See `InverseMassDistributor` for background.
    """
    def map_einsum(self, expr, *args, **kwargs):
        iarg_stiffness = None
        iarg_mass_inverse = None

        new_args = []
        for iarg, arg in enumerate(expr.args):
            if arg.tags_of_type(TensorProductMassOperatorInverseTag):
                iarg_mass_inverse = iarg
            elif arg.tags_of_type(TensorProductStiffnessOperatorTag):
                iarg_stiffness = iarg
            new_args.append(self.rec(arg))

        # create a new operator using mass inverse times stiffness
        if iarg_stiffness is not None and iarg_mass_inverse is not None:
            iarg_data = list(set(
                range(len(expr.args))) - {iarg_stiffness, iarg_mass_inverse})
            assert len(iarg_data) == 1
            iarg_data = iarg_data[0]

            stiffness = expr.args[iarg_stiffness]
            mass_inverse = expr.args[iarg_mass_inverse]
            data = self.rec(expr.args[iarg_data])

            d = mass_inverse @ stiffness

            from grudge.transform.metadata import AxisIgnoredForPropagationTag
            d_axes = []
            for ax in d.axes:
                d_axes.append(ax.tagged(AxisIgnoredForPropagationTag()))
            d = d.copy(axes=tuple(d_axes))

            new_args = [d, data]
            new_access_descriptors = [
                expr.access_descriptors[iarg_stiffness],
                expr.access_descriptors[iarg_data]]

            return Einsum(
                tuple(new_access_descriptors),
                tuple(new_args),
                axes=expr.axes,
                redn_axis_to_redn_descr=expr.redn_axis_to_redn_descr,
                tags=expr.tags,
                non_equality_tags=expr.non_equality_tags)

        return expr.copy(args=tuple(new_args))


class RedundantMassTimesMassInverseRemover(CopyMapperWithExtraArgs):
    """
    This is part two of a three-part transformation pipeline. Removes einsum
    nodes that contain both a mass inverse and mass operator.

    See `InverseMassDistributor` for more information.
    """
    def map_einsum(self, expr, *args, **kwargs):
        found_mass = False
        found_mass_inverse = False
        new_args = []
        for arg in expr.args:
            if arg.tags_of_type(TensorProductMassOperatorInverseTag):
                found_mass_inverse = True
            elif arg.tags_of_type(TensorProductMassOperatorTag):
                found_mass = True
            new_args.append(self.rec(arg))

        if found_mass and found_mass_inverse:
            for arg in expr.args:
                if not (arg.tags_of_type(TensorProductMassOperatorInverseTag) or
                        arg.tags_of_type(TensorProductMassOperatorTag)):
                    return self.rec(arg)

        return expr.copy(args=tuple(new_args))


class InverseMassDistributor(CopyMapperWithExtraArgs):
    r"""
    Implements one part of a three-part transformation pipeline to realize an
    algebraic simplification arising in tensor-product discretizations.

    Specifically, one can represent a weak derivative operator associated with
    a tensor-product discretization as a Kronecker product of a 1D weak
    derivative operator with a variable number of 1D mass matrices, which
    depends on the dimension of the problem.

    Let $S$ be the full weak operator, $\hat{S}$ as the 1D weak derivative
    operator and $\hat{M}$ as the 1D mass operator. For example, consider a 2D
    tensor-product discretization. The weak $x$-derivative operator can be
    expressed as

    ..math::

        S_x = \hat{S} \otimes \hat{M}

    Since we are using a tensor-product discretization, the mass operator can be
    decomposed as a tensor-product of a variable number of mass operators.
    Hence, the mass inverse operator can also be expressed via a tensor-product.

    ..math::

    M^{-1} S_x^k = (\hat{M}^{-1} \otimes \hat{M}^{-1})(\hat{S} \otimes \hat{M})

    By associativity of the tensor-product,

    .. math::

        M^{-1} S_x^k = \hat{M}^{-1}\hat{S} \otimes \hat{M}^{-1}\hat{M}

    Thus, we can instead apply the operator as

    ..math::

        M^{-1} S_x^k = \hat{M}^{-1}\hat{S} \otimes \hat{I}

    where $\hat{I}$ is a 1D identity operator. This results in both a reduction
    in the total number of operations and the required storage for the
    operators.

    This transformation takes care of the distribution of the mass inverse
    operator through the DAG to other tensor-product application routines
    Moreover, the mass inverse is distribtued to the face mass terms (if
    included in the original einsum), and properly reshapes to and from
    tensor-product form to apply the 1D mass operator.
    """
    def map_einsum(self, expr, *args, **kwargs):
        new_args = []
        new_access_descrs = []
        for iarg, arg in enumerate(expr.args):
            # we assume that the 0th argument will be the one with this tag
            if arg.tags_of_type(TensorProductMassOperatorInverseTag):
                return self.rec(
                    expr.args[1], arg, expr.access_descriptors[iarg])
            else:
                if len(args) > 0:
                    new_args.append(self.rec(arg, args[0], args[1]))
                else:
                    new_args.append(self.rec(arg))
            new_access_descrs.append(expr.access_descriptors[iarg])

        if len(args) > 0:
            from pytato.analysis import is_einsum_similar_to_subscript
            if is_einsum_similar_to_subscript(expr, "ifj,fej,fej->ei"):
                nfaces, nelts, _ = expr.args[1].shape
                ndofs = expr.shape[1]

                from math import ceil
                dim = ceil(nfaces/2)
                ndofs_1d = ceil(ndofs**(1/dim))
                expr = expr.reshape(nelts, *(ndofs_1d,)*dim)

                for axis in range(dim):
                    operator_spec = "ij"
                    data_spec = f"e{"abcd"[:axis]}j{"opqr"[:dim-axis-1]}"
                    out_spec = f"e{"abcd"[:axis]}i{"opqr"[:dim-axis-1]}"
                    spec = operator_spec + "," + data_spec + "->" + out_spec

                    expr = pt.einsum(spec, args[0], expr)

                return expr.reshape(nelts, ndofs)
            else:
                new_args.append(args[0])
                new_access_descrs.append(args[1])

        return expr.copy(args=tuple(new_args),
                         access_descriptors=tuple(new_access_descrs))

# }}}
