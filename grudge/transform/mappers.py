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
from pytato.array import Einsum, EinsumElementwiseAxis
from pytato.transform import CombineMapper, CopyMapperWithExtraArgs

from grudge.transform.metadata import (
    TensorProductMassOperatorInverseTag,
    TensorProductMassOperatorTag,
    TensorProductStiffnessOperatorTag
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


class InverseMassRemover(CopyMapperWithExtraArgs):
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


class InverseMassPropagator(CopyMapperWithExtraArgs):
    r"""
    Applying a full mass inverse operator when using a tensor-product
    discretization results in redundant work. For example, suppose we have a 3D
    tensor-product discretization. Then the weak differentiation operator
    associated with the r-coordinate direction can be expressed as

    .. math::

        K_r = \hat{K} \otimes \hat{M} \otimes \hat{M},

    where :math:`\hat{M}` is a 1D mass operator and :math:`\hat{K}` is a 1D weak
    differentiation operator. Similarly, the inverse mass operator can be
    expressed as

    .. math::

        M^{-1} = \hat{M}^{-1} \otimes \hat{M}^{-1} \otimes \hat{M}^{-1},

    By the properties of the tensor-product, multiplying the weak derivative
    operator on the left by the inverse mass operator results in

    .. math::

        M^{-1} K_r = \hat{M}^{-1} \hat{K}_r \otimes \hat{I} \otimes \hat{I},

    where :math:`\hat{I}` is a 1D identity operator.

    The goal of this mapper is to remove redundant mass-times-mass inverse
    operations from an expression graph of operations involved with a
    tensor-product discretization.

    Once an inverse mass operator is identified, this mapper uses
    :class:`MassRemoverMapper` to find and remove the corresponding mass
    operator based on the output axis of the einsum that the inverse mass is an
    argument of.
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

                if nfaces == 2:
                    dim = 1
                elif nfaces == 4:
                    dim = 2
                elif nfaces == 6:
                    dim = 3
                else:
                    raise ValueError("Unable to determine the dimension of the",
                                     " hypercube to apply transformations.")

                from math import ceil
                ndofs_1d = int(ceil(ndofs**(1/dim)))
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
