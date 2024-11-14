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


from pytato.array import EinsumElementwiseAxis
from pytato.transform import CombineMapper, CopyMapperWithExtraArgs

from grudge.transform.metadata import (
    TensorProductMassOperatorInverseTag,
    TensorProductMassOperatorTag,
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

class MassRemoverMapper(CopyMapperWithExtraArgs):
    """
    See :class:`InverseMassRemoverMapper`.
    """
    def map_einsum(self, expr, *args, **kwargs):
        new_args = []
        out_access_descr, = args
        for arg in expr.args:
            if arg.tags_of_type(TensorProductMassOperatorTag):
                if expr.access_descriptors[0][0] == out_access_descr:
                    return expr.args[1]

            new_args.append(self.rec(arg, out_access_descr))

        return expr.copy(args=tuple(new_args))


class InverseMassRemoverMapper(CopyMapperWithExtraArgs):
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
    mass_remover = MassRemoverMapper()

    def map_einsum(self, expr, *args, **kwargs):
        new_args = []
        for arg in expr.args:
            if arg.tags_of_type(TensorProductMassOperatorInverseTag):
                out_access_descr = expr.access_descriptors[0][0]
                assert isinstance(out_access_descr, EinsumElementwiseAxis)

                new_expr = self.mass_remover(expr.args[1], out_access_descr)
                if new_expr != expr.args[1]:
                    return self.rec(new_expr, arg, out_access_descr)

            new_args.append(self.rec(arg))

        return expr.copy(args=tuple(new_args))

# }}}
