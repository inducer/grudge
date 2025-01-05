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


from meshmode.discretization import DiscretizationDOFAxisTag, DiscretizationElementAxisTag
import pytato as pt
from pytato import Array, Axis, ReductionDescriptor, Reshape
from pytato.analysis import is_einsum_similar_to_subscript
from pytato.array import Einsum, EinsumAxisDescriptor, EinsumElementwiseAxis, EinsumReductionAxis, immutabledict
from pytato.transform import (
    CombineMapper,
    CopyMapper,
    CopyMapperWithExtraArgs,
)

from grudge.transform.metadata import (
    TensorProductDOFAxisTag,
    TensorProductMassOperatorInverseTag,
    TensorProductMassOperatorTag,
    TensorProductOperatorAxisTag,
    TensorProductStiffnessOperatorTag,
)


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


class InverseMassAttacher(CopyMapperWithExtraArgs):
    def map_einsum(
            self,
            expr: Einsum,
            inv_mass: Array,
            access_descr: tuple[EinsumAxisDescriptor]
        ) -> Array:

        new_args = []
        for iarg, arg in enumerate(expr.args):
            if arg.tags_of_type(TensorProductMassOperatorTag):
                if expr.access_descriptors[iarg] == access_descr:
                    a = inv_mass @ arg
                    a = a.copy(axes=tuple(
                        ax.tagged(TensorProductOperatorAxisTag())
                        for ax in a.axes
                    ))
                    new_args.append(a)
                    continue

            elif arg.tags_of_type(TensorProductStiffnessOperatorTag):
                if expr.access_descriptors[iarg] == access_descr:
                    a = inv_mass @ arg
                    a = a.copy(axes=tuple(
                        ax.tagged(TensorProductOperatorAxisTag(),)
                        for ax in a.axes
                    ))
                    new_args.append(a)
                    continue

            new_args.append(self.rec(arg, inv_mass, access_descr))

        return expr.copy(args=tuple(new_args))

    def map_reshape(
            self,
            expr: Reshape,
            inv_mass: Array,
            access_descr: tuple[EinsumAxisDescriptor]
        ) -> Array:

        if isinstance(expr.array, Einsum):
            if is_einsum_similar_to_subscript(expr.array, "ifj,fej->ei"):
                _, nfaces, _ = expr.array.args[0].shape

                dim = int(nfaces / 2)

                output_ax = tuple(
                    descr for descr in access_descr
                    if isinstance(descr, EinsumElementwiseAxis)
                )
                assert len(output_ax) == 1
                output_ax, = output_ax

                data_access_descrs = []
                for i in range(dim+1):
                    if i != output_ax.dim:
                        data_access_descrs.append(EinsumElementwiseAxis(i))
                    else:
                        data_access_descrs.append(EinsumReductionAxis(0))
                access_descriptors = (access_descr, tuple(data_access_descrs))

                redn_axis_to_redn_descr = immutabledict({
                    EinsumReductionAxis(0):
                    ReductionDescriptor(tags=frozenset())
                })

                axes = tuple(
                    Axis(tags=frozenset((DiscretizationElementAxisTag(),)))
                    if i == 0
                    else Axis(tags=frozenset((TensorProductDOFAxisTag(i-1),)))
                    for i in range(dim+1)
                )

                return Einsum(
                    access_descriptors=access_descriptors,
                    args=(inv_mass, expr),
                    axes=axes,
                    redn_axis_to_redn_descr=redn_axis_to_redn_descr,
                    tags=frozenset()
                )

        return expr.copy(array=self.rec(expr.array, inv_mass, access_descr))


class InverseMassDistributor(CopyMapper):
    def map_einsum(self, expr: Einsum) -> Array:
        for iarg, arg in enumerate(expr.args):
            if not arg.tags_of_type(TensorProductMassOperatorInverseTag):
                iarg_rec = iarg
                break

        new_args = []
        for iarg, arg in enumerate(expr.args):
            if arg.tags_of_type(TensorProductMassOperatorInverseTag):
                return InverseMassAttacher()(
                    self.rec(expr.args[iarg_rec]),
                    arg,
                    expr.access_descriptors[iarg]
                )

            else:
                new_args.append(self.rec(arg))

        return expr.copy(args=tuple(new_args))


def check_redundant_mass(expr: Einsum) -> bool:
    found_mass = False
    found_inverse_mass = False

    for arg in expr.args:
        if arg.tags_of_type(TensorProductMassOperatorInverseTag):
            found_inverse_mass = True
        elif arg.tags_of_type(TensorProductMassOperatorTag):
            found_mass = True

    return (found_inverse_mass and found_mass)


class RedundantMassRemover(CopyMapper):
    def map_einsum(self, expr: Einsum) -> Array:
        new_args = []
        for arg in expr.args:
            if isinstance(arg, Einsum):
                if check_redundant_mass(arg):
                    continue
            new_args.append(self.rec(arg))

        if len(new_args) == 1:
            return self.rec(new_args[0])

        return expr.copy(args=tuple(new_args))


class FaceMassResultReshaper(CopyMapper):
    def map_einsum(self, expr: Einsum) -> Array:

        new_args = [self.rec(arg) for arg in expr.args]
        new_expr = expr.copy(args=tuple(new_args))

        if is_einsum_similar_to_subscript(new_expr, "ifj,fej->ei"):
            nfaces, nelts, _ = new_expr.args[1].shape
            ndofs = expr.shape[1]

            from math import ceil
            dim = ceil(nfaces / 2)
            ndofs_1d = ceil(ndofs**(1/dim))

            new_expr = new_expr.reshape(nelts, *(ndofs_1d,)*dim, order="F")
            for i in range(dim+1):
                new_expr = new_expr.with_tagged_axis(
                    i, (
                        DiscretizationElementAxisTag() if i == 0 else
                        TensorProductDOFAxisTag(i-1)
                    )
                )

            new_expr = new_expr.reshape(nelts, ndofs_1d**dim, order="F")
            new_expr = new_expr.with_tagged_axis(0,
                                                 DiscretizationElementAxisTag())
            new_expr = new_expr.with_tagged_axis(1, DiscretizationDOFAxisTag())

        return new_expr


def tensor_product_algebraic_transforms(dag):
    # 0. preprocess face mass result (reshape to tp -> reshape from tp)
    dag = FaceMassResultReshaper()(dag)

    # 1. distribute the inverse mass to:
    #   - einsums with stiffness
    #   - einsums with mass
    #   - face mass (insert applications between reshapes)
    dag = InverseMassDistributor()(dag)

    # 2. remove einsums with mass and mass inverse
    dag = RedundantMassRemover()(dag)

    # done
    return dag
