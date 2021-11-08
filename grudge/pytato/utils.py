import kanren
import pytato as pt
import unification

from meshmode.transform_metadata import DiscretizationEntityAxisTag
from pytato.loopy import LoopyCall
from pytato.transform import ArrayOrNames
from arraycontext import ArrayContainer
from arraycontext.container.traversal import rec_map_array_container
from typing import Set, Mapping, Tuple, Union


# {{{ solve for discretization metadata for arrays' axes

class DiscretizationEntityConstraintCollector(pt.transform.Mapper):
    """
    .. warning::

        Instances of this mapper type store state that are only for visiting a
        single DAG. Using a single instance for collecting the constraints on
        multiple DAGs is undefined behavior.
    """
    def __init__(self):
        super().__init__()
        self._visited_ids: Set[int] = set()

        # axis_to_var: mapping from (array, iaxis) to the kanren variable to be
        # used for unification.
        self.axis_to_tag_var: Mapping[Tuple[pt.Array, int],
                                      unification.variable.Var] = {}
        self.variables_to_solve: Set[unification.variable.Var] = set()
        self.constraints = []

    # type-ignore reason: CachedWalkMapper.rec's type does not match
    # WalkMapper.rec's type
    def rec(self, expr: ArrayOrNames) -> None:  # type: ignore
        if id(expr) in self._visited_ids:
            return

        # type-ignore reason: super().rec expects either 'Array' or
        # 'AbstractResultWithNamedArrays', passed 'ArrayOrNames'
        super().rec(expr)  # type: ignore
        self._visited_ids.add(id(expr))

    def get_kanren_var_for_axis_tag(self,
                                    expr: pt.Array,
                                    iaxis: int
                                    ) -> unification.variable.Var:
        key = (expr, iaxis)

        if key not in self.axis_to_tag_var:
            self.axis_to_tag_var[key] = kanren.var()

        return self.axis_to_tag_var[key]

    def _record_all_axes_to_be_solved_if_impl_stored(self, expr):
        if expr.tags_of_type(pt.tags.ImplStored):
            for iaxis in range(expr.ndim):
                self.variables_to_solve.add(self.get_kanren_var_for_axis_tag(expr,
                                                                             iaxis))

    def _record_all_axes_to_be_solved(self, expr):
        for iaxis in range(expr.ndim):
            self.variables_to_solve.add(self.get_kanren_var_for_axis_tag(expr,
                                                                         iaxis))

    def record_eq_constraints_from_tags(self, expr: pt.Array) -> None:
        for iaxis, axis in enumerate(expr.axes):
            if axis.tags_of_type(DiscretizationEntityAxisTag):
                discr_tag, = axis.tags_of_type(DiscretizationEntityAxisTag)
                axis_var = self.get_kanren_var_for_axis_tag(expr, iaxis)
                self.constraints.append(kanren.eq(axis_var, discr_tag))

    def _map_input_base(self, expr: pt.InputArgumentBase
                        ) -> None:
        self.record_eq_constraints_from_tags(expr)
        self._record_all_axes_to_be_solved_if_impl_stored(expr)

        for dim in expr.shape:
            if isinstance(dim, pt.Array):
                self.rec(dim)

    map_placeholder = _map_input_base
    map_data_wrapper = _map_input_base
    map_size_param = _map_input_base

    def map_index_lambda(self, expr: pt.IndexLambda) -> None:
        from pytato.utils import are_shape_components_equal
        from pytato.normalization import index_lambda_to_high_level_op
        from pytato.normalization import (BinaryOp, FullOp, ComparisonOp,
                                          WhereOp, BroadcastOp, C99CallOp,
                                          ReduceOp)

        # {{{ record constraints for expr and its subexprs.

        self.record_eq_constraints_from_tags(expr)
        self._record_all_axes_to_be_solved_if_impl_stored(expr)

        for dim in expr.shape:
            if isinstance(dim, pt.Array):
                self.rec(dim)

        for bnd in expr.bindings.values():
            self.rec(bnd)

        # }}}

        hlo = index_lambda_to_high_level_op(expr)

        if isinstance(hlo, BinaryOp):
            subexprs = (hlo.x1, hlo.x2)
        elif isinstance(hlo, ComparisonOp):
            subexprs = (hlo.left, hlo.right)
        elif isinstance(hlo, WhereOp):
            subexprs = (hlo.condition, hlo.then, hlo.else_)
        elif isinstance(hlo, FullOp):
            # A full-op does not impose any constraints
            subexprs = ()
        elif isinstance(hlo, BroadcastOp):
            subexprs = (hlo.x,)
        elif isinstance(hlo, C99CallOp):
            subexprs = hlo.args
        elif isinstance(hlo, ReduceOp):
            # {{{ ReduceOp doesn't quite involve broadcasting

            i_out_axis = 0
            for i_in_axis in range(hlo.x.ndim):
                if i_in_axis not in hlo.axes:
                    in_tag_var = self.get_kanren_var_for_axis_tag(hlo.x,
                                                                  i_in_axis)
                    out_tag_var = self.get_kanren_var_for_axis_tag(expr,
                                                                   i_out_axis)
                    self.constraints.append(kanren.eq(in_tag_var,
                                                      out_tag_var))
                    i_out_axis += 1

            assert i_out_axis == expr.ndim
            return

            # }}}
        else:
            raise NotImplementedError(type(hlo))

        for subexpr in subexprs:
            if isinstance(subexpr, pt.Array):
                for i_in_axis, i_out_axis in zip(
                        range(subexpr.ndim),
                        range(expr.ndim-subexpr.ndim, expr.ndim)):
                    in_dim = subexpr.shape[i_in_axis]
                    out_dim = expr.shape[i_out_axis]
                    if are_shape_components_equal(in_dim, out_dim):
                        in_tag_var = self.get_kanren_var_for_axis_tag(subexpr,
                                                                      i_in_axis)
                        out_tag_var = self.get_kanren_var_for_axis_tag(expr,
                                                                       i_out_axis)

                        self.constraints.append(kanren.eq(in_tag_var, out_tag_var))
                    else:
                        # broadcasted axes, cannot belong to the same
                        # discretization entity.
                        assert are_shape_components_equal(in_dim, 1)

    def map_matrix_product(self, expr: pt.MatrixProduct) -> None:
        self.record_eq_constraints_from_tags(expr)
        self._record_all_axes_to_be_solved_if_impl_stored(expr)
        self.rec(expr.x1)
        self.rec(expr.x2)
        raise NotImplementedError

    def map_stack(self, expr: pt.Stack) -> None:
        self.record_eq_constraints_from_tags(expr)
        self._record_all_axes_to_be_solved_if_impl_stored(expr)
        # TODO; I think the axis corresponding to 'axis' need not be solved.
        for ary in expr.arrays:
            self.rec(ary)

        for iaxis in range(expr.ndim):
            for ary in expr.arrays:
                if iaxis < expr.axis:
                    in_tag_var = self.get_kanren_var_for_axis_tag(ary,
                                                                  iaxis)
                    out_tag_var = self.get_kanren_var_for_axis_tag(expr,
                                                                   iaxis)

                    self.constraints.append(kanren.eq(in_tag_var, out_tag_var))
                elif iaxis == expr.axis:
                    pass
                elif iaxis > expr.axis:
                    in_tag_var = self.get_kanren_var_for_axis_tag(ary,
                                                                  iaxis-1)
                    out_tag_var = self.get_kanren_var_for_axis_tag(expr,
                                                                   iaxis)

                    self.constraints.append(kanren.eq(in_tag_var, out_tag_var))
                else:
                    raise AssertionError

    def map_concatenate(self, expr: pt.Concatenate) -> None:
        self.record_eq_constraints_from_tags(expr)
        # TODO; I think the axis corresponding to 'axis' need not be solved.
        for ary in expr.arrays:
            self.rec(ary)
        raise NotImplementedError

    def map_axis_permutation(self, expr: pt.AxisPermutation
                             ) -> None:
        self.record_eq_constraints_from_tags(expr)
        self._record_all_axes_to_be_solved_if_impl_stored(expr)
        self.rec(expr.array)

        assert expr.ndim == expr.array.ndim

        for out_axis in range(expr.ndim):
            in_axis = expr.axis_permutation[out_axis]
            out_tag = self.get_kanren_var_for_axis_tag(expr, out_axis)
            in_tag = self.get_kanren_var_for_axis_tag(expr, in_axis)
            self.constraints.append(kanren.eq(out_tag, in_tag))

    def _map_index_base(self, expr: pt.IndexBase) -> None:
        from pytato.array import NormalizedSlice

        self.record_eq_constraints_from_tags(expr)
        self._record_all_axes_to_be_solved_if_impl_stored(expr)
        self.rec(expr.array)
        for idx in expr.indices:
            if isinstance(idx, pt.Array):
                self.rec(idx)

        for idx in expr.indices:
            if isinstance(idx, int):
                pass
            elif isinstance(idx, NormalizedSlice):
                # For normalized slices, impose the conditions for everything
                # else, just bail...?
                raise NotImplementedError("Basic Indices not supported")
            else:
                # There's almost no constraint we could impose here...
                assert isinstance(idx, pt.Array)

    map_basic_index = _map_index_base
    map_contiguous_advanced_index = _map_index_base
    map_non_contiguous_advanced_index = _map_index_base

    def map_reshape(self, expr: pt.Reshape) -> None:
        self.record_eq_constraints_from_tags(expr)
        self._record_all_axes_to_be_solved_if_impl_stored(expr)
        self.rec(expr.array)
        # we can add constraints to reshape that only include new axes in its
        # reshape.
        # Other reshapes do not 'conserve' the types in our type-system.
        # Well *what if*. Let's just say this type inference fails for
        # non-trivial 'reshapes'. So, what are the 'trivial' reshapes?
        # trivial reshapes:
        # (x1, x2, ... xn) -> ((1,)*, x1, (1,)*, x2, (1,)*, x3, (1,)*, ..., xn, 1*)
        # given all(x1!=1, x2!=1, x3!=1, .. xn!= 1)
        if ((1 not in (expr.array.shape))  # leads to ambiguous newaxis
                and (set(expr.shape) <= (set(expr.array.shape) | {1}))):
            i_in_axis = 0
            for i_out_axis, dim in enumerate(expr.shape):
                if dim != 1:
                    self.variables_to_solve.add(
                        self.get_kanren_var_for_axis_tag(expr,
                                                         i_out_axis))
                    assert dim == expr.array.shape[i_in_axis]
                    i_in_axis_tag = self.get_kanren_var_for_axis_tag(expr.array,
                                                                     i_in_axis)
                    i_out_axis_tag = self.get_kanren_var_for_axis_tag(expr,
                                                                      i_out_axis)
                    self.constraints.append(kanren.eq(i_in_axis_tag,
                                                      i_out_axis_tag))
                    i_in_axis += 1
        else:
            print(f"Skipping {expr.array.shape} -> {expr.shape}")
            # Wacky reshape => bail.
            return

    def map_einsum(self, expr: pt.Einsum) -> None:
        from pytato.array import ElementwiseAxis

        self.record_eq_constraints_from_tags(expr)
        self._record_all_axes_to_be_solved_if_impl_stored(expr)

        for arg in expr.args:
            self.rec(arg)

        descr_to_tag = {}
        for iaxis in range(expr.ndim):
            descr_to_tag[ElementwiseAxis(iaxis)] = (
                self.get_kanren_var_for_axis_tag(expr, iaxis))

        for access_descrs, arg in zip(expr.access_descriptors,
                                      expr.args):
            for iarg_axis, descr in enumerate(access_descrs):
                in_tag_var = self.get_kanren_var_for_axis_tag(arg,
                                                              iarg_axis)
                if descr in descr_to_tag:
                    self.constraints.append(kanren.eq(descr_to_tag[descr],
                                                      in_tag_var))
                else:
                    descr_to_tag[descr] = in_tag_var

    def map_dict_of_named_arrays(self, expr: pt.DictOfNamedArrays
                                 ) -> None:
        for _, subexpr in sorted(expr._data.items()):
            self.rec(subexpr)
            self._record_all_axes_to_be_solved(subexpr)

    def map_loopy_call(self, expr: LoopyCall) -> None:
        for _, subexpr in sorted(expr.bindings.items()):
            if isinstance(subexpr, pt.Array):
                if not isinstance(subexpr, pt.InputArgumentBase):
                    self._record_all_axes_to_be_solved(subexpr)
                self.rec(subexpr)

        # there's really no good way to propagate the metadata in this case.

    def map_named_array(self, expr: pt.NamedArray) -> None:
        self.record_eq_constraints_from_tags(expr)
        self.rec(expr._container)


def unify_discretization_entity_tags(expr: Union[ArrayContainer, ArrayOrNames]
                                     ) -> ArrayOrNames:
    if not isinstance(expr, (pt.Array, pt.DictOfNamedArrays)):
        return rec_map_array_container(unify_discretization_entity_tags,
                                       expr)

    from collections import defaultdict
    discr_unification_helper = DiscretizationEntityConstraintCollector()
    discr_unification_helper(expr)
    tag_var_to_axis = {}
    variables_to_solve = []

    for (axis, var) in discr_unification_helper.axis_to_tag_var.items():
        if var in discr_unification_helper.variables_to_solve:
            tag_var_to_axis[var] = axis
            variables_to_solve.append(var)

    solutions = kanren.run(0,
                           variables_to_solve,
                           *discr_unification_helper.constraints)

    # There should be only solution
    assert len(solutions) == 1

    # ary_to_axes_tags: mapping from array to a mapping from iaxis to the
    # solved tag.
    ary_to_axes_tags = defaultdict(dict)
    for var, value in zip(variables_to_solve, solutions[0]):
        ary, axis = tag_var_to_axis[var]
        ary_to_axes_tags[ary][axis] = value

    def attach_tags(expr: ArrayOrNames) -> ArrayOrNames:
        if not isinstance(expr, pt.Array):
            return expr

        old_expr = expr
        for iaxis, solved_tag in ary_to_axes_tags[expr].items():
            if expr.axes[iaxis].tags_of_type(DiscretizationEntityAxisTag):
                discr_tag, = (expr
                              .axes[iaxis]
                              .tags_of_type(DiscretizationEntityAxisTag))
                assert discr_tag == solved_tag
            else:
                if not isinstance(solved_tag, DiscretizationEntityAxisTag):
                    print(ary_to_axes_tags[old_expr])
                    import pudb; pu.db
                    2/0
                    raise ValueError(f"In {expr!r}, axis={iaxis}'s type cannot be "
                                     "inferred.")
                expr = expr.with_tagged_axis(iaxis, solved_tag)

        return expr

    return pt.transform.map_and_copy(expr, attach_tags)

# }}}


class UnInferredStoredArrayCatcher(pt.transform.CachedWalkMapper):
    """
    Raises a :class:`ValueError` if a stored array has axes without a
    :class:`DiscretizationEntityAxisTag` tagged to it.
    """
    def post_visit(self, expr: ArrayOrNames) -> None:
        if (isinstance(expr, pt.Array)
                and expr.tags_of_type(pt.tags.ImplStored)):
            if any(len(axis.tags_of_type(DiscretizationEntityAxisTag)) != 1
                   for axis in expr.axes):
                raise ValueError(f"{expr!r} doesn't have all its axes inferred.")

        if isinstance(expr, pt.DictOfNamedArrays):
            if any(any(len(axis.tags_of_type(DiscretizationEntityAxisTag)) != 1
                       for axis in subexpr.axes)
                   for subexpr in expr._data.values()):
                raise ValueError(f"{expr!r} doesn't have all its axes inferred.")

        from pytato.loopy import LoopyCall

        if isinstance(expr, LoopyCall):
            if any(any(len(axis.tags_of_type(DiscretizationEntityAxisTag)) != 1
                       for axis in subexpr.axes)
                   for subexpr in expr.bindings.values()
                   if (isinstance(subexpr, pt.Array)
                       and not isinstance(subexpr, pt.InputArgumentBase)
                       and subexpr.ndim != 0)):
                raise ValueError(f"{expr!r} doesn't have all its axes inferred.")


def are_all_stored_arrays_inferred(expr: ArrayOrNames):
    UnInferredStoredArrayCatcher()(expr)
