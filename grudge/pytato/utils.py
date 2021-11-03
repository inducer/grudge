import kanren
import pytato as pt
import unification

from grudge.metadata import DiscretizationEntityTag
from pytato.transform import ArrayOrNames
from arraycontext import ArrayContainer
from arraycontext.container.traversal import rec_map_array_container
from typing import Set, Mapping, Tuple


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

    def record_all_axes_to_be_solved(self, expr):
        for iaxis in range(expr.ndim):
            self.variables_to_solve.add(self.get_kanren_var_for_axis_tag(expr,
                                                                     iaxis))

    def record_eq_constraints_from_tags(self, expr: pt.Array) -> None:
        for iaxis, axis in enumerate(expr.axes):
            if axis.tags_of_type(DiscretizationEntityTag):
                discr_tag, = axis.tags_of_type(DiscretizationEntityTag)
                axis_var = self.get_kanren_var_for_axis_tag(expr, iaxis)
                self.constraints.append(kanren.eq(axis_var, discr_tag))

    def _map_input_base(self, expr: pt.InputArgumentBase
                        ) -> None:
        self.record_eq_constraints_from_tags(expr)
        self.record_all_axes_to_be_solved(expr)

        for dim in expr.shape:
            if isinstance(dim, pt.Array):
                self.rec(dim)

    map_placeholder = _map_input_base
    map_data_wrapper = _map_input_base
    map_size_param = _map_input_base

    def map_index_lambda(self, expr: pt.IndexLambda) -> None:
        from pytato.normalization import index_lambda_to_high_level_op
        from pytato.normalization import BinaryOp, FullOp

        # {{{ record constraints for expr and its subexprs.

        self.record_eq_constraints_from_tags(expr)
        self.record_all_axes_to_be_solved(expr)

        for dim in expr.shape:
            if isinstance(dim, pt.Array):
                self.rec(dim)

        for bnd in expr.bindings.values():
            self.rec(bnd)

        # }}}

        hlo = index_lambda_to_high_level_op(expr)

        if isinstance(hlo, BinaryOp):
            for subexpr in [hlo.x1, hlo.x2]:
                if isinstance(subexpr, pt.Array):
                    if subexpr.shape != expr.shape:
                        # Some broadcasting logic that we don't handle yet.
                        raise NotImplementedError

                    for iaxis in range(expr.ndim):
                        in_var = self.get_kanren_var_for_axis_tag(subexpr, iaxis)
                        out_var = self.get_kanren_var_for_axis_tag(expr, iaxis)
                        self.constraints.append(kanren.eq(in_var, out_var))
        elif isinstance(hlo, FullOp):
            # A full-op does not impose any constraints
            pass
        else:
            raise NotImplementedError(type(hlo))

    def map_matrix_product(self, expr: pt.MatrixProduct) -> None:
        self.record_eq_constraints_from_tags(expr)
        self.record_all_axes_to_be_solved(expr)
        self.rec(expr.x1)
        self.rec(expr.x2)
        raise NotImplementedError

    def map_stack(self, expr: pt.Stack) -> None:
        self.record_eq_constraints_from_tags(expr)
        # TODO; I think the axis correpsonding to 'axis' need not be solved.
        for ary in expr.arrays:
            self.rec(ary)

        raise NotImplementedError

    def map_concatenate(self, expr: pt.Concatenate) -> None:
        self.record_eq_constraints_from_tags(expr)
        # TODO; I think the axis correpsonding to 'axis' need not be solved.
        for ary in expr.arrays:
            self.rec(ary)
        raise NotImplementedError

    def map_axis_permutation(self, expr: pt.AxisPermutation
                             ) -> None:
        self.record_eq_constraints_from_tags(expr)
        self.record_all_axes_to_be_solved(expr)
        self.rec(expr.array)

        assert expr.ndim == expr.array.ndim

        for out_axis in range(expr.ndim):
            in_axis = expr.axis_permutation[out_axis]
            out_tag = self.get_kanren_var_for_axis_tag(expr, out_axis)
            in_tag = self.get_kanren_var_for_axis_tag(expr, in_axis)
            self.constraints.append(kanren.eq(out_tag, in_tag))

    def _map_index_base(self, expr: pt.IndexBase) -> None:
        self.record_eq_constraints_from_tags(expr)
        self.rec(expr.array)
        for idx in expr.indices:
            if isinstance(idx, pt.Array):
                self.rec(idx)

        raise NotImplementedError

    map_basic_index = _map_index_base
    map_contiguous_advanced_index = _map_index_base
    map_non_contiguous_advanced_index = _map_index_base

    def map_reshape(self, expr: pt.Reshape) -> None:
        # For "newaxes" reshapes the propagation rules are trivial.
        raise NotImplementedError

    def map_einsum(self, expr: pt.Einsum) -> None:
        from pytato.array import ElementwiseAxis

        self.record_eq_constraints_from_tags(expr)
        self.record_all_axes_to_be_solved(expr)

        for arg in expr.args:
            self.rec(arg)

        descr_to_tag = {}
        for iaxis in range(expr.ndim):
            descr_to_tag[ElementwiseAxis(iaxis)] = (
                self.get_kanren_var_for_axis_tag(expr, iaxis))

        for access_descrs, arg in zip(expr.access_descriptors,
                                      expr.args):
            for iarg_axis, descr in enumerate(access_descrs):
                in_tag_var = self.get_kanren_var_for_axis_tag(arg, iaxis)
                if descr in descr_to_tag:
                    self.constraints.append(kanren.eq(descr_to_tag[descr],
                                                      in_tag_var))
                else:
                    descr_to_tag[descr] = in_tag_var

    def map_dict_of_named_arrays(self, expr: pt.DictOfNamedArrays
                                 ) -> None:
        for _, subexpr in sorted(expr._data.items()):
            self.rec(subexpr)


def _unify_discretization_entity_tags(expr: ArrayOrNames
                                      ) -> ArrayOrNames:
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

        for iaxis, solved_tag in ary_to_axes_tags[expr].items():
            if expr.axes[iaxis].tags_of_type(DiscretizationEntityTag):
                discr_tag, = expr.axes[iaxis].tags_of_type(DiscretizationEntityTag)
                assert discr_tag == solved_tag
            else:
                expr = expr.with_tagged_axis(iaxis, solved_tag)

        return expr

    return pt.transform.map_and_copy(expr, attach_tags)


def unify_discretization_entity_tags(ary: ArrayContainer
                                     ) -> ArrayContainer:
    return rec_map_array_container(_unify_discretization_entity_tags,
                                   ary)

# }}}
