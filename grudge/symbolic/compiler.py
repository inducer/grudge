"""Compiler to turn operator expression tree into (imperative) bytecode."""

__copyright__ = "Copyright (C) 2008-15 Andreas Kloeckner"

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

import numpy as np

from pytools import Record, memoize_method, memoize
from pytools.obj_array import obj_array_vectorize

from grudge import sym
import grudge.symbolic.mappers as mappers
from pymbolic.primitives import Variable, Subscript
from sys import intern
from functools import reduce

from loopy.version import LOOPY_USE_LANGUAGE_VERSION_2018_2  # noqa: F401


# {{{ instructions

class Instruction(Record):
    priority = 0
    neglect_for_dofdesc_inference = False

    def get_assignees(self):
        raise NotImplementedError("no get_assignees in %s" % self.__class__)

    def get_dependencies(self):
        raise NotImplementedError("no get_dependencies in %s" % self.__class__)

    def __str__(self):
        raise NotImplementedError

    def __hash__(self):
        return id(self)

    def __eq__(self, other):
        return self is other

    def __ne__(self, other):
        return not self.__eq__(other)


@memoize
def _make_dep_mapper(include_subscripts):
    return mappers.DependencyMapper(
            include_operator_bindings=False,
            include_subscripts=include_subscripts,
            include_calls="descend_args")


# {{{ loopy kernel instruction

class LoopyKernelDescriptor:
    def __init__(self, loopy_kernel, input_mappings, output_mappings,
            fixed_arguments, governing_dd):
        self.loopy_kernel = loopy_kernel
        self.input_mappings = input_mappings
        self.output_mappings = output_mappings
        self.fixed_arguments = fixed_arguments
        self.governing_dd = governing_dd

    @memoize_method
    def scalar_args(self):
        import loopy as lp
        return [arg.name for arg in self.loopy_kernel.args
                if isinstance(arg, lp.ValueArg)
                and arg.name not in ["nelements", "nunit_dofs"]]


class LoopyKernelInstruction(Instruction):
    comment = ""
    scope_indicator = ""

    def __init__(self, kernel_descriptor):
        super().__init__()
        self.kernel_descriptor = kernel_descriptor

    @memoize_method
    def get_assignees(self):
        return {k for k in self.kernel_descriptor.output_mappings.keys()}

    @memoize_method
    def get_dependencies(self):
        from pymbolic.primitives import Variable, Subscript
        result = set()
        for v in self.kernel_descriptor.input_mappings.values():
            if isinstance(v, Variable):
                result.add(v)
            elif isinstance(v, Subscript):
                result.add(v.aggregate)
        return result

    def __str__(self):
        knl_str = "\n".join(
                f"{insn.assignee} = {insn.expression}"
                for insn in self.kernel_descriptor.loopy_kernel.instructions)

        knl_str = knl_str.replace("grdg_", "")

        return "{ /* loopy */\n  %s\n}" % knl_str.replace("\n", "\n  ")

    mapper_method = "map_insn_loopy_kernel"

# }}}


class AssignBase(Instruction):
    comment = ""
    scope_indicator = ""

    def __str__(self):
        comment = self.comment
        if len(self.names) == 1:
            if comment:
                comment = "/* %s */ " % comment

            return "{} <-{} {}{}".format(
                    self.names[0], self.scope_indicator, comment,
                    self.exprs[0])
        else:
            if comment:
                comment = " /* %s */" % comment

            lines = []
            lines.append("{" + comment)
            for n, e, dnr in zip(self.names, self.exprs, self.do_not_return):
                if dnr:
                    dnr_indicator = "-#"
                else:
                    dnr_indicator = ""

                lines.append("  {} <{}-{} {}".format(
                    n, dnr_indicator, self.scope_indicator, e))
            lines.append("}")
            return "\n".join(lines)


class Assign(AssignBase):
    """
    .. attribute:: names
    .. attribute:: exprs
    .. attribute:: do_not_return

        a list of bools indicating whether the corresponding entry in names and
        exprs describes an expression that is not needed beyond this assignment

    .. attribute:: priority
    """

    def __init__(self, names, exprs, **kwargs):
        Instruction.__init__(self, names=names, exprs=exprs, **kwargs)

        if not hasattr(self, "do_not_return"):
            self.do_not_return = [False] * len(names)

    @memoize_method
    def flop_count(self):
        return sum(mappers.FlopCounter()(expr) for expr in self.exprs)

    def get_assignees(self):
        return set(self.names)

    @memoize_method
    def get_dependencies(self, each_vector=False):
        dep_mapper = _make_dep_mapper(include_subscripts=False)

        from operator import or_
        deps = reduce(
                or_, (dep_mapper(expr)
                for expr in self.exprs))

        from pymbolic.primitives import Variable
        deps -= {Variable(name) for name in self.names}

        if not each_vector:
            self._dependencies = deps

        return deps

    mapper_method = intern("map_insn_assign")


class RankDataSwapAssign(Instruction):
    """
    .. attribute:: name
    .. attribute:: field
    .. attribute:: i_remote_rank

        The number of the remote rank that this instruction swaps data with.

    .. attribute:: dd_out
    .. attribute:: comment
    """
    MPI_TAG_GRUDGE_DATA_BASE = 15165

    def __init__(self, name, field, op):
        self.name = name
        self.field = field
        self.i_remote_rank = op.i_remote_part
        self.dd_out = op.dd_out
        self.send_tag = self.MPI_TAG_GRUDGE_DATA_BASE + op.unique_id
        self.recv_tag = self.MPI_TAG_GRUDGE_DATA_BASE + op.unique_id
        self.comment = f"Swap data with rank {self.i_remote_rank:02d}"

    @memoize_method
    def get_assignees(self):
        return {self.name}

    @memoize_method
    def get_dependencies(self):
        return _make_dep_mapper(include_subscripts=False)(self.field)

    def __str__(self):
        return ("{\n"
              + f"   /* {self.comment} */\n"
              + f"   send_tag = {self.send_tag}\n"
              + f"   recv_tag = {self.recv_tag}\n"
              + f"   {self.name} <- {self.field}\n"
              + "}")

    mapper_method = intern("map_insn_rank_data_swap")


class ToDiscretizationScopedAssign(Assign):
    scope_indicator = "(to discr)-"

    mapper_method = intern("map_insn_assign_to_discr_scoped")


class FromDiscretizationScopedAssign(AssignBase):
    scope_indicator = "(discr)-"
    neglect_for_dofdesc_inference = True

    def __init__(self, name, **kwargs):
        super().__init__(name=name, **kwargs)

    @memoize_method
    def flop_count(self):
        return 0

    def get_assignees(self):
        return frozenset([self.name])

    def get_dependencies(self):
        return frozenset()

    def __str__(self):
        return "%s <-(from discr)" % self.name

    mapper_method = intern("map_insn_assign_from_discr_scoped")


class DiffBatchAssign(Instruction):
    """
    :ivar names:
    :ivar operators:

        .. note ::

            All operators here are guaranteed to satisfy
            :meth:`grudge.symbolic.operators.DiffOperatorBase.
            equal_except_for_axis`.

    :ivar field:
    """

    def get_assignees(self):
        return frozenset(self.names)

    @memoize_method
    def get_dependencies(self):
        return _make_dep_mapper(include_subscripts=False)(self.field)

    def __str__(self):
        lines = []

        if len(self.names) > 1:
            lines.append("{")
            for n, d in zip(self.names, self.operators):
                lines.append(f"  {n} <- {d}({self.field})")
            lines.append("}")
        else:
            for n, d in zip(self.names, self.operators):
                lines.append(f"{n} <- {d}({self.field})")

        return "\n".join(lines)

    mapper_method = intern("map_insn_diff_batch_assign")

# }}}


# {{{ graphviz/dot dataflow graph drawing

def dot_dataflow_graph(code, max_node_label_length=30,
        label_wrap_width=50):
    origins = {}
    node_names = {}

    result = [
            'initial [label="initial"]'
            'result [label="result"]']

    for num, insn in enumerate(code.instructions):
        node_name = "node%d" % num
        node_names[insn] = node_name
        node_label = str(insn)

        if (max_node_label_length is not None
                and not isinstance(insn, LoopyKernelInstruction)):
            node_label = node_label[:max_node_label_length]

        if label_wrap_width is not None:
            from pytools import word_wrap
            node_label = word_wrap(node_label, label_wrap_width,
                    wrap_using="\n      ")

        node_label = node_label.replace("\n", "\\l") + "\\l"

        result.append(f"{node_name} [ "
                f'label="p{insn.priority}: {node_label}" shape=box ];')

        for assignee in insn.get_assignees():
            origins[assignee] = node_name

    def get_orig_node(expr):
        from pymbolic.primitives import Variable
        if isinstance(expr, Variable):
            return origins.get(expr.name, "initial")
        else:
            return "initial"

    def gen_expr_arrow(expr, target_node):
        orig_node = get_orig_node(expr)
        result.append(f'{orig_node} -> {target_node} [label="{expr}"];')

    for insn in code.instructions:
        for dep in insn.get_dependencies():
            gen_expr_arrow(dep, node_names[insn])

    if isinstance(code.result, np.ndarray) and code.result.dtype.char == "O":
        for subexp in code.result:
            gen_expr_arrow(subexp, "result")
    else:
        gen_expr_arrow(code.result, "result")

    return "digraph dataflow {\n%s\n}\n" % "\n".join(result)

# }}}


# {{{ code representation

class Code:
    def __init__(self, instructions, result):
        self.instructions = instructions
        self.result = result
        # self.last_schedule = None
        self.static_schedule_attempts = 5

    def dump_dataflow_graph(self, name=None):
        from pytools.debug import open_unique_debug_file

        if name is None:
            stem = "dataflow"
        else:
            stem = "dataflow-%s" % name

        outf, _ = open_unique_debug_file(stem, ".dot")
        with outf:
            outf.write(dot_dataflow_graph(self, max_node_label_length=None))

    def __str__(self):
        var_to_writer = {
                var_name: insn
                for insn in self.instructions
                for var_name in insn.get_assignees()}

        # {{{ topological sort

        added_insns = set()
        ordered_insns = []

        def insert_insn(insn):
            if insn in added_insns:
                return

            for dep in insn.get_dependencies():
                try:
                    if isinstance(dep, Subscript):
                        dep_name = dep.aggregate.name
                    else:
                        dep_name = dep.name

                    writer = var_to_writer[dep_name]
                except KeyError:
                    # input variables won't be found
                    pass
                else:
                    insert_insn(writer)

            ordered_insns.append(insn)
            added_insns.add(insn)

        for insn in self.instructions:
            insert_insn(insn)

        assert len(ordered_insns) == len(self.instructions)
        assert len(added_insns) == len(self.instructions)

        # }}}

        lines = []
        for insn in ordered_insns:
            lines.extend(str(insn).split("\n"))
        lines.append("RESULT: " + str(self.result))

        return "\n".join(lines)

    # {{{ dynamic scheduler (generates static schedules by self-observation)

    class NoInstructionAvailable(Exception):
        pass

    @memoize_method
    def get_next_step(self, available_names, done_insns):
        from pytools import argmax2
        available_insns = [
                (insn, insn.priority) for insn in self.instructions
                if insn not in done_insns
                and all(dep.name in available_names
                    for dep in insn.get_dependencies())]

        if not available_insns:
            raise self.NoInstructionAvailable

        from pytools import flatten
        discardable_vars = set(available_names) - set(flatten(
            [dep.name for dep in insn.get_dependencies()]
            for insn in self.instructions
            if insn not in done_insns))

        # {{{ make sure results do not get discarded

        dm = mappers.DependencyMapper(composite_leaves=False)

        def remove_result_variable(result_expr):
            # The extra dependency mapper run is necessary
            # because, for instance, subscripts can make it
            # into the result expression, which then does
            # not consist of just variables.

            for var in dm(result_expr):
                assert isinstance(var, Variable)
                discardable_vars.discard(var.name)

        obj_array_vectorize(remove_result_variable, self.result)

        # }}}

        return argmax2(available_insns), discardable_vars

    def execute(self, exec_mapper, pre_assign_check=None, profile_data=None,
                log_quantities=None):
        if profile_data is not None:
            from time import time
            start_time = time()
            if profile_data == {}:
                profile_data["insn_eval_time"] = 0
                profile_data["future_eval_time"] = 0
                profile_data["busy_wait_time"] = 0
                profile_data["total_time"] = 0
        if log_quantities is not None:
            exec_sub_timer = log_quantities["exec_timer"].start_sub_timer()
        context = exec_mapper.context

        futures = []
        done_insns = set()

        while True:
            try:
                if profile_data is not None:
                    insn_start_time = time()
                if log_quantities is not None:
                    insn_sub_timer = \
                            log_quantities["insn_eval_timer"].start_sub_timer()

                insn, discardable_vars = self.get_next_step(
                    frozenset(list(context.keys())),
                    frozenset(done_insns))

                done_insns.add(insn)
                for name in discardable_vars:
                    del context[name]

                mapper_method = getattr(exec_mapper, insn.mapper_method)
                if log_quantities is not None:
                    if isinstance(insn, RankDataSwapAssign):
                        from logpyle import time_and_count_function
                        mapper_method = time_and_count_function(
                                mapper_method,
                                log_quantities["rank_data_swap_timer"],
                                log_quantities["rank_data_swap_counter"])

                assignments, new_futures = mapper_method(insn, profile_data)

                for target, value in assignments:
                    if pre_assign_check is not None:
                        pre_assign_check(target, value)
                    context[target] = value

                futures.extend(new_futures)
                if profile_data is not None:
                    profile_data["insn_eval_time"] += time() - insn_start_time
                if log_quantities is not None:
                    insn_sub_timer.stop().submit()
            except self.NoInstructionAvailable:
                if not futures:
                    # No more instructions or futures. We are done.
                    break

                # Busy wait for a new future
                if profile_data is not None:
                    busy_wait_start_time = time()
                if log_quantities is not None:
                    busy_sub_timer =\
                            log_quantities["busy_wait_timer"].start_sub_timer()

                did_eval_future = False
                while not did_eval_future:
                    for i in range(len(futures)):
                        if futures[i].is_ready():
                            if profile_data is not None:
                                profile_data["busy_wait_time"] +=\
                                        time() - busy_wait_start_time
                                future_start_time = time()
                            if log_quantities is not None:
                                busy_sub_timer.stop().submit()
                                future_sub_timer =\
                                            log_quantities["future_eval_timer"]\
                                                                .start_sub_timer()

                            future = futures.pop(i)
                            assignments, new_futures = future()

                            for target, value in assignments:
                                if pre_assign_check is not None:
                                    pre_assign_check(target, value)
                                context[target] = value

                            futures.extend(new_futures)
                            did_eval_future = True

                            if profile_data is not None:
                                profile_data["future_eval_time"] +=\
                                        time() - future_start_time
                            if log_quantities is not None:
                                future_sub_timer.stop().submit()
                            break

        if len(done_insns) < len(self.instructions):
            raise RuntimeError("not all instructions are reachable"
                    "--did you forget to pass a value for a placeholder?")

        if log_quantities is not None:
            exec_sub_timer.stop().submit()
        if profile_data is not None:
            profile_data["total_time"] = time() - start_time
            return (obj_array_vectorize(exec_mapper, self.result),
                    profile_data)
        return obj_array_vectorize(exec_mapper, self.result)

# }}}

# }}}


# {{{ assignment aggregration pass

def aggregate_assignments(inf_mapper, instructions, result,
        max_vectors_in_batch_expr):
    from pymbolic.primitives import Variable

    function_registry = inf_mapper.function_registry

    # {{{ aggregation helpers

    def get_complete_origins_set(insn, skip_levels=0):
        try:
            return insn_to_origins_cache[insn]
        except KeyError:
            pass

        if skip_levels < 0:
            skip_levels = 0

        result = set()
        for dep in insn.get_dependencies():
            if isinstance(dep, Variable):
                dep_origin = origins_map.get(dep.name, None)
                if dep_origin is not None:
                    if skip_levels <= 0:
                        result.add(dep_origin)
                    result |= get_complete_origins_set(
                            dep_origin, skip_levels-1)

        insn_to_origins_cache[insn] = result

        return result

    var_assignees_cache = {}

    def get_var_assignees(insn):
        try:
            return var_assignees_cache[insn]
        except KeyError:
            result = {Variable(assignee) for assignee in insn.get_assignees()}
            var_assignees_cache[insn] = result
            return result

    def aggregate_two_assignments(ass_1, ass_2):
        names = ass_1.names + ass_2.names

        from pymbolic.primitives import Variable
        deps = (ass_1.get_dependencies() | ass_2.get_dependencies()) \
                - {Variable(name) for name in names}

        return Assign(
                names=names, exprs=ass_1.exprs + ass_2.exprs,
                _dependencies=deps,
                priority=max(ass_1.priority, ass_2.priority))

    # }}}

    # {{{ main aggregation pass

    insn_to_origins_cache = {}

    origins_map = {
                assignee: insn
                for insn in instructions
                for assignee in insn.get_assignees()}

    from pytools import partition
    from grudge.symbolic.primitives import DTAG_SCALAR

    unprocessed_assigns, other_insns = partition(
            lambda insn: (
                isinstance(insn, Assign)
                and not isinstance(insn, ToDiscretizationScopedAssign)
                and not isinstance(insn, FromDiscretizationScopedAssign)
                and not is_external_call(insn.exprs[0], function_registry)
                and not any(
                    inf_mapper.infer_for_name(n).domain_tag == DTAG_SCALAR
                    for n in insn.names)),
            instructions)

    # filter out zero-flop-count assigns--no need to bother with those
    processed_assigns, unprocessed_assigns = partition(
            lambda ass: ass.flop_count() == 0,
            unprocessed_assigns)

    # filter out zero assignments
    from grudge.tools import is_zero

    i = 0

    while i < len(unprocessed_assigns):
        my_assign = unprocessed_assigns[i]
        if any(is_zero(expr) for expr in my_assign.exprs):
            processed_assigns.append(unprocessed_assigns.pop(i))
        else:
            i += 1

    # greedy aggregation
    while unprocessed_assigns:
        my_assign = unprocessed_assigns.pop()

        my_deps = my_assign.get_dependencies()
        my_assignees = get_var_assignees(my_assign)

        agg_candidates = []
        for i, other_assign in enumerate(unprocessed_assigns):
            other_deps = other_assign.get_dependencies()
            other_assignees = get_var_assignees(other_assign)

            if ((my_deps & other_deps
                    or my_deps & other_assignees
                    or other_deps & my_assignees)
                    and my_assign.priority == other_assign.priority):
                agg_candidates.append((i, other_assign))

        did_work = False

        if agg_candidates:
            my_indirect_origins = get_complete_origins_set(
                    my_assign, skip_levels=1)

            for other_assign_index, other_assign in agg_candidates:
                if max_vectors_in_batch_expr is not None:
                    new_assignee_count = len(
                            set(my_assign.get_assignees())
                            | set(other_assign.get_assignees()))
                    new_dep_count = len(
                            my_assign.get_dependencies(
                                each_vector=True)
                            | other_assign.get_dependencies(
                                each_vector=True))

                    if (new_assignee_count + new_dep_count
                            > max_vectors_in_batch_expr):
                        continue

                other_indirect_origins = get_complete_origins_set(
                        other_assign, skip_levels=1)

                if (my_assign not in other_indirect_origins
                        and other_assign not in my_indirect_origins):
                    did_work = True

                    # aggregate the two assignments
                    new_assignment = aggregate_two_assignments(
                            my_assign, other_assign)
                    del unprocessed_assigns[other_assign_index]
                    unprocessed_assigns.append(new_assignment)
                    for assignee in new_assignment.get_assignees():
                        origins_map[assignee] = new_assignment

                    break

        if not did_work:
            processed_assigns.append(my_assign)

    externally_used_names = {
            expr
            for insn in processed_assigns + other_insns
            for expr in insn.get_dependencies()}

    if isinstance(result, np.ndarray) and result.dtype.char == "O":
        externally_used_names |= {expr for expr in result}
    else:
        externally_used_names |= {result}

    def schedule_and_finalize_assignment(ass):
        dep_mapper = _make_dep_mapper(include_subscripts=False)

        names_exprs = list(zip(ass.names, ass.exprs))

        my_assignees = {name for name, expr in names_exprs}
        names_exprs_deps = [
                (name, expr,
                    {dep.name for dep in dep_mapper(expr) if
                        isinstance(dep, Variable)} & my_assignees)
                for name, expr in names_exprs]

        ordered_names_exprs = []
        available_names = set()

        while names_exprs_deps:
            schedulable = []

            i = 0
            while i < len(names_exprs_deps):
                name, expr, deps = names_exprs_deps[i]

                unsatisfied_deps = deps - available_names

                if not unsatisfied_deps:
                    schedulable.append((str(expr), name, expr))
                    del names_exprs_deps[i]
                else:
                    i += 1

            # make sure these come out in a constant order
            schedulable.sort()

            if schedulable:
                for key, name, expr in schedulable:
                    ordered_names_exprs.append((name, expr))
                    available_names.add(name)
            else:
                raise RuntimeError("aggregation resulted in an "
                        "impossible assignment")

        return Assign(
                names=[name for name, expr in ordered_names_exprs],
                exprs=[expr for name, expr in ordered_names_exprs],
                do_not_return=[Variable(name) not in externally_used_names
                    for name, expr in ordered_names_exprs],
                priority=ass.priority)

    return [schedule_and_finalize_assignment(ass)
        for ass in processed_assigns] + other_insns

    # }}}

# }}}


# {{{ to-loopy mapper

def is_external_call(expr, function_registry):
    from pymbolic.primitives import Call
    if not isinstance(expr, Call):
        return False
    return not is_function_loopyable(expr.function, function_registry)


def is_function_loopyable(function, function_registry):
    from grudge.symbolic.primitives import FunctionSymbol
    assert isinstance(function, FunctionSymbol)
    return function_registry[function.name].supports_codegen


class ToLoopyExpressionMapper(mappers.IdentityMapper):
    def __init__(self, dd_inference_mapper, temp_names, subscript):
        self.dd_inference_mapper = dd_inference_mapper
        self.function_registry = dd_inference_mapper.function_registry
        self.temp_names = temp_names
        self.subscript = subscript

        self.expr_to_name = {}
        self.used_names = set()
        self.non_scalar_vars = []

    def map_name(self, name):
        dot_idx = name.find(".")
        if dot_idx != -1:
            return "grdg_sub_{}_{}".format(name[:dot_idx], name[dot_idx+1:])
        else:
            return name

    def map_variable_ref_expr(self, expr, name_prefix):
        from pymbolic import var
        dd = self.dd_inference_mapper(expr)

        try:
            name = self.expr_to_name[expr]
        except KeyError:
            name_prefix = self.map_name(name_prefix)
            name = name_prefix

            suffix_nr = 0
            while name in self.used_names:
                name = f"{name_prefix}_{suffix_nr}"
                suffix_nr += 1
            self.used_names.add(name)

            self.expr_to_name[expr] = name

        from grudge.symbolic.primitives import DTAG_SCALAR
        if dd.domain_tag == DTAG_SCALAR or name in self.temp_names:
            return var(name)
        else:
            self.non_scalar_vars.append(name)
            return var(name)[self.subscript]

    def map_variable(self, expr):
        return self.map_variable_ref_expr(expr, expr.name)

    def map_grudge_variable(self, expr):
        return self.map_variable_ref_expr(expr, expr.name)

    def map_subscript(self, expr):
        subscript = expr.index
        if isinstance(subscript, tuple):
            assert len(subscript) == 1
            subscript, = subscript

        assert isinstance(subscript, int)

        return self.map_variable_ref_expr(
                expr,
                "%s_%d" % (expr.aggregate.name, subscript))

    def map_call(self, expr):
        if is_function_loopyable(expr.function, self.function_registry):
            from pymbolic import var

            func_name = expr.function.name
            if func_name == "fabs":
                func_name = "abs"

            return var(func_name)(
                    *[self.rec(par) for par in expr.parameters])
        else:
            raise NotImplementedError(
                    "do not know how to map function '%s' into loopy"
                    % expr.function)

    def map_ones(self, expr):
        return 1.0

    def map_node_coordinate_component(self, expr):
        return self.map_variable_ref_expr(
                expr, "grdg_ncc%d" % expr.axis)

    def map_common_subexpression(self, expr):
        raise ValueError("not expecting CSEs at this stage in the "
                "compilation process")


# {{{ bessel handling

BESSEL_PREAMBLE = """//CL//
#include <pyopencl-bessel-j.cl>
#include <pyopencl-bessel-y.cl>
"""


def bessel_preamble_generator(preamble_info):
    from loopy.target.pyopencl import PyOpenCLTarget
    if not isinstance(preamble_info.kernel.target, PyOpenCLTarget):
        raise NotImplementedError("Only the PyOpenCLTarget is supported as of now")

    if any(func.name in ["bessel_j", "bessel_y"]
            for func in preamble_info.seen_functions):
        yield ("50-grudge-bessel", BESSEL_PREAMBLE)


def bessel_function_mangler(kernel, name, arg_dtypes):
    from loopy.types import NumpyType
    if name == "bessel_j" and len(arg_dtypes) == 2:
        n_dtype, x_dtype, = arg_dtypes

        # *technically* takes a float, but let's not worry about that.
        if n_dtype.numpy_dtype.kind != "i":
            raise TypeError("%s expects an integer first argument")

        from loopy.kernel.data import CallMangleInfo
        return CallMangleInfo(
                "bessel_jv",
                (NumpyType(np.float64),),
                (NumpyType(np.int32), NumpyType(np.float64)),
                )

    elif name == "bessel_y" and len(arg_dtypes) == 2:
        n_dtype, x_dtype, = arg_dtypes

        # *technically* takes a float, but let's not worry about that.
        if n_dtype.numpy_dtype.kind != "i":
            raise TypeError("%s expects an integer first argument")

        from loopy.kernel.data import CallMangleInfo
        return CallMangleInfo(
                "bessel_yn",
                (NumpyType(np.float64),),
                (NumpyType(np.int32), NumpyType(np.float64)),
                )

    return None

# }}}


class ToLoopyInstructionMapper:
    def __init__(self, dd_inference_mapper):
        self.dd_inference_mapper = dd_inference_mapper
        self.function_registry = dd_inference_mapper.function_registry
        self.insn_count = 0

    def map_insn_assign(self, insn):
        from grudge.symbolic.primitives import OperatorBinding

        if (
                len(insn.exprs) == 1
                and (
                    isinstance(insn.exprs[0], OperatorBinding)
                    or is_external_call(
                        insn.exprs[0], self.function_registry))):
            return insn

        # FIXME: These names and the size names could clash with user-given names.
        # Need better metadata tracking in loopy.
        iel = "iel"
        idof = "idof"

        temp_names = [
                name
                for name, dnr in zip(insn.names, insn.do_not_return)
                if dnr]

        from pymbolic import var
        expr_mapper = ToLoopyExpressionMapper(
                self.dd_inference_mapper, temp_names, (var(iel), var(idof)))
        insns = []

        import loopy as lp
        from pymbolic import var
        for name, expr, dnr in zip(insn.names, insn.exprs, insn.do_not_return):
            insns.append(
                    lp.Assignment(
                        expr_mapper(var(name)),
                        expr_mapper(expr),
                        temp_var_type=lp.Optional(None) if dnr else lp.Optional(),
                        no_sync_with=frozenset([
                            ("*", "any"),
                            ]),
                        ))

        if not expr_mapper.non_scalar_vars:
            return insn

        knl = lp.make_kernel(
                "{[%(iel)s, %(idof)s]: "
                "0 <= %(iel)s < nelements and 0 <= %(idof)s < nunit_dofs}"
                % {"iel": iel, "idof": idof},
                insns,

                name="grudge_assign_%d" % self.insn_count,

                # Single-insn kernels may have their no_sync_with resolve to an
                # empty set, that's OK.
                options=lp.Options(
                    check_dep_resolution=False,
                    return_dict=True,
                    no_numpy=True,
                    )
                )

        self.insn_count += 1

        from pytools import single_valued
        governing_dd = single_valued(
                self.dd_inference_mapper(expr)
                for expr in insn.exprs)

        knl = lp.register_preamble_generators(knl,
                [bessel_preamble_generator])
        knl = lp.register_function_manglers(knl,
                [bessel_function_mangler])

        input_mappings = {}
        output_mappings = {}

        from grudge.symbolic.mappers import DependencyMapper
        dep_mapper = DependencyMapper(composite_leaves=False)

        for expr, name in expr_mapper.expr_to_name.items():
            deps = dep_mapper(expr)
            assert len(deps) <= 1
            if not deps:
                is_output = False
            else:
                dep, = deps
                is_output = dep.name in insn.names

            if is_output:
                tgt_dict = output_mappings
            else:
                tgt_dict = input_mappings

            tgt_dict[name] = expr

        return LoopyKernelInstruction(
            LoopyKernelDescriptor(
                loopy_kernel=knl,
                input_mappings=input_mappings,
                output_mappings=output_mappings,
                fixed_arguments={},
                governing_dd=governing_dd)
            )

    def map_insn_rank_data_swap(self, insn):
        return insn

    def map_insn_assign_to_discr_scoped(self, insn):
        return insn

    def map_insn_assign_from_discr_scoped(self, insn):
        return insn

    def map_insn_diff_batch_assign(self, insn):
        return insn


def rewrite_insn_to_loopy_insns(inf_mapper, insn_list):
    insn_mapper = ToLoopyInstructionMapper(inf_mapper)

    return [
            getattr(insn_mapper, insn.mapper_method)(insn)
            for insn in insn_list]

# }}}


# {{{ compiler

class CodeGenerationState(Record):
    """
    .. attribute:: generating_discr_code
    """

    def get_code_list(self, compiler):
        if self.generating_discr_code:
            return compiler.discr_code
        else:
            return compiler.eval_code


class OperatorCompiler(mappers.IdentityMapper):
    def __init__(self, discr, function_registry,
            prefix="_expr", max_vectors_in_batch_expr=None):
        super().__init__()
        self.prefix = prefix

        self.max_vectors_in_batch_expr = max_vectors_in_batch_expr

        self.discr_code = []
        self.discr_scope_names_created = set()
        self.discr_scope_names_copied_to_eval = set()

        self.eval_code = []
        self.expr_to_var = {}

        self.assigned_names = set()

        self.discr = discr
        self.function_registry = function_registry

        from pytools import UniqueNameGenerator
        self.name_gen = UniqueNameGenerator()

    # {{{ collect various optemplate components

    def collect_diff_ops(self, expr):
        return mappers.BoundOperatorCollector(sym.RefDiffOperatorBase)(expr)

    # }}}

    # {{{ top-level driver

    def __call__(self, expr):
        # Put the result expressions into variables as well.
        expr = sym.cse(expr, "_result")

        # from grudge.symbolic.mappers.type_inference import TypeInferrer
        # self.typedict = TypeInferrer()(expr)

        # Used for diff batching
        self.diff_ops = self.collect_diff_ops(expr)

        codegen_state = CodeGenerationState(generating_discr_code=False)
        # Finally, walk the expression and build the code.
        result = super().__call__(expr, codegen_state)

        eval_code = self.eval_code
        del self.eval_code
        discr_code = self.discr_code
        del self.discr_code

        from grudge.symbolic.dofdesc_inference import DOFDescInferenceMapper
        inf_mapper = DOFDescInferenceMapper(
                discr_code + eval_code, self.function_registry)

        eval_code = aggregate_assignments(
                inf_mapper, eval_code, result, self.max_vectors_in_batch_expr)

        discr_code = rewrite_insn_to_loopy_insns(inf_mapper, discr_code)
        eval_code = rewrite_insn_to_loopy_insns(inf_mapper, eval_code)

        from pytools.obj_array import make_obj_array
        return (
                Code(discr_code,
                    make_obj_array(
                        [Variable(name)
                            for name in self.discr_scope_names_copied_to_eval])),
                Code(eval_code, result))

    # }}}

    # {{{ variables and names

    def assign_to_new_var(self, codegen_state, expr, priority=0, prefix=None):
        # Observe that the only things that can be legally subscripted in
        # grudge are variables. All other expressions are broken down into
        # their scalar components.
        if isinstance(expr, (Variable, Subscript)):
            return expr

        new_name = self.name_gen(prefix if prefix is not None else "expr")
        codegen_state.get_code_list(self).append(Assign(
            (new_name,), (expr,), priority=priority))

        return Variable(new_name)

    # }}}

    # {{{ map_xxx routines

    def map_common_subexpression(self, expr, codegen_state):
        def get_rec_child(codegen_state):
            if isinstance(expr.child, sym.OperatorBinding):
                # We need to catch operator bindings here and
                # treat them specially. They get assigned to their
                # own variable by default, which would mean the
                # CSE prefix would be omitted, making the resulting
                # code less readable.
                return self.map_operator_binding(
                        expr.child, codegen_state, name_hint=expr.prefix)
            else:
                return self.rec(expr.child, codegen_state)

        if expr.scope == sym.cse_scope.DISCRETIZATION:
            from pymbolic import var
            try:
                expr_name = self.discr._discr_scoped_subexpr_to_name[expr.child]
            except KeyError:
                expr_name = "discr." + self.discr._discr_scoped_name_gen(
                        expr.prefix if expr.prefix is not None else "expr")
                self.discr._discr_scoped_subexpr_to_name[expr.child] = expr_name

            assert expr_name.startswith("discr.")

            priority = getattr(expr, "priority", 0)

            if expr_name not in self.discr_scope_names_created:
                new_codegen_state = codegen_state.copy(generating_discr_code=True)
                rec_child = get_rec_child(new_codegen_state)

                new_codegen_state.get_code_list(self).append(
                        ToDiscretizationScopedAssign(
                            (expr_name,), (rec_child,), priority=priority))

                self.discr_scope_names_created.add(expr_name)

            if codegen_state.generating_discr_code:
                return var(expr_name)
            else:
                if expr_name in self.discr_scope_names_copied_to_eval:
                    return var(expr_name)

                self.eval_code.append(
                        FromDiscretizationScopedAssign(
                            expr_name, priority=priority))

                self.discr_scope_names_copied_to_eval.add(expr_name)

                return var(expr_name)

        else:
            try:
                return self.expr_to_var[expr.child]
            except KeyError:
                priority = getattr(expr, "priority", 0)

                rec_child = get_rec_child(codegen_state)

                cse_var = self.assign_to_new_var(
                        codegen_state, rec_child,
                        priority=priority, prefix=expr.prefix)

                self.expr_to_var[expr.child] = cse_var
                return cse_var

    def map_operator_binding(self, expr, codegen_state, name_hint=None):
        if isinstance(expr.op, sym.RefDiffOperatorBase):
            return self.map_ref_diff_op_binding(expr, codegen_state)
        elif isinstance(expr.op, sym.OppositePartitionFaceSwap):
            return self.map_rank_data_swap_binding(expr, codegen_state, name_hint)
        else:
            # make sure operator assignments stand alone and don't get muddled
            # up in vector math
            field_var = self.assign_to_new_var(
                    codegen_state,
                    self.rec(expr.field, codegen_state))
            result_var = self.assign_to_new_var(
                    codegen_state,
                    expr.op(field_var),
                    prefix=name_hint)
            return result_var

    def map_call(self, expr, codegen_state):
        if is_function_loopyable(expr.function, self.function_registry):
            return super().map_call(expr, codegen_state)
        else:
            # If it's not a C-level function, it shouldn't get muddled up into
            # a vector math expression.

            return self.assign_to_new_var(
                    codegen_state,
                    type(expr)(
                        expr.function,
                        [self.assign_to_new_var(
                            codegen_state,
                            self.rec(par, codegen_state))
                            for par in expr.parameters]))

    def map_ref_diff_op_binding(self, expr, codegen_state):
        try:
            return self.expr_to_var[expr]
        except KeyError:
            all_diffs = [diff
                    for diff in self.diff_ops
                    if diff.op.equal_except_for_axis(expr.op)
                    and diff.field == expr.field]

            names = [self.name_gen("expr") for d in all_diffs]

            from pytools import single_valued
            op_class = single_valued(type(d.op) for d in all_diffs)

            codegen_state.get_code_list(self).append(
                    DiffBatchAssign(
                        names=names,
                        op_class=op_class,
                        operators=[d.op for d in all_diffs],
                        field=self.rec(
                            single_valued(d.field for d in all_diffs),
                            codegen_state)))

            from pymbolic import var
            for n, d in zip(names, all_diffs):
                self.expr_to_var[d] = var(n)

            return self.expr_to_var[expr]

    def map_rank_data_swap_binding(self, expr, codegen_state, name_hint):
        try:
            return self.expr_to_var[expr]
        except KeyError:
            field = self.rec(expr.field, codegen_state)
            name = self.name_gen("raw_rank%02d_bdry_data" % expr.op.i_remote_part)
            field_insn = RankDataSwapAssign(name=name, field=field, op=expr.op)
            codegen_state.get_code_list(self).append(field_insn)
            field_var = Variable(field_insn.name)
            self.expr_to_var[expr] = self.assign_to_new_var(codegen_state,
                                                            expr.op(field_var),
                                                            prefix=name_hint)
            return self.expr_to_var[expr]

    # }}}

# }}}


# vim: foldmethod=marker
