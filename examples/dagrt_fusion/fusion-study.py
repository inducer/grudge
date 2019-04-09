#!/usr/bin/env python3
"""A study of operator fusion for time integration operators in Grudge.
"""

from __future__ import division, print_function

__copyright__ = """
Copyright (C) 2015 Andreas Kloeckner
Copyright (C) 2019 Matt Wala
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


import logging
import numpy as np
import six
import pyopencl as cl
import pyopencl.array  # noqa
import pytest

import dagrt.language as lang
import pymbolic.primitives as p
import grudge.symbolic.mappers as gmap
import grudge.symbolic.operators as op
from grudge.execution import ExecutionMapper
from pymbolic.mapper.evaluator import EvaluationMapper \
        as PymbolicEvaluationMapper

from grudge import sym, bind, DGDiscretizationWithBoundaries
from leap.rk import LSRK4Method

from pyopencl.tools import (  # noqa
        pytest_generate_tests_for_pyopencl as pytest_generate_tests)


logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)

logger.setLevel(logging.DEBUG)


# {{{ topological sort

def topological_sort(stmts, root_deps):
    id_to_stmt = {stmt.id: stmt for stmt in stmts}

    ordered_stmts = []
    satisfied = set()

    def satisfy_dep(name):
        if name in satisfied:
            return

        stmt = id_to_stmt[name]
        for dep in sorted(stmt.depends_on):
            satisfy_dep(dep)
        ordered_stmts.append(stmt)
        satisfied.add(name)

    for d in root_deps:
        satisfy_dep(d)

    return ordered_stmts

# }}}


# {{{ leap to grudge translation

# Use evaluation, not identity mappers to propagate symbolic vectors to
# outermost level.

class DagrtToGrudgeRewriter(PymbolicEvaluationMapper):
    def __init__(self, context):
        self.context = context

    def map_variable(self, expr):
        return self.context[expr.name]

    def map_call(self, expr):
        raise ValueError("function call not expected")


class GrudgeArgSubstitutor(gmap.SymbolicEvaluator):
    def __init__(self, args):
        super().__init__(context={})
        self.args = args

    def map_grudge_variable(self, expr):
        if expr.name in self.args:
            return self.args[expr.name]
        return super().map_variable(expr)


def transcribe_phase(dag, field_var_name, field_components, phase_name,
                     sym_operator):
    """Arguments:

        dag: The Dagrt code for the time integrator

        field_var_name: The name of the simulation variable

        field_components: The number of components (fields) in the variable

        phase_name: The name of the phase to transcribe in the Dagrt code

        sym_operator: The Grudge symbolic expression to substitue for the
            right-hand side evaluation in the Dagrt code
    """
    sym_operator = gmap.OperatorBinder()(sym_operator)
    phase = dag.phases[phase_name]

    ctx = {
            "<t>": sym.var("input_t", sym.DD_SCALAR),
            "<dt>": sym.var("input_dt", sym.DD_SCALAR),
            f"<state>{field_var_name}": sym.make_sym_array(
                f"input_{field_var_name}", field_components),
            f"<p>residual": sym.make_sym_array(
                "input_residual", field_components),
    }

    rhs_name = f"<func>{field_var_name}"
    output_vars = [v for v in ctx]
    yielded_states = []

    from dagrt.codegen.transform import isolate_function_calls_in_phase
    ordered_stmts = topological_sort(
            isolate_function_calls_in_phase(
                phase,
                dag.get_stmt_id_generator(),
                dag.get_var_name_generator()).statements,
            phase.depends_on)

    for stmt in ordered_stmts:
        if stmt.condition is not True:
            raise NotImplementedError(
                "non-True condition (in statement '%s') not supported"
                % stmt.id)

        if isinstance(stmt, lang.Nop):
            pass

        elif isinstance(stmt, lang.AssignExpression):
            if not isinstance(stmt.lhs, p.Variable):
                raise NotImplementedError("lhs of statement %s is not a variable: %s"
                        % (stmt.id, stmt.lhs))
            ctx[stmt.lhs.name] = sym.cse(
                DagrtToGrudgeRewriter(ctx)(stmt.rhs),
                (
                    stmt.lhs.name
                    .replace("<", "")
                    .replace(">", "")))

        elif isinstance(stmt, lang.AssignFunctionCall):
            if stmt.function_id != rhs_name:
                raise NotImplementedError(
                        "statement '%s' calls unsupported function '%s'"
                        % (stmt.id, stmt.function_id))

            if stmt.parameters:
                raise NotImplementedError(
                    "statement '%s' calls function '%s' with positional arguments"
                    % (stmt.id, stmt.function_id))

            kwargs = {name: sym.cse(DagrtToGrudgeRewriter(ctx)(arg))
                      for name, arg in stmt.kw_parameters.items()}

            if len(stmt.assignees) != 1:
                raise NotImplementedError(
                    "statement '%s' calls function '%s' "
                    "with more than one LHS"
                    % (stmt.id, stmt.function_id))

            assignee, = stmt.assignees
            ctx[assignee] = GrudgeArgSubstitutor(kwargs)(sym_operator)

        elif isinstance(stmt, lang.YieldState):
            d2g = DagrtToGrudgeRewriter(ctx)
            yielded_states.append(
                (stmt.time_id, d2g(stmt.time), stmt.component_id,
                    d2g(stmt.expression)))

        else:
            raise NotImplementedError("statement %s is of unsupported type ''%s'"
                        % (stmt.id, type(stmt).__name__))

    return output_vars, [ctx[ov] for ov in output_vars], yielded_states

# }}}


# {{{ time integrator implementations

class RK4TimeStepperBase(object):

    def __init__(self, queue, component_getter):
        self.queue = queue
        self.component_getter = component_getter

    def get_initial_context(self, fields, t_start, dt):
        from pytools.obj_array import join_fields

        # Flatten fields.
        flattened_fields = []
        for field in fields:
            if isinstance(field, list):
                flattened_fields.extend(field)
            else:
                flattened_fields.append(field)
        flattened_fields = join_fields(*flattened_fields)
        del fields

        return {
                "input_t": t_start,
                "input_dt": dt,
                self.state_name: flattened_fields,
                "input_residual": flattened_fields,
        }

    def set_up_stepper(self, discr, field_var_name, sym_rhs, num_fields,
                       exec_mapper_factory=ExecutionMapper):
        dt_method = LSRK4Method(component_id=field_var_name)
        dt_code = dt_method.generate()
        self.field_var_name = field_var_name
        self.state_name = f"input_{field_var_name}"

        # Transcribe the phase.
        output_vars, results, yielded_states = transcribe_phase(
                dt_code, field_var_name, num_fields,
                "primary", sym_rhs)

        # Build the bound operator for the time integrator.
        output_t = results[0]
        output_dt = results[1]
        output_states = results[2]
        output_residuals = results[3]

        assert len(output_states) == num_fields
        assert len(output_states) == len(output_residuals)

        from pytools.obj_array import join_fields
        flattened_results = join_fields(output_t, output_dt, *output_states)

        self.bound_op = bind(
                discr, flattened_results, exec_mapper_factory=exec_mapper_factory)

    def run(self, fields, t_start, dt, t_end, return_profile_data=False):
        context = self.get_initial_context(fields, t_start, dt)

        t = t_start

        while t <= t_end:
            if return_profile_data:
                profile_data = dict()
            else:
                profile_data = None

            results = self.bound_op(
                    self.queue,
                    profile_data=profile_data,
                    **context)

            if return_profile_data:
                results = results[0]

            t = results[0]
            context["input_t"] = t
            context["input_dt"] = results[1]
            output_states = results[2:]
            context[self.state_name] = output_states

            result = (t, self.component_getter(output_states))
            if return_profile_data:
                result += (profile_data,)

            yield result


class RK4TimeStepper(RK4TimeStepperBase):

    def __init__(self, queue, discr, field_var_name, grudge_bound_op,
                 num_fields, component_getter, exec_mapper_factory=ExecutionMapper):
        """Arguments:

            field_var_name: The name of the simulation variable

            grudge_bound_op: The BoundExpression for the right-hand side

            num_fields: The number of components in the simulation variable

            component_getter: A function, which, given an object array
               representing the simulation variable, splits the array into
               its components

        """
        super().__init__(queue, component_getter)

        from pymbolic import var

        # Construct sym_rhs to have the effect of replacing the RHS calls in the
        # dagrt code with calls of the grudge operator.
        from grudge.symbolic.primitives import ExternalCall, Variable
        call = sym.cse(ExternalCall(
                var("grudge_op"),
                (
                    (Variable("t", dd=sym.DD_SCALAR),)
                    + tuple(
                        Variable(field_var_name, dd=sym.DD_VOLUME)[i]
                        for i in range(num_fields))),
                dd=sym.DD_VOLUME))

        from pytools.obj_array import join_fields
        sym_rhs = join_fields(*(call[i] for i in range(num_fields)))

        self.queue = queue
        self.grudge_bound_op = grudge_bound_op
        self.set_up_stepper(
                discr, field_var_name, sym_rhs, num_fields, exec_mapper_factory)
        self.component_getter = component_getter

    def _bound_op(self, t, *args, profile_data=None):
        from pytools.obj_array import join_fields
        context = {
                "t": t,
                self.field_var_name: join_fields(*args)}
        result = self.grudge_bound_op(
                self.queue, profile_data=profile_data, **context)
        if profile_data is not None:
            result = result[0]
        return result

    def get_initial_context(self, fields, t_start, dt):
        context = super().get_initial_context(fields, t_start, dt)
        context["grudge_op"] = self._bound_op
        return context


class FusedRK4TimeStepper(RK4TimeStepperBase):

    def __init__(self, queue, discr, field_var_name, sym_rhs, num_fields,
                 component_getter, exec_mapper_factory=ExecutionMapper):
        super().__init__(queue, component_getter)
        self.set_up_stepper(
                discr, field_var_name, sym_rhs, num_fields, exec_mapper_factory)

# }}}


# {{{ problem setup code

def get_strong_wave_op_with_discr(cl_ctx, dims=2, order=4):
    from meshmode.mesh.generation import generate_regular_rect_mesh
    mesh = generate_regular_rect_mesh(
            a=(-0.5,)*dims,
            b=(0.5,)*dims,
            n=(256,)*dims)

    logger.debug("%d elements" % mesh.nelements)

    discr = DGDiscretizationWithBoundaries(cl_ctx, mesh, order=order)

    source_center = np.array([0.1, 0.22, 0.33])[:dims]
    source_width = 0.05
    source_omega = 3

    sym_x = sym.nodes(mesh.dim)
    sym_source_center_dist = sym_x - source_center
    sym_t = sym.ScalarVariable("t")

    from grudge.models.wave import StrongWaveOperator
    from meshmode.mesh import BTAG_ALL, BTAG_NONE
    op = StrongWaveOperator(-0.1, dims,
            source_f=(
                sym.sin(source_omega*sym_t)
                * sym.exp(
                    -np.dot(sym_source_center_dist, sym_source_center_dist)
                    / source_width**2)),
            dirichlet_tag=BTAG_NONE,
            neumann_tag=BTAG_NONE,
            radiation_tag=BTAG_ALL,
            flux_type="upwind")

    op.check_bc_coverage(mesh)

    return (op, discr)


def get_strong_wave_component(state_component):
    return (state_component[0], state_component[1:])

# }}}


# {{{ equivalence check between fused and non-fused versions

def test_stepper_equivalence(ctx_factory, order=4):
    cl_ctx = ctx_factory()
    queue = cl.CommandQueue(cl_ctx)

    dims = 2

    op, discr = get_strong_wave_op_with_discr(cl_ctx, dims=dims, order=order)

    if dims == 2:
        dt = 0.04
    elif dims == 3:
        dt = 0.02

    from pytools.obj_array import join_fields
    ic = join_fields(discr.zeros(queue),
            [discr.zeros(queue) for i in range(discr.dim)])

    bound_op = bind(discr, op.sym_operator())

    stepper = RK4TimeStepper(
            queue, discr, "w", bound_op, 1 + discr.dim, get_strong_wave_component)

    fused_stepper = FusedRK4TimeStepper(
            queue, discr, "w", op.sym_operator(), 1 + discr.dim,
            get_strong_wave_component)

    t_start = 0
    t_end = 0.5
    nsteps = int(np.ceil((t_end + 1e-9) / dt))
    print("dt=%g nsteps=%d" % (dt, nsteps))

    step = 0

    norm = bind(discr, sym.norm(2, sym.var("u_ref") - sym.var("u")))

    fused_steps = fused_stepper.run(ic, t_start, dt, t_end)

    for t_ref, (u_ref, v_ref) in stepper.run(ic, t_start, dt, t_end):
        step += 1
        logger.debug("step %d/%d", step, nsteps)
        t, (u, v) = next(fused_steps)
        assert t == t_ref, step
        assert norm(queue, u=u, u_ref=u_ref) <= 1e-13, step

# }}}


# {{{ mem op counter implementation

class ExecutionMapperWithMemOpCounting(ExecutionMapper):
    # This is a skeleton implementation that only has just enough functionality
    # for the wave-min example to work.

    def __init__(self, queue, context, bound_op):
        super().__init__(queue, context, bound_op)

    def map_external_call(self, expr):
        # Should have been caught by our op counter
        assert False, ("map_external_call called: %s" % expr)

    # {{{ expressions

    def map_profiled_external_call(self, expr, profile_data):
        from pymbolic.primitives import Variable
        assert isinstance(expr.function, Variable)
        args = [self.rec(p) for p in expr.parameters]
        return self.context[expr.function.name](*args, profile_data=profile_data)

    def map_profiled_essentially_elementwise_linear(self, op, field_expr,
                                                    profile_data):
        result = getattr(self, op.mapper_method)(op, field_expr)

        if profile_data is not None:
            # We model the cost to load the input and write the output.  In
            # particular, we assume the elementwise matrices are negligible in
            # size and thus ignorable.

            field = self.rec(field_expr)
            profile_data["bytes_read"] = (
                    profile_data.get("bytes_read", 0) + field.nbytes)
            profile_data["bytes_written"] = (
                    profile_data.get("bytes_written", 0) + result.nbytes)

        return result

    # }}}

    # {{{ instruction mappings

    def process_assignment_expr(self, expr, profile_data):
        if isinstance(expr, sym.ExternalCall):
            assert expr.mapper_method == "map_external_call"
            val = self.map_profiled_external_call(expr, profile_data)

        elif isinstance(expr, sym.OperatorBinding):
            if isinstance(
                    expr.op,
                    (
                        # TODO: Not comprehensive.
                        op.InterpolationOperator,
                        op.RefFaceMassOperator,
                        op.RefInverseMassOperator,
                        op.OppositeInteriorFaceSwap)):
                val = self.map_profiled_essentially_elementwise_linear(
                        expr.op, expr.field, profile_data)

            else:
                assert False, ("unknown operator: %s" % expr.op)

        else:
            logger.debug("assignment not profiled: %s", expr)
            val = self.rec(expr)

        return val

    def map_insn_assign(self, insn, profile_data):
        result = []
        for name, expr in zip(insn.names, insn.exprs):
            result.append((name, self.process_assignment_expr(expr, profile_data)))
        return result, []

    def map_insn_loopy_kernel(self, insn, profile_data):
        kwargs = {}
        kdescr = insn.kernel_descriptor
        for name, expr in six.iteritems(kdescr.input_mappings):
            val = self.rec(expr)
            kwargs[name] = val
            assert not isinstance(val, np.ndarray)
            if profile_data is not None and isinstance(val, pyopencl.array.Array):
                profile_data["bytes_read"] = (
                        profile_data.get("bytes_read", 0) + val.nbytes)
                profile_data["bytes_read_within_assignments"] = (
                        profile_data.get("bytes_read_within_assignments", 0)
                        + val.nbytes)

        discr = self.discrwb.discr_from_dd(kdescr.governing_dd)
        for name in kdescr.scalar_args():
            v = kwargs[name]
            if isinstance(v, (int, float)):
                kwargs[name] = discr.real_dtype.type(v)
            elif isinstance(v, complex):
                kwargs[name] = discr.complex_dtype.type(v)
            elif isinstance(v, np.number):
                pass
            else:
                raise ValueError("unrecognized scalar type for variable '%s': %s"
                        % (name, type(v)))

        kwargs["grdg_n"] = discr.nnodes
        evt, result_dict = kdescr.loopy_kernel(self.queue, **kwargs)

        for val in result_dict.values():
            assert not isinstance(val, np.ndarray)
            if profile_data is not None and isinstance(val, pyopencl.array.Array):
                profile_data["bytes_written"] = (
                        profile_data.get("bytes_written", 0) + val.nbytes)
                profile_data["bytes_written_within_assignments"] = (
                        profile_data.get("bytes_written_within_assignments", 0)
                        + val.nbytes)

        return list(result_dict.items()), []

    def map_insn_assign_to_discr_scoped(self, insn, profile_data=None):
        assignments = []

        for name, expr in zip(insn.names, insn.exprs):
            logger.debug("assignment not profiled: %s <- %s", name, expr)
            value = self.rec(expr)
            self.discrwb._discr_scoped_subexpr_name_to_value[name] = value
            assignments.append((name, value))

        return assignments, []

    def map_insn_assign_from_discr_scoped(self, insn, profile_data=None):
        return [(insn.name,
            self.discrwb._discr_scoped_subexpr_name_to_value[insn.name])], []

    def map_insn_rank_data_swap(self, insn, profile_data):
        raise NotImplementedError("no profiling for instruction: %s" % insn)

    def map_insn_diff_batch_assign(self, insn, profile_data):
        assignments, futures = super().map_insn_diff_batch_assign(insn)

        if profile_data is not None:
            # We model the cost to load the input and write the output.  In
            # particular, we assume the elementwise matrices are negligible in
            # size and thus ignorable.

            field = self.rec(insn.field)
            profile_data["bytes_read"] = (
                    profile_data.get("bytes_read", 0) + field.nbytes)

            for _, value in assignments:
                profile_data["bytes_written"] = (
                        profile_data.get("bytes_written", 0) + value.nbytes)

        return assignments, futures

    # }}}

# }}}


# {{{ mem op counter check

@pytest.mark.parametrize("use_fusion", (True, False))
def test_stepper_mem_ops(ctx_factory, use_fusion):
    cl_ctx = ctx_factory()
    queue = cl.CommandQueue(cl_ctx)

    dims = 2

    op, discr = get_strong_wave_op_with_discr(cl_ctx, dims=dims, order=3)

    t_start = 0
    dt = 0.04
    t_end = 0.2

    from pytools.obj_array import join_fields
    ic = join_fields(discr.zeros(queue),
            [discr.zeros(queue) for i in range(discr.dim)])

    if not use_fusion:
        bound_op = bind(
                discr, op.sym_operator(),
                exec_mapper_factory=ExecutionMapperWithMemOpCounting)

        stepper = RK4TimeStepper(
                queue, discr, "w", bound_op, 1 + discr.dim,
                get_strong_wave_component,
                exec_mapper_factory=ExecutionMapperWithMemOpCounting)

    else:
        stepper = FusedRK4TimeStepper(
                queue, discr, "w", op.sym_operator(), 1 + discr.dim,
                get_strong_wave_component,
                exec_mapper_factory=ExecutionMapperWithMemOpCounting)

    step = 0

    nsteps = int(np.ceil((t_end + 1e-9) / dt))
    for (_, _, profile_data) in stepper.run(
            ic, t_start, dt, t_end, return_profile_data=True):
        step += 1
        logger.info("step %d/%d", step, nsteps)

    logger.info("fusion? %s", use_fusion)
    logger.info("bytes read: %d", profile_data["bytes_read"])
    logger.info("bytes written: %d", profile_data["bytes_written"])
    logger.info("bytes total: %d",
            profile_data["bytes_read"] + profile_data["bytes_written"])

# }}}


# {{{ execution mapper with timing

def time_insn(f):
    from pytools import ProcessTimer
    time_field_name = "time_%s" % f.__name__

    def wrapper(self, insn, profile_data):
        timer = ProcessTimer()
        retval = f(self, insn, profile_data)
        timer.done()

        if profile_data is not None:
            profile_data[time_field_name] = (
                    profile_data.get(time_field_name, 0)
                    + timer.wall_elapsed)

        return retval

    return wrapper


class ExecutionMapperWithTiming(ExecutionMapper):

    def map_external_call(self, expr):
        # Should have been caught by our implementation.
        assert False, ("map_external_call called: %s" % (expr))
        
    def map_operator_binding(self, expr):
        # Should have been caught by our implementation.
        assert False, ("map_operator_binding called: %s" % expr)

    def map_profiled_external_call(self, expr, profile_data):
        from pymbolic.primitives import Variable
        assert isinstance(expr.function, Variable)
        args = [self.rec(p) for p in expr.parameters]
        return self.context[expr.function.name](*args, profile_data=profile_data)

    def map_profiled_operator_binding(self, expr, profile_data):
        from pytools import ProcessTimer
        timer = ProcessTimer()
        retval = super().map_operator_binding(expr)
        timer.done()
        if profile_data is not None:
            time_field_name = "time_op_%s" % expr.op.mapper_method
            profile_data[time_field_name] = (
                    profile_data.get(time_field_name, 0)
                    + timer.wall_elapsed)
        return retval

    @time_insn
    def map_insn_loopy_kernel(self, *args, **kwargs):
        return super().map_insn_loopy_kernel(*args, **kwargs)

    @time_insn
    def map_insn_assign(self, insn, profile_data):
        if len(insn.exprs) == 1:
            if isinstance(insn.exprs[0], sym.ExternalCall):
                assert insn.exprs[0].mapper_method == "map_external_call"
                val = self.map_profiled_external_call(insn.exprs[0], profile_data)
                return [(insn.names[0], val)], []
            elif isinstance(insn.exprs[0], sym.OperatorBinding):
                assert insn.exprs[0].mapper_method == "map_operator_binding"
                val = self.map_profiled_operator_binding(insn.exprs[0], profile_data)
                return [(insn.names[0], val)], []
                                                  
        return super().map_insn_assign(insn, profile_data)

    @time_insn
    def map_insn_assign_to_discr_scoped(self, insn, profile_data):
        return super().map_insn_assign_to_discr_scoped(insn, profile_data)

    @time_insn
    def map_insn_assign_from_discr_scoped(self, insn, profile_data):
        return super().map_insn_assign_from_discr_scoped(insn, profile_data)

    @time_insn
    def map_insn_diff_batch_assign(self, insn, profile_data):
        return super().map_insn_diff_batch_assign(insn, profile_data)

# }}}


# {{{ timing check

@pytest.mark.parametrize("use_fusion", (True, False))
def test_stepper_timing(ctx_factory, use_fusion):
    cl_ctx = ctx_factory()
    queue = cl.CommandQueue(cl_ctx)

    dims = 2

    op, discr = get_strong_wave_op_with_discr(cl_ctx, dims=dims, order=3)

    t_start = 0
    dt = 0.04
    t_end = 0.1

    from pytools.obj_array import join_fields
    ic = join_fields(discr.zeros(queue),
            [discr.zeros(queue) for i in range(discr.dim)])

    if not use_fusion:
        bound_op = bind(
                discr, op.sym_operator(),
                exec_mapper_factory=ExecutionMapperWithTiming)

        stepper = RK4TimeStepper(
                queue, discr, "w", bound_op, 1 + discr.dim,
                get_strong_wave_component,
                exec_mapper_factory=ExecutionMapperWithTiming)

    else:
        stepper = FusedRK4TimeStepper(
                queue, discr, "w", op.sym_operator(), 1 + discr.dim,
                get_strong_wave_component,
                exec_mapper_factory=ExecutionMapperWithTiming)

    step = 0

    import time
    t = time.time()
    nsteps = int(np.ceil((t_end + 1e-9) / dt))
    for (_, _, profile_data) in stepper.run(
            ic, t_start, dt, t_end, return_profile_data=True):
        step += 1
        tn = time.time()
        logger.info("step %d/%d: %f", step, nsteps, tn - t)
        t = tn

    logger.info("fusion? %s", use_fusion)
    logger.info(profile_data)

# }}}


# {{{ paper outputs

def get_example_stepper(queue, dims=2, order=3, use_fusion=True,
                        exec_mapper_factory=ExecutionMapper,
                        return_ic=False):
    op, discr = get_strong_wave_op_with_discr(queue.context, dims=dims, order=3)

    if not use_fusion:
        bound_op = bind(
                discr, op.sym_operator(),
                exec_mapper_factory=exec_mapper_factory)

        stepper = RK4TimeStepper(
                queue, discr, "w", bound_op, 1 + discr.dim,
                get_strong_wave_component,
                exec_mapper_factory=exec_mapper_factory)

    else:
        stepper = FusedRK4TimeStepper(
                queue, discr, "w", op.sym_operator(), 1 + discr.dim,
                get_strong_wave_component,
                exec_mapper_factory=exec_mapper_factory)

    if return_ic:
        from pytools.obj_array import join_fields
        ic = join_fields(discr.zeros(queue),
                [discr.zeros(queue) for i in range(discr.dim)])
        return stepper, ic

    return stepper


def statement_counts_table():
    cl_ctx = cl.create_some_context()
    queue = cl.CommandQueue(cl_ctx)

    fused_stepper = get_example_stepper(queue, use_fusion=True)
    stepper = get_example_stepper(queue, use_fusion=False)

    print(r"\begin{tabular}{lr}")
    print(r"\toprule")
    print(r"Operator & Grudge Node Count \\")
    print(r"\midrule")
    print(
            r"Time integration (not fused) & %d \\"
            % len(stepper.bound_op.eval_code.instructions))
    print(
            r"Right-hand side (not fused) & %d \\"
            % len(stepper.grudge_bound_op.eval_code.instructions))
    print(
            r"Fused operator & %d \\"
            % len(fused_stepper.bound_op.eval_code.instructions))
    print(r"\bottomrule")
    print(r"\end{tabular}")


"""
def graphs():
    cl_ctx = cl.create_some_context()
    queue = cl.CommandQueue(cl_ctx)

    fused_stepper = get_example_stepper(queue, use_fusion=True)
    stepper = get_example_stepper(queue, use_fusion=False)

    from grudge.symbolic.compiler import dot_dataflow_graph
"""


def mem_ops_table():
    cl_ctx = cl.create_some_context()
    queue = cl.CommandQueue(cl_ctx)

    fused_stepper = get_example_stepper(
            queue,
            use_fusion=True,
            exec_mapper_factory=ExecutionMapperWithMemOpCounting)

    stepper, ic = get_example_stepper(
            queue,
            use_fusion=False,
            exec_mapper_factory=ExecutionMapperWithMemOpCounting,
            return_ic=True)

    t_start = 0
    dt = 0.02
    t_end = 0.02

    for (_, _, profile_data) in stepper.run(
            ic, t_start, dt, t_end, return_profile_data=True):
        pass

    nonfused_bytes_read = profile_data["bytes_read"]
    nonfused_bytes_written = profile_data["bytes_written"]
    nonfused_bytes_total = nonfused_bytes_read + nonfused_bytes_written

    for (_, _, profile_data) in fused_stepper.run(
            ic, t_start, dt, t_end, return_profile_data=True):
        pass

    fused_bytes_read = profile_data["bytes_read"]
    fused_bytes_written = profile_data["bytes_written"]
    fused_bytes_total = fused_bytes_read + fused_bytes_written

    print(r"\begin{tabular}{lrrr}")
    print(r"\toprule")
    print(r"Operator & Bytes Read & Bytes Written & Total (\% of Baseline) \\")
    print(r"\midrule")
    print(
            r"Baseline & \num{%d} & \num{%d} & \num{%d} (100) \\"
            % (
                nonfused_bytes_read,
                nonfused_bytes_written,
                nonfused_bytes_total))
    print(
            r"Fused & \num{%d} & \num{%d} & \num{%d} (%.1f) \\"
            % (
                fused_bytes_read,
                fused_bytes_written,
                fused_bytes_total,
                100 * fused_bytes_total / nonfused_bytes_total))
    print(r"\bottomrule")
    print(r"\end{tabular}")


def mem_ops_table():
    cl_ctx = cl.create_some_context()
    queue = cl.CommandQueue(cl_ctx)

    fused_stepper = get_example_stepper(
            queue,
            use_fusion=True,
            exec_mapper_factory=ExecutionMapperWithMemOpCounting)

    stepper, ic = get_example_stepper(
            queue,
            use_fusion=False,
            exec_mapper_factory=ExecutionMapperWithMemOpCounting,
            return_ic=True)

    t_start = 0
    dt = 0.02
    t_end = 0.02

    for (_, _, profile_data) in stepper.run(
            ic, t_start, dt, t_end, return_profile_data=True):
        pass

    nonfused_bytes_read = profile_data["bytes_read"]
    nonfused_bytes_written = profile_data["bytes_written"]
    nonfused_bytes_total = nonfused_bytes_read + nonfused_bytes_written

    for (_, _, profile_data) in fused_stepper.run(
            ic, t_start, dt, t_end, return_profile_data=True):
        pass

    fused_bytes_read = profile_data["bytes_read"]
    fused_bytes_written = profile_data["bytes_written"]
    fused_bytes_total = fused_bytes_read + fused_bytes_written

    print(r"\begin{tabular}{lrrr}")
    print(r"\toprule")
    print(r"Operator & Bytes Read & Bytes Written & Total (\% of Baseline) \\")
    print(r"\midrule")
    print(
            r"Baseline & \num{%d} & \num{%d} & \num{%d} (100) \\"
            % (
                nonfused_bytes_read,
                nonfused_bytes_written,
                nonfused_bytes_total))
    print(
            r"Fused & \num{%d} & \num{%d} & \num{%d} (%.1f) \\"
            % (
                fused_bytes_read,
                fused_bytes_written,
                fused_bytes_total,
                100 * fused_bytes_total / nonfused_bytes_total))
    print(r"\bottomrule")
    print(r"\end{tabular}")

# }}}


if __name__ == "__main__":
    #test_stepper_equivalence()
    #test_stepper_mem_ops(True)
    #test_stepper_mem_ops(False)
    #statement_counts_table()
    mem_ops_table()
    #pass
    #test_stepper_timing(cl._csc, False)
