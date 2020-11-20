#!/usr/bin/env python3
"""Study of operator fusion (inlining) for time integration operators in Grudge.
"""

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

# FIXME:
# Results before https://github.com/inducer/grudge/pull/15 were better:
#
# Operator     | \parbox{1in}{\centering \% Memory Ops. Due to Scalar Assignments}
# -------------+-------------------------------------------------------------------
# 2D: Baseline | 51.1
# 2D: Inlined  | 48.9
# 3D: Baseline | 50.1
# 3D: Inlined  | 48.6
# INFO:__main__:Wrote '<stdout>'
# ==== Scalar Assigment Inlining Impact ====
# Operator     | Bytes Read | Bytes Written | Total      | \% of Baseline
# -------------+------------+---------------+------------+----------------
# 2D: Baseline | 9489600    | 3348000       | 12837600   | 100
# 2D: Inlined  | 8949600    | 2808000       | 11757600   | 91.6
# 3D: Baseline | 1745280000 | 505440000     | 2250720000 | 100
# 3D: Inlined  | 1680480000 | 440640000     | 2121120000 | 94.2
# INFO:__main__:Wrote '<stdout>'


import contextlib
import logging
import numpy as np
import os
import sys
import pyopencl as cl
import pyopencl.array  # noqa
import pytest

import dagrt.language as lang
import pymbolic.primitives as p

from meshmode.dof_array import DOFArray
from meshmode.array_context import PyOpenCLArrayContext

import grudge.symbolic.mappers as gmap
import grudge.symbolic.operators as sym_op
from grudge.execution import ExecutionMapper
from grudge.function_registry import base_function_registry
from pymbolic.mapper import Mapper
from pymbolic.mapper.evaluator import EvaluationMapper \
        as PymbolicEvaluationMapper
from pytools import memoize
from pytools.obj_array import flat_obj_array

from grudge import sym, bind, DGDiscretizationWithBoundaries
from leap.rk import LSRK4MethodBuilder

from pyopencl.tools import (  # noqa
        pytest_generate_tests_for_pyopencl as pytest_generate_tests)


logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)


SKIP_TESTS = int(os.environ.get("SKIP_TESTS", 0))
PAPER_OUTPUT = int(os.environ.get("PAPER_OUTPUT", 0))
OUT_DIR = os.environ.get("OUT_DIR", ".")


@contextlib.contextmanager
def open_output_file(filename):
    if not PAPER_OUTPUT:
        yield sys.stdout
        sys.stdout.flush()
    else:
        try:
            outfile = open(os.path.join(OUT_DIR, filename), "w")
            yield outfile
        finally:
            outfile.close()


def dof_array_nbytes(ary: np.ndarray):
    if isinstance(ary, np.ndarray) and ary.dtype.char == "O":
        return sum(
                dof_array_nbytes(ary[idx])
                for idx in np.ndindex(ary.shape))
    elif isinstance(ary, DOFArray):
        return sum(dof_array_nbytes(ary_i) for ary_i in ary)
    else:
        return ary.nbytes


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
    """Generate a Grudge operator for a Dagrt time integrator phase.

    Arguments:

        dag: The Dagrt code object for the time integrator

        field_var_name: The name of the simulation variable

        field_components: The number of components (fields) in the variable

        phase_name: The name of the phase to transcribe

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
            "<p>residual": sym.make_sym_array(
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

        elif isinstance(stmt, lang.Assign):
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
                    (
                        stmt.time_id,
                        d2g(stmt.time),
                        stmt.component_id,
                        d2g(stmt.expression)))

        else:
            raise NotImplementedError("statement %s is of unsupported type ''%s'"
                        % (stmt.id, type(stmt).__name__))

    return output_vars, [ctx[ov] for ov in output_vars], yielded_states

# }}}


# {{{ time integrator implementations

class RK4TimeStepperBase:

    def __init__(self, component_getter):
        self.component_getter = component_getter

    def get_initial_context(self, fields, t_start, dt):

        # Flatten fields.
        flattened_fields = []
        for field in fields:
            if isinstance(field, list):
                flattened_fields.extend(field)
            else:
                flattened_fields.append(field)
        flattened_fields = flat_obj_array(*flattened_fields)
        del fields

        return {
                "input_t": t_start,
                "input_dt": dt,
                self.state_name: flattened_fields,
                "input_residual": flattened_fields,
        }

    def set_up_stepper(self, discr, field_var_name, sym_rhs, num_fields,
                       function_registry=base_function_registry,
                       exec_mapper_factory=ExecutionMapper):
        dt_method = LSRK4MethodBuilder(component_id=field_var_name)
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

        flattened_results = flat_obj_array(output_t, output_dt, *output_states)

        self.bound_op = bind(
                discr, flattened_results,
                function_registry=function_registry,
                exec_mapper_factory=exec_mapper_factory)

    def run(self, fields, t_start, dt, t_end, return_profile_data=False):
        context = self.get_initial_context(fields, t_start, dt)

        t = t_start

        while t <= t_end:
            if return_profile_data:
                profile_data = dict()
            else:
                profile_data = None

            results = self.bound_op(
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

    def __init__(self, discr, field_var_name, grudge_bound_op,
                 num_fields, component_getter, exec_mapper_factory=ExecutionMapper):
        """Arguments:

            field_var_name: The name of the simulation variable

            grudge_bound_op: The BoundExpression for the right-hand side

            num_fields: The number of components in the simulation variable

            component_getter: A function, which, given an object array
               representing the simulation variable, splits the array into
               its components

        """
        super().__init__(component_getter)

        # Construct sym_rhs to have the effect of replacing the RHS calls in the
        # dagrt code with calls of the grudge operator.
        from grudge.symbolic.primitives import FunctionSymbol, Variable
        call = sym.cse(
                FunctionSymbol("grudge_op")(*(
                    (Variable("t", dd=sym.DD_SCALAR),)
                    + tuple(
                        Variable(field_var_name, dd=sym.DD_VOLUME)[i]
                        for i in range(num_fields)))))
        sym_rhs = flat_obj_array(*(call[i] for i in range(num_fields)))

        self.grudge_bound_op = grudge_bound_op

        from grudge.function_registry import register_external_function

        freg = register_external_function(
                base_function_registry,
                "grudge_op",
                implementation=self._bound_op,
                dd=sym.DD_VOLUME)

        self.set_up_stepper(
                discr, field_var_name, sym_rhs, num_fields,
                freg,
                exec_mapper_factory)

        self.component_getter = component_getter

    def _bound_op(self, array_context, t, *args, profile_data=None):
        context = {
                "t": t,
                self.field_var_name: flat_obj_array(*args)}
        result = self.grudge_bound_op(
                array_context, profile_data=profile_data, **context)
        if profile_data is not None:
            result = result[0]
        return result

    def get_initial_context(self, fields, t_start, dt):
        context = super().get_initial_context(fields, t_start, dt)
        context["grudge_op"] = self._bound_op
        return context


class FusedRK4TimeStepper(RK4TimeStepperBase):

    def __init__(self, discr, field_var_name, sym_rhs, num_fields,
                 component_getter, exec_mapper_factory=ExecutionMapper):
        super().__init__(component_getter)
        self.set_up_stepper(
                discr, field_var_name, sym_rhs, num_fields,
                base_function_registry,
                exec_mapper_factory)

# }}}


# {{{ problem setup code

def _get_source_term(dims):
    source_center = np.array([0.1, 0.22, 0.33])[:dims]
    source_width = 0.05
    source_omega = 3

    sym_x = sym.nodes(dims)
    sym_source_center_dist = sym_x - source_center
    sym_t = sym.ScalarVariable("t")

    return (
                sym.sin(source_omega*sym_t)
                * sym.exp(
                    -np.dot(sym_source_center_dist, sym_source_center_dist)
                    / source_width**2))


def get_wave_op_with_discr(actx, dims=2, order=4):
    from meshmode.mesh.generation import generate_regular_rect_mesh
    mesh = generate_regular_rect_mesh(
            a=(-0.5,)*dims,
            b=(0.5,)*dims,
            n=(16,)*dims)

    logger.debug("%d elements", mesh.nelements)

    discr = DGDiscretizationWithBoundaries(actx, mesh, order=order)

    from grudge.models.wave import WeakWaveOperator
    from meshmode.mesh import BTAG_ALL, BTAG_NONE
    op = WeakWaveOperator(0.1, dims,
            source_f=_get_source_term(dims),
            dirichlet_tag=BTAG_NONE,
            neumann_tag=BTAG_NONE,
            radiation_tag=BTAG_ALL,
            flux_type="upwind")

    op.check_bc_coverage(mesh)

    return (op.sym_operator(), discr)


def get_wave_component(state_component):
    return (state_component[0], state_component[1:])

# }}}


# {{{ equivalence check between fused and non-fused versions

def test_stepper_equivalence(ctx_factory, order=4):
    cl_ctx = ctx_factory()
    queue = cl.CommandQueue(cl_ctx)
    actx = PyOpenCLArrayContext(queue)

    dims = 2

    sym_operator, discr = get_wave_op_with_discr(
            actx, dims=dims, order=order)
    #sym_operator_direct, discr = get_wave_op_with_discr_direct(
    #        actx, dims=dims, order=order)

    if dims == 2:
        dt = 0.04
    elif dims == 3:
        dt = 0.02

    ic = flat_obj_array(discr.zeros(actx),
            [discr.zeros(actx) for i in range(discr.dim)])

    bound_op = bind(discr, sym_operator)

    stepper = RK4TimeStepper(
            discr, "w", bound_op, 1 + discr.dim, get_wave_component)

    fused_stepper = FusedRK4TimeStepper(
            discr, "w", sym_operator, 1 + discr.dim,
            get_wave_component)

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
        assert norm(u=u, u_ref=u_ref) <= 1e-13, step

# }}}


# {{{ execution mapper wrapper

class ExecutionMapperWrapper(Mapper):

    def __init__(self, array_context, context, bound_op):
        self.inner_mapper = ExecutionMapper(array_context, context, bound_op)
        self.array_context = array_context
        self.context = context
        self.bound_op = bound_op

    def map_variable(self, expr):
        # Needed, because bound op execution can ask for variable values.
        return self.inner_mapper.map_variable(expr)

    def map_node_coordinate_component(self, expr):
        return self.inner_mapper.map_node_coordinate_component(expr)

    def map_grudge_variable(self, expr):
        # See map_variable()
        return self.inner_mapper.map_grudge_variable(expr)

# }}}


# {{{ mem op counter implementation

class ExecutionMapperWithMemOpCounting(ExecutionMapperWrapper):
    # This is a skeleton implementation that only has just enough functionality
    # for the wave-min example to work.

    # {{{ expressions

    def map_profiled_call(self, expr, profile_data):
        args = [self.inner_mapper.rec(p) for p in expr.parameters]
        return self.inner_mapper.function_registry[expr.function.name](
                self.array_context, *args, profile_data=profile_data)

    def map_profiled_essentially_elementwise_linear(self, op, field_expr,
                                                    profile_data):
        result = getattr(self.inner_mapper, op.mapper_method)(op, field_expr)

        if profile_data is not None:
            # We model the cost to load the input and write the output.  In
            # particular, we assume the elementwise matrices are negligible in
            # size and thus ignorable.

            field = self.inner_mapper.rec(field_expr)
            profile_data["bytes_read"] = (
                    profile_data.get("bytes_read", 0)
                    + dof_array_nbytes(field))
            profile_data["bytes_written"] = (
                    profile_data.get("bytes_written", 0)
                    + dof_array_nbytes(result))

            if op.mapper_method == "map_projection":
                profile_data["interp_bytes_read"] = (
                        profile_data.get("interp_bytes_read", 0)
                        + dof_array_nbytes(field))
                profile_data["interp_bytes_written"] = (
                        profile_data.get("interp_bytes_written", 0)
                        + dof_array_nbytes(result))

        return result

    # }}}

    # {{{ instruction mappings

    def process_assignment_expr(self, expr, profile_data):
        if isinstance(expr, p.Call):
            assert expr.mapper_method == "map_call"
            val = self.map_profiled_call(expr, profile_data)

        elif isinstance(expr, sym.OperatorBinding):
            if isinstance(
                    expr.op,
                    (
                        # TODO: Not comprehensive.
                        sym_op.ProjectionOperator,
                        sym_op.RefFaceMassOperator,
                        sym_op.RefMassOperator,
                        sym_op.RefInverseMassOperator,
                        sym_op.OppositeInteriorFaceSwap)):
                val = self.map_profiled_essentially_elementwise_linear(
                        expr.op, expr.field, profile_data)

            else:
                assert False, ("unknown operator: %s" % expr.op)

        else:
            logger.debug("assignment not profiled: %s", expr)
            val = self.inner_mapper.rec(expr)

        return val

    def map_insn_assign(self, insn, profile_data):
        result = []
        for name, expr in zip(insn.names, insn.exprs):
            result.append((name, self.process_assignment_expr(expr, profile_data)))
        return result, []

    def map_insn_loopy_kernel(self, insn, profile_data):
        kdescr = insn.kernel_descriptor
        discr = self.inner_mapper.discrwb.discr_from_dd(kdescr.governing_dd)

        dof_array_kwargs = {}
        other_kwargs = {}

        for name, expr in kdescr.input_mappings.items():
            v = self.inner_mapper.rec(expr)
            if isinstance(v, DOFArray):
                dof_array_kwargs[name] = v

                if profile_data is not None:
                    size = dof_array_nbytes(v)
                    profile_data["bytes_read"] = (
                            profile_data.get("bytes_read", 0) + size)
                    profile_data["bytes_read_by_scalar_assignments"] = (
                            profile_data.get("bytes_read_by_scalar_assignments", 0)
                            + size)
            else:
                assert not isinstance(v, np.ndarray)
                other_kwargs[name] = v

        for name in kdescr.scalar_args():
            v = other_kwargs[name]
            if isinstance(v, (int, float)):
                other_kwargs[name] = discr.real_dtype.type(v)
            elif isinstance(v, complex):
                other_kwargs[name] = discr.complex_dtype.type(v)
            elif isinstance(v, np.number):
                pass
            else:
                raise ValueError("unrecognized scalar type for variable '%s': %s"
                        % (name, type(v)))

        result = {}
        for grp in discr.groups:
            kwargs = other_kwargs.copy()
            kwargs["nelements"] = grp.nelements
            kwargs["nunit_dofs"] = grp.nunit_dofs

            for name, ary in dof_array_kwargs.items():
                kwargs[name] = ary[grp.index]

            knl_result = self.inner_mapper.array_context.call_loopy(
                    kdescr.loopy_kernel, **kwargs)

            for name, val in knl_result.items():
                result.setdefault(name, []).append(val)

        result = {
                name: DOFArray(self.inner_mapper.array_context, tuple(val))
                for name, val in result.items()}

        for val in result.values():
            assert isinstance(val, DOFArray)
            if profile_data is not None:
                size = dof_array_nbytes(val)
                profile_data["bytes_written"] = (
                        profile_data.get("bytes_written", 0) + size)
                profile_data["bytes_written_by_scalar_assignments"] = (
                        profile_data.get("bytes_written_by_scalar_assignments", 0)
                        + size)

        return list(result.items()), []

    def map_insn_assign_to_discr_scoped(self, insn, profile_data=None):
        assignments = []

        for name, expr in zip(insn.names, insn.exprs):
            logger.debug("assignment not profiled: %s <- %s", name, expr)
            inner_mapper = self.inner_mapper
            value = inner_mapper.rec(expr)
            inner_mapper.discrwb._discr_scoped_subexpr_name_to_value[name] = value
            assignments.append((name, value))

        return assignments, []

    def map_insn_assign_from_discr_scoped(self, insn, profile_data=None):
        return [(
            insn.name,
            self.inner_mapper.
                discrwb._discr_scoped_subexpr_name_to_value[insn.name])], []

    def map_insn_rank_data_swap(self, insn, profile_data):
        raise NotImplementedError("no profiling for instruction: %s" % insn)

    def map_insn_diff_batch_assign(self, insn, profile_data):
        assignments, futures = self.inner_mapper.map_insn_diff_batch_assign(insn)

        if profile_data is not None:
            # We model the cost to load the input and write the output.  In
            # particular, we assume the elementwise matrices are negligible in
            # size and thus ignorable.

            field = self.inner_mapper.rec(insn.field)
            profile_data["bytes_read"] = (
                    profile_data.get("bytes_read", 0) + dof_array_nbytes(field))

            for _, value in assignments:
                profile_data["bytes_written"] = (
                        profile_data.get("bytes_written", 0)
                        + dof_array_nbytes(value))

        return assignments, futures

    # }}}

# }}}


# {{{ mem op counter check

def test_assignment_memory_model(ctx_factory):
    cl_ctx = ctx_factory()
    queue = cl.CommandQueue(cl_ctx)
    actx = PyOpenCLArrayContext(queue)

    _, discr = get_wave_op_with_discr(actx, dims=2, order=3)

    # Assignment instruction
    bound_op = bind(
            discr,
            sym.Variable("input0", sym.DD_VOLUME)
            + sym.Variable("input1", sym.DD_VOLUME),
            exec_mapper_factory=ExecutionMapperWithMemOpCounting)

    input0 = discr.zeros(actx)
    input1 = discr.zeros(actx)

    result, profile_data = bound_op(
            profile_data={},
            input0=input0,
            input1=input1)

    assert profile_data["bytes_read"] == \
            dof_array_nbytes(input0) + dof_array_nbytes(input1)
    assert profile_data["bytes_written"] == dof_array_nbytes(result)


@pytest.mark.parametrize("use_fusion", (True, False))
def test_stepper_mem_ops(ctx_factory, use_fusion):
    cl_ctx = ctx_factory()
    queue = cl.CommandQueue(cl_ctx)
    actx = PyOpenCLArrayContext(queue)

    dims = 2

    sym_operator, discr = get_wave_op_with_discr(
            actx, dims=dims, order=3)

    t_start = 0
    dt = 0.04
    t_end = 0.2

    ic = flat_obj_array(discr.zeros(actx),
            [discr.zeros(actx) for i in range(discr.dim)])

    if not use_fusion:
        bound_op = bind(
                discr, sym_operator,
                exec_mapper_factory=ExecutionMapperWithMemOpCounting)

        stepper = RK4TimeStepper(
                discr, "w", bound_op, 1 + discr.dim,
                get_wave_component,
                exec_mapper_factory=ExecutionMapperWithMemOpCounting)

    else:
        stepper = FusedRK4TimeStepper(
                discr, "w", sym_operator, 1 + discr.dim,
                get_wave_component,
                exec_mapper_factory=ExecutionMapperWithMemOpCounting)

    step = 0

    nsteps = int(np.ceil((t_end + 1e-9) / dt))
    for (_, _, profile_data) in stepper.run(
            ic, t_start, dt, t_end, return_profile_data=True):
        step += 1
        logger.info("step %d/%d", step, nsteps)

    logger.info("using fusion? %s", use_fusion)
    logger.info("bytes read: %d", profile_data["bytes_read"])
    logger.info("bytes written: %d", profile_data["bytes_written"])
    logger.info("bytes total: %d",
            profile_data["bytes_read"] + profile_data["bytes_written"])

# }}}


# {{{ execution mapper with timing

SECONDS_PER_NANOSECOND = 10**9


class TimingFuture:

    def __init__(self, start_event, stop_event):
        self.start_event = start_event
        self.stop_event = stop_event

    def elapsed(self):
        cl.wait_for_events([self.start_event, self.stop_event])
        return (
                self.stop_event.profile.end
                - self.start_event.profile.end) / SECONDS_PER_NANOSECOND


from collections.abc import MutableSequence


class TimingFutureList(MutableSequence):

    def __init__(self, *args, **kwargs):
        self._list = list(*args, **kwargs)

    def __len__(self):
        return len(self._list)

    def __getitem__(self, idx):
        return self._list[idx]

    def __setitem__(self, idx, val):
        self._list[idx] = val

    def __delitem__(self, idx):
        del self._list[idx]

    def insert(self, idx, val):
        self._list.insert(idx, val)

    def elapsed(self):
        return sum(future.elapsed() for future in self._list)


def time_insn(f):
    time_field_name = "time_%s" % f.__name__

    def wrapper(self, insn, profile_data):
        if profile_data is None:
            return f(self, insn, profile_data)

        start = cl.enqueue_marker(self.array_context.queue)
        retval = f(self, insn, profile_data)
        end = cl.enqueue_marker(self.array_context.queue)
        profile_data\
                .setdefault(time_field_name, TimingFutureList())\
                .append(TimingFuture(start, end))

        return retval

    return wrapper


class ExecutionMapperWithTiming(ExecutionMapperWrapper):

    def map_profiled_call(self, expr, profile_data):
        args = [self.inner_mapper.rec(p) for p in expr.parameters]
        return self.inner_mapper.function_registry[expr.function.name](
                self.array_context, *args, profile_data=profile_data)

    def map_profiled_operator_binding(self, expr, profile_data):
        if profile_data is None:
            return self.inner_mapper.map_operator_binding(expr)

        start = cl.enqueue_marker(self.array_context.queue)
        retval = self.inner_mapper.map_operator_binding(expr)
        end = cl.enqueue_marker(self.array_context.queue)
        time_field_name = "time_op_%s" % expr.op.mapper_method
        profile_data\
                .setdefault(time_field_name, TimingFutureList())\
                .append(TimingFuture(start, end))

        return retval

    def map_insn_assign_to_discr_scoped(self, insn, profile_data):
        return self.inner_mapper.map_insn_assign_to_discr_scoped(insn, profile_data)

    def map_insn_assign_from_discr_scoped(self, insn, profile_data):
        return self.\
            inner_mapper.map_insn_assign_from_discr_scoped(insn, profile_data)

    @time_insn
    def map_insn_loopy_kernel(self, *args, **kwargs):
        return self.inner_mapper.map_insn_loopy_kernel(*args, **kwargs)

    def map_insn_assign(self, insn, profile_data):
        if len(insn.exprs) == 1:
            if isinstance(insn.exprs[0], p.Call):
                assert insn.exprs[0].mapper_method == "map_call"
                val = self.map_profiled_call(insn.exprs[0], profile_data)
                return [(insn.names[0], val)], []
            elif isinstance(insn.exprs[0], sym.OperatorBinding):
                assert insn.exprs[0].mapper_method == "map_operator_binding"
                val = self.map_profiled_operator_binding(insn.exprs[0], profile_data)
                return [(insn.names[0], val)], []

        return self.inner_mapper.map_insn_assign(insn, profile_data)

    @time_insn
    def map_insn_diff_batch_assign(self, insn, profile_data):
        return self.inner_mapper.map_insn_diff_batch_assign(insn, profile_data)

# }}}


# {{{ timing check

@pytest.mark.parametrize("use_fusion", (True, False))
def test_stepper_timing(ctx_factory, use_fusion):
    cl_ctx = ctx_factory()
    queue = cl.CommandQueue(
            cl_ctx,
            properties=cl.command_queue_properties.PROFILING_ENABLE)
    actx = PyOpenCLArrayContext(queue)

    dims = 3

    sym_operator, discr = get_wave_op_with_discr(
            actx, dims=dims, order=3)

    t_start = 0
    dt = 0.04
    t_end = 0.1

    ic = flat_obj_array(discr.zeros(actx),
            [discr.zeros(actx) for i in range(discr.dim)])

    if not use_fusion:
        bound_op = bind(
                discr, sym_operator,
                exec_mapper_factory=ExecutionMapperWithTiming)

        stepper = RK4TimeStepper(
                discr, "w", bound_op, 1 + discr.dim,
                get_wave_component,
                exec_mapper_factory=ExecutionMapperWithTiming)

    else:
        stepper = FusedRK4TimeStepper(
                discr, "w", sym_operator, 1 + discr.dim,
                get_wave_component,
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
    for key, value in profile_data.items():
        if isinstance(value, TimingFutureList):
            print(key, value.elapsed())

# }}}


# {{{ paper outputs

def get_example_stepper(actx, dims=2, order=3, use_fusion=True,
                        exec_mapper_factory=ExecutionMapper,
                        return_ic=False):
    sym_operator, discr = get_wave_op_with_discr(
            actx, dims=dims, order=3)

    if not use_fusion:
        bound_op = bind(
                discr, sym_operator,
                exec_mapper_factory=exec_mapper_factory)

        stepper = RK4TimeStepper(
                discr, "w", bound_op, 1 + discr.dim,
                get_wave_component,
                exec_mapper_factory=exec_mapper_factory)

    else:
        stepper = FusedRK4TimeStepper(
                discr, "w", sym_operator, 1 + discr.dim,
                get_wave_component,
                exec_mapper_factory=exec_mapper_factory)

    if return_ic:
        ic = flat_obj_array(discr.zeros(actx),
                [discr.zeros(actx) for i in range(discr.dim)])
        return stepper, ic

    return stepper


def latex_table(table_format, header, rows):
    result = []
    _ = result.append
    _(rf"\begin{{tabular}}{{{table_format}}}")
    _(r"\toprule")
    _(" & ".join(rf"\multicolumn{{1}}{{c}}{{{item}}}" for item in header) + r" \\")
    _(r"\midrule")
    for row in rows:
        _(" & ".join(row) + r" \\")
    _(r"\bottomrule")
    _(r"\end{tabular}")
    return "\n".join(result)


def ascii_table(table_format, header, rows):
    from pytools import Table
    table = Table()
    table.add_row(header)

    for input_row in rows:
        row = []
        for item in input_row:
            if item.startswith(r"\num{"):
                # Strip \num{...} formatting
                row.append(item[5:-1])
            else:
                row.append(item)
        table.add_row(row)

    return str(table)


if not PAPER_OUTPUT:
    table = ascii_table
else:
    table = latex_table


def problem_stats(order=3):
    cl_ctx = cl.create_some_context()
    queue = cl.CommandQueue(cl_ctx)
    actx = PyOpenCLArrayContext(queue)

    with open_output_file("grudge-problem-stats.txt") as outf:
        _, dg_discr_2d = get_wave_op_with_discr(
            actx, dims=2, order=order)
        print("Number of 2D elements:", dg_discr_2d.mesh.nelements, file=outf)
        vol_discr_2d = dg_discr_2d.discr_from_dd("vol")
        dofs_2d = {group.nunit_dofs for group in vol_discr_2d.groups}
        from pytools import one
        print("Number of DOFs per 2D element:", one(dofs_2d), file=outf)

        _, dg_discr_3d = get_wave_op_with_discr(
            actx, dims=3, order=order)
        print("Number of 3D elements:", dg_discr_3d.mesh.nelements, file=outf)
        vol_discr_3d = dg_discr_3d.discr_from_dd("vol")
        dofs_3d = {group.nunit_dofs for group in vol_discr_3d.groups}
        from pytools import one
        print("Number of DOFs per 3D element:", one(dofs_3d), file=outf)

    logger.info("Wrote '%s'", outf.name)


def statement_counts_table():
    cl_ctx = cl.create_some_context()
    queue = cl.CommandQueue(cl_ctx)
    actx = PyOpenCLArrayContext(queue)

    fused_stepper = get_example_stepper(actx, use_fusion=True)
    stepper = get_example_stepper(actx, use_fusion=False)

    with open_output_file("statement-counts.tex") as outf:
        if not PAPER_OUTPUT:
            print("==== Statement Counts ====", file=outf)

        print(table(
            "lr",
            ("Operator", "Grudge Node Count"),
            (
                ("Time integration: baseline",
                 r"\num{%d}"
                     % len(stepper.bound_op.eval_code.instructions)),
                ("Right-hand side: baseline",
                 r"\num{%d}"
                     % len(stepper.grudge_bound_op.eval_code.instructions)),
                ("Inlined operator",
                 r"\num{%d}"
                     % len(fused_stepper.bound_op.eval_code.instructions))
            )),
            file=outf)

    logger.info("Wrote '%s'", outf.name)


@memoize(key=lambda queue, dims: dims)
def mem_ops_results(actx, dims):
    fused_stepper = get_example_stepper(
            actx,
            dims=dims,
            use_fusion=True,
            exec_mapper_factory=ExecutionMapperWithMemOpCounting)

    stepper, ic = get_example_stepper(
            actx,
            dims=dims,
            use_fusion=False,
            exec_mapper_factory=ExecutionMapperWithMemOpCounting,
            return_ic=True)

    t_start = 0
    dt = 0.02
    t_end = 0.02

    result = {}

    for (_, _, profile_data) in stepper.run(
            ic, t_start, dt, t_end, return_profile_data=True):
        pass

    result["nonfused_bytes_read"] = profile_data["bytes_read"]
    result["nonfused_bytes_written"] = profile_data["bytes_written"]
    result["nonfused_bytes_total"] = \
            result["nonfused_bytes_read"] \
            + result["nonfused_bytes_written"]

    result["nonfused_bytes_read_by_scalar_assignments"] = \
            profile_data["bytes_read_by_scalar_assignments"]
    result["nonfused_bytes_written_by_scalar_assignments"] = \
            profile_data["bytes_written_by_scalar_assignments"]
    result["nonfused_bytes_total_by_scalar_assignments"] = \
            result["nonfused_bytes_read_by_scalar_assignments"] \
            + result["nonfused_bytes_written_by_scalar_assignments"]

    for (_, _, profile_data) in fused_stepper.run(
            ic, t_start, dt, t_end, return_profile_data=True):
        pass

    result["fused_bytes_read"] = profile_data["bytes_read"]
    result["fused_bytes_written"] = profile_data["bytes_written"]
    result["fused_bytes_total"] = \
            result["fused_bytes_read"] \
            + result["fused_bytes_written"]

    result["fused_bytes_read_by_scalar_assignments"] = \
            profile_data["bytes_read_by_scalar_assignments"]
    result["fused_bytes_written_by_scalar_assignments"] = \
            profile_data["bytes_written_by_scalar_assignments"]
    result["fused_bytes_total_by_scalar_assignments"] = \
            result["fused_bytes_read_by_scalar_assignments"] \
            + result["fused_bytes_written_by_scalar_assignments"]

    return result


def scalar_assignment_percent_of_total_mem_ops_table():
    cl_ctx = cl.create_some_context()
    queue = cl.CommandQueue(cl_ctx)
    actx = PyOpenCLArrayContext(queue)

    result2d = mem_ops_results(actx, 2)
    result3d = mem_ops_results(actx, 3)

    with open_output_file("scalar-assignments-mem-op-percentage.tex") as outf:
        if not PAPER_OUTPUT:
            print("==== Scalar Assigment % of Total Mem Ops ====", file=outf)

        print(
            table(
                "lr",
                ("Operator",
                 r"\parbox{1in}{\centering \% Memory Ops. "
                 r"Due to Scalar Assignments}"),
                (
                    ("2D: Baseline",
                     "%.1f" % (
                         100 * result2d["nonfused_bytes_total_by_scalar_assignments"]
                         / result2d["nonfused_bytes_total"])),
                    ("2D: Inlined",
                     "%.1f" % (
                         100 * result2d["fused_bytes_total_by_scalar_assignments"]
                         / result2d["fused_bytes_total"])),
                    ("3D: Baseline",
                     "%.1f" % (
                         100 * result3d["nonfused_bytes_total_by_scalar_assignments"]
                         / result3d["nonfused_bytes_total"])),
                    ("3D: Inlined",
                     "%.1f" % (
                         100 * result3d["fused_bytes_total_by_scalar_assignments"]
                         / result3d["fused_bytes_total"])),
                )),
            file=outf)

    logger.info("Wrote '%s'", outf.name)


def scalar_assignment_effect_of_fusion_mem_ops_table():
    cl_ctx = cl.create_some_context()
    queue = cl.CommandQueue(cl_ctx)

    result2d = mem_ops_results(queue, 2)
    result3d = mem_ops_results(queue, 3)

    with open_output_file("scalar-assignments-fusion-impact.tex") as outf:
        if not PAPER_OUTPUT:
            print("==== Scalar Assigment Inlining Impact ====", file=outf)

        print(
            table(
                "lrrrr",
                ("Operator",
                 r"Bytes Read",
                 r"Bytes Written",
                 r"Total",
                 r"\% of Baseline"),
                (
                    ("2D: Baseline",
                     r"\num{%d}" % (
                         result2d["nonfused_bytes_read_by_scalar_assignments"]),
                     r"\num{%d}" % (
                         result2d["nonfused_bytes_written_by_scalar_assignments"]),
                     r"\num{%d}" % (
                         result2d["nonfused_bytes_total_by_scalar_assignments"]),
                     "100"),
                    ("2D: Inlined",
                     r"\num{%d}" % (
                         result2d["fused_bytes_read_by_scalar_assignments"]),
                     r"\num{%d}" % (
                         result2d["fused_bytes_written_by_scalar_assignments"]),
                     r"\num{%d}" % (
                         result2d["fused_bytes_total_by_scalar_assignments"]),
                     r"%.1f" % (
                         100 * result2d["fused_bytes_total_by_scalar_assignments"]
                         / result2d["nonfused_bytes_total_by_scalar_assignments"])),
                    ("3D: Baseline",
                     r"\num{%d}" % (
                         result3d["nonfused_bytes_read_by_scalar_assignments"]),
                     r"\num{%d}" % (
                         result3d["nonfused_bytes_written_by_scalar_assignments"]),
                     r"\num{%d}" % (
                         result3d["nonfused_bytes_total_by_scalar_assignments"]),
                     "100"),
                    ("3D: Inlined",
                     r"\num{%d}" % (
                         result3d["fused_bytes_read_by_scalar_assignments"]),
                     r"\num{%d}" % (
                         result3d["fused_bytes_written_by_scalar_assignments"]),
                     r"\num{%d}" % (
                         result3d["fused_bytes_total_by_scalar_assignments"]),
                     r"%.1f" % (
                         100 * result3d["fused_bytes_total_by_scalar_assignments"]
                         / result3d["nonfused_bytes_total_by_scalar_assignments"])),
                )),
            file=outf)
    logger.info("Wrote '%s'", outf.name)

# }}}


def main():
    import sys
    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        if not SKIP_TESTS:
            # Run tests.
            from py.test import main
            result = main([__file__])
            assert result == 0

        # Run examples.
        problem_stats()
        statement_counts_table()
        scalar_assignment_percent_of_total_mem_ops_table()
        scalar_assignment_effect_of_fusion_mem_ops_table()


if __name__ == "__main__":
    main()

# vim: foldmethod=marker
