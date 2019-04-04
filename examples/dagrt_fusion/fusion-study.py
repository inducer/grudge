from __future__ import division, print_function

__copyright__ = "Copyright (C) 2015 Andreas Kloeckner"

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
import pyopencl as cl

import dagrt.language as lang
import pymbolic.primitives as p
import grudge.symbolic.mappers as gmap
from pymbolic.mapper.evaluator import EvaluationMapper \
        as PymbolicEvaluationMapper

from grudge import sym, bind, DGDiscretizationWithBoundaries
from leap.rk import LSRK4Method


logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)


# {{{ topological sort

def topological_sort(stmts, root_deps):
    id_to_stmt = {stmt.id: stmt for stmt in stmts}

    ordered_stmts = []
    satisfied = set()

    def satisfy_dep(name):
        if name in satisfied:
            return

        stmt = id_to_stmt[name]
        for dep in stmt.depends_on:
            satisfy_dep(dep)
        ordered_stmts.append(stmt)
        satisfied.add(name)

    for d in root_deps:
        satisfy_dep(d)

    return ordered_stmts

# }}}


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


def get_strong_wave_op_with_discr(cl_ctx, dims=3, order=4):
    from meshmode.mesh.generation import generate_regular_rect_mesh
    mesh = generate_regular_rect_mesh(
            a=(-0.5,)*dims,
            b=(0.5,)*dims,
            n=(16,)*dims)

    logger.info("%d elements" % mesh.nelements)

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


class RK4TimeStepperBase(object):

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

    def set_up_stepper(self, discr, field_var_name, sym_rhs, num_fields):
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

        self.bound_op = bind(discr, flattened_results)


class RK4TimeStepper(RK4TimeStepperBase):

    def __init__(self, queue, discr, field_var_name, grudge_bound_op,
                 num_fields, component_getter):
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
        self.set_up_stepper(discr, field_var_name, sym_rhs, num_fields)
        self.component_getter = component_getter

    def _bound_op(self, t, *args):
        from pytools.obj_array import join_fields
        context = {
                "t": t,
                self.field_var_name: join_fields(*args)}
        return self.grudge_bound_op(self.queue, **context)

    def get_initial_context(self, fields, t_start, dt):
        context = super().get_initial_context(fields, t_start, dt)
        context["grudge_op"] = self._bound_op
        return context

    def run(self, fields, t_start, dt, t_end):
        context = self.get_initial_context(fields, t_start, dt)

        t = t_start

        while t <= t_end:
            results = self.bound_op(self.queue, **context)
            t = results[0]
            context["input_t"] = t
            context["input_dt"] = results[1]
            output_states = results[2:]
            context[self.state_name] = output_states
            yield (t, self.component_getter(output_states))


class FusedRK4TimeStepper(RK4TimeStepperBase):

    def __init__(self, queue, discr, field_var_name, sym_rhs, num_fields,
                 component_getter):
        self.queue = queue
        self.set_up_stepper(discr, field_var_name, sym_rhs, num_fields)
        self.component_getter = component_getter

    def run(self, fields, t_start, dt, t_end):
        context = self.get_initial_context(fields, t_start, dt)

        t = t_start

        while t <= t_end:
            results = self.bound_op(self.queue, **context)
            t = results[0]
            context["input_t"] = t
            context["input_dt"] = results[1]
            output_states = results[2:]
            context[self.state_name] = output_states
            yield (t, self.component_getter(output_states))


def get_strong_wave_component(state_component):
    return (state_component[0], state_component[1:])


# {{{ equivalence check

def test_stepper_equivalence(order=4):
    cl_ctx = cl.create_some_context()
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
    nsteps = int(t_end/dt)
    print("dt=%g nsteps=%d" % (dt, nsteps))

    step = 0

    norm = bind(discr, sym.norm(2, sym.var("u_ref") - sym.var("u")))

    fused_steps = fused_stepper.run(ic, t_start, dt, t_end)

    for t_ref, (u_ref, v_ref) in stepper.run(ic, t_start, dt, t_end):
        step += 1
        logger.info("step %d", step)
        t, (u, v) = next(fused_steps)
        assert t == t_ref, step
        assert norm(queue, u=u, u_ref=u_ref) <= 1e-13, step

# }}}


if __name__ == "__main__":
    test_stepper_equivalence()
