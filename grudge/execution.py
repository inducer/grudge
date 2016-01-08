from __future__ import division, absolute_import

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

import six
import numpy as np
import loopy as lp
import pyopencl as cl
import pyopencl.array  # noqa
from pytools import memoize_in

import grudge.symbolic.mappers as mappers
from grudge import sym

import logging
logger = logging.getLogger(__name__)


# {{{ exec mapper

class ExecutionMapper(mappers.Evaluator,
        mappers.BoundOpMapperMixin,
        mappers.LocalOpReducerMixin):
    def __init__(self, queue, context, bound_op):
        super(ExecutionMapper, self).__init__(context)
        self.discr = bound_op.discr
        self.bound_op = bound_op
        self.queue = queue

    def get_discr(self, dd):
        qtag = dd.quadrature_tag
        if qtag is None:
            # FIXME: Remove once proper quadrature support arrives
            qtag = sym.QTAG_NONE

        if dd.is_volume():
            if qtag is not sym.QTAG_NONE:
                # FIXME
                raise NotImplementedError("quadrature")
            return self.discr.volume_discr

        elif dd.domain_tag is sym.FRESTR_ALL_FACES:
            return self.discr.all_faces_discr(qtag)
        elif dd.domain_tag is sym.FRESTR_INTERIOR_FACES:
            return self.discr.interior_faces_discr(qtag)
        elif dd.is_boundary():
            return self.discr.boundary_discr(dd.domain_tag, qtag)
        else:
            raise ValueError("DOF desc tag not understood: " + str(dd))

    # {{{ expression mappings -------------------------------------------------

    def map_ones(self, expr):
        discr = self.get_discr(expr.dd)

        result = discr.empty(self.queue, allocator=self.bound_op.allocator)
        result.fill(1)
        return result

    def map_node_coordinate_component(self, expr):
        discr = self.get_discr(expr.dd)
        return discr.nodes()[expr.axis].with_queue(self.queue)

    def map_boundarize(self, op, field_expr):
        return self.discr.boundarize_volume_field(
                self.rec(field_expr), tag=op.tag,
                kind=self.discr.compute_kind)

    def map_grudge_variable(self, expr):
        return self.context[expr.name]

    def map_call(self, expr):
        from pymbolic.primitives import Variable
        assert isinstance(expr.function, Variable)

        # FIXME: Make a way to register functions
        import pyopencl.clmath as clmath
        func = getattr(clmath, expr.function.name)

        return func(*[self.rec(p) for p in expr.parameters])

    def map_nodal_sum(self, op, field_expr):
        return cl.array.sum(self.rec(field_expr))

    def map_nodal_max(self, op, field_expr):
        return cl.array.max(self.rec(field_expr))

    def map_nodal_min(self, op, field_expr):
        return cl.array.min(self.rec(field_expr))

    def map_if(self, expr):
        bool_crit = self.rec(expr.condition)

        then = self.rec(expr.then)
        else_ = self.rec(expr.else_)

        result = cl.array.empty_like(then, queue=self.queue,
                allocator=self.bound_op.allocator)
        cl.array.if_positive(bool_crit, then, else_, out=result,
                queue=self.queue)

        return result

    def map_ref_diff_base(self, op, field_expr):
        raise NotImplementedError(
                "differentiation should be happening in batched form")

    def map_elementwise_linear(self, op, field_expr):
        field = self.rec(field_expr)

        from grudge.tools import is_zero
        if is_zero(field):
            return 0

        @memoize_in(self.bound_op, "elwise_linear_knl")
        def knl():
            knl = lp.make_kernel(
                """{[k,i,j]:
                    0<=k<nelements and
                    0<=i,j<ndiscr_nodes}""",
                "result[k,i] = sum(j, mat[i, j] * vec[k, j])",
                default_offset=lp.auto, name="diff")

            knl = lp.split_iname(knl, "i", 16, inner_tag="l.0")
            return lp.tag_inames(knl, dict(k="g.0"))

        discr = self.get_discr(op.dd_in)

        # FIXME: This shouldn't really assume that it's dealing with a volume
        # input. What about quadrature? What about boundaries?
        result = discr.empty(
                queue=self.queue,
                dtype=field.dtype, allocator=self.bound_op.allocator)

        for grp in discr.groups:
            cache_key = "elwise_linear", grp, op, field.dtype
            try:
                matrix = self.bound_op.operator_data_cache[cache_key]
            except KeyError:
                matrix = (
                        cl.array.to_device(
                            self.queue,
                            np.asarray(op.matrix(grp), dtype=field.dtype))
                        .with_queue(None))

                self.bound_op.operator_data_cache[cache_key] = matrix

            knl()(self.queue, mat=matrix, result=grp.view(result),
                    vec=grp.view(field))

        return result

    def map_elementwise_max(self, op, field_expr):
        from grudge._internal import perform_elwise_max
        field = self.rec(field_expr)

        out = self.discr.volume_zeros(dtype=field.dtype)
        for eg in self.discr.element_groups:
            perform_elwise_max(eg.ranges, field, out)

        return out

    def map_interpolation(self, op, field_expr):
        if op.dd_in.quadrature_tag not in [None, sym.QTAG_NONE]:
            raise ValueError("cannot interpolate *from* a quadrature grid")

        dd_in = op.dd_in
        dd_out = op.dd_out

        qtag = dd_out.quadrature_tag
        if qtag is None:
            # FIXME: Remove once proper quadrature support arrives
            qtag = sym.QTAG_NONE

        if dd_in.is_volume():
            if dd_out.domain_tag is sym.FRESTR_ALL_FACES:
                conn = self.discr.all_faces_connection(qtag)
            elif dd_out.domain_tag is sym.FRESTR_INTERIOR_FACES:
                conn = self.discr.interior_faces_connection(qtag)
            elif dd_out.is_boundary():
                conn = self.discr.boundary_connection(dd_out.domain_tag, qtag)
            else:
                raise ValueError("cannot interpolate from volume to: " + str(dd_out))

        elif dd_in.domain_tag is sym.FRESTR_INTERIOR_FACES:
            if dd_out.domain_tag is sym.FRESTR_ALL_FACES:
                conn = self.discr.all_faces_connection(None, qtag)
            else:
                raise ValueError(
                        "cannot interpolate from interior faces to: "
                        + str(dd_out))

        elif dd_in.is_boundary():
            if dd_out.domain_tag is sym.FRESTR_ALL_FACES:
                conn = self.discr.all_faces_connection(dd_in.domain_tag, qtag)
            else:
                raise ValueError(
                        "cannot interpolate from interior faces to: "
                        + str(dd_out))

        else:
            raise ValueError("cannot interpolate from: " + str(dd_in))

        return conn(self.queue, self.rec(field_expr)).with_queue(self.queue)

    def map_opposite_interior_face_swap(self, op, field_expr):
        dd = op.dd_in

        qtag = dd.quadrature_tag
        if qtag is None:
            # FIXME: Remove once proper quadrature support arrives
            qtag = sym.QTAG_NONE

        return self.discr.opposite_face_connection(qtag)(
                self.queue, self.rec(field_expr)).with_queue(self.queue)

    def map_face_mass_operator(self, op, field_expr):
        raise NotImplementedError

    # }}}

    # {{{ code execution functions

    def exec_assign(self, insn):
        return [(name, self.rec(expr))
                for name, expr in zip(insn.names, insn.exprs)], []

    def exec_vector_expr_assign(self, insn):
        if self.bound_op.instrumented:
            def stats_callback(n, vec_expr):
                self.bound_op.vector_math_flop_counter.add(n*insn.flop_count())
                return self.bound_op.vector_math_timer
        else:
            stats_callback = None

        # FIXME: Reenable compiled vector exprs
        if True:  # insn.flop_count() == 0:
            return [(name, self(expr))
                for name, expr in zip(insn.names, insn.exprs)], []
        else:
            compiled = insn.compiled(self.bound_op)
            return zip(compiled.result_names(),
                    compiled(self, stats_callback)), []

    def exec_diff_batch_assign(self, insn):
        field = self.rec(insn.field)
        repr_op = insn.operators[0]
        if not isinstance(repr_op, sym.RefDiffOperator):
            # FIXME
            raise NotImplementedError()

        # FIXME: There's no real reason why differentiation is special,
        # execution-wise.
        # This should be unified with map_elementwise_linear, which should
        # be extended to support batching.

        discr = self.get_discr(repr_op.dd_in)

        return [
            (name, discr.num_reference_derivative(
                self.queue, (op.rst_axis,), field)
                .with_queue(self.queue))
            for name, op in zip(insn.names, insn.operators)], []

    # }}}

# }}}


# {{{ bound operator

class BoundOperator(object):
    def __init__(self, discr, code, debug_flags, allocator=None):
        self.discr = discr
        self.code = code
        self.operator_data_cache = {}
        self.debug_flags = debug_flags
        self.allocator = allocator

    def __call__(self, queue, **context):
        import pyopencl.array as cl_array

        def replace_queue(a):
            if isinstance(a, cl_array.Array):
                return a.with_queue(queue)
            else:
                return a

        from pytools.obj_array import with_object_array_or_scalar

        new_context = {}
        for name, var in six.iteritems(context):
            new_context[name] = with_object_array_or_scalar(replace_queue, var)

        return self.code.execute(ExecutionMapper(queue, new_context, self))

# }}}


# {{{ process_sym_operator function

def process_sym_operator(sym_operator, post_bind_mapper=None,
        dumper=lambda name, sym_operator: None, mesh=None):

    import grudge.symbolic.mappers as mappers

    dumper("before-bind", sym_operator)
    sym_operator = mappers.OperatorBinder()(sym_operator)

    mappers.ErrorChecker(mesh)(sym_operator)

    if post_bind_mapper is not None:
        dumper("before-postbind", sym_operator)
        sym_operator = post_bind_mapper(sym_operator)

    if mesh is not None:
        dumper("before-empty-flux-killer", sym_operator)
        sym_operator = mappers.EmptyFluxKiller(mesh)(sym_operator)

    dumper("before-cfold", sym_operator)
    sym_operator = mappers.CommutativeConstantFoldingMapper()(sym_operator)

    # Ordering restriction:
    #
    # - Must run constant fold before first type inference pass, because zeros,
    # while allowed, violate typing constraints (because they can't be assigned
    # a unique type), and need to be killed before the type inferrer sees them.

    # FIXME: Reenable type inference

    # from grudge.symbolic.mappers.type_inference import TypeInferrer
    # dumper("before-specializer", sym_operator)
    # sym_operator = mappers.OperatorSpecializer(
    #         TypeInferrer()(sym_operator)
    #         )(sym_operator)

    # Ordering restriction:
    #
    # - Must run OperatorSpecializer before performing the GlobalToReferenceMapper,
    # because otherwise it won't differentiate the type of grids (node or quadrature
    # grids) that the operators will apply on.

    assert mesh is not None
    dumper("before-global-to-reference", sym_operator)
    sym_operator = mappers.GlobalToReferenceMapper(mesh.ambient_dim)(sym_operator)

    # Ordering restriction:
    #
    # - Must specialize quadrature operators before performing inverse mass
    # contraction, because there are no inverse-mass-contracted variants of the
    # quadrature operators.

    dumper("before-imass", sym_operator)
    sym_operator = mappers.InverseMassContractor()(sym_operator)

    dumper("before-cfold-2", sym_operator)
    sym_operator = mappers.CommutativeConstantFoldingMapper()(sym_operator)

    # FIXME: Reenable derivative joiner
    # dumper("before-derivative-join", sym_operator)
    # sym_operator = mappers.DerivativeJoiner()(sym_operator)

    dumper("process-finished", sym_operator)

    return sym_operator

# }}}


def bind(discr, sym_operator, post_bind_mapper=lambda x: x,
        debug_flags=set(), allocator=None):
    # from grudge.symbolic.mappers import QuadratureUpsamplerRemover
    # sym_operator = QuadratureUpsamplerRemover(self.quad_min_degrees)(
    #         sym_operator)

    stage = [0]

    def dump_optemplate(name, sym_operator):
        if "dump_optemplate_stages" in debug_flags:
            from grudge.tools import open_unique_debug_file
            from grudge.optemplate import pretty
            open_unique_debug_file("%02d-%s" % (stage[0], name), ".txt").write(
                    pretty(sym_operator))
            stage[0] += 1

    sym_operator = process_sym_operator(sym_operator,
            post_bind_mapper=post_bind_mapper,
            dumper=dump_optemplate,
            mesh=discr.mesh)

    from grudge.symbolic.compiler import OperatorCompiler
    code = OperatorCompiler()(sym_operator)

    bound_op = BoundOperator(discr, code,
            debug_flags=debug_flags, allocator=allocator)

    if "dump_op_code" in debug_flags:
        from grudge.tools import open_unique_debug_file
        open_unique_debug_file("op-code", ".txt").write(
                str(code))

    if "dump_dataflow_graph" in debug_flags:
        bound_op.code.dump_dataflow_graph()

    return bound_op

# vim: foldmethod=marker
