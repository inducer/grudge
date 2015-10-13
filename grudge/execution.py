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

import numpy as np
import grudge.symbolic.mappers as mappers

import logging
logger = logging.getLogger(__name__)


# {{{ exec mapper

class ExecutionMapper(mappers.Evaluator,
        mappers.BoundOpMapperMixin,
        mappers.LocalOpReducerMixin):
    def __init__(self, queue, context, executor):
        super(ExecutionMapper, self).__init__(context)
        self.discr = executor.discr
        self.executor = executor
        self.queue = queue

    # {{{ expression mappings -------------------------------------------------

    def map_ones(self, expr):
        # FIXME
        if expr.quadrature_tag is not None:
            raise NotImplementedError("ones on quad. grids")

        result = self.discr.empty(self.queue)
        result.fill(1)
        return result

    def map_node_coordinate_component(self, expr):
        # FIXME
        if expr.quadrature_tag is not None:
            raise NotImplementedError("node coordinate components on quad. grids")

        return self.discr.volume_discr.nodes()[expr.axis] \
                .with_queue(self.queue)

    def map_normal_component(self, expr):
        if expr.quadrature_tag is not None:
            raise NotImplementedError("normal components on quad. grids")
        return self.discr.boundary_normals(expr.boundary_tag)[expr.axis]

    def map_boundarize(self, op, field_expr):
        return self.discr.boundarize_volume_field(
                self.rec(field_expr), tag=op.tag,
                kind=self.discr.compute_kind)

    def map_scalar_parameter(self, expr):
        return self.context[expr.name]

    def map_jacobian(self, expr):
        return self.discr.volume_jacobians(expr.quadrature_tag)

    def map_forward_metric_derivative(self, expr):
        return (self.discr.forward_metric_derivatives(expr.quadrature_tag)
                    [expr.xyz_axis][expr.rst_axis])

    def map_inverse_metric_derivative(self, expr):
        return (self.discr.inverse_metric_derivatives(expr.quadrature_tag)
                    [expr.xyz_axis][expr.rst_axis])

    def map_call(self, expr):
        from pymbolic.primitives import Variable
        assert isinstance(expr.function, Variable)
        func_name = expr.function.name

        try:
            func = self.discr.exec_functions[func_name]
        except KeyError:
            func = getattr(np, expr.function.name)

        return func(*[self.rec(p) for p in expr.parameters])

    def map_nodal_sum(self, op, field_expr):
        return np.sum(self.rec(field_expr))

    def map_nodal_max(self, op, field_expr):
        return np.max(self.rec(field_expr))

    def map_nodal_min(self, op, field_expr):
        return np.min(self.rec(field_expr))

    def map_if(self, expr):
        bool_crit = self.rec(expr.condition)
        then = self.rec(expr.then)
        else_ = self.rec(expr.else_)

        true_indices = np.nonzero(bool_crit)
        false_indices = np.nonzero(~bool_crit)

        result = self.discr.volume_empty(
                kind=self.discr.compute_kind)

        if isinstance(then, np.ndarray):
            then = then[true_indices]
        if isinstance(else_, np.ndarray):
            else_ = else_[false_indices]

        result[true_indices] = then
        result[false_indices] = else_
        return result

    def map_ref_diff_base(self, op, field_expr):
        raise NotImplementedError(
                "differentiation should be happening in batched form")

    def map_elementwise_linear(self, op, field_expr):
        field = self.rec(field_expr)

        from grudge.tools import is_zero
        if is_zero(field):
            return 0

        out = self.discr.volume_zeros()
        self.executor.do_elementwise_linear(op, field, out)
        return out

    def map_ref_quad_mass(self, op, field_expr):
        field = self.rec(field_expr)

        from grudge.tools import is_zero
        if is_zero(field):
            return 0

        qtag = op.quadrature_tag

        from grudge._internal import perform_elwise_operator

        out = self.discr.volume_zeros()
        for eg in self.discr.element_groups:
            eg_quad_info = eg.quadrature_info[qtag]

            perform_elwise_operator(eg_quad_info.ranges, eg.ranges,
                    eg_quad_info.ldis_quad_info.mass_matrix(),
                    field, out)

        return out

    def map_quad_grid_upsampler(self, op, field_expr):
        field = self.rec(field_expr)

        from grudge.tools import is_zero
        if is_zero(field):
            return 0

        qtag = op.quadrature_tag

        from grudge._internal import perform_elwise_operator
        quad_info = self.discr.get_quadrature_info(qtag)

        out = np.zeros(quad_info.node_count, field.dtype)
        for eg in self.discr.element_groups:
            eg_quad_info = eg.quadrature_info[qtag]

            perform_elwise_operator(eg.ranges, eg_quad_info.ranges,
                eg_quad_info.ldis_quad_info.volume_up_interpolation_matrix(),
                field, out)

        return out

    def map_quad_int_faces_grid_upsampler(self, op, field_expr):
        field = self.rec(field_expr)

        from grudge.tools import is_zero
        if is_zero(field):
            return 0

        qtag = op.quadrature_tag

        from grudge._internal import perform_elwise_operator
        quad_info = self.discr.get_quadrature_info(qtag)

        out = np.zeros(quad_info.int_faces_node_count, field.dtype)
        for eg in self.discr.element_groups:
            eg_quad_info = eg.quadrature_info[qtag]

            perform_elwise_operator(eg.ranges, eg_quad_info.el_faces_ranges,
                eg_quad_info.ldis_quad_info.volume_to_face_up_interpolation_matrix()
                .copy(),
                field, out)

        return out

    def map_quad_bdry_grid_upsampler(self, op, field_expr):
        field = self.rec(field_expr)

        from grudge.tools import is_zero
        if is_zero(field):
            return 0

        bdry = self.discr.get_boundary(op.boundary_tag)
        bdry_q_info = bdry.get_quadrature_info(op.quadrature_tag)

        out = np.zeros(bdry_q_info.node_count, field.dtype)

        from grudge._internal import perform_elwise_operator
        for fg, from_ranges, to_ranges, ldis_quad_info in zip(
                bdry.face_groups,
                bdry.fg_ranges,
                bdry_q_info.fg_ranges,
                bdry_q_info.fg_ldis_quad_infos):
            perform_elwise_operator(from_ranges, to_ranges,
                ldis_quad_info.face_up_interpolation_matrix(),
                field, out)

        return out

    def map_elementwise_max(self, op, field_expr):
        from grudge._internal import perform_elwise_max
        field = self.rec(field_expr)

        out = self.discr.volume_zeros(dtype=field.dtype)
        for eg in self.discr.element_groups:
            perform_elwise_max(eg.ranges, field, out)

        return out

    # }}}

    # {{{ code execution functions --------------------------------------------
    def exec_assign(self, insn):
        return [(name, self.rec(expr))
                for name, expr in zip(insn.names, insn.exprs)], []

    def exec_vector_expr_assign(self, insn):
        if self.executor.instrumented:
            def stats_callback(n, vec_expr):
                self.executor.vector_math_flop_counter.add(n*insn.flop_count())
                return self.executor.vector_math_timer
        else:
            stats_callback = None

        if insn.flop_count() == 0:
            return [(name, self(expr))
                for name, expr in zip(insn.names, insn.exprs)], []
        else:
            compiled = insn.compiled(self.executor)
            return zip(compiled.result_names(),
                    compiled(self, stats_callback)), []

    def exec_flux_batch_assign(self, insn):
        from pymbolic.primitives import is_zero

        class ZeroSpec:
            pass

        class BoundaryZeros(ZeroSpec):
            pass

        class VolumeZeros(ZeroSpec):
            pass

        def eval_arg(arg_spec):
            arg_expr, is_int = arg_spec
            arg = self.rec(arg_expr)
            if is_zero(arg):
                if insn.is_boundary and not is_int:
                    return BoundaryZeros()
                else:
                    return VolumeZeros()
            else:
                return arg

        args = [eval_arg(arg_expr)
                for arg_expr in insn.flux_var_info.arg_specs]

        from pytools import common_dtype
        max_dtype = common_dtype(
                [a.dtype for a in args if not isinstance(a, ZeroSpec)],
                self.discr.default_scalar_type)

        def cast_arg(arg):
            if isinstance(arg, BoundaryZeros):
                return self.discr.boundary_zeros(
                        insn.repr_op.boundary_tag, dtype=max_dtype)
            elif isinstance(arg, VolumeZeros):
                return self.discr.volume_zeros(
                        dtype=max_dtype)
            elif isinstance(arg, np.ndarray):
                return np.asarray(arg, dtype=max_dtype)
            else:
                return arg

        args = [cast_arg(arg) for arg in args]

        if insn.quadrature_tag is None:
            if insn.is_boundary:
                face_groups = self.discr.get_boundary(insn.repr_op.boundary_tag)\
                        .face_groups
            else:
                face_groups = self.discr.face_groups
        else:
            if insn.is_boundary:
                face_groups = self.discr.get_boundary(insn.repr_op.boundary_tag)\
                        .get_quadrature_info(insn.quadrature_tag).face_groups
            else:
                face_groups = self.discr.get_quadrature_info(insn.quadrature_tag) \
                        .face_groups

        result = []

        for fg in face_groups:
            # grab module
            module = insn.get_module(self.discr, max_dtype)
            func = module.gather_flux

            # set up argument structure
            arg_struct = module.ArgStruct()
            for arg_name, arg in zip(insn.flux_var_info.arg_names, args):
                setattr(arg_struct, arg_name, arg)
            for arg_num, scalar_arg_expr in enumerate(
                    insn.flux_var_info.scalar_parameters):
                setattr(arg_struct,
                        "_scalar_arg_%d" % arg_num,
                        self.rec(scalar_arg_expr))

            fof_shape = (fg.face_count*fg.face_length()*fg.element_count(),)
            all_fluxes_on_faces = [
                    np.zeros(fof_shape, dtype=max_dtype)
                    for f in insn.expressions]
            for i, fof in enumerate(all_fluxes_on_faces):
                setattr(arg_struct, "flux%d_on_faces" % i, fof)

            # make sure everything ended up in Boost.Python attributes
            # (i.e. empty __dict__)
            assert not arg_struct.__dict__, arg_struct.__dict__.keys()

            # perform gather
            func(fg, arg_struct)

            # do lift, produce output
            for name, flux_bdg, fluxes_on_faces in zip(insn.names, insn.expressions,
                    all_fluxes_on_faces):

                if insn.quadrature_tag is None:
                    if flux_bdg.op.is_lift:
                        mat = fg.ldis_loc.lifting_matrix()
                        scaling = fg.local_el_inverse_jacobians
                    else:
                        mat = fg.ldis_loc.multi_face_mass_matrix()
                        scaling = None
                else:
                    assert not flux_bdg.op.is_lift
                    mat = fg.ldis_loc_quad_info.multi_face_mass_matrix()
                    scaling = None

                out = self.discr.volume_zeros(dtype=fluxes_on_faces.dtype)
                self.executor.lift_flux(fg, mat, scaling, fluxes_on_faces, out)

                if self.discr.instrumented:
                    from grudge.tools import lift_flops

                    # correct for quadrature, too.
                    self.discr.lift_flop_counter.add(lift_flops(fg))

                result.append((name, out))

        if not face_groups:
            # No face groups? Still assign context variables.
            for name, flux_bdg in zip(insn.names, insn.expressions):
                result.append((name, self.discr.volume_zeros()))

        return result, []

    def exec_diff_batch_assign(self, insn):
        rst_diff = self.executor.diff(insn.operators, self.rec(insn.field))

        return [(name, diff) for name, diff in zip(insn.names, rst_diff)], []

    exec_quad_diff_batch_assign = exec_diff_batch_assign

    # }}}

# }}}


# {{{ executor

class Executor(object):
    def __init__(self, discr, code, debug_flags, instrumented):
        self.discr = discr
        self.code = code
        self.elwise_linear_cache = {}
        self.debug_flags = debug_flags

        if "dump_op_code" in debug_flags:
            from grudge.tools import open_unique_debug_file
            open_unique_debug_file("op-code", ".txt").write(
                    str(self.code))

        self.instrumented = instrumented

    def instrument(self):
        discr = self.discr
        assert discr.instrumented

        from pytools.log import time_and_count_function
        from grudge.tools import time_count_flop

        from grudge.tools import diff_rst_flops, mass_flops

        if discr.quad_min_degrees:
            from warnings import warn
            warn("flop counts for quadrature may be wrong")

        self.diff_rst = \
                time_count_flop(
                        self.diff_rst,
                        discr.diff_timer,
                        discr.diff_counter,
                        discr.diff_flop_counter,
                        diff_rst_flops(discr))

        self.do_elementwise_linear = \
                time_count_flop(
                        self.do_elementwise_linear,
                        discr.el_local_timer,
                        discr.el_local_counter,
                        discr.el_local_flop_counter,
                        mass_flops(discr))

        self.lift_flux = \
                time_and_count_function(
                        self.lift_flux,
                        discr.lift_timer,
                        discr.lift_counter)

    def lift_flux(self, fgroup, matrix, scaling, field, out):
        from grudge._internal import lift_flux
        from pytools import to_uncomplex_dtype
        lift_flux(fgroup,
                matrix.astype(to_uncomplex_dtype(field.dtype)),
                scaling, field, out)

    def diff_rst(self, op, field):
        result = self.discr.volume_zeros(dtype=field.dtype)

        from grudge._internal import perform_elwise_operator
        for eg in self.discr.element_groups:
            perform_elwise_operator(op.preimage_ranges(eg), eg.ranges,
                    op.matrices(eg)[op.rst_axis].astype(field.dtype),
                    field, result)

        return result

    def diff_builtin(self, operators, field):
        """For the batch of reference differentiation operators in
        *operators*, return the local corresponding derivatives of
        *field*.
        """

        return [self.diff_rst(op, field) for op in operators]

    def do_elementwise_linear(self, op, field, out):
        for eg in self.discr.element_groups:
            try:
                matrix, coeffs = self.elwise_linear_cache[eg, op, field.dtype]
            except KeyError:
                matrix = np.asarray(op.matrix(eg), dtype=field.dtype)
                coeffs = op.coefficients(eg)
                self.elwise_linear_cache[eg, op, field.dtype] = matrix, coeffs

            from grudge._internal import (
                    perform_elwise_scaled_operator,
                    perform_elwise_operator)

            if coeffs is None:
                perform_elwise_operator(eg.ranges, eg.ranges,
                        matrix, field, out)
            else:
                perform_elwise_scaled_operator(eg.ranges, eg.ranges,
                        coeffs, matrix, field, out)

    def __call__(self, queue, **context):
        return self.code.execute(ExecutionMapper(queue, context, self))

# }}}


# {{{ process_sym_operator function

def process_sym_operator(sym_operator, post_bind_mapper=None,
        dumper=lambda name, sym_operator: None, mesh=None,
        type_hints={}):

    import grudge.symbolic.mappers as mappers
    from grudge.symbolic.mappers.bc_to_flux import BCToFluxRewriter
    from grudge.symbolic.mappers.type_inference import TypeInferrer

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

    dumper("before-bc2flux", sym_operator)
    sym_operator = BCToFluxRewriter()(sym_operator)

    # Ordering restriction:
    #
    # - Must run constant fold before first type inference pass, because zeros,
    # while allowed, violate typing constraints (because they can't be assigned
    # a unique type), and need to be killed before the type inferrer sees them.
    #
    # - Must run BC-to-flux before first type inferrer run so that zeros in
    # flux arguments can be removed.

    dumper("before-specializer", sym_operator)
    sym_operator = mappers.OperatorSpecializer(
            TypeInferrer()(sym_operator, type_hints)
            )(sym_operator)

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

    dumper("before-derivative-join", sym_operator)
    sym_operator = mappers.DerivativeJoiner()(sym_operator)

    dumper("before-boundary-combiner", sym_operator)
    sym_operator = mappers.BoundaryCombiner(mesh)(sym_operator)

    dumper("process-finished", sym_operator)

    return sym_operator

# }}}


def bind(discr, sym_operator, post_bind_mapper=lambda x: x, type_hints={},
        debug_flags=set(), instrumented=False):
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
            mesh=discr.mesh,
            type_hints=type_hints)

    from grudge.symbolic.compiler import OperatorCompiler
    code = OperatorCompiler()(sym_operator, type_hints)

    ex = Executor(discr, code, type_hints, instrumented=instrumented)

    if "dump_dataflow_graph" in debug_flags:
        ex.code.dump_dataflow_graph()

    return ex

# vim: foldmethod=marker
