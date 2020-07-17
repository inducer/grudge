from __future__ import division, absolute_import

__copyright__ = "Copyright (C) 2015-2017 Andreas Kloeckner, Bogdan Enache"

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
from grudge.function_registry import base_function_registry

import logging
logger = logging.getLogger(__name__)

from loopy.version import LOOPY_USE_LANGUAGE_VERSION_2018_2  # noqa: F401


MPI_TAG_SEND_TAGS = 1729


# {{{ exec mapper

class ExecutionMapper(mappers.Evaluator,
        mappers.BoundOpMapperMixin,
        mappers.LocalOpReducerMixin):
    def __init__(self, queue, context, bound_op):
        super(ExecutionMapper, self).__init__(context)
        self.discrwb = bound_op.discrwb
        self.bound_op = bound_op
        self.function_registry = bound_op.function_registry
        self.queue = queue

    # {{{ expression mappings

    def map_ones(self, expr):
        if expr.dd.is_scalar():
            return 1

        discr = self.discrwb.discr_from_dd(expr.dd)

        result = discr.empty(self.queue, allocator=self.bound_op.allocator)
        result.fill(1)
        return result

    def map_node_coordinate_component(self, expr):
        discr = self.discrwb.discr_from_dd(expr.dd)
        return discr.nodes()[expr.axis].with_queue(self.queue)

    def map_grudge_variable(self, expr):
        from numbers import Number

        value = self.context[expr.name]
        if not expr.dd.is_scalar() and isinstance(value, Number):
            discr = self.discrwb.discr_from_dd(expr.dd)
            ary = discr.empty(self.queue)
            ary.fill(value)
            value = ary

        return value

    def map_subscript(self, expr):
        value = super(ExecutionMapper, self).map_subscript(expr)

        if isinstance(expr.aggregate, sym.Variable):
            dd = expr.aggregate.dd

            from numbers import Number
            if not dd.is_scalar() and isinstance(value, Number):
                discr = self.discrwb.discr_from_dd(dd)
                ary = discr.empty(self.queue)
                ary.fill(value)
                value = ary
        return value

    def map_call(self, expr):
        args = [self.rec(p) for p in expr.parameters]
        return self.function_registry[expr.function.name](self.queue, *args)

    # }}}

    # {{{ elementwise reductions

    def _map_elementwise_reduction(self, op_name, field_expr, dd):
        @memoize_in(self, "elementwise_%s_knl" % op_name)
        def knl():
            knl = lp.make_kernel(
                "{[el, idof, jdof]: 0<=el<nelements and 0<=idof, jdof<ndofs}",
                """
                result[el, idof] = %s(jdof, operand[el, jdof])
                """ % op_name,
                default_offset=lp.auto,
                name="elementwise_%s_knl" % op_name)

            return lp.tag_inames(knl, "el:g.0,idof:l.0")

        field = self.rec(field_expr)
        discr = self.discrwb.discr_from_dd(dd)
        assert field.shape == (discr.nnodes,)

        result = discr.empty(queue=self.queue, dtype=field.dtype,
                allocator=self.bound_op.allocator)
        for grp in discr.groups:
            knl()(self.queue,
                    operand=grp.view(field),
                    result=grp.view(result))

        return result

    def map_elementwise_sum(self, op, field_expr):
        return self._map_elementwise_reduction("sum", field_expr, op.dd_in)

    def map_elementwise_min(self, op, field_expr):
        return self._map_elementwise_reduction("min", field_expr, op.dd_in)

    def map_elementwise_max(self, op, field_expr):
        return self._map_elementwise_reduction("max", field_expr, op.dd_in)

    # }}}

    # {{{ nodal reductions

    def map_nodal_sum(self, op, field_expr):
        # FIXME: Could allow array scalars
        return cl.array.sum(self.rec(field_expr)).get()[()]

    def map_nodal_max(self, op, field_expr):
        # FIXME: Could allow array scalars
        return cl.array.max(self.rec(field_expr)).get()[()]

    def map_nodal_min(self, op, field_expr):
        # FIXME: Could allow array scalars
        return cl.array.min(self.rec(field_expr)).get()[()]

    # }}}

    def map_if(self, expr):
        bool_crit = self.rec(expr.condition)

        then = self.rec(expr.then)
        else_ = self.rec(expr.else_)

        import pymbolic.primitives as p
        var = p.Variable

        i = var("i")
        if isinstance(then,  pyopencl.array.Array):
            sym_then = var("a")[i]
        elif isinstance(then,  np.number):
            sym_then = var("a")
        else:
            raise TypeError(
                "Expected parameter to be of type np.number or pyopencl.array.Array")

        if isinstance(else_,  pyopencl.array.Array):
            sym_else = var("b")[i]
        elif isinstance(else_,  np.number):
            sym_else = var("b")
        else:
            raise TypeError(
                "Expected parameter to be of type np.number or pyopencl.array.Array")

        @memoize_in(self.bound_op, "map_if_knl")
        def knl():
            knl = lp.make_kernel(
                "{[i]: 0<=i<n}",
                [
                    lp.Assignment(var("out")[i],
                        p.If(var("crit")[i], sym_then, sym_else))
                ])
            return lp.split_iname(knl, "i", 128, outer_tag="g.0", inner_tag="l.0")

        evt, (out,) = knl()(self.queue, crit=bool_crit, a=then, b=else_)

        return out

    # {{{ elementwise linear operators

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
                    0<=i<ndiscr_nodes_out and
                    0<=j<ndiscr_nodes_in}""",
                "result[k,i] = sum(j, mat[i, j] * vec[k, j])",
                default_offset=lp.auto, name="diff")

            knl = lp.split_iname(knl, "i", 16, inner_tag="l.0")
            knl = lp.tag_array_axes(knl, "mat", "stride:auto,stride:auto")
            return lp.tag_inames(knl, dict(k="g.0"))

        in_discr = self.discrwb.discr_from_dd(op.dd_in)
        out_discr = self.discrwb.discr_from_dd(op.dd_out)

        result = out_discr.empty(
                queue=self.queue,
                dtype=field.dtype, allocator=self.bound_op.allocator)

        for in_grp, out_grp in zip(in_discr.groups, out_discr.groups):

            cache_key = "elwise_linear", in_grp, out_grp, op, field.dtype
            try:
                matrix = self.bound_op.operator_data_cache[cache_key]
            except KeyError:
                matrix = (
                    cl.array.to_device(
                        self.queue,
                        np.asarray(op.matrix(out_grp, in_grp), dtype=field.dtype))
                    .with_queue(None))

                self.bound_op.operator_data_cache[cache_key] = matrix

            knl()(self.queue, mat=matrix, result=out_grp.view(result),
                    vec=in_grp.view(field))

        return result

    def map_projection(self, op, field_expr):
        conn = self.discrwb.connection_from_dds(op.dd_in, op.dd_out)
        return conn(self.queue, self.rec(field_expr)).with_queue(self.queue)

    def map_opposite_partition_face_swap(self, op, field_expr):
        assert op.dd_in == op.dd_out
        bdry_conn = self.discrwb.get_distributed_boundary_swap_connection(op.dd_in)
        remote_bdry_vec = self.rec(field_expr)  # swapped by RankDataSwapAssign
        return bdry_conn(self.queue, remote_bdry_vec).with_queue(self.queue)

    def map_opposite_interior_face_swap(self, op, field_expr):
        return self.discrwb.opposite_face_connection()(
                self.queue, self.rec(field_expr)).with_queue(self.queue)

    # }}}

    # {{{ face mass operator

    def map_ref_face_mass_operator(self, op, field_expr):
        field = self.rec(field_expr)

        from grudge.tools import is_zero
        if is_zero(field):
            return 0

        @memoize_in(self.bound_op, "face_mass_knl")
        def knl():
            knl = lp.make_kernel(
                """{[k,i,f,j]:
                    0<=k<nelements and
                    0<=f<nfaces and
                    0<=i<nvol_nodes and
                    0<=j<nface_nodes}""",
                "result[k,i] = sum(f, sum(j, mat[i, f, j] * vec[f, k, j]))",
                default_offset=lp.auto, name="face_mass")

            knl = lp.split_iname(knl, "i", 16, inner_tag="l.0")
            return lp.tag_inames(knl, dict(k="g.0"))

        all_faces_conn = self.discrwb.connection_from_dds("vol", op.dd_in)
        all_faces_discr = all_faces_conn.to_discr
        vol_discr = all_faces_conn.from_discr

        result = vol_discr.empty(
                queue=self.queue,
                dtype=field.dtype, allocator=self.bound_op.allocator)

        assert len(all_faces_discr.groups) == len(vol_discr.groups)

        for afgrp, volgrp in zip(all_faces_discr.groups, vol_discr.groups):
            cache_key = "face_mass", afgrp, op, field.dtype

            nfaces = volgrp.mesh_el_group.nfaces

            try:
                matrix = self.bound_op.operator_data_cache[cache_key]
            except KeyError:
                matrix = op.matrix(afgrp, volgrp, field.dtype)
                matrix = (
                        cl.array.to_device(self.queue, matrix)
                        .with_queue(None))

                self.bound_op.operator_data_cache[cache_key] = matrix

            input_view = afgrp.view(field).reshape(
                    nfaces, volgrp.nelements, afgrp.nunit_nodes)
            knl()(self.queue, mat=matrix, result=volgrp.view(result),
                    vec=input_view)

        return result

    def map_signed_face_ones(self, expr):
        assert expr.dd.is_trace()
        face_discr = self.discrwb.discr_from_dd(expr.dd)
        assert face_discr.dim == 0

        # NOTE: ignore quadrature_tags on expr.dd, since we only care about
        # the face_id here
        all_faces_conn = self.discrwb.connection_from_dds(
                sym.DD_VOLUME,
                sym.DOFDesc(expr.dd.domain_tag))

        field = face_discr.empty(self.queue,
                dtype=self.discrwb.real_dtype,
                allocator=self.bound_op.allocator)
        field.fill(1)

        for grp in all_faces_conn.groups:
            for batch in grp.batches:
                i = batch.to_element_indices.with_queue(self.queue)
                field[i] = (2.0 * (batch.to_element_face % 2) - 1.0) * field[i]

        return field

    # }}}

    # {{{ instruction execution functions

    def map_insn_rank_data_swap(self, insn, profile_data=None):
        local_data = self.rec(insn.field).get(self.queue)
        comm = self.discrwb.mpi_communicator

        # print("Sending data to rank %d with tag %d"
        #             % (insn.i_remote_rank, insn.send_tag))
        send_req = comm.Isend(local_data, insn.i_remote_rank, tag=insn.send_tag)

        remote_data_host = np.empty_like(local_data)
        recv_req = comm.Irecv(remote_data_host, insn.i_remote_rank, insn.recv_tag)

        return [], [
                MPIRecvFuture(recv_req, insn.name, remote_data_host, self.queue),
                MPISendFuture(send_req)]

    def map_insn_loopy_kernel(self, insn, profile_data=None):
        kwargs = {}
        kdescr = insn.kernel_descriptor
        for name, expr in six.iteritems(kdescr.input_mappings):
            kwargs[name] = self.rec(expr)

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
        return list(result_dict.items()), []

    def map_insn_assign(self, insn, profile_data=None):
        return [(name, self.rec(expr))
                for name, expr in zip(insn.names, insn.exprs)], []

    def map_insn_assign_to_discr_scoped(self, insn, profile_data=None):
        assignments = []
        for name, expr in zip(insn.names, insn.exprs):
            value = self.rec(expr)
            self.discrwb._discr_scoped_subexpr_name_to_value[name] = value
            assignments.append((name, value))

        return assignments, []

    def map_insn_assign_from_discr_scoped(self, insn, profile_data=None):
        return [(insn.name,
            self.discrwb._discr_scoped_subexpr_name_to_value[insn.name])], []

    def map_insn_diff_batch_assign(self, insn, profile_data=None):
        field = self.rec(insn.field)
        repr_op = insn.operators[0]
        # FIXME: There's no real reason why differentiation is special,
        # execution-wise.
        # This should be unified with map_elementwise_linear, which should
        # be extended to support batching.

        assert repr_op.dd_in.domain_tag == repr_op.dd_out.domain_tag

        @memoize_in(self.discrwb, "reference_derivative_knl")
        def knl():
            knl = lp.make_kernel(
                """{[imatrix,k,i,j]:
                    0<=imatrix<nmatrices and
                    0<=k<nelements and
                    0<=i<nunit_nodes_out and
                    0<=j<nunit_nodes_in}""",
                """
                result[imatrix, k, i] = sum(
                        j, diff_mat[imatrix, i, j] * vec[k, j])
                """,
                default_offset=lp.auto, name="diff")

            knl = lp.split_iname(knl, "i", 16, inner_tag="l.0")
            return lp.tag_inames(knl, dict(k="g.0"))

        noperators = len(insn.operators)

        in_discr = self.discrwb.discr_from_dd(repr_op.dd_in)
        out_discr = self.discrwb.discr_from_dd(repr_op.dd_out)

        result = out_discr.empty(
                queue=self.queue,
                dtype=field.dtype, extra_dims=(noperators,),
                allocator=self.bound_op.allocator)

        for in_grp, out_grp in zip(in_discr.groups, out_discr.groups):

            if in_grp.nelements == 0:
                continue

            matrices = repr_op.matrices(out_grp, in_grp)

            # FIXME: Should transfer matrices to device and cache them
            matrices_ary = np.empty((
                noperators, out_grp.nunit_nodes, in_grp.nunit_nodes))
            for i, op in enumerate(insn.operators):
                matrices_ary[i] = matrices[op.rst_axis]

            knl()(self.queue,
                    diff_mat=matrices_ary,
                    result=out_grp.view(result), vec=in_grp.view(field))

        return [(name, result[i]) for i, name in enumerate(insn.names)], []

    # }}}

# }}}


# {{{ futures

class MPIRecvFuture(object):
    def __init__(self, recv_req, insn_name, remote_data_host, queue):
        self.receive_request = recv_req
        self.insn_name = insn_name
        self.remote_data_host = remote_data_host
        self.queue = queue

    def is_ready(self):
        return self.receive_request.Test()

    def __call__(self):
        self.receive_request.Wait()
        remote_data = cl.array.to_device(self.queue, self.remote_data_host)
        return [(self.insn_name, remote_data)], []


class MPISendFuture(object):
    def __init__(self, send_request):
        self.send_request = send_request

    def is_ready(self):
        return self.send_request.Test()

    def __call__(self):
        self.send_request.wait()
        return [], []

# }}}


# {{{ bound operator

class BoundOperator(object):

    def __init__(self, discrwb, discr_code, eval_code, debug_flags,
            function_registry, exec_mapper_factory, allocator=None):
        self.discrwb = discrwb
        self.discr_code = discr_code
        self.eval_code = eval_code
        self.operator_data_cache = {}
        self.debug_flags = debug_flags
        self.function_registry = function_registry
        self.allocator = allocator
        self.exec_mapper_factory = exec_mapper_factory

    def __str__(self):
        sep = 75 * "=" + "\n"
        return (
                sep
                + "DISCRETIZATION-SCOPE CODE\n"
                + sep
                + str(self.discr_code) + "\n"
                + sep
                + "PER-EVALUATION CODE\n"
                + sep
                + str(self.eval_code))

    def __call__(self, queue, profile_data=None, log_quantities=None, **context):
        import pyopencl.array as cl_array

        def replace_queue(a):
            if isinstance(a, cl_array.Array):
                return a.with_queue(queue)
            else:
                return a

        from pytools.obj_array import with_object_array_or_scalar

        # {{{ discrwb-scope evaluation

        if any(
                (result_var.name not in
                    self.discrwb._discr_scoped_subexpr_name_to_value)
                for result_var in self.discr_code.result):
            # need to do discrwb-scope evaluation
            discrwb_eval_context = {}
            self.discr_code.execute(
                    self.exec_mapper_factory(queue, discrwb_eval_context, self))

        # }}}

        new_context = {}
        for name, var in six.iteritems(context):
            new_context[name] = with_object_array_or_scalar(replace_queue, var)

        return self.eval_code.execute(
                self.exec_mapper_factory(queue, new_context, self),
                profile_data=profile_data,
                log_quantities=log_quantities)

# }}}


# {{{ process_sym_operator function

def process_sym_operator(discrwb, sym_operator, post_bind_mapper=None,
        dumper=lambda name, sym_operator: None):

    orig_sym_operator = sym_operator
    import grudge.symbolic.mappers as mappers

    dumper("before-bind", sym_operator)
    sym_operator = mappers.OperatorBinder()(sym_operator)

    mappers.ErrorChecker(discrwb.mesh)(sym_operator)

    sym_operator = \
            mappers.OppositeInteriorFaceSwapUniqueIDAssigner()(sym_operator)

    # {{{ broadcast root rank's symn_operator

    # also make sure all ranks had same orig_sym_operator

    if discrwb.mpi_communicator is not None:
        (mgmt_rank_orig_sym_operator, mgmt_rank_sym_operator) = \
                discrwb.mpi_communicator.bcast(
                    (orig_sym_operator, sym_operator),
                    discrwb.get_management_rank_index())

        from pytools.obj_array import is_equal as is_oa_equal
        if not is_oa_equal(mgmt_rank_orig_sym_operator, orig_sym_operator):
            raise ValueError("rank %d received a different symbolic "
                    "operator to bind from rank %d"
                    % (discrwb.mpi_communicator.Get_rank(),
                        discrwb.get_management_rank_index()))

        sym_operator = mgmt_rank_sym_operator

    # }}}

    if post_bind_mapper is not None:
        dumper("before-postbind", sym_operator)
        sym_operator = post_bind_mapper(sym_operator)

    dumper("before-empty-flux-killer", sym_operator)
    sym_operator = mappers.EmptyFluxKiller(discrwb.mesh)(sym_operator)

    dumper("before-cfold", sym_operator)
    sym_operator = mappers.CommutativeConstantFoldingMapper()(sym_operator)

    dumper("before-qcheck", sym_operator)
    sym_operator = mappers.QuadratureCheckerAndRemover(
            discrwb.quad_tag_to_group_factory)(sym_operator)

    # Work around https://github.com/numpy/numpy/issues/9438
    #
    # The idea is that we need 1j as an expression to survive
    # until code generation time. If it is evaluated and combined
    # with other constants, we will need to determine its size
    # (as np.complex64/128) within the expression. But because
    # of the above numpy bug, sized numbers are not likely to survive
    # expression building--so that's why we step in here to fix that.

    dumper("before-csize", sym_operator)
    sym_operator = mappers.ConstantToNumpyConversionMapper(
            real_type=discrwb.real_dtype.type,
            complex_type=discrwb.complex_dtype.type,
            )(sym_operator)

    dumper("before-global-to-reference", sym_operator)
    sym_operator = mappers.GlobalToReferenceMapper(discrwb.ambient_dim)(sym_operator)

    dumper("before-distributed", sym_operator)

    volume_mesh = discrwb.discr_from_dd("vol").mesh
    from meshmode.distributed import get_connected_partitions
    connected_parts = get_connected_partitions(volume_mesh)

    if connected_parts:
        sym_operator = mappers.DistributedMapper(connected_parts)(sym_operator)

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
        function_registry=base_function_registry,
        exec_mapper_factory=ExecutionMapper,
        debug_flags=frozenset(), allocator=None):
    # from grudge.symbolic.mappers import QuadratureUpsamplerRemover
    # sym_operator = QuadratureUpsamplerRemover(self.quad_min_degrees)(
    #         sym_operator)

    stage = [0]

    def dump_sym_operator(name, sym_operator):
        if "dump_sym_operator_stages" in debug_flags:
            from pytools.debug import open_unique_debug_file
            outf, name = open_unique_debug_file("%02d-%s" % (stage[0], name), ".txt")
            with outf:
                outf.write(sym.pretty(sym_operator))

            stage[0] += 1

    sym_operator = process_sym_operator(
            discr,
            sym_operator,
            post_bind_mapper=post_bind_mapper,
            dumper=dump_sym_operator)

    from grudge.symbolic.compiler import OperatorCompiler
    discr_code, eval_code = OperatorCompiler(discr, function_registry)(sym_operator)

    bound_op = BoundOperator(discr, discr_code, eval_code,
            function_registry=function_registry,
            exec_mapper_factory=exec_mapper_factory,
            debug_flags=debug_flags,
            allocator=allocator)

    if "dump_op_code" in debug_flags:
        from pytools.debug import open_unique_debug_file
        outf, _ = open_unique_debug_file("op-code", ".txt")
        with outf:
            outf.write(str(bound_op))

    if "dump_dataflow_graph" in debug_flags:
        discr_code.dump_dataflow_graph("discr")
        eval_code.dump_dataflow_graph("eval")

    return bound_op

# vim: foldmethod=marker
