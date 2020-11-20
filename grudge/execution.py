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

from typing import Optional, Union, Dict
from numbers import Number
import numpy as np

from pytools import memoize_in
from pytools.obj_array import make_obj_array

import loopy as lp
import pyopencl as cl
import pyopencl.array  # noqa

from meshmode.dof_array import DOFArray, thaw, flatten, unflatten
from meshmode.array_context import ArrayContext, make_loopy_program

import grudge.symbolic.mappers as mappers
from grudge import sym
from grudge.function_registry import base_function_registry

import logging
logger = logging.getLogger(__name__)

from loopy.version import LOOPY_USE_LANGUAGE_VERSION_2018_2  # noqa: F401


ResultType = Union[DOFArray, Number]


# {{{ exec mapper

class ExecutionMapper(mappers.Evaluator,
        mappers.BoundOpMapperMixin,
        mappers.LocalOpReducerMixin):
    def __init__(self, array_context, context, bound_op):
        super().__init__(context)
        self.discrwb = bound_op.discrwb
        self.bound_op = bound_op
        self.function_registry = bound_op.function_registry
        self.array_context = array_context

    # {{{ expression mappings

    def map_ones(self, expr):
        if expr.dd.is_scalar():
            return 1

        discr = self.discrwb.discr_from_dd(expr.dd)

        result = discr.empty(self.array_context)
        for grp_ary in result:
            grp_ary.fill(1.0)
        return result

    def map_node_coordinate_component(self, expr):
        discr = self.discrwb.discr_from_dd(expr.dd)
        return thaw(self.array_context, discr.nodes()[expr.axis])

    def map_grudge_variable(self, expr):
        from numbers import Number

        value = self.context[expr.name]
        if not expr.dd.is_scalar() and isinstance(value, Number):
            discr = self.discrwb.discr_from_dd(expr.dd)
            ary = discr.empty(self.array_context)
            for grp_ary in ary:
                grp_ary.fill(value)
            value = ary

        return value

    def map_subscript(self, expr):
        value = super().map_subscript(expr)

        if isinstance(expr.aggregate, sym.Variable):
            dd = expr.aggregate.dd

            from numbers import Number
            if not dd.is_scalar() and isinstance(value, Number):
                discr = self.discrwb.discr_from_dd(dd)
                ary = discr.empty(self.array_context)
                for grp_ary in ary:
                    grp_ary.fill(value)
                value = ary
        return value

    def map_call(self, expr):
        args = [self.rec(p) for p in expr.parameters]
        return self.function_registry[expr.function.name](self.array_context, *args)

    # }}}

    # {{{ elementwise reductions

    def _map_elementwise_reduction(self, op_name, field_expr, dd):
        @memoize_in(self.array_context,
                (ExecutionMapper, "elementwise_%s_prg" % op_name))
        def prg():
            return make_loopy_program(
                "{[iel, idof, jdof]: 0<=iel<nelements and 0<=idof, jdof<ndofs}",
                """
                result[iel, idof] = %s(jdof, operand[iel, jdof])
                """ % op_name,
                name="grudge_elementwise_%s" % op_name)

        field = self.rec(field_expr)
        discr = self.discrwb.discr_from_dd(dd)
        assert field.shape == (len(discr.groups),)

        result = discr.empty(self.array_context, dtype=field.entry_dtype)
        for grp in discr.groups:
            assert field[grp.index].shape == (grp.nelements, grp.nunit_dofs)
            self.array_context.call_loopy(
                    prg(),
                    operand=field[grp.index],
                    result=result[grp.index])

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
        # FIXME: Fix CL-specific-ness
        return sum([
                cl.array.sum(grp_ary).get()[()]
                for grp_ary in self.rec(field_expr)
                ])

    def map_nodal_max(self, op, field_expr):
        # FIXME: Could allow array scalars
        # FIXME: Fix CL-specific-ness
        return np.max([
            cl.array.max(grp_ary).get()[()]
            for grp_ary in self.rec(field_expr)])

    def map_nodal_min(self, op, field_expr):
        # FIXME: Could allow array scalars
        # FIXME: Fix CL-specific-ness
        return np.min([
            cl.array.min(grp_ary).get()[()]
            for grp_ary in self.rec(field_expr)])

    # }}}

    def map_if(self, expr):
        bool_crit = self.rec(expr.condition)

        if isinstance(bool_crit, DOFArray):
            # continues below
            pass
        elif isinstance(bool_crit, (np.bool_, np.bool, np.number)):
            if bool_crit:
                return self.rec(expr.then)
            else:
                return self.rec(expr.else_)
        else:
            raise TypeError(
                "Expected criterion to be of type np.number or DOFArray")

        assert isinstance(bool_crit, DOFArray)
        ngroups = len(bool_crit)

        from pymbolic import var
        iel = var("iel")
        idof = var("idof")
        subscript = (iel, idof)

        then = self.rec(expr.then)
        else_ = self.rec(expr.else_)

        import pymbolic.primitives as p
        var = p.Variable

        if isinstance(then, DOFArray):
            sym_then = var("a")[subscript]

            def get_then(igrp):
                return then[igrp]
        elif isinstance(then, np.number):
            sym_then = var("a")

            def get_then(igrp):
                return then
        else:
            raise TypeError(
                "Expected 'then' to be of type np.number or DOFArray")

        if isinstance(else_, DOFArray):
            sym_else = var("b")[subscript]

            def get_else(igrp):
                return else_[igrp]
        elif isinstance(else_, np.number):
            sym_else = var("b")

            def get_else(igrp):
                return else_
        else:
            raise TypeError(
                "Expected 'else' to be of type np.number or DOFArray")

        @memoize_in(self.array_context, (ExecutionMapper, "map_if_knl"))
        def knl(sym_then, sym_else):
            return make_loopy_program(
                "{[iel, idof]: 0<=iel<nelements and 0<=idof<nunit_dofs}",
                [
                    lp.Assignment(var("out")[iel, idof],
                        p.If(var("crit")[iel, idof], sym_then, sym_else))
                ])

        return DOFArray(self.array_context, tuple(
            self.array_context.call_loopy(
                knl(sym_then, sym_else),
                crit=bool_crit[igrp],
                a=get_then(igrp),
                b=get_else(igrp))
            for igrp in range(ngroups)))

    # {{{ elementwise linear operators

    def map_ref_diff_base(self, op, field_expr):
        raise NotImplementedError(
                "differentiation should be happening in batched form")

    def map_elementwise_linear(self, op, field_expr):
        field = self.rec(field_expr)

        from grudge.tools import is_zero
        if is_zero(field):
            return 0

        @memoize_in(self.array_context, (ExecutionMapper, "elwise_linear_knl"))
        def prg():
            result = make_loopy_program(
                """{[iel, idof, j]:
                    0<=iel<nelements and
                    0<=idof<ndiscr_nodes_out and
                    0<=j<ndiscr_nodes_in}""",
                "result[iel, idof] = sum(j, mat[idof, j] * vec[iel, j])",
                name="diff")

            result = lp.tag_array_axes(result, "mat", "stride:auto,stride:auto")
            return result

        in_discr = self.discrwb.discr_from_dd(op.dd_in)
        out_discr = self.discrwb.discr_from_dd(op.dd_out)

        result = out_discr.empty(self.array_context, dtype=field.entry_dtype)

        for in_grp, out_grp in zip(in_discr.groups, out_discr.groups):

            cache_key = "elwise_linear", in_grp, out_grp, op, field.entry_dtype
            try:
                matrix = self.bound_op.operator_data_cache[cache_key]
            except KeyError:
                matrix = self.array_context.freeze(
                        self.array_context.from_numpy(
                            np.asarray(
                                op.matrix(out_grp, in_grp),
                                dtype=field.entry_dtype)))

                self.bound_op.operator_data_cache[cache_key] = matrix

            self.array_context.call_loopy(
                    prg(),
                    mat=matrix,
                    result=result[out_grp.index],
                    vec=field[in_grp.index])

        return result

    def map_projection(self, op, field_expr):
        conn = self.discrwb.connection_from_dds(op.dd_in, op.dd_out)
        return conn(self.rec(field_expr))

    def map_opposite_partition_face_swap(self, op, field_expr):
        assert op.dd_in == op.dd_out
        bdry_conn = self.discrwb.get_distributed_boundary_swap_connection(op.dd_in)
        remote_bdry_vec = self.rec(field_expr)  # swapped by RankDataSwapAssign
        return bdry_conn(remote_bdry_vec)

    def map_opposite_interior_face_swap(self, op, field_expr):
        return self.discrwb.opposite_face_connection()(self.rec(field_expr))

    # }}}

    # {{{ face mass operator

    def map_ref_face_mass_operator(self, op, field_expr):
        field = self.rec(field_expr)

        from grudge.tools import is_zero
        if is_zero(field):
            return 0

        @memoize_in(self.array_context, (ExecutionMapper, "face_mass_knl"))
        def prg():
            return make_loopy_program(
                """{[iel,idof,f,j]:
                    0<=iel<nelements and
                    0<=f<nfaces and
                    0<=idof<nvol_nodes and
                    0<=j<nface_nodes}""",
                """
                result[iel,idof] = sum(f, sum(j, mat[idof, f, j] * vec[f, iel, j]))
                """,
                name="face_mass")

        all_faces_conn = self.discrwb.connection_from_dds("vol", op.dd_in)
        all_faces_discr = all_faces_conn.to_discr
        vol_discr = all_faces_conn.from_discr

        result = vol_discr.empty(self.array_context, dtype=field.entry_dtype)

        assert len(all_faces_discr.groups) == len(vol_discr.groups)

        for afgrp, volgrp in zip(all_faces_discr.groups, vol_discr.groups):
            cache_key = "face_mass", afgrp, op, field.entry_dtype

            nfaces = volgrp.mesh_el_group.nfaces

            try:
                matrix = self.bound_op.operator_data_cache[cache_key]
            except KeyError:
                matrix = op.matrix(afgrp, volgrp, field.entry_dtype)
                matrix = self.array_context.freeze(
                        self.array_context.from_numpy(matrix))

                self.bound_op.operator_data_cache[cache_key] = matrix

            input_view = field[afgrp.index].reshape(
                    nfaces, volgrp.nelements, afgrp.nunit_dofs)
            self.array_context.call_loopy(
                    prg(),
                    mat=matrix,
                    result=result[volgrp.index],
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

        field = face_discr.empty(self.array_context, dtype=self.discrwb.real_dtype)
        for grp_ary in field:
            grp_ary.fill(1)

        for igrp, grp in enumerate(all_faces_conn.groups):
            for batch in grp.batches:
                i = self.array_context.thaw(batch.to_element_indices)
                grp_field = field[igrp].reshape(-1)
                grp_field[i] = \
                        (2.0 * (batch.to_element_face % 2) - 1.0) * grp_field[i]

        return field

    # }}}

    # {{{ instruction execution functions

    def map_insn_rank_data_swap(self, insn, profile_data=None):
        local_data = self.array_context.to_numpy(flatten(self.rec(insn.field)))
        comm = self.discrwb.mpi_communicator

        # print("Sending data to rank %d with tag %d"
        #             % (insn.i_remote_rank, insn.send_tag))
        send_req = comm.Isend(local_data, insn.i_remote_rank, tag=insn.send_tag)

        remote_data_host = np.empty_like(local_data)
        recv_req = comm.Irecv(remote_data_host, insn.i_remote_rank, insn.recv_tag)

        return [], [
                MPIRecvFuture(
                    array_context=self.array_context,
                    bdry_discr=self.discrwb.discr_from_dd(insn.dd_out),
                    recv_req=recv_req,
                    insn_name=insn.name,
                    remote_data_host=remote_data_host),
                MPISendFuture(send_req)]

    def map_insn_loopy_kernel(self, insn, profile_data=None):
        kdescr = insn.kernel_descriptor
        discr = self.discrwb.discr_from_dd(kdescr.governing_dd)

        dof_array_kwargs = {}
        other_kwargs = {}

        for name, expr in kdescr.input_mappings.items():
            v = self.rec(expr)
            if isinstance(v, DOFArray):
                dof_array_kwargs[name] = v
            else:
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

            knl_result = self.array_context.call_loopy(
                    kdescr.loopy_kernel, **kwargs)

            for name, val in knl_result.items():
                result.setdefault(name, []).append(val)

        result = {
                name: DOFArray(self.array_context, tuple(val))
                for name, val in result.items()}

        return list(result.items()), []

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

        @memoize_in(self.array_context,
                (ExecutionMapper, "reference_derivative_prg"))
        def prg(nmatrices):
            result = make_loopy_program(
                """{[imatrix, iel, idof, j]:
                    0<=imatrix<nmatrices and
                    0<=iel<nelements and
                    0<=idof<nunit_nodes_out and
                    0<=j<nunit_nodes_in}""",
                """
                result[imatrix, iel, idof] = sum(
                        j, diff_mat[imatrix, idof, j] * vec[iel, j])
                """,
                name="diff")

            result = lp.fix_parameters(result, nmatrices=nmatrices)
            result = lp.tag_inames(result, "imatrix: unr")
            result = lp.tag_array_axes(result, "result", "sep,c,c")
            return result

        noperators = len(insn.operators)

        in_discr = self.discrwb.discr_from_dd(repr_op.dd_in)
        out_discr = self.discrwb.discr_from_dd(repr_op.dd_out)

        result = make_obj_array([
            out_discr.empty(self.array_context, dtype=field.entry_dtype)
            for idim in range(noperators)])

        for in_grp, out_grp in zip(in_discr.groups, out_discr.groups):
            if in_grp.nelements == 0:
                continue

            # Cache operator
            cache_key = "diff_batch", in_grp, out_grp, tuple(insn.operators),\
                field.entry_dtype
            try:
                matrices_ary_dev = self.bound_op.operator_data_cache[cache_key]
            except KeyError:
                matrices = repr_op.matrices(out_grp, in_grp)
                matrices_ary = np.empty(
                    (noperators, out_grp.nunit_dofs, in_grp.nunit_dofs),
                    dtype=field.entry_dtype)
                for i, op in enumerate(insn.operators):
                    matrices_ary[i] = matrices[op.rst_axis]
                matrices_ary_dev = self.array_context.from_numpy(matrices_ary)
                self.bound_op.operator_data_cache[cache_key] = matrices_ary_dev

            self.array_context.call_loopy(
                    prg(noperators),
                    diff_mat=matrices_ary_dev,
                    result=make_obj_array([
                        result[iop][out_grp.index]
                        for iop in range(noperators)
                        ]), vec=field[in_grp.index])

        return [(name, result[i]) for i, name in enumerate(insn.names)], []

    # }}}

# }}}


# {{{ futures

class MPIRecvFuture:
    def __init__(self, array_context, bdry_discr, recv_req, insn_name,
            remote_data_host):
        self.array_context = array_context
        self.bdry_discr = bdry_discr
        self.receive_request = recv_req
        self.insn_name = insn_name
        self.remote_data_host = remote_data_host

    def is_ready(self):
        return self.receive_request.Test()

    def __call__(self):
        self.receive_request.Wait()
        actx = self.array_context
        remote_data = unflatten(self.array_context, self.bdry_discr,
                actx.from_numpy(self.remote_data_host))
        return [(self.insn_name, remote_data)], []


class MPISendFuture:
    def __init__(self, send_request):
        self.send_request = send_request

    def is_ready(self):
        return self.send_request.Test()

    def __call__(self):
        self.send_request.Wait()
        return [], []

# }}}


# {{{ bound operator

class BoundOperator:
    def __init__(self, discrwb, discr_code, eval_code, debug_flags,
            function_registry, exec_mapper_factory):
        self.discrwb = discrwb
        self.discr_code = discr_code
        self.eval_code = eval_code
        self.operator_data_cache = {}
        self.debug_flags = debug_flags
        self.function_registry = function_registry
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

    def __call__(self, array_context: Optional[ArrayContext] = None,
            *, profile_data=None, log_quantities=None, **context):
        """
        :arg array_context: only needs to be supplied if no instances of
            :class:`~meshmode.dof_array.DOFArray` with a
            :class:`~meshmode.array_context.ArrayContext`
            are supplied as part of *context*.
        """

        # {{{ figure array context

        array_contexts = []
        if array_context is not None:
            if not isinstance(array_context, ArrayContext):
                raise TypeError(
                        "first positional argument (if supplied) must be "
                        "an ArrayContext")

            array_contexts.append(array_context)
        del array_context

        def look_for_array_contexts(ary):
            if isinstance(ary, DOFArray):
                if ary.array_context is not None:
                    array_contexts.append(ary.array_context)
            elif isinstance(ary, np.ndarray) and ary.dtype.char == "O":
                for idx in np.ndindex(ary.shape):
                    look_for_array_contexts(ary[idx])
            else:
                pass

        for key, val in context.items():
            look_for_array_contexts(val)

        if array_contexts:
            from pytools import is_single_valued
            if not is_single_valued(array_contexts):
                raise ValueError("arguments do not agree on an array context")

            array_context = array_contexts[0]
        else:
            raise ValueError("no array context given or available from arguments")

        # }}}

        # {{{ discrwb-scope evaluation

        if any(
                (result_var.name not in
                    self.discrwb._discr_scoped_subexpr_name_to_value)
                for result_var in self.discr_code.result):
            # need to do discrwb-scope evaluation
            discrwb_eval_context: Dict[str, ResultType] = {}
            self.discr_code.execute(
                    self.exec_mapper_factory(
                        array_context, discrwb_eval_context, self))

        # }}}

        return self.eval_code.execute(
                self.exec_mapper_factory(array_context, context, self),
                profile_data=profile_data,
                log_quantities=log_quantities)

# }}}


# {{{ process_sym_operator function

def process_sym_operator(discrwb, sym_operator, post_bind_mapper=None, dumper=None,
        local_only=None):
    if local_only is None:
        local_only = False

    if dumper is None:
        def dumper(name, sym_operator):
            return

    orig_sym_operator = sym_operator
    import grudge.symbolic.mappers as mappers

    dumper("before-bind", sym_operator)
    sym_operator = mappers.OperatorBinder()(sym_operator)

    mappers.ErrorChecker(discrwb.mesh)(sym_operator)

    sym_operator = \
            mappers.OppositeInteriorFaceSwapUniqueIDAssigner()(sym_operator)

    if not local_only:
        # {{{ broadcast root rank's symn_operator

        # also make sure all ranks had same orig_sym_operator

        if discrwb.mpi_communicator is not None:
            (mgmt_rank_orig_sym_operator, mgmt_rank_sym_operator) = \
                    discrwb.mpi_communicator.bcast(
                        (orig_sym_operator, sym_operator),
                        discrwb.get_management_rank_index())

            if not np.array_equal(mgmt_rank_orig_sym_operator, orig_sym_operator):
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
    sym_operator = mappers.GlobalToReferenceMapper(discrwb)(sym_operator)

    dumper("before-distributed", sym_operator)

    if not local_only:
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


def bind(discr, sym_operator, *, post_bind_mapper=lambda x: x,
        function_registry=base_function_registry,
        exec_mapper_factory=ExecutionMapper,
        debug_flags=frozenset(), local_only=None):
    """
    :param local_only: If *True*, *sym_operator* should oly be evaluated on the
        local part of the mesh. No inter-rank communication will take place.
        (However rank boundaries, tagged :class:`~meshmode.mesh.BTAG_PARTITION`,
        will not automatically be considered part of the domain boundary.)
    """
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
            dumper=dump_sym_operator,
            local_only=local_only)

    from grudge.symbolic.compiler import OperatorCompiler
    discr_code, eval_code = OperatorCompiler(discr, function_registry)(sym_operator)

    bound_op = BoundOperator(discr, discr_code, eval_code,
            function_registry=function_registry,
            exec_mapper_factory=exec_mapper_factory,
            debug_flags=debug_flags)

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
