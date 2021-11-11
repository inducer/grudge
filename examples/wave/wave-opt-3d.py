"""Minimal example of a grudge driver."""

__copyright__ = """
Copyright (C) 2015 Andreas Kloeckner
Copyright (C) 2021 University of Illinois Board of Trustees
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


import numpy as np
import pyopencl as cl
import pyopencl.tools as cl_tools

from arraycontext import thaw, freeze
from grudge.array_context import PyOpenCLArrayContext
from meshmode.array_context import (
    SingleGridWorkBalancingPytatoArrayContext)

from typing import Union
from time import time
from grudge import DiscretizationCollection

from pytools.obj_array import flat_obj_array
from loopy.symbolic import (get_dependencies, CombineMapper)
from meshmode.transform_metadata import (DiscretizationEntityAxisTag,
                                         DiscretizationElementAxisTag,
                                         DiscretizationDOFAxisTag,
                                         DiscretizationFaceAxisTag)

import grudge.op as op
import kanren
from kanren.constraints import neq as kanren_neq

import logging
logger = logging.getLogger(__name__)


REWRITE_DIFF_EINSUM = False


class IndirectAccessChecker(CombineMapper):
    """
    On calling returns *True* iff the *array_name* was accessed indirectly.
    """
    def __init__(self, array_name, all_inames):
        self.array_name = array_name
        self.all_inames = all_inames

    def combine(self, values):
        return any(values)

    def map_subscript(self, expr):
        if expr.aggregate.name == self.array_name:
            return not (get_dependencies(expr.index_tuple) <= self.all_inames)
        else:
            return super().map_subscript(expr)

    def map_variable(self, expr):
        return False

    def map_constant(self, expr):
        return False

    def map_resolved_function(self, expr):
        return False


def transform_face_mass(t_unit):
    knl = t_unit.default_entrypoint

    import loopy as lp
    from loopy.transform.data import add_prefetch_for_single_kernel

    n_elements_per_wg = 8
    n_work_items_per_element = 16

    knl = lp.split_iname(knl, "iel_face_mass", n_elements_per_wg,
                         outer_tag="g.0")
    knl = lp.split_iname(knl, "idof_face_mass",
                         n_work_items_per_element)

    # {{{ metadata

    if REWRITE_DIFF_EINSUM:
        iface_dof = {"flux": "_pt_sum_r1",
                     "flux_0": "_pt_sum_r1_0",
                     "flux_1": "_pt_sum_r1_1",
                     "flux_2": "_pt_sum_r1_2"}
    else:
        iface_dof = {"flux": "_pt_sum_r1_0",
                     "flux_0": "_pt_sum_r1_2",
                     "flux_1": "_pt_sum_r1_6",
                     "flux_2": "_pt_sum_r1_8"}

    accumulators = sorted({tv.name for tv in knl.temporary_variables.values()
                           if (tv.name.startswith("acc__")
                               and (list(knl
                                         .writer_map()[tv.name])[0]
                                    .startswith("face_mass")))})
    # one accumulator for each diff
    assert len(accumulators) == 4

    # }}}

    # Preftch flux values into local address space
    for dof_var_name in ["flux", "flux_0",
                         "flux_1", "flux_2"]:
        new_insn_id = f"{dof_var_name}_prftch"
        knl = add_prefetch_for_single_kernel(
                    knl, t_unit.callables_table,
                    var_name=dof_var_name,
                    sweep_inames=["iel_face_mass_inner",
                                  iface_dof[dof_var_name]],
                    fetch_outer_inames=frozenset({"iel_face_mass_outer",
                                                  "iface_face_mass"}),
                    temporary_address_space=lp.AddressSpace.LOCAL,
                    prefetch_insn_id=new_insn_id,
                    dim_arg_names=[f"iprftch_{dof_var_name}_0",
                                   f"iprftch_{dof_var_name}_1"],
                    default_tag=None)
        knl = lp.join_inames(knl, (f"iprftch_{dof_var_name}_1",
                                   f"{dof_var_name}_dim_2"),
                            f"iprftch_{dof_var_name}")

        knl = lp.split_iname(
            lp.split_iname(knl, f"iprftch_{dof_var_name}",
                           n_work_items_per_element * n_elements_per_wg),
            f"iprftch_{dof_var_name}_inner",
            n_work_items_per_element,
            inner_tag="l.0", outer_tag="l.1")

        # new_insns = [insn.copy(depends_on=frozenset({"g_barrier_2",
        #                                              f"{dof_var_name}_store"}))
        #              if insn.id == new_insn_id else insn
        #              for insn in knl.instructions]
        # knl = knl.copy(instructions=new_insns)

    # {{{ inside each iface_face_mass prefetch the reference lifting matrix

    knl = add_prefetch_for_single_kernel(
                knl, t_unit.callables_table,
                var_name="_pt_in_12",
                sweep_inames=(["idof_face_mass_inner",
                               "idof_face_mass_outer"]
                              + sorted(iface_dof.values())),
                fetch_outer_inames=frozenset({"iel_face_mass_outer",
                                              "iface_face_mass"}),
                temporary_address_space=lp.AddressSpace.LOCAL,
                dim_arg_names=["iprftch_ref_lift_0",
                               "iprftch_ref_lift_1"],
                default_tag=None)
    knl = lp.join_inames(knl, ("iprftch_ref_lift_0", "_pt_in_12_dim_2"),
                        "iprftch_ref_lift")
    knl = lp.split_iname(
        lp.split_iname(knl, "iprftch_ref_lift",
                       n_work_items_per_element * n_elements_per_wg),
        "iprftch_ref_lift_inner",
        n_work_items_per_element,
        inner_tag="l.0", outer_tag="l.1")

    # }}}

    # {{{ inside each iface_face_mass, prefetch the jacobian to shared

    # knl = add_prefetch_for_single_kernel(
    #             knl, t_unit.callables_table,
    #             var_name="_pt_in_13",
    #             sweep_inames=["iel_face_mass_inner"],
    #             fetch_outer_inames=frozenset({"iel_face_mass_outer",
    #                                           "idof_face_mass_inner",
    #                                           "iface_face_mass"}),
    #             temporary_address_space=lp.AddressSpace.PRIVATE,
    #             prefetch_insn_id="jac_prftch",
    #             default_tag=None)

    # }}}

    # {{{ make iface the outermost loop in the reductions

    knl = lp.privatize_temporaries_with_inames(
            knl, "idof_face_mass_outer",
            only_var_names=accumulators)
    knl = lp.duplicate_inames(
        knl,
        inames=("idof_face_mass_outer",),
        new_inames=["idof_face_mass_outer_init"],
        within=("("
                + " or ".join(f"writes:{acc}"
                              for acc in accumulators)
               + ")"
               + " and not iname:iface_face_mass"))
    knl = lp.duplicate_inames(knl,
                              inames=("idof_face_mass_outer",),
                              new_inames=["idof_face_mass_outer_update"],
                              within="writes:face_mass*")

    # }}}

    knl = lp.tag_inames(knl, {"iel_face_mass_inner": "l.1",
                              "idof_face_mass_inner": "l.0"})

    for redn_iname in iface_dof.values():
        knl = lp.tag_inames(knl, {redn_iname: "unr"})

    t_unit = t_unit.with_kernel(knl)
    # print(lp.generate_code_v2(t_unit).device_code())

    return t_unit


def transform_diff(t_unit):
    import loopy as lp
    from loopy.transform.data import add_prefetch_for_single_kernel

    knl = t_unit.default_entrypoint

    # {{{ metadata

    ref_diff_mat = "_pt_in_2"

    jac_det = "_pt_in"
    jacobians = ["_pt_in_3", "_pt_in_4", "_pt_in_5",
                 "_pt_in_6", "_pt_in_7", "_pt_in_8",
                 "_pt_in_9", "_pt_in_10", "_pt_in_11"]

    accumulators = sorted({tv.name for tv in knl.temporary_variables.values()
                           if (tv.name.startswith("acc__")
                               and (list(knl
                                         .writer_map()[tv.name])[0]
                                    .startswith("mass_inv")))})
    # one accumulator for each diff
    assert len(accumulators) == 6

    # }}}

    n_elements_per_wg = 8
    n_work_items_per_element = 16

    # {{{ have a single ref coord axis-loop that goes over all the divs

    # grad
    knl = lp.rename_iname(knl, "_pt_sum_r0_0", "irefcord", existing_ok=True)
    knl = lp.rename_iname(knl, "_pt_sum_r0_3", "irefcord", existing_ok=True)
    knl = lp.rename_iname(knl, "_pt_sum_r0_11", "irefcord", existing_ok=True)

    # div
    knl = lp.rename_iname(knl, "_pt_sum_r0_6", "irefcord", existing_ok=True)
    knl = lp.rename_iname(knl, "_pt_sum_r0_7", "irefcord", existing_ok=True)
    knl = lp.rename_iname(knl, "_pt_sum_r0_8", "irefcord", existing_ok=True)

    # }}}

    # {{{ fuse the grad inner dof reduction loop

    knl = lp.rename_iname(knl, "_pt_sum_r1",  "igrad_dof_redn", existing_ok=True)
    knl = lp.rename_iname(knl, "_pt_sum_r1_1",  "igrad_dof_redn", existing_ok=True)
    knl = lp.rename_iname(knl, "_pt_sum_r1_7",  "igrad_dof_redn", existing_ok=True)

    knl = lp.rename_iname(knl, "_pt_sum_r1_3", "idiv_dof_redn0", existing_ok=True)
    knl = lp.rename_iname(knl, "_pt_sum_r1_4", "idiv_dof_redn1", existing_ok=True)
    knl = lp.rename_iname(knl, "_pt_sum_r1_5", "idiv_dof_redn2", existing_ok=True)

    # }}}

    knl = lp.split_iname(knl, "iel_diff", n_elements_per_wg,
                         outer_tag="g.0")
    knl = lp.split_iname(knl, "idof_diff",
                         n_work_items_per_element)

    # {{{ prefetch the jacobians into private memory

    # each work-items needs only one (double prec.) register for storing each of
    # these jacobians

    for jac_var_name in [jac_det] + jacobians:
        new_insn_id = f"{jac_var_name}_prftch_insn"
        knl = add_prefetch_for_single_kernel(
                    knl, t_unit.callables_table,
                    var_name=jac_var_name,
                    sweep_inames=[],
                    fetch_outer_inames=frozenset({"iel_diff_outer",
                                                  "iel_diff_inner",
                                                  "idof_diff_inner"}),
                    prefetch_insn_id=new_insn_id,
                    temporary_address_space=lp.AddressSpace.PRIVATE,
                    within="iname:iel_diff_outer",
                    default_tag=None)
        knl = lp.add_dependency(knl, f"id:{new_insn_id}", "g_barrier_3")

    # }}}

    # {{{ prefetch ref_diff_mat within ref coord

    knl = add_prefetch_for_single_kernel(
                knl, t_unit.callables_table,
                var_name=ref_diff_mat,
                sweep_inames=(["idof_diff_inner",
                               "idof_diff_outer",
                               "igrad_dof_redn",
                               "idiv_dof_redn0",
                               "idiv_dof_redn1",
                               "idiv_dof_redn2"]),
                fetch_outer_inames=frozenset({"iel_diff_outer",
                                              "irefcord"}),
                temporary_address_space=lp.AddressSpace.LOCAL,
                dim_arg_names=["iprftch_ref_diff_0",
                               "iprftch_ref_diff_1"],
                default_tag=None)
    knl = lp.join_inames(knl, ("iprftch_ref_diff_1", f"{ref_diff_mat}_dim_2"),
                        "iprftch_ref_diff")
    knl = lp.split_iname(
        lp.split_iname(knl, "iprftch_ref_diff",
                       n_work_items_per_element * n_elements_per_wg),
        "iprftch_ref_diff_inner",
        n_work_items_per_element,
        inner_tag="l.0", outer_tag="l.1")

    # }}}

    #  {{{ prefetch the inputs

    # Not sure if we want to do this yet.

    # }}}

    # {{{ privatize temporaries to pull "iref_cord" as the outer loop

    knl = lp.privatize_temporaries_with_inames(knl, "idof_diff_outer",
                                               only_var_names=accumulators)

    knl = lp.duplicate_inames(knl,
                              inames=["idof_diff_outer"],
                              new_inames=["idof_diff_outer_init"],
                              within=("("
                                      + " or ".join(f"writes:{acc}"
                                                    for acc in accumulators)
                                     + ")"
                                     + " and not iname:irefcord"))
    knl = lp.duplicate_inames(knl,
                              inames=["idof_diff_outer"],
                              new_inames=["idof_diff_outer_store"],
                              within="writes:mass_inv*")

    # }}}

    knl = lp.tag_inames(knl, {"iel_diff_inner": "l.1",
                              "idof_diff_inner": "l.0"})

    t_unit = t_unit.with_kernel(knl)
    return t_unit


class HopefullySmartPytatoArrayContext(
        SingleGridWorkBalancingPytatoArrayContext):

    DO_CSE = True

    def transform_dag(self, dag):
        import pytato as pt

        # {{{ CSE

        if self.DO_CSE:
            dag = pt.transform.materialize_with_mpms(dag)

            def mark_materialized_nodes_as_cse(
                        ary: Union[pt.Array,
                                   pt.AbstractResultWithNamedArrays]) -> pt.Array:
                if isinstance(ary, pt.AbstractResultWithNamedArrays):
                    return ary

                if ary.tags_of_type(pt.tags.ImplStored):
                    return ary.tagged(pt.tags.PrefixNamed("cse"))
                else:
                    return ary

            dag = pt.transform.map_and_copy(dag, mark_materialized_nodes_as_cse)

        # }}}

        # {{{ collapse data wrappers

        data_wrapper_cache = {}

        def cached_data_wrapper_if_present(ary):
            if isinstance(ary, pt.DataWrapper):
                cache_key = (ary.data.data.int_ptr, ary.data.offset,
                             ary.shape, ary.data.strides)
                try:
                    result = data_wrapper_cache[cache_key]
                except KeyError:
                    result = ary
                    data_wrapper_cache[cache_key] = result

                return result
            else:
                return ary

        dag = pt.transform.map_and_copy(dag, cached_data_wrapper_if_present)

        # }}}

        # {{{ get rid of copies for different views of a cl-array

        def eliminate_reshapes_of_data_wrappers(ary):
            if (isinstance(ary, pt.Reshape)
                    and isinstance(ary.array, pt.DataWrapper)):
                return (pt.make_data_wrapper(ary.array.data.reshape(ary.shape))
                        .tagged(ary.tags))
            else:
                return ary

        dag = pt.transform.map_and_copy(dag,
                                        eliminate_reshapes_of_data_wrappers)

        # }}}

        # {{{ face_mass: materialize einsum args

        def materialize_face_mass_input_and_output(expr):
            if isinstance(expr, pt.Einsum):
                my_tag, = expr.tags_of_type(pt.tags.EinsumInfo)
                if my_tag.spec == "ifj,fej,fej->ei":
                    mat, jac, vec = expr.args
                    return (pt.einsum("ifj,fej,fej->ei",
                                      mat,
                                      jac,
                                      vec.tagged((pt.tags.ImplStored(),
                                                  pt.tags.PrefixNamed("flux"))))
                            .tagged((pt.tags.ImplStored(),
                                     pt.tags.PrefixNamed("face_mass"))))
                else:
                    return expr
            else:
                return expr

        dag = pt.transform.map_and_copy(dag, materialize_face_mass_input_and_output)

        # }}}

        # {{{ materialize inverse mass inputs

        def materialize_inverse_mass_inputs(expr):
            if isinstance(expr, pt.Einsum):
                my_tag, = expr.tags_of_type(pt.tags.EinsumInfo)
                if my_tag.spec == "ei,ij,ej->ei":
                    arg1, arg2, arg3 = expr.args
                    return pt.einsum(my_tag.spec,
                                     arg1,
                                     arg2,
                                     arg3.tagged(
                                         (pt.tags.ImplStored(),
                                          pt.tags.PrefixNamed("mass_inv_inp"))))
                else:
                    return expr
            else:
                return expr

        dag = pt.transform.map_and_copy(dag, materialize_inverse_mass_inputs)

        # }}}

        # {{{ rewrite einsum node of divs

        def rewrite_einsum_exprs(expr):
            if isinstance(expr, pt.Einsum):
                my_tag, = expr.tags_of_type(pt.tags.EinsumInfo)
                if my_tag.spec == "dij,ej,ej,dej->ei":
                    arg0, arg1, arg2, arg3 = expr.args
                    return (pt.einsum("ej,ej,eij->ei", arg1, arg2,
                                      pt.einsum("dij,dej->eij", arg0, arg3))
                            .tagged((tag
                                     for tag in expr.tags if tag != my_tag)))
                else:
                    return expr
            else:
                return expr

        if REWRITE_DIFF_EINSUM:
            # early scheduling of loops is bad
            dag = pt.transform.map_and_copy(dag, rewrite_einsum_exprs)

        # }}}

        # {{{ infer axis types

        from grudge.pytato.utils import (unify_discretization_entity_tags,
                                         are_all_stored_arrays_inferred)

        dag = unify_discretization_entity_tags(dag)
        are_all_stored_arrays_inferred(dag)

        # }}}

        # {{{ /!\ Remove tags from Loopy call results.
        # See <https://www.github.com/inducer/pytato/issues/195>

        def untag_loopy_call_results(expr):
            from pytato.loopy import LoopyCallResult
            if isinstance(expr, LoopyCallResult):
                return expr.copy(tags=frozenset(),
                                 axes=(pt.Axis(frozenset()),)*expr.ndim)
            else:
                return expr

        dag = pt.transform.map_and_copy(dag, untag_loopy_call_results)

        # }}}

        # {{{ make the discretization entity tags loopy iname tags

        def make_discretization_tags_iname_tags(expr):
            if isinstance(expr, pt.Array):
                for iaxis, axis in enumerate(expr.axes):
                    for tag in axis.tags_of_type(DiscretizationEntityAxisTag):
                        expr = expr.with_tagged_axis(iaxis,
                                                     pt.tags.AxisToLoopyInameTag(tag)
                                                     )

            return expr

        dag = pt.transform.map_and_copy(dag, make_discretization_tags_iname_tags)

        # }}}

        return dag

    def transform_loopy_program(self, t_unit):
        from arraycontext.impl.pytato.compile import FromArrayContextCompile
        if t_unit.default_entrypoint.tags_of_type(FromArrayContextCompile):
            import loopy as lp
            from loopy.transform.precompute import precompute_for_single_kernel
            from loopy.transform.instruction import simplify_indices

            t_unit = simplify_indices(t_unit)
            knl = t_unit.default_entrypoint

            # {{{ loop fusion

            for mesh_entity, fused_loop_name in [
                    (DiscretizationFaceAxisTag, "iface"),
                    (DiscretizationElementAxisTag, "iel"),
                    (DiscretizationDOFAxisTag, "idof")]:
                taggedo = lp.relations.get_taggedo_of_type(knl, mesh_entity)

                # tag_k: tag of type 'mesh_entity'
                tag_k = kanren.var()
                tags = kanren.run(0,
                                  tag_k,
                                  taggedo(kanren.var(), tag_k),
                                  results_filter=frozenset)
                for tag in tags:
                    # iname_k: iname tagged with 'tag'
                    iname_k = kanren.var()
                    inames = kanren.run(0,
                                        iname_k,
                                        taggedo(iname_k, tag),
                                        results_filter=frozenset)

                    knl = lp.rename_inames_in_batch(
                        knl,
                        lp.get_kennedy_unweighted_fusion_candidates(
                            knl, inames, prefix=fused_loop_name))

            # }}}

            # {{{ array contraction

            tempo = lp.relations.get_tempo(knl)
            producero = lp.relations.get_producero(knl)
            consumero = lp.relations.get_consumero(knl)
            withino = lp.relations.get_withino(knl)
            reduce_insno = lp.relations.get_reduce_insno(knl)

            all_temps = frozenset(knl.temporary_variables)

            # temp_k: temporary variable that cannot be contracted
            temp_k = kanren.var()
            producer_insn_k = kanren.var()
            consumer_insn_k = kanren.var()
            producer_loops_k = kanren.var()
            consumer_loops_k = kanren.var()
            temps_not_to_contract = kanren.run(0,
                                               temp_k,
                                               tempo(temp_k),
                                               producero(producer_insn_k, temp_k),
                                               consumero(consumer_insn_k, temp_k),
                                               withino(producer_insn_k,
                                                       producer_loops_k),
                                               withino(consumer_insn_k,
                                                       consumer_loops_k),
                                               kanren.lany(
                                                   kanren_neq(producer_loops_k,
                                                              consumer_loops_k),
                                                   reduce_insno(producer_insn_k),
                                                   reduce_insno(consumer_insn_k)),
                                               results_filter=frozenset)

            for temp in sorted(all_temps - frozenset(temps_not_to_contract)):
                knl = lp.assignment_to_subst(knl, temp)
                knl = precompute_for_single_kernel(
                    knl, t_unit.callables_table, f"{temp}_subst",
                    sweep_inames=(),
                    temporary_address_space=lp.AddressSpace.PRIVATE)

            # }}}

            # {{{ Stats

            if 0:
                from loopy.kernel.array import ArrayBase
                from pytools import product
                t_unit = t_unit.with_kernel(knl)

                op_map = lp.get_op_map(t_unit,
                                       subgroup_size=32)
                f64_ops = op_map.filter_by(dtype=[np.float64],
                                           kernel_name="_pt_kernel").eval_and_sum({})

                # {{{ footprint gathering

                nfootprint_bytes = 0

                for ary in knl.args:
                    if (isinstance(ary, ArrayBase)
                            and ary.address_space == lp.AddressSpace.GLOBAL):
                        nfootprint_bytes += (product(ary.shape)
                                             * ary.dtype.itemsize)

                for ary in knl.temporary_variables.values():
                    if ary.address_space == lp.AddressSpace.GLOBAL:
                        # global temps would be written once and read once
                        nfootprint_bytes += (2 * product(ary.shape)
                                             * ary.dtype.itemsize)

                # }}}

                print("Double-prec. GFlOps:", f64_ops * 1e-9)
                print("Footprint GBs:",  nfootprint_bytes * 1e-9)
                1/0

            # }}}

            # add the 6 gbarriers
            knl = lp.add_barrier(knl,
                                 insn_before="tag:cse_knl1",
                                 insn_after="tag:cse_knl2")
            knl = lp.add_barrier(knl,
                                 insn_before="tag:cse_knl2",
                                 insn_after="tag:cse_knl3")
            knl = lp.add_barrier(knl,
                                 insn_before="tag:cse_knl3",
                                 insn_after="writes:flux*")
            knl = lp.add_barrier(knl,
                                 insn_before="writes:flux*",
                                 insn_after="writes:face_mass*")
            knl = lp.add_barrier(knl,
                                 insn_before="writes:face_mass*",
                                 insn_after="writes:mass_inv_inp*")
            knl = lp.add_barrier(knl,
                                 insn_before="writes:mass_inv_inp*",
                                 insn_after="writes:_pt_out*")

            # {{{ fuse elementwise loops

            # Since u, v_0, v_1, v_2 all correspond to the same function space
            # we fuse the loop nests corresponding to v_0, v_1, v_2 into the
            # semantic equivalent loop of 'u'.

            # idof == 0, implies v_0
            # idof == 1, implies v_1
            # idof == 2, implies v_2

            # fuse all restriction loops
            for idof in range(3):
                # face loop
                knl = lp.rename_iname(knl,
                                      f"flux_{idof}_dim0",
                                      "flux_dim0",
                                      existing_ok=True)

                # element loop
                knl = lp.rename_iname(knl,
                                      f"flux_{idof}_dim1",
                                      "flux_dim1",
                                      existing_ok=True)

                # face loop
                knl = lp.rename_iname(knl,
                                      f"flux_{idof}_dim2",
                                      "flux_dim2",
                                      existing_ok=True)
            knl = lp.rename_iname(knl, "flux_dim0", "iface_flux")
            knl = lp.rename_iname(knl, "flux_dim1", "iel_flux")
            knl = lp.rename_iname(knl, "flux_dim2", "iface_dof_flux")

            # fuse all face-mass loops
            knl = lp.rename_iname(knl, "face_mass_dim0", "iel_face_mass")
            knl = lp.rename_iname(knl, "face_mass_dim1", "idof_face_mass")

            for idof in range(3):
                # element loop
                knl = lp.rename_iname(knl,
                                      f"face_mass_{idof}_dim0",
                                      "iel_face_mass",
                                      existing_ok=True)

                # vol. dof
                knl = lp.rename_iname(knl,
                                      f"face_mass_{idof}_dim1",
                                      "idof_face_mass",
                                      existing_ok=True)

            # fuse all div/grad + lift terms
            knl = lp.rename_iname(knl,
                                  "mass_inv_inp_dim0",
                                  "iel_diff",
                                  existing_ok=True)

            # vol. dof
            knl = lp.rename_iname(knl,
                                  "mass_inv_inp_dim1",
                                  "idof_diff",
                                  existing_ok=True)
            for idof in range(3):
                # element loop
                knl = lp.rename_iname(knl,
                                      f"mass_inv_inp_{idof}_dim0",
                                      "iel_diff",
                                      existing_ok=True)

                # vol. dof
                knl = lp.rename_iname(knl,
                                      f"mass_inv_inp_{idof}_dim1",
                                      "idof_diff",
                                      existing_ok=True)

            # fuse all final output loops (result of inverse mass)
            for idof in range(4):
                # element loop
                knl = lp.rename_iname(knl,
                                      f"_pt_out_{idof}_0_dim0",
                                      "iel_out",
                                      existing_ok=True)

                # vol. dof
                knl = lp.rename_iname(knl,
                                      f"_pt_out_{idof}_0_dim1",
                                      "idof_out",
                                      existing_ok=True)

            t_unit = t_unit.with_kernel(knl)

            # }}}

            # {{{ fuse reduction loops

            # we aren't going to parallelize the reductions, realize them so
            # that we can do apply other loop transformations to them.
            t_unit = lp.realize_reduction(t_unit)
            knl = t_unit.default_entrypoint

            # }}}

            l_one_size = 4
            l_zero_size = 16

            # {{{ elementwise CSE kernels: elementwise parallelization

            for i in range(3):
                knl = lp.split_iname(knl, f"iel_cse_{i}", l_one_size,
                                     inner_tag="l.1", outer_tag="g.0")
                knl = lp.split_iname(knl, f"idof_cse_{i}", l_zero_size,
                                     inner_tag="l.0")

            # }}}

            # {{{ elementwise compute at face: elementwise parallelization

            knl = lp.split_iname(knl, "iel_flux", l_one_size,
                                 inner_tag="l.1", outer_tag="g.0")
            knl = lp.split_iname(knl, "iface_dof_flux", l_zero_size,
                                 inner_tag="l.0")

            # }}}

            # {{{ face mass transformations

            # Have a single loop over the faces during face_mass.
            if REWRITE_DIFF_EINSUM:
                knl = lp.rename_iname(knl, "_pt_sum_r0_2", "iface_face_mass")
                knl = lp.rename_iname(knl, "_pt_sum_r0_6", "iface_face_mass",
                                      existing_ok=True)
                knl = lp.rename_iname(knl, "_pt_sum_r0_12", "iface_face_mass",
                                      existing_ok=True)
                knl = lp.rename_iname(knl, "_pt_sum_r0_15", "iface_face_mass",
                                      existing_ok=True)
            else:
                knl = lp.rename_iname(knl, "_pt_sum_r0_1", "iface_face_mass")
                knl = lp.rename_iname(knl, "_pt_sum_r0_4", "iface_face_mass",
                                      existing_ok=True)
                knl = lp.rename_iname(knl, "_pt_sum_r0_9", "iface_face_mass",
                                      existing_ok=True)
                knl = lp.rename_iname(knl, "_pt_sum_r0_12", "iface_face_mass",
                                      existing_ok=True)

            if 1:
                t_unit = t_unit.with_kernel(knl)
                t_unit = transform_face_mass(t_unit)
                knl = t_unit.default_entrypoint
            else:
                knl = lp.split_iname(knl, "iel_face_mass", l_one_size,
                                     inner_tag="l.1", outer_tag="g.0")
                knl = lp.split_iname(knl, "idof_face_mass", l_zero_size,
                                     inner_tag="l.0")

            # }}}

            # {{{ diff transformation

            if not REWRITE_DIFF_EINSUM:
                t_unit = t_unit.with_kernel(knl)
                t_unit = transform_diff(t_unit)
                knl = t_unit.default_entrypoint
            else:
                knl = lp.split_iname(knl, "iel_diff", l_one_size,
                                     inner_tag="l.1", outer_tag="g.0")
                knl = lp.split_iname(knl, "idof_diff", l_zero_size,
                                     inner_tag="l.0")

            # }}}

            # {{{ einsums

            knl = lp.split_iname(knl, "iel_out", l_one_size,
                                 inner_tag="l.1", outer_tag="g.0")
            knl = lp.split_iname(knl, "idof_out", l_zero_size,
                                 inner_tag="l.0")

            # }}}

            t_unit = t_unit.with_kernel(knl)

            return t_unit
        else:
            return super().transform_loopy_program(t_unit)


def rk4_step(y, t, h, f):
    k1 = f(t, y)
    k2 = f(t+h/2, y + h/2*k1)
    k3 = f(t+h/2, y + h/2*k2)
    k4 = f(t+h, y + h*k3)
    return y + h/6*(k1 + 2*k2 + 2*k3 + k4)


def main(ctx_factory, dim=2, order=4, visualize=False,
         actx_class=PyOpenCLArrayContext):
    cl_ctx = ctx_factory()
    queue = cl.CommandQueue(cl_ctx)
    actx = actx_class(
        queue,
        allocator=cl_tools.MemoryPool(cl_tools.ImmediateAllocator(queue)),
    )

    from meshmode.mesh.generation import generate_regular_rect_mesh
    mesh = generate_regular_rect_mesh(
            a=(-0.5,)*dim,
            b=(0.5,)*dim,
            nelements_per_axis=(20,)*dim)

    dcoll = DiscretizationCollection(actx, mesh, order=order)

    def source_f(actx, dcoll, t=0):
        source_center = np.array([0.1, 0.22, 0.33])[:dcoll.dim]
        source_width = 0.05
        source_omega = 3
        nodes = thaw(dcoll.nodes(), actx)
        source_center_dist = flat_obj_array(
            [nodes[i] - source_center[i] for i in range(dcoll.dim)]
        )
        return (
            actx.np.sin(source_omega*t)
            * actx.np.exp(
                -np.dot(source_center_dist, source_center_dist)
                / source_width**2
            )
        )

    x = thaw(dcoll.nodes(), actx)
    ones = dcoll.zeros(actx) + 1
    c = actx.np.where(actx.np.less(np.dot(x, x), 0.15), 0.1 * ones, 0.2 * ones)

    from grudge.models.wave import VariableCoefficientWeakWaveOperator
    from meshmode.mesh import BTAG_ALL, BTAG_NONE

    wave_op = VariableCoefficientWeakWaveOperator(
        dcoll,
        c,
        source_f=source_f,
        dirichlet_tag=BTAG_NONE,
        neumann_tag=BTAG_NONE,
        radiation_tag=BTAG_ALL,
        flux_type="upwind"
    )

    fields = flat_obj_array(
        dcoll.zeros(actx),
        [dcoll.zeros(actx) for i in range(dcoll.dim)]
    )

    wave_op.check_bc_coverage(mesh)

    def rhs(t, w):
        return wave_op.operator(t, w)

    dt = 1/3 * wave_op.estimate_rk4_timestep(actx, dcoll, fields=fields)

    final_t = 1
    nsteps = int(final_t/dt) + 1

    logger.info(f"{mesh.nelements} elements, dt={dt}, nsteps={nsteps}")

    from grudge.shortcuts import make_visualizer
    vis = make_visualizer(dcoll)

    t = 0.
    step = 1

    def norm(u):
        return op.norm(dcoll, u, 2)

    t_last_step = time()

    if visualize:
        u = fields[0]
        v = fields[1:]
        vis.write_vtk_file(
            f"fld-var-propogation-speed-{step:04d}.vtu",
            [
                ("u", u),
                ("v", v),
                ("c", c),
            ]
        )

    compiled_rhs = actx.compile(rhs)

    while t < final_t:
        # thaw+freeze to see similar expression graphs in rk4
        fields = thaw(freeze(fields, actx), actx)

        fields = rk4_step(fields, t, dt, compiled_rhs)

        if step % 10 == 0:
            actx.queue.finish()
            logger.info(f"step: {step} t: {time()-t_last_step} secs. "
                        f"L2: {actx.to_numpy(norm(u=fields[0]))}")

            if visualize:
                vis.write_vtk_file(
                    f"fld-var-propogation-speed-{step:04d}.vtu",
                    [
                        ("u", fields[0]),
                        ("v", fields[1:]),
                        ("c", c),
                    ]
                )

            assert actx.to_numpy(norm(u=fields[0])) < 1
            t_last_step = time()

        t += dt
        step += 1


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dim", default=3, type=int)
    parser.add_argument("--order", default=4, type=int)
    parser.add_argument("--visualize", action="store_true", default=False)
    parser.add_argument("--dumblazy", action="store_true", default=False)
    parser.add_argument("--hopefullysmartlazy", action="store_true", default=False)
    args = parser.parse_args()

    assert not (args.dumblazy and args.hopefullysmartlazy)

    if args.dumblazy:
        actx_class = SingleGridWorkBalancingPytatoArrayContext
    elif args.hopefullysmartlazy:
        actx_class = HopefullySmartPytatoArrayContext
    else:
        actx_class = PyOpenCLArrayContext

    logging.basicConfig(level=logging.INFO)
    main(cl.create_some_context,
         dim=args.dim,
         order=args.order,
         visualize=args.visualize,
         actx_class=actx_class)

# vim: fdm=marker
