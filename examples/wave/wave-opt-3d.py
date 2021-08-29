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
from arraycontext.impl.pytato.compile import FromActxCompile

from time import time
from grudge import DiscretizationCollection

from pytools.obj_array import flat_obj_array
from loopy.symbolic import (get_dependencies, CombineMapper)

import grudge.op as op

import logging
logger = logging.getLogger(__name__)


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
    import loopy as lp
    from loopy.transform.data import add_prefetch_for_single_kernel
    knl = t_unit.default_entrypoint

    n_elements_per_wg = 4
    n_work_items_per_element = 8

    knl = lp.split_iname(knl, "iel_face_mass", n_elements_per_wg,
                         outer_tag="g.0")
    knl = lp.split_iname(knl, "idof_face_mass",
                         n_work_items_per_element)

    # Preftch rstrct vals into private address space
    for dof_var_name in ["rstrct_vals", "rstrct_vals_0",
                         "rstrct_vals_1", "rstrct_vals_2"]:
        knl = add_prefetch_for_single_kernel(
                    knl, t_unit.callables_table,
                    var_name=dof_var_name,
                    sweep_inames=["face_jdof"],
                    fetch_outer_inames=frozenset({"iel_face_mass_outer",
                                                  "iel_face_mass_inner",
                                                  "idof_face_mass_inner",
                                                  "face_f"}),
                    temporary_address_space=lp.AddressSpace.PRIVATE,
                    default_tag=None)

    # {{{ inside each face_f prefetch the reference lifting matrix

    knl = add_prefetch_for_single_kernel(
                knl, t_unit.callables_table,
                var_name="_pt_in_13",
                sweep_inames=["idof_face_mass_inner",
                              "idof_face_mass_outer",
                              "face_jdof"],
                fetch_outer_inames=frozenset({"iel_face_mass_outer",
                                              "face_f"}),
                temporary_address_space=lp.AddressSpace.LOCAL,
                dim_arg_names=["iprftch_ref_mat_0",
                               "iprftch_ref_mat_1"],
                default_tag=None)
    knl = lp.join_inames(knl, ("iprftch_ref_mat_0", "_pt_in_13_dim_2"),
                        "iprftch_ref_mat")
    knl = lp.split_iname(
        lp.split_iname(knl, "iprftch_ref_mat",
                       n_work_items_per_element * n_elements_per_wg),
        "iprftch_ref_mat_inner",
        n_work_items_per_element,
        inner_tag="l.0", outer_tag="l.1")

    # }}}

    # {{{ inside each face_f, prefetch the jacobian to shared

    knl = add_prefetch_for_single_kernel(
                knl, t_unit.callables_table,
                var_name="_pt_in_12",
                sweep_inames=["iel_face_mass_inner",
                              "face_jdof"],
                fetch_outer_inames=frozenset({"iel_face_mass_outer", "face_f"}),
                temporary_address_space=lp.AddressSpace.LOCAL,
                dim_arg_names=["iprftch_jac_0",
                               "iprftch_jac_1"],
                default_tag=None)

    knl = lp.join_inames(knl, ("iprftch_jac_1", "_pt_in_12_dim_2"),
                        "iprftch_jac")
    knl = lp.split_iname(
        lp.split_iname(knl, "iprftch_jac",
                       n_work_items_per_element * n_elements_per_wg),
        "iprftch_jac_inner",
        n_work_items_per_element,
        inner_tag="l.0", outer_tag="l.1")

    # }}}

    # {{{

    knl = lp.privatize_temporaries_with_inames(knl, "idof_face_mass_outer",
                                               only_var_names={"acc_face_f",
                                                               "acc_face_f_0",
                                                               "acc_face_f_1",
                                                               "acc_face_f_2"})
    knl = lp.duplicate_inames(knl,
                              inames=("idof_face_mass_outer",),
                              new_inames=["idof_face_mass_outer_init"],
                              within=("id:face_insn_face_f_init"
                                      " or id:face_insn_0_face_f_0_init"
                                      " or id:face_insn_1_face_f_1_init"
                                      " or id:face_insn_2_face_f_2_init"))
    knl = lp.duplicate_inames(knl,
                              inames=("idof_face_mass_outer",),
                              new_inames=["idof_face_mass_outer_update"],
                              within=("id:face_insn"
                                      " or id:face_insn_0"
                                      " or id:face_insn_1"
                                      " or id:face_insn_2"))

    # }}}

    knl = lp.tag_inames(knl, {"iel_face_mass_inner": "l.1",
                              "idof_face_mass_inner": "l.0"})

    t_unit = t_unit.with_kernel(knl)
    # print(lp.generate_code_v2(t_unit).device_code())

    return t_unit


class HopefullySmartPytatoArrayContext(
        SingleGridWorkBalancingPytatoArrayContext):

    DO_CSE = True

    def transform_dag(self, dag):
        import pytato as pt

        # {{{ CSE

        if self.DO_CSE:
            nusers = pt.analysis.get_nusers(dag)

            def materialize(ary: pt.Array) -> pt.Array:
                if ((not isinstance(ary, (pt.InputArgumentBase, pt.NamedArray)))
                        and nusers[ary] > 1):
                    return ary.tagged(pt.tags.ImplementAs(pt.tags.ImplStored("cse")))

                return ary

            dag = pt.transform.map_and_copy(dag, materialize)

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

        # {{{ materialize inverse mass inputs

        def materialize_inverse_mass_inputs(expr):
            if isinstance(expr, pt.Einsum):
                my_tag, = expr.tags_of_type(pt.tags.EinsumInfo)
                if my_tag.spec == "ei,ij,ej->ei":
                    arg1, arg2, arg3 = expr.args
                    return pt.einsum(my_tag.spec,
                                     arg1,
                                     arg2,
                                     arg3.tagged(pt.tags.ImplementAs(
                                         pt.tags.ImplStored("mass_inv_inp"))))
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

        dag = pt.transform.map_and_copy(dag, rewrite_einsum_exprs)

        # }}}

        return dag

    def transform_loopy_program(self, t_unit):
        if t_unit.default_entrypoint.tags_of_type(FromActxCompile):
            import loopy as lp
            from loopy.transform.precompute import precompute_for_single_kernel
            from loopy.transform.instruction import simplify_indices

            t_unit = lp.inline_callable_kernel(t_unit, "face_mass")
            t_unit = simplify_indices(t_unit)

            knl = t_unit.default_entrypoint

            # get rid of noops
            noops = {insn.id for insn in knl.instructions
                     if isinstance(insn, lp.NoOpInstruction)}
            knl = lp.remove_instructions(knl, noops)

            # {{{ CSE kernels

            cse_kernel1 = {"cse", "cse_0", "cse_6", "cse_8", "cse_4"}
            cse_kernel2 = {"cse_1", "cse_2", "cse_3", "cse_5", "cse_7", "cse_9",
                           "cse_10", "cse_11", "cse_31", "cse_33", "cse_35"}
            cse_kernel3 = {"cse_12", "cse_13", "cse_14", "cse_15", "cse_16",
                           "cse_17", "cse_18", "cse_19", "cse_20", "cse_21",
                           "cse_22", "cse_23", "cse_24", "cse_25", "cse_26",
                           "cse_27", "cse_28", "cse_29", "cse_30", "cse_32",
                           "cse_34", "cse_36"}

            # Why a list and not a 'set': Loopy's model of substitutions not
            # having dependencies forces it to add fake dependencies to
            # instructions. So, in short "order matters".

            assert len(cse_kernel1 & cse_kernel2) == 0
            assert len(cse_kernel3 & cse_kernel2) == 0
            assert len(cse_kernel3 & cse_kernel1) == 0
            assert len(cse_kernel1 | cse_kernel2 | cse_kernel3) == 38
            assert ((cse_kernel1 | cse_kernel2 | cse_kernel3)
                    < set(knl.temporary_variables))

            priv_cse_kernel2_vars = ["cse_3", "cse_10",
                                     "cse_1", "cse_2", "cse_7", "cse_9", "cse_5"]

            priv_cse_kernel3_vars = ["cse_24", "cse_29",
                                     "cse_26", "cse_27", "cse_28", "cse_13",
                                     "cse_23", "cse_25",
                                     "cse_12", "cse_14", "cse_22",
                                     "cse_19", "cse_21",
                                     "cse_16", "cse_18", "cse_20",
                                     "cse_15", "cse_17"]

            for i, var_names in enumerate((cse_kernel1, cse_kernel2, cse_kernel3)):
                for var_name in var_names:
                    knl = lp.rename_iname(knl, f"{var_name}_dim0", f"iel_cse_{i}",
                                          existing_ok=True)
                    knl = lp.rename_iname(knl, f"{var_name}_dim1", f"idof_cse_{i}",
                                          existing_ok=True)

            for i, var_names in enumerate((priv_cse_kernel2_vars,
                                           priv_cse_kernel3_vars)):
                for var_name in var_names:
                    knl = lp.assignment_to_subst(knl, var_name)
                for var_name in var_names[::-1]:
                    knl = precompute_for_single_kernel(
                        knl, t_unit.callables_table, f"{var_name}_subst",
                        sweep_inames=(),
                        temporary_address_space=lp.AddressSpace.PRIVATE,
                        compute_insn_id=f"cse_knl{i+2}_prcmpt_{var_name}")

            knl = lp.map_instructions(knl,
                      " or ".join(f"writes:{var_name}"
                                  for var_name in cse_kernel1),
                      lambda x: x.tagged(lp.LegacyStringInstructionTag("cse_knl1")))

            knl = lp.map_instructions(knl,
                      " or ".join([f"writes:{var_name}"
                                   for var_name in cse_kernel2]
                                  + ["id:cse_knl2_prcmpt_*"]),
                      lambda x: x.tagged(lp.LegacyStringInstructionTag("cse_knl2")))

            knl = lp.map_instructions(knl,
                      " or ".join([f"writes:{var_name}"
                                   for var_name in cse_kernel3]
                                  + ["id:cse_knl3_prcmpt_*"]),
                      lambda x: x.tagged(lp.LegacyStringInstructionTag("cse_knl3")))

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

                for ary in knl.args + list(knl.temporary_variables.values()):
                    if (isinstance(ary, ArrayBase)
                            and ary.address_space == lp.AddressSpace.GLOBAL):
                        nfootprint_bytes += (product(ary.shape)
                                             * ary.dtype.itemsize)

                # }}}

                print("Double-prec. GFlOps:", f64_ops * 1e-9)
                print("Footprint GBs:", nfootprint_bytes * 1e-9)
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
                                 insn_after="writes:rstrct_vals*")
            knl = lp.add_barrier(knl,
                                 insn_before="writes:rstrct_vals*",
                                 insn_after="iname:face_iel*")
            knl = lp.add_barrier(knl,
                                 insn_before="iname:face_iel*",
                                 insn_after="writes:mass_inv_inp*")
            knl = lp.add_barrier(knl,
                                 insn_before="writes:mass_inv_inp*",
                                 insn_after="writes:_pt_out*")

            # {{{ Plot the digraph of the CSEs

            if 0 and self.DO_CSE:
                rmap = knl.reader_map()
                print("digraph {")
                for arg in knl.args:
                    print(f"  {arg.name} [shape=record]")

                for tv in (knl.args
                           + list(knl.temporary_variables.values())):
                    indirect_access_checker = IndirectAccessChecker(tv.name,
                                                                    knl.all_inames())
                    for insn_id in rmap.get(tv.name, ()):
                        insn = knl.id_to_insn[insn_id]
                        if indirect_access_checker(insn.expression):
                            color = "red"
                        else:
                            color = "blue"

                        print(f"  {tv.name} -> {insn.assignee_name}"
                              f"[color={color}]")

                print("}")
                1/0

            # }}}

            # {{{ loop fusion

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
                                      f"rstrct_vals_{idof}_dim0",
                                      "rstrct_vals_dim0",
                                      existing_ok=True)

                # element loop
                knl = lp.rename_iname(knl,
                                      f"rstrct_vals_{idof}_dim1",
                                      "rstrct_vals_dim1",
                                      existing_ok=True)

                # face loop
                knl = lp.rename_iname(knl,
                                      f"rstrct_vals_{idof}_dim2",
                                      "rstrct_vals_dim2",
                                      existing_ok=True)
            knl = lp.rename_iname(knl, "rstrct_vals_dim0", "iface_rstrct")
            knl = lp.rename_iname(knl, "rstrct_vals_dim1", "iel_rstrct")
            knl = lp.rename_iname(knl, "rstrct_vals_dim2", "iface_dof_rstrct")

            # fuse all face-mass loops
            for idof in range(3):
                # element loop
                knl = lp.rename_iname(knl,
                                      f"face_iel_{idof}",
                                      "face_iel",
                                      existing_ok=True)

                # vol. dof
                knl = lp.rename_iname(knl,
                                      f"face_idof_{idof}",
                                      "face_idof",
                                      existing_ok=True)

            knl = lp.rename_iname(knl, "face_iel", "iel_face_mass")
            knl = lp.rename_iname(knl, "face_idof", "idof_face_mass")

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

            # we aren't going to parallelize the reductions, realize them so
            # that we can do apply other loop transformations to them.
            t_unit = t_unit.with_kernel(knl)
            t_unit = lp.realize_reduction(t_unit)
            knl = t_unit.default_entrypoint

            for i in range(3):
                knl = lp.rename_iname(knl, f"face_f_{i}", "face_f",
                                      existing_ok=True)
                knl = lp.rename_iname(knl, f"face_jdof_{i}", "face_jdof",
                                      existing_ok=True)

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

            knl = lp.split_iname(knl, "iel_rstrct", l_one_size,
                                 inner_tag="l.1", outer_tag="g.0")
            knl = lp.split_iname(knl, "iface_dof_rstrct", l_zero_size,
                                 inner_tag="l.0")

            # }}}

            # {{{ face mass transformations

            if 0:
                t_unit = t_unit.with_kernel(knl)
                t_unit = transform_face_mass(t_unit)
                knl = t_unit.default_entrypoint
            else:
                knl = lp.split_iname(knl, "iel_face_mass", l_one_size,
                                     inner_tag="l.1", outer_tag="g.0")
                knl = lp.split_iname(knl, "idof_face_mass", l_zero_size,
                                     inner_tag="l.0")

            # }}}

            # {{{ einsums

            knl = lp.split_iname(knl, "iel_diff", l_one_size,
                                 inner_tag="l.1", outer_tag="g.0")
            knl = lp.split_iname(knl, "idof_diff", l_zero_size,
                                 inner_tag="l.0")

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
