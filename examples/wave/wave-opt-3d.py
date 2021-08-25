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

    def map_constant(self, exrpr):
        return False


class HopefullySmartPytatoArrayContext(
        SingleGridWorkBalancingPytatoArrayContext):

    DO_CSE = False

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

        return dag

    def transform_loopy_program(self, t_unit):
        if t_unit.default_entrypoint.tags_of_type(FromActxCompile):
            import loopy as lp
            t_unit = lp.inline_callable_kernel(t_unit, "face_mass")
            knl = t_unit.default_entrypoint

            # get rid of noops
            noops = {insn.id for insn in knl.instructions
                     if isinstance(insn, lp.NoOpInstruction)}
            knl = lp.remove_instructions(knl, noops)

            # add the 2 gbarriers
            knl = lp.add_barrier(knl,
                                 insn_before="writes:rstrct_vals*",
                                 insn_after="iname:face_iel*",
                                 within_inames=frozenset())
            knl = lp.add_barrier(knl,
                                 insn_before="iname:face_iel*",
                                 insn_after="writes:_pt_out*",
                                 within_inames=frozenset())

            # {{{ Plot the digraph of the CSEs

            if 0 and self.DO_CSE:
                rmap = knl.reader_map()
                print("digraph {")

                for tv in knl.temporary_variables.values():
                    indirect_access_checker = IndirectAccessChecker(tv.name,
                                                                    knl.all_inames())
                    if tv.name.startswith("cse"):
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

            # fuse all final grad/div einsums
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

            # }}}

            return t_unit.with_kernel(knl)
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
