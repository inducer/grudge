import numpy as np  # noqa: F401
import pyopencl as cl
from typing import Union
from meshmode.mesh import BTAG_ALL
from meshmode.mesh.generation import generate_regular_rect_mesh
from arraycontext.metadata import NameHint
from meshmode.array_context import (PytatoPyOpenCLArrayContext,
                                    PyOpenCLArrayContext)
from pytato.transform import CombineMapper
from pytato.array import (Placeholder, DataWrapper, SizeParam, IndexBase,
                          Array, DictOfNamedArrays, BasicIndex)
from meshmode.discretization.connection import (FACE_RESTR_INTERIOR,
                                                FACE_RESTR_ALL)
from pytools.obj_array import make_obj_array
from pyopencl.tools import (  # noqa
        pytest_generate_tests_for_pyopencl as pytest_generate_tests)
import grudge
import grudge.op as op


# {{{ utilities for test_push_indirections_*

class _IndexeeArraysMaterializedChecker(CombineMapper[bool]):
    def combine(self, *args: bool) -> bool:
        return all(args)

    def map_placeholder(self, expr: Placeholder) -> bool:
        return True

    def map_data_wrapper(self, expr: DataWrapper) -> bool:
        return True

    def map_size_param(self, expr: SizeParam) -> bool:
        return True

    def _map_index_base(self, expr: IndexBase) -> bool:
        from grudge.pytato_transforms.pytato_indirection_transforms import (
            _is_materialized)
        return self.combine(
            _is_materialized(expr.array) or isinstance(expr.array, BasicIndex),
            self.rec(expr.array)
        )


def are_all_indexees_materialized_nodes(
        expr: Union[Array, DictOfNamedArrays]) -> bool:
    """
    Returns *True* only if all indexee arrays are either materialized nodes,
    OR, other indexing nodes that have materialized indexees.
    """
    return _IndexeeArraysMaterializedChecker()(expr)


class _IndexerArrayDatawrapperChecker(CombineMapper[bool]):
    def combine(self, *args: bool) -> bool:
        return all(args)

    def map_placeholder(self, expr: Placeholder) -> bool:
        return True

    def map_data_wrapper(self, expr: DataWrapper) -> bool:
        return True

    def map_size_param(self, expr: SizeParam) -> bool:
        return True

    def _map_index_base(self, expr: IndexBase) -> bool:
        return self.combine(
            *[isinstance(idx, DataWrapper)
              for idx in expr.indices
              if isinstance(idx, Array)],
            super()._map_index_base(expr),
        )


def are_all_indexer_arrays_datawrappers(
        expr: Union[Array, DictOfNamedArrays]) -> bool:
    """
    Returns *True* only if all indexer arrays are instances of
    :class:`~pytato.array.DataWrapper`.
    """
    return _IndexerArrayDatawrapperChecker()(expr)

# }}}


def _evaluate_dict_of_named_arrays(actx, dict_of_named_arrays):
    container = make_obj_array([dict_of_named_arrays._data[name]
                                for name in sorted(dict_of_named_arrays.keys())])

    evaluated_container = actx.thaw(actx.freeze(container))

    return {name: evaluated_container[i]
            for i, name in enumerate(sorted(dict_of_named_arrays.keys()))}


class FluxOptimizerActx(PytatoPyOpenCLArrayContext):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.check_completed = False

    def transform_dag(self, dag):
        from grudge.pytato_transforms.pytato_indirection_transforms import (
            fuse_dof_pick_lists, fold_constant_indirections)
        from pytato.tags import PrefixNamed

        if all(PrefixNamed("flux_container") in v.tags for v in dag._data.values()):
            assert not are_all_indexer_arrays_datawrappers(dag)
            assert not are_all_indexees_materialized_nodes(dag)
            self.check_completed = True

        dag = fuse_dof_pick_lists(dag)
        dag = fold_constant_indirections(
            dag, lambda x: _evaluate_dict_of_named_arrays(self, x))

        if all(PrefixNamed("flux_container") in v.tags for v in dag._data.values()):
            assert are_all_indexer_arrays_datawrappers(dag)
            assert are_all_indexees_materialized_nodes(dag)
            self.check_completed = True

        return dag


# {{{ test_resampling_indirections_are_fused_0

def _compute_flux_0(dcoll, actx, u):
    u_interior_tpair, = op.interior_trace_pairs(dcoll, u)
    flux_on_interior_faces = u_interior_tpair.avg
    flux_on_all_faces = op.project(
        dcoll, FACE_RESTR_INTERIOR, FACE_RESTR_ALL, flux_on_interior_faces)

    flux_on_all_faces = actx.tag(NameHint("flux_container"), flux_on_all_faces)
    return flux_on_all_faces


def test_resampling_indirections_are_fused_0(ctx_factory):
    cl_ctx = ctx_factory()
    cq = cl.CommandQueue(cl_ctx)

    ref_actx = PyOpenCLArrayContext(cq)
    actx = FluxOptimizerActx(cq)

    dim = 3
    nel_1d = 4
    mesh = generate_regular_rect_mesh(
        a=(-0.5,)*dim,
        b=(0.5,)*dim,
        nelements_per_axis=(nel_1d,)*dim,
        boundary_tag_to_face={"bdry": ["-x", "+x",
                                       "-y", "+y",
                                       "-z", "+z"]}
    )
    dcoll = grudge.make_discretization_collection(ref_actx, mesh, order=2)

    x, _, _ = dcoll.nodes()
    compiled_flux_0 = actx.compile(lambda ary: _compute_flux_0(dcoll, actx, ary))

    ref_output = ref_actx.to_numpy(
        _compute_flux_0(dcoll, ref_actx, ref_actx.thaw(x)))
    output = actx.to_numpy(
        compiled_flux_0(actx.thaw(x)))

    np.testing.assert_allclose(ref_output[0], output[0])
    assert actx.check_completed

# }}}


# {{{ test_resampling_indirections_are_fused_1

def _compute_flux_1(dcoll, actx, u):
    u_interior_tpair, = op.interior_trace_pairs(dcoll, u)
    flux_on_interior_faces = u_interior_tpair.avg
    flux_on_bdry = op.project(dcoll, "vol", BTAG_ALL, u)
    flux_on_all_faces = (
        op.project(dcoll,
                   FACE_RESTR_INTERIOR,
                   FACE_RESTR_ALL,
                   flux_on_interior_faces)
        + op.project(dcoll, BTAG_ALL, FACE_RESTR_ALL, flux_on_bdry)
    )

    result = op.inverse_mass(dcoll, op.face_mass(dcoll, flux_on_all_faces))

    result = actx.tag(NameHint("flux_container"), result)
    return result


def test_resampling_indirections_are_fused_1(ctx_factory):
    cl_ctx = ctx_factory()
    cq = cl.CommandQueue(cl_ctx)

    ref_actx = PyOpenCLArrayContext(cq)
    actx = FluxOptimizerActx(cq)

    dim = 3
    nel_1d = 4
    mesh = generate_regular_rect_mesh(
        a=(-0.5,)*dim,
        b=(0.5,)*dim,
        nelements_per_axis=(nel_1d,)*dim,
        boundary_tag_to_face={"bdry": ["-x", "+x",
                                       "-y", "+y",
                                       "-z", "+z"]}
    )
    dcoll = grudge.make_discretization_collection(ref_actx, mesh, order=2)

    x, _, _ = dcoll.nodes()
    compiled_flux_1 = actx.compile(lambda ary: _compute_flux_1(dcoll, actx, ary))

    ref_output = ref_actx.to_numpy(
        _compute_flux_1(dcoll, ref_actx, ref_actx.thaw(x)))
    output = actx.to_numpy(
        compiled_flux_1(actx.thaw(x)))

    np.testing.assert_allclose(ref_output[0], output[0])
    assert actx.check_completed

# }}}


# {{{ test_resampling_indirections_are_fused_2

def _compute_flux_2(dcoll, actx, u):
    u_interior_tpair, = op.interior_trace_pairs(dcoll, u)
    normal_on_interior_faces = actx.thaw(dcoll.normal(u_interior_tpair.dd))
    normal_on_bdry_faces = actx.thaw(dcoll.normal(BTAG_ALL))
    flux_on_interior_faces = u_interior_tpair.avg * normal_on_interior_faces
    flux_on_bdry = op.project(dcoll, "vol", BTAG_ALL, u) * normal_on_bdry_faces
    flux_on_all_faces = (
        op.project(dcoll,
                   FACE_RESTR_INTERIOR,
                   FACE_RESTR_ALL,
                   flux_on_interior_faces)
        + op.project(dcoll, BTAG_ALL, FACE_RESTR_ALL, flux_on_bdry)
    )

    result = op.inverse_mass(dcoll, op.face_mass(dcoll, flux_on_all_faces))

    result = actx.tag(NameHint("flux_container"), result)
    return result


def test_resampling_indirections_are_fused_2(ctx_factory):
    cl_ctx = ctx_factory()
    cq = cl.CommandQueue(cl_ctx)

    ref_actx = PyOpenCLArrayContext(cq)
    actx = FluxOptimizerActx(cq)

    dim = 2
    nel_1d = 4
    mesh = generate_regular_rect_mesh(
        a=(-0.5,)*dim,
        b=(0.5,)*dim,
        nelements_per_axis=(nel_1d,)*dim,
        boundary_tag_to_face={"bdry": ["-x", "+x",
                                       "-y", "+y"]}
    )
    dcoll = grudge.make_discretization_collection(ref_actx, mesh, order=2)

    x, _ = dcoll.nodes()
    compiled_flux_2 = actx.compile(lambda ary: _compute_flux_2(dcoll, actx, ary))

    ref_output = ref_actx.to_numpy(
        _compute_flux_2(dcoll, ref_actx, ref_actx.thaw(x)))
    output = actx.to_numpy(
        compiled_flux_2(actx.thaw(x)))

    np.testing.assert_allclose(ref_output[0], output[0])
    assert actx.check_completed

# }}}

# vim: fdm=marker
