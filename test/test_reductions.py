__copyright__ = """
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

from dataclasses import dataclass

from arraycontext import (
    with_container_arithmetic,
    dataclass_array_container,
    pytest_generate_tests_for_array_contexts
)

from meshmode.dof_array import DOFArray

from grudge.array_context import PytestPyOpenCLArrayContextFactory

pytest_generate_tests = pytest_generate_tests_for_array_contexts(
        [PytestPyOpenCLArrayContextFactory])

from grudge import DiscretizationCollection

import grudge.op as op

from meshmode.dof_array import flatten

from pytools.obj_array import make_obj_array

import pytest

import logging


logger = logging.getLogger(__name__)


@pytest.mark.parametrize(("mesh_size", "with_initial"), [
    (4, False),
    (4, True),
    (0, False),
    (0, True)
])
def test_nodal_reductions(actx_factory, mesh_size, with_initial):
    actx = actx_factory()

    from mesh_data import BoxMeshBuilder
    builder = BoxMeshBuilder(ambient_dim=1)

    mesh = builder.get_mesh(mesh_size, builder.mesh_order)
    dcoll = DiscretizationCollection(actx, mesh, order=builder.order)
    x = actx.thaw(dcoll.nodes())

    def f(x):
        return -actx.np.sin(10*x[0])

    def g(x):
        return actx.np.cos(2*x[0])

    def h(x):
        return -actx.np.tan(5*x[0])

    fields = make_obj_array([f(x), g(x), h(x)])

    f_ref = actx.to_numpy(flatten(fields[0]))
    g_ref = actx.to_numpy(flatten(fields[1]))
    h_ref = actx.to_numpy(flatten(fields[2]))
    concat_fields = np.concatenate([f_ref, g_ref, h_ref])

    for grudge_op, np_op in [(op.nodal_max, np.max),
                             (op.nodal_min, np.min),
                             (op.nodal_sum, np.sum)]:
        extra_kwargs = {}
        if with_initial:
            if grudge_op is op.nodal_max:
                extra_kwargs["initial"] = -100.
            elif grudge_op is op.nodal_min:
                extra_kwargs["initial"] = 100.

        # nodal_min/nodal_max have default initial values, so they behave
        # differently from numpy in the empty case
        extra_np_only_kwargs = {}
        if mesh_size == 0 and not with_initial:
            if grudge_op is op.nodal_max:
                extra_np_only_kwargs["initial"] = -np.inf
            elif grudge_op is op.nodal_min:
                extra_np_only_kwargs["initial"] = np.inf

        # Componentwise reduction checks
        assert np.isclose(
            actx.to_numpy(grudge_op(dcoll, "vol", fields[0], **extra_kwargs)),
            np_op(f_ref, **extra_kwargs, **extra_np_only_kwargs),
            rtol=1e-13)
        assert np.isclose(
            actx.to_numpy(grudge_op(dcoll, "vol", fields[1], **extra_kwargs)),
            np_op(g_ref, **extra_kwargs, **extra_np_only_kwargs),
            rtol=1e-13)
        assert np.isclose(
            actx.to_numpy(grudge_op(dcoll, "vol", fields[2], **extra_kwargs)),
            np_op(h_ref, **extra_kwargs, **extra_np_only_kwargs),
            rtol=1e-13)

        # Test nodal reductions work on object arrays
        assert np.isclose(
            actx.to_numpy(grudge_op(dcoll, "vol", fields, **extra_kwargs)),
            np_op(concat_fields, **extra_kwargs, **extra_np_only_kwargs),
            rtol=1e-13)


def test_elementwise_reductions(actx_factory):
    actx = actx_factory()

    from mesh_data import BoxMeshBuilder
    builder = BoxMeshBuilder(ambient_dim=1)

    nelements = 4
    mesh = builder.get_mesh(nelements, builder.mesh_order)
    dcoll = DiscretizationCollection(actx, mesh, order=builder.order)
    x = actx.thaw(dcoll.nodes())

    def f(x):
        return actx.np.sin(x[0])

    field = f(x)
    mins = []
    maxs = []
    sums = []
    for grp_f in field:
        min_res = np.empty(grp_f.shape)
        max_res = np.empty(grp_f.shape)
        sum_res = np.empty(grp_f.shape)
        for eidx in range(mesh.nelements):
            element_data = actx.to_numpy(grp_f[eidx])
            min_res[eidx, :] = np.min(element_data)
            max_res[eidx, :] = np.max(element_data)
            sum_res[eidx, :] = np.sum(element_data)
        mins.append(actx.from_numpy(min_res))
        maxs.append(actx.from_numpy(max_res))
        sums.append(actx.from_numpy(sum_res))

    ref_mins = DOFArray(actx, data=tuple(mins))
    ref_maxs = DOFArray(actx, data=tuple(maxs))
    ref_sums = DOFArray(actx, data=tuple(sums))

    elem_mins = op.elementwise_min(dcoll, field)
    elem_maxs = op.elementwise_max(dcoll, field)
    elem_sums = op.elementwise_sum(dcoll, field)

    assert actx.to_numpy(op.norm(dcoll, elem_mins - ref_mins, np.inf)) < 1.e-15
    assert actx.to_numpy(op.norm(dcoll, elem_maxs - ref_maxs, np.inf)) < 1.e-15
    assert actx.to_numpy(op.norm(dcoll, elem_sums - ref_sums, np.inf)) < 1.e-15


# {{{ Array container tests

@with_container_arithmetic(bcast_obj_array=False,
        eq_comparison=False, rel_comparison=False,
        _cls_has_array_context_attr=True)
@dataclass_array_container
@dataclass(frozen=True)
class MyContainer:
    name: str
    mass: DOFArray
    momentum: np.ndarray
    enthalpy: DOFArray

    @property
    def array_context(self):
        return self.mass.array_context


def test_nodal_reductions_with_container(actx_factory):
    actx = actx_factory()

    from mesh_data import BoxMeshBuilder
    builder = BoxMeshBuilder(ambient_dim=2)

    mesh = builder.get_mesh(4, builder.mesh_order)
    dcoll = DiscretizationCollection(actx, mesh, order=builder.order)
    x = actx.thaw(dcoll.nodes())

    def f(x):
        return -actx.np.sin(10*x[0]) * actx.np.cos(2*x[1])

    def g(x):
        return actx.np.cos(2*x[0]) * actx.np.sin(10*x[1])

    def h(x):
        return -actx.np.tan(5*x[0]) * actx.np.tan(0.5*x[1])

    mass = f(x) + g(x)
    momentum = make_obj_array([f(x)/g(x), h(x)])
    enthalpy = h(x) - g(x)

    ary_container = MyContainer(name="container",
                                mass=mass,
                                momentum=momentum,
                                enthalpy=enthalpy)

    mass_ref = actx.to_numpy(flatten(mass))
    momentum_ref = np.concatenate([actx.to_numpy(mom_i)
                                   for mom_i in flatten(momentum)])
    enthalpy_ref = actx.to_numpy(flatten(enthalpy))
    concat_fields = np.concatenate([mass_ref, momentum_ref, enthalpy_ref])

    for grudge_op, np_op in [(op.nodal_sum, np.sum),
                             (op.nodal_max, np.max),
                             (op.nodal_min, np.min)]:

        assert np.isclose(actx.to_numpy(grudge_op(dcoll, "vol", ary_container)),
                          np_op(concat_fields), rtol=1e-13)

    # Check norm reduction
    assert np.isclose(actx.to_numpy(op.norm(dcoll, ary_container, np.inf)),
                      np.linalg.norm(concat_fields, ord=np.inf),
                      rtol=1e-13)


def test_elementwise_reductions_with_container(actx_factory):
    actx = actx_factory()

    from mesh_data import BoxMeshBuilder
    builder = BoxMeshBuilder(ambient_dim=2)

    nelements = 4
    mesh = builder.get_mesh(nelements, builder.mesh_order)
    dcoll = DiscretizationCollection(actx, mesh, order=builder.order)
    x = actx.thaw(dcoll.nodes())

    def f(x):
        return actx.np.sin(x[0]) * actx.np.sin(x[1])

    def g(x):
        return actx.np.cos(x[0]) * actx.np.cos(x[1])

    def h(x):
        return actx.np.cos(x[0]) * actx.np.sin(x[1])

    mass = 2*f(x) + 0.5*g(x)
    momentum = make_obj_array([f(x)/g(x), h(x)])
    enthalpy = 3*h(x) - g(x)

    ary_container = MyContainer(name="container",
                                mass=mass,
                                momentum=momentum,
                                enthalpy=enthalpy)

    def _get_ref_data(field):
        mins = []
        maxs = []
        sums = []
        for grp_f in field:
            min_res = np.empty(grp_f.shape)
            max_res = np.empty(grp_f.shape)
            sum_res = np.empty(grp_f.shape)
            for eidx in range(mesh.nelements):
                element_data = actx.to_numpy(grp_f[eidx])
                min_res[eidx, :] = np.min(element_data)
                max_res[eidx, :] = np.max(element_data)
                sum_res[eidx, :] = np.sum(element_data)
            mins.append(actx.from_numpy(min_res))
            maxs.append(actx.from_numpy(max_res))
            sums.append(actx.from_numpy(sum_res))
        min_field = DOFArray(actx, data=tuple(mins))
        max_field = DOFArray(actx, data=tuple(maxs))
        sums_field = DOFArray(actx, data=tuple(sums))
        return min_field, max_field, sums_field

    min_mass, max_mass, sums_mass = _get_ref_data(mass)
    min_enthalpy, max_enthalpy, sums_enthalpy = _get_ref_data(enthalpy)
    min_mom_x, max_mom_x, sums_mom_x = _get_ref_data(momentum[0])
    min_mom_y, max_mom_y, sums_mom_y = _get_ref_data(momentum[1])
    min_momentum = make_obj_array([min_mom_x, min_mom_y])
    max_momentum = make_obj_array([max_mom_x, max_mom_y])
    sums_momentum = make_obj_array([sums_mom_x, sums_mom_y])

    reference_min = MyContainer(
        name="Reference min",
        mass=min_mass,
        momentum=min_momentum,
        enthalpy=min_enthalpy
    )

    reference_max = MyContainer(
        name="Reference max",
        mass=max_mass,
        momentum=max_momentum,
        enthalpy=max_enthalpy
    )

    reference_sum = MyContainer(
        name="Reference sums",
        mass=sums_mass,
        momentum=sums_momentum,
        enthalpy=sums_enthalpy
    )

    elem_mins = op.elementwise_min(dcoll, ary_container)
    elem_maxs = op.elementwise_max(dcoll, ary_container)
    elem_sums = op.elementwise_sum(dcoll, ary_container)

    assert actx.to_numpy(op.norm(dcoll, elem_mins - reference_min, np.inf)) < 1.e-14
    assert actx.to_numpy(op.norm(dcoll, elem_maxs - reference_max, np.inf)) < 1.e-14
    assert actx.to_numpy(op.norm(dcoll, elem_sums - reference_sum, np.inf)) < 1.e-14

# }}}


# You can test individual routines by typing
# $ python test_grudge.py 'test_routine()'

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        pytest.main([__file__])
