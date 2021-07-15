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

from arraycontext import thaw

from grudge.array_context import PytestPyOpenCLArrayContextFactory
from arraycontext import pytest_generate_tests_for_array_contexts
pytest_generate_tests = pytest_generate_tests_for_array_contexts(
        [PytestPyOpenCLArrayContextFactory])

from grudge import make_discretization_collection

import grudge.op as op

from meshmode.dof_array import flatten

from pytools.obj_array import make_obj_array

import pytest

import logging


logger = logging.getLogger(__name__)


def test_nodal_reductions(actx_factory):
    actx = actx_factory()

    from mesh_data import BoxMeshBuilder
    builder = BoxMeshBuilder(ambient_dim=1)

    mesh = builder.get_mesh(4, builder.mesh_order)
    dcoll = make_discretization_collection(actx, mesh, order=builder.order)
    x = thaw(dcoll.nodes(), actx)

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

    for inner_grudge_op, np_op in [(op.nodal_sum, np.sum),
                             (op.nodal_max, np.max),
                             (op.nodal_min, np.min)]:

        # FIXME: Remove this once all grudge reductions return device scalars
        def grudge_op(dcoll, dd, vec):
            res = inner_grudge_op(dcoll, dd, vec)

            from numbers import Number
            if not isinstance(res, Number):
                return actx.to_numpy(res)
            else:
                return res

        # Componentwise reduction checks
        assert np.isclose(grudge_op(dcoll, "vol", fields[0]),
                          np_op(f_ref), rtol=1e-13)
        assert np.isclose(grudge_op(dcoll, "vol", fields[1]),
                          np_op(g_ref), rtol=1e-13)
        assert np.isclose(grudge_op(dcoll, "vol", fields[2]),
                          np_op(h_ref), rtol=1e-13)

        # Test nodal reductions work on object arrays
        assert np.isclose(grudge_op(dcoll, "vol", fields),
                          np_op(concat_fields), rtol=1e-13)


def test_elementwise_reductions(actx_factory):
    actx = actx_factory()

    from mesh_data import BoxMeshBuilder
    builder = BoxMeshBuilder(ambient_dim=1)

    nelements = 4
    mesh = builder.get_mesh(nelements, builder.mesh_order)
    dcoll = make_discretization_collection(actx, mesh, order=builder.order)
    x = thaw(dcoll.nodes(), actx)

    def f(x):
        return actx.np.sin(x[0])

    field = f(x)
    mins = []
    maxs = []
    sums = []
    for gidx, grp_f in enumerate(field):
        min_res = np.empty(grp_f.shape)
        max_res = np.empty(grp_f.shape)
        sum_res = np.empty(grp_f.shape)
        for eidx in range(dcoll.discr_from_dd("vol").groups[gidx].nelements):
            element_data = actx.to_numpy(grp_f[eidx])
            min_res[eidx, :] = np.min(element_data)
            max_res[eidx, :] = np.max(element_data)
            sum_res[eidx, :] = np.sum(element_data)
        mins.append(actx.from_numpy(min_res))
        maxs.append(actx.from_numpy(max_res))
        sums.append(actx.from_numpy(sum_res))

    from meshmode.dof_array import DOFArray, flat_norm

    ref_mins = DOFArray(actx, data=tuple(mins))
    ref_maxs = DOFArray(actx, data=tuple(maxs))
    ref_sums = DOFArray(actx, data=tuple(sums))

    elem_mins = op.elementwise_min(dcoll, field)
    elem_maxs = op.elementwise_max(dcoll, field)
    elem_sums = op.elementwise_sum(dcoll, field)

    assert flat_norm(elem_mins - ref_mins, ord=np.inf) < 1.e-15
    assert flat_norm(elem_maxs - ref_maxs, ord=np.inf) < 1.e-15
    assert flat_norm(elem_sums - ref_sums, ord=np.inf) < 1.e-15


# You can test individual routines by typing
# $ python test_grudge.py 'test_routine()'

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        pytest.main([__file__])
