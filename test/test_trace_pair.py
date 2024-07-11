__copyright__ = "Copyright (C) 2021 University of Illinois Board of Trustees"

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


import logging

import numpy as np

import meshmode.mesh.generation as mgen
from arraycontext import pytest_generate_tests_for_array_contexts
from meshmode.dof_array import DOFArray

from grudge.array_context import PytestPyOpenCLArrayContextFactory
from grudge.discretization import make_discretization_collection
from grudge.trace_pair import TracePair


logger = logging.getLogger(__name__)
pytest_generate_tests = pytest_generate_tests_for_array_contexts(
        [PytestPyOpenCLArrayContextFactory])


def test_trace_pair(actx_factory):
    """Simple smoke test for :class:`grudge.trace_pair.TracePair`."""
    actx = actx_factory()
    dim = 3
    order = 1
    n = 4

    mesh = mgen.generate_regular_rect_mesh(
        a=(-1,)*dim, b=(1,)*dim,
        nelements_per_axis=(n,)*dim)

    dcoll = make_discretization_collection(actx, mesh, order=order)

    rng = np.random.default_rng(1234)

    def rand():
        return DOFArray(
                actx,
                tuple(actx.from_numpy(
                    rng.uniform(size=(grp.nelements, grp.nunit_dofs)))
                    for grp in dcoll.discr_from_dd("vol").groups))

    from grudge.dof_desc import DD_VOLUME_ALL
    interior = rand()
    exterior = rand()
    tpair = TracePair(DD_VOLUME_ALL, interior=interior, exterior=exterior)

    import grudge.op as op
    assert op.norm(dcoll, tpair.avg - 0.5*(exterior + interior), np.inf) == 0
    assert op.norm(dcoll, tpair.diff - (exterior - interior), np.inf) == 0
    assert op.norm(dcoll, tpair.int - interior, np.inf) == 0
    assert op.norm(dcoll, tpair.ext - exterior, np.inf) == 0
