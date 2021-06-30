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


import numpy as np
from grudge.trace_pair import TracePair
import meshmode.mesh.generation as mgen

from grudge import DiscretizationCollection

from grudge.array_context import PytestPyOpenCLArrayContextFactory
from arraycontext import pytest_generate_tests_for_array_contexts
pytest_generate_tests = pytest_generate_tests_for_array_contexts(
        [PytestPyOpenCLArrayContextFactory])

import logging

logger = logging.getLogger(__name__)


def test_trace_pair(actx_factory):
    """Simple smoke test for :class:`grudge.trace_pair.TracePair`."""
    actx = actx_factory()
    dim = 3
    order = 1
    n = 4

    mesh = mgen.generate_regular_rect_mesh(
        a=(-1,)*dim, b=(1,)*dim,
        nelements_per_axis=(n,)*dim)

    dcoll = DiscretizationCollection(actx, mesh, order=order)

    def rand():
        ary = dcoll.zeros(actx)
        for grp_ary in ary:
            grp_ary.set(np.random.rand(*grp_ary.shape))
        return ary

    interior = rand()
    exterior = rand()
    tpair = TracePair("vol", interior=interior, exterior=exterior)

    assert tpair.avg == 0.5*(interior + exterior)
    assert tpair.diff == (exterior - interior)
    assert tpair.int == interior
    assert tpair.ext == exterior
