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
from grudge.trace_pair import TracePair, CommTag
import meshmode.mesh.generation as mgen
from meshmode.dof_array import DOFArray
from dataclasses import dataclass

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
        return DOFArray(
                actx,
                tuple(actx.from_numpy(
                    np.random.rand(grp.nelements, grp.nunit_dofs))
                    for grp in dcoll.discr_from_dd("vol").groups))

    interior = rand()
    exterior = rand()
    tpair = TracePair("vol", interior=interior, exterior=exterior)

    import grudge.op as op
    assert op.norm(dcoll, tpair.avg - 0.5*(exterior + interior), np.inf) == 0
    assert op.norm(dcoll, tpair.diff - (exterior - interior), np.inf) == 0
    assert op.norm(dcoll, tpair.int - interior, np.inf) == 0
    assert op.norm(dcoll, tpair.ext - exterior, np.inf) == 0


def test_commtag(actx_factory):

    class DerivedCommTag(CommTag):
        pass

    class DerivedDerivedCommTag(DerivedCommTag):
        pass

    # {{{ test equality and hash consistency

    ct = CommTag()
    ct2 = CommTag()
    dct = DerivedCommTag()
    dct2 = DerivedCommTag()
    ddct = DerivedDerivedCommTag()

    assert ct == ct2
    assert ct != dct
    assert dct == dct2
    assert dct != ddct
    assert ddct != dct
    assert (ct, dct) != (dct, ct)

    assert hash(ct) == hash(ct2)
    assert hash(ct) != hash(dct)
    assert hash(dct) != hash(ddct)

    # }}}

    # {{{ test hash stability

    assert hash(ct) == 4644528671524962420
    assert hash(dct) == -1013583671995716582
    assert hash(ddct) == 626392264874077479

    assert hash((ct, 123)) == -578844573019921397
    assert hash((dct, 123)) == -8009406276367324841
    assert hash((dct, ct)) == 6599529611285265043

    # }}}

    # {{{ test using derived dataclasses

    @dataclass(frozen=True)
    class DataCommTag(CommTag):
        data: int

    @dataclass(frozen=True)
    class DataCommTag2(CommTag):
        data: int

    d1 = DataCommTag(1)
    d2 = DataCommTag(2)
    d3 = DataCommTag(1)

    assert d1 != d2
    assert hash(d1) != hash(d2)
    assert d1 == d3
    assert hash(d1) == hash(d3)

    d4 = DataCommTag2(1)
    assert d1 != d4

    # }}}
