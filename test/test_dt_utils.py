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

from grudge.array_context import (
    PytestPyOpenCLArrayContextFactory,
    PytestPytatoPyOpenCLArrayContextFactory
)
from arraycontext import pytest_generate_tests_for_array_contexts
pytest_generate_tests = pytest_generate_tests_for_array_contexts(
        [PytestPyOpenCLArrayContextFactory,
         PytestPytatoPyOpenCLArrayContextFactory])

from grudge import DiscretizationCollection

import grudge.op as op

import pytest

import logging


logger = logging.getLogger(__name__)


@pytest.mark.parametrize("name", ["interval", "box2d", "box3d"])
def test_geometric_factors_regular_refinement(actx_factory, name):
    from grudge.dt_utils import dt_geometric_factors

    actx = actx_factory()

    # {{{ cases

    if name == "interval":
        from mesh_data import BoxMeshBuilder
        builder = BoxMeshBuilder(ambient_dim=1)
    elif name == "box2d":
        from mesh_data import BoxMeshBuilder
        builder = BoxMeshBuilder(ambient_dim=2)
    elif name == "box3d":
        from mesh_data import BoxMeshBuilder
        builder = BoxMeshBuilder(ambient_dim=3)
    else:
        raise ValueError("unknown geometry name: %s" % name)

    # }}}

    min_factors = []
    for resolution in builder.resolutions:
        mesh = builder.get_mesh(resolution, builder.mesh_order)
        dcoll = DiscretizationCollection(actx, mesh, order=builder.order)
        min_factors.append(
            actx.to_numpy(
                op.nodal_min(dcoll, "vol", thaw(dt_geometric_factors(dcoll), actx)))
        )

    # Resolution is doubled each refinement, so the ratio of consecutive
    # geometric factors should satisfy: gfi+1 / gfi = 2
    min_factors = np.asarray(min_factors)
    ratios = min_factors[:-1] / min_factors[1:]
    assert np.all(np.isclose(ratios, 2))


@pytest.mark.parametrize("name", ["interval", "box2d", "box3d"])
def test_non_geometric_factors(actx_factory, name):
    from grudge.dt_utils import dt_non_geometric_factors

    actx = actx_factory()

    # {{{ cases

    if name == "interval":
        from mesh_data import BoxMeshBuilder
        builder = BoxMeshBuilder(ambient_dim=1)
    elif name == "box2d":
        from mesh_data import BoxMeshBuilder
        builder = BoxMeshBuilder(ambient_dim=2)
    elif name == "box3d":
        from mesh_data import BoxMeshBuilder
        builder = BoxMeshBuilder(ambient_dim=3)
    else:
        raise ValueError("unknown geometry name: %s" % name)

    # }}}

    factors = []
    degrees = list(range(1, 8))
    for degree in degrees:
        mesh = builder.get_mesh(1, degree)
        dcoll = DiscretizationCollection(actx, mesh, order=degree)
        factors.append(min(dt_non_geometric_factors(dcoll)))

    # Crude estimate, factors should behave like 1/N**2
    factors = np.asarray(factors)
    lower_bounds = 1/(np.asarray(degrees)**2)
    upper_bounds = 6.295*lower_bounds

    assert all(lower_bounds <= factors)
    assert all(factors <= upper_bounds)


# You can test individual routines by typing
# $ python test_grudge.py 'test_routine()'

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        pytest.main([__file__])
