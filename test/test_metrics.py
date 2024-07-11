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

import logging

import numpy as np
import pytest

import meshmode.mesh.generation as mgen
from arraycontext import pytest_generate_tests_for_array_contexts
from meshmode.dof_array import flat_norm

from grudge.array_context import (
    PytestPyOpenCLArrayContextFactory,
    PytestPytatoPyOpenCLArrayContextFactory,
)
from grudge.discretization import make_discretization_collection


logger = logging.getLogger(__name__)
pytest_generate_tests = pytest_generate_tests_for_array_contexts(
        [PytestPyOpenCLArrayContextFactory,
         PytestPytatoPyOpenCLArrayContextFactory])


# {{{ inverse metric

@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("nonaffine", [False, True])
@pytest.mark.parametrize("use_quad", [False, True])
def test_inverse_metric(actx_factory, dim, nonaffine, use_quad):
    actx = actx_factory()

    order = 3
    mesh = mgen.generate_regular_rect_mesh(a=(-0.5,)*dim, b=(0.5,)*dim,
            nelements_per_axis=(6,)*dim, order=order)

    if nonaffine:
        def m(x):
            result = np.empty_like(x)
            result[0] = (
                    1.5*x[0] + np.cos(x[0])
                    + 0.1*np.sin(10*x[1]))
            result[1] = (
                    0.05*np.cos(10*x[0])
                    + 1.3*x[1] + np.sin(x[1]))
            if len(x) == 3:
                result[2] = x[2]
            return result

        from meshmode.mesh.processing import map_mesh
        mesh = map_mesh(mesh, m)

    from meshmode.discretization.poly_element import (
        QuadratureSimplexGroupFactory,
        default_simplex_group_factory,
    )

    from grudge.dof_desc import DISCR_TAG_BASE, DISCR_TAG_QUAD, as_dofdesc

    dcoll = make_discretization_collection(
        actx, mesh,
        discr_tag_to_group_factory={
            DISCR_TAG_BASE: default_simplex_group_factory(base_dim=dim, order=order),
            DISCR_TAG_QUAD: QuadratureSimplexGroupFactory(2*order + 1),
        }
    )

    from grudge.geometry import (
        forward_metric_derivative_mat,
        inverse_metric_derivative_mat,
    )

    dd = as_dofdesc("vol")
    if use_quad:
        dd = dd.with_discr_tag(DISCR_TAG_QUAD)

    mat = forward_metric_derivative_mat(
        actx, dcoll, dd,
        _use_geoderiv_connection=actx.supports_nonscalar_broadcasting).dot(
        inverse_metric_derivative_mat(
            actx, dcoll, dd,
            _use_geoderiv_connection=actx.supports_nonscalar_broadcasting))

    for i in range(mesh.dim):
        for j in range(mesh.dim):
            tgt = 1 if i == j else 0

            err = actx.to_numpy(flat_norm(mat[i, j] - tgt, ord=np.inf))
            logger.info("error[%d, %d]: %.5e", i, j, err)
            assert err < 1.0e-12, (i, j, err)

# }}}


# You can test individual routines by typing
# $ python test_metrics.py 'test_routine()'

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        pytest.main([__file__])

# vim: fdm=marker
