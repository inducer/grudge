__copyright__ = "Copyright (C) 2021 Andreas Kloeckner"

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
import numpy.linalg as la
import pytest

import meshmode.mesh.generation as mgen
from grudge import DiscretizationCollection

from arraycontext import (  # noqa
    pytest_generate_tests_for_pyopencl_array_context
    as pytest_generate_tests
)


@pytest.mark.parametrize("n", [2, 3])
@pytest.mark.parametrize("seed", [5, 17, 2222, 23432])
def test_eigenvalue_formula(n, seed):
    np.random.seed(seed)
    a = np.random.randn(n, n)
    a = (a @ a.T).astype(np.complex128)

    _np = np

    class FakeArrayContext:
        np = _np

    from grudge.dt_finding import symmetric_eigenvalues
    eigval_formula = np.real_if_close(
            symmetric_eigenvalues(FakeArrayContext(), a))
    assert eigval_formula.dtype.kind != "c"
    eigval_formula = sorted(eigval_formula)

    eigval_lapack = sorted(la.eigvalsh(a))

    assert np.allclose(eigval_formula, eigval_lapack)


@pytest.mark.parametrize("dim", [1, 2, 3])
def test_mapping_jacobian_min_sing_val(actx_factory, dim):
    actx = actx_factory()

    mesh = mgen.generate_regular_rect_mesh(a=(-0.5,)*dim, b=(0.5,)*dim,
            nelements_per_axis=(6,)*dim, order=4)

    dcoll = DiscretizationCollection(actx, mesh, order=4)

    from grudge.dt_finding import min_singular_value_of_mapping_jacobian

    min_singular_value_of_mapping_jacobian(actx, dcoll)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        pytest.main([__file__])
