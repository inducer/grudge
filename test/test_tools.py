__copyright__ = """
Copyright (C) 2022 University of Illinois Board of Trustees
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

from dataclasses import dataclass

import numpy as np
import numpy.linalg as la  # noqa

from arraycontext import pytest_generate_tests_for_array_contexts

from grudge.array_context import PytestPyOpenCLArrayContextFactory


pytest_generate_tests = pytest_generate_tests_for_array_contexts(
        [PytestPyOpenCLArrayContextFactory])

import logging

import pytest

from pytools.obj_array import make_obj_array


logger = logging.getLogger(__name__)


# {{{ map_subarrays and rec_map_subarrays

@dataclass(frozen=True, eq=True)
class _DummyScalar:
    val: int


def test_map_subarrays():
    """Test map_subarrays."""
    from grudge.tools import map_subarrays

    # Scalar
    result = map_subarrays(
        lambda x: np.array([x, 2*x]), (), (2,), 1)
    assert result.dtype == int
    assert np.all(result == np.array([1, 2]))

    # Scalar, nested
    result = map_subarrays(
        lambda x: np.array([x, 2*x]), (), (2,), 1, return_nested=True)
    assert result.dtype == int
    assert np.all(result == np.array([1, 2]))  # Same as non-nested

    # in_shape is whole array
    result = map_subarrays(
        lambda x: 2*x, (2,), (2,), np.array([1, 2]))
    assert result.dtype == int
    assert np.all(result == np.array([2, 4]))

    # in_shape is whole array, nested
    result = map_subarrays(
        lambda x: 2*x, (2,), (2,), np.array([1, 2]), return_nested=True)
    assert result.dtype == int
    assert np.all(result == np.array([2, 4]))  # Same as non-nested

    # len(out_shape) == 0
    result = map_subarrays(
        np.sum, (2,), (), np.array([[1, 2], [2, 4]]))
    assert result.dtype == int
    assert np.all(result == np.array([3, 6]))

    # len(out_shape) == 0, nested output
    result = map_subarrays(
        np.sum, (2,), (), np.array([[1, 2], [2, 4]]), return_nested=True)
    assert np.all(result == np.array([3, 6]))  # Same as non-nested output

    # len(out_shape) == 0, non-numerical scalars
    result = map_subarrays(
        lambda x: _DummyScalar(x[0].val + x[1].val), (2,), (),
        np.array([
            [_DummyScalar(1), _DummyScalar(2)],
            [_DummyScalar(2), _DummyScalar(4)]]))
    assert result.dtype == object
    assert result.shape == (2,)
    assert result[0] == _DummyScalar(3)
    assert result[1] == _DummyScalar(6)

    # len(out_shape) != 0
    result = map_subarrays(
        lambda x: np.array([x, 2*x]), (), (2,), np.array([1, 2]))
    assert result.dtype == int
    assert np.all(result == np.array([[1, 2], [2, 4]]))

    # len(out_shape) != 0, nested
    result = map_subarrays(
        lambda x: np.array([x, 2*x]), (), (2,), np.array([1, 2]), return_nested=True)
    assert result.dtype == object
    assert result.shape == (2,)
    assert np.all(result[0] == np.array([1, 2]))
    assert np.all(result[1] == np.array([2, 4]))

    # len(out_shape) != 0, non-numerical scalars
    result = map_subarrays(
        lambda x: np.array([_DummyScalar(x), _DummyScalar(2*x)]), (), (2,),
        np.array([1, 2]))
    assert result.dtype == object
    assert result.shape == (2, 2)
    assert np.all(result[0] == np.array([_DummyScalar(1), _DummyScalar(2)]))
    assert np.all(result[1] == np.array([_DummyScalar(2), _DummyScalar(4)]))

    # Zero-size input array
    result = map_subarrays(
        lambda x: np.array([x, 2*x]), (), (2,), np.empty((2, 0)))
    assert result.dtype == object
    assert result.shape == (2, 0, 2)

    # Zero-size input array, nested
    result = map_subarrays(
        lambda x: np.array([x, 2*x]), (), (2,), np.empty((2, 0)),
        return_nested=True)
    assert result.dtype == object
    assert result.shape == (2, 0)


def test_rec_map_subarrays():
    """Test rec_map_subarrays."""
    from grudge.tools import rec_map_subarrays

    # Scalar
    result = rec_map_subarrays(
        lambda x: np.array([x, 2*x]), (), (2,), 1)
    assert result.dtype == int
    assert np.all(result == np.array([1, 2]))

    # Scalar, non-numerical
    result = rec_map_subarrays(
        lambda x: np.array([x.val, 2*x.val]), (), (2,), _DummyScalar(1),
        scalar_cls=_DummyScalar)
    assert result.dtype == int
    assert np.all(result == np.array([1, 2]))

    # Array of scalars
    result = rec_map_subarrays(
        np.sum, (2,), (), np.array([[1, 2], [2, 4]]))
    assert result.dtype == int
    assert np.all(result == np.array([3, 6]))

    # Array of scalars, non-numerical
    result = rec_map_subarrays(
        lambda x: x[0].val + x[1].val, (2,), (),
        np.array([
            [_DummyScalar(1), _DummyScalar(2)],
            [_DummyScalar(2), _DummyScalar(4)]]),
        scalar_cls=_DummyScalar)
    assert result.dtype == int
    assert np.all(result == np.array([3, 6]))

    # Array container
    result = rec_map_subarrays(
        np.sum, (2,), (), make_obj_array([np.array([1, 2]), np.array([2, 4])]))
    assert result.dtype == object
    assert result[0] == 3
    assert result[1] == 6

    # Array container, non-numerical scalars
    result = rec_map_subarrays(
        lambda x: x[0].val + x[1], (2,), (),
        make_obj_array([
            np.array([_DummyScalar(1), 2]),
            np.array([_DummyScalar(2), 4])]),
        scalar_cls=_DummyScalar)
    assert result.dtype == object
    assert result[0] == 3
    assert result[1] == 6

# }}}


# You can test individual routines by typing
# $ python test_tools.py 'test_routine()'

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        pytest.main([__file__])

# vim: fdm=marker
