"""
.. autofunction:: build_jacobian
.. autofunction:: map_subarrays
.. autofunction:: rec_map_subarrays
"""

from __future__ import annotations


__copyright__ = "Copyright (C) 2007 Andreas Kloeckner"

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

from functools import partial
from typing import Any, Callable

import numpy as np

from arraycontext import ArrayContext, ArrayOrContainer, ArrayOrContainerT
from pytools import product


# {{{ build_jacobian

def build_jacobian(
        actx: ArrayContext,
        f: Callable[[ArrayOrContainerT], ArrayOrContainerT],
        base_state: ArrayOrContainerT,
        stepsize: float) -> np.ndarray:
    """Returns a Jacobian matrix of *f* determined by a one-sided finite
    difference approximation with *stepsize*.

    :arg f: a callable with a single argument, any array or
        :class:`arraycontext.ArrayContainer` supported by *actx*.
    :arg base_state: The point at which the Jacobian is to be
        calculated. May be any array or :class:`arraycontext.ArrayContainer`
        supported by *actx*.
    :returns: a two-dimensional :class:`numpy.ndarray`.
    """
    from arraycontext import flatten, unflatten
    flat_base_state = flatten(base_state, actx)

    n, = flat_base_state.shape

    mat = np.empty((n, n), dtype=flat_base_state.dtype)

    f_base = f(base_state)

    for i in range(n):
        unit_i_flat = np.zeros(n, dtype=mat.dtype)
        unit_i_flat[i] = stepsize

        f_unit_i = f(base_state + unflatten(
            base_state, actx.from_numpy(unit_i_flat), actx))

        mat[:, i] = actx.to_numpy(flatten((f_unit_i - f_base) / stepsize, actx))

    return mat

# }}}


# {{{ map_subarrays

def map_subarrays(
        f: Callable[[Any], Any],
        in_shape: tuple[int, ...], out_shape: tuple[int, ...],
        ary: Any, *, return_nested: bool = False) -> Any:
    """
    Apply a function *f* to subarrays of shape *in_shape* of an
    :class:`numpy.ndarray`, typically (but not necessarily) of dtype
    :class:`object`. Return an :class:`numpy.ndarray` with the corresponding
    subarrays replaced by the return values of *f*, and with the shape adapted
    to reflect *out_shape*.

    Similar to :class:`numpy.vectorize`.

    *Example 1:* given a function *f* that maps arrays of shape ``(2, 2)`` to scalars
    and an input array *ary* of shape ``(3, 2, 2)``, the call::

        map_subarrays(f, (2, 2), (), ary)

    will produce an array of shape ``(3,)`` containing the results of calling *f* on
    the 3 subarrays of shape ``(2, 2)`` in *ary*.

    *Example 2:* given a function *f* that maps arrays of shape ``(2,)`` to arrays of
    shape ``(2, 2)`` and an input array *ary* of shape ``(3, 2)``, the call::

        map_subarrays(f, (2,), (2, 2), ary)

    will produce an array of shape ``(3, 2, 2)`` containing the results of calling
    *f* on the 3 subarrays of shape ``(2,)`` in *ary*. The call::

        map_subarrays(f, (2,), (2, 2), ary, return_nested=True)

    will instead produce an array of shape ``(3,)`` with each entry containing an
    array of shape ``(2, 2)``.

    :arg f: the function to be called.
    :arg in_shape: the shape of any inputs to *f*.
    :arg out_shape: the shape of the result of calling *f* on an array of shape
        *in_shape*.
    :arg ary: a :class:`numpy.ndarray` instance.
    :arg return_nested: if *out_shape* is nontrivial, this flag indicates whether
        to return a nested array (containing one entry for each application of *f*),
        or to return a single, higher-dimensional array.

    :returns: an array with the subarrays of shape *in_shape* replaced with
        subarrays of shape *out_shape* resulting from the application of *f*.
    """
    if not isinstance(ary, np.ndarray):
        if len(in_shape) != 0:
            raise ValueError(f"found scalar, expected array of shape {in_shape}")
        return f(ary)
    else:
        if (
                ary.ndim < len(in_shape)
                or ary.shape[ary.ndim-len(in_shape):] != in_shape):
            raise ValueError(
                f"array of shape {ary.shape} is incompatible with function "
                f"expecting input shape {in_shape}")
        base_shape = ary.shape[:ary.ndim-len(in_shape)]
        if len(base_shape) == 0:
            return f(ary)
        elif product(base_shape) == 0:
            if return_nested:
                return np.empty(base_shape, dtype=object)
            else:
                return np.empty(base_shape + out_shape, dtype=object)
        else:
            in_slice = tuple(slice(0, n) for n in in_shape)
            result_entries = np.empty(base_shape, dtype=object)
            for idx in np.ndindex(base_shape):
                result_entries[idx] = f(ary[idx + in_slice])
            if len(out_shape) == 0:
                out_entry_template = result_entries.flat[0]
                if np.isscalar(out_entry_template):
                    return result_entries.astype(type(out_entry_template))
                else:
                    return result_entries
            else:
                if return_nested:
                    return result_entries
                else:
                    out_slice = tuple(slice(0, n) for n in out_shape)
                    out_entry_template = result_entries.flat[0]
                    result = np.empty(
                        base_shape + out_shape, dtype=out_entry_template.dtype)
                    for idx in np.ndindex(base_shape):
                        result[idx + out_slice] = result_entries[idx]
                    return result

# }}}


# {{{ rec_map_subarrays

def rec_map_subarrays(
        f: Callable[[Any], Any],
        in_shape: tuple[int, ...],
        out_shape: tuple[int, ...],
        ary: ArrayOrContainer, *,
        scalar_cls: type | tuple[type] | None = None,
        return_nested: bool = False) -> ArrayOrContainer:
    r"""
    Like :func:`map_subarrays`, but with support for
    :class:`arraycontext.ArrayContainer`\ s.

    :arg scalar_cls: An array container of this type will be considered a scalar
        and arrays of it will be passed to *f* without further destructuring.
    """
    if scalar_cls is not None:
        def is_scalar(x):
            return np.isscalar(x) or isinstance(x, scalar_cls)
    else:
        def is_scalar(x):
            return np.isscalar(x)

    def is_array_of_scalars(a):
        return (
            isinstance(a, np.ndarray)
            and (
                a.dtype != object
                or all(is_scalar(a[idx]) for idx in np.ndindex(a.shape))))

    if is_scalar(ary) or is_array_of_scalars(ary):
        return map_subarrays(
            f, in_shape, out_shape, ary, return_nested=return_nested)
    else:
        from arraycontext import map_array_container
        return map_array_container(
            partial(
                rec_map_subarrays, f, in_shape, out_shape, scalar_cls=scalar_cls,
                return_nested=return_nested),
            ary)

# }}}

# vim: foldmethod=marker
