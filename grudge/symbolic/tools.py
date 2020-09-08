"""Operator templates: extra bits of functionality."""

__copyright__ = "Copyright (C) 2008 Andreas Kloeckner"

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


# {{{ symbolic operator tools

def is_scalar(expr):
    from grudge import sym
    if isinstance(expr, sym.Variable) and expr.dd.domain_tag is sym.DTAG_SCALAR:
        return True

    return isinstance(expr, (int, float, complex))


def split_sym_operator_for_multirate(state_vector, sym_operator,
        index_groups):
    class IndexGroupKillerSubstMap:
        def __init__(self, kill_set):
            self.kill_set = kill_set

        def __call__(self, expr):
            if expr in kill_set:
                return 0
            else:
                return None

    # make IndexGroupKillerSubstMap that kill everything
    # *except* what's in that index group
    killers = []
    for i in range(len(index_groups)):
        kill_set = set()
        for j in range(len(index_groups)):
            if i != j:
                kill_set |= set(index_groups[j])

        killers.append(IndexGroupKillerSubstMap(kill_set))

    from grudge.symbolic import \
            SubstitutionMapper, \
            CommutativeConstantFoldingMapper

    return [
            CommutativeConstantFoldingMapper()(
                SubstitutionMapper(killer)(
                    sym_operator[ig]))
            for ig in index_groups
            for killer in killers]


def ptwise_mul(a, b):
    from pytools.obj_array import log_shape
    a_log_shape = log_shape(a)
    b_log_shape = log_shape(b)

    from pytools import indices_in_shape

    if a_log_shape == ():
        result = np.empty(b_log_shape, dtype=object)
        for i in indices_in_shape(b_log_shape):
            result[i] = a*b[i]
    elif b_log_shape == ():
        result = np.empty(a_log_shape, dtype=object)
        for i in indices_in_shape(a_log_shape):
            result[i] = a[i]*b
    else:
        raise ValueError("ptwise_mul can't handle two non-scalars")

    return result


def ptwise_dot(logdims1, logdims2, a1, a2):
    a1_log_shape = a1.shape[:logdims1]
    a2_log_shape = a1.shape[:logdims2]

    assert a1_log_shape[-1] == a2_log_shape[0]
    len_k = a2_log_shape[0]

    result = np.empty(a1_log_shape[:-1]+a2_log_shape[1:], dtype=object)

    from pytools import indices_in_shape
    for a1_i in indices_in_shape(a1_log_shape[:-1]):
        for a2_i in indices_in_shape(a2_log_shape[1:]):
            result[a1_i+a2_i] = sum(
                    a1[a1_i+(k,)] * a2[(k,)+a2_i]
                    for k in range(len_k)
                    )

    if result.shape == ():
        return result[()]
    else:
        return result

# }}}


# {{{ pretty printing

def pretty(sym_operator):
    from grudge.symbolic.mappers import PrettyStringifyMapper

    stringify_mapper = PrettyStringifyMapper()
    from pymbolic.mapper.stringifier import PREC_NONE
    result = stringify_mapper(sym_operator, PREC_NONE)

    splitter = "="*75 + "\n"

    cse_strs = stringify_mapper.get_cse_strings()
    if cse_strs:
        result = "\n".join(cse_strs)+"\n"+splitter+result

    return result

# }}}


# vim: foldmethod=marker
