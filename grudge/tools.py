"""Miscellaneous helper facilities."""

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

import numpy as np
from pytools import levi_civita


def is_zero(x):
    # DO NOT try to replace this with an attempted "== 0" comparison.
    # This will become an elementwise numpy operation and not do what
    # you want.

    if np.isscalar(x):
        return x == 0
    else:
        return False


# {{{ SubsettableCrossProduct

class SubsettableCrossProduct:
    """A cross product that can operate on an arbitrary subsets of its
    two operands and return an arbitrary subset of its result.
    """

    full_subset = (True, True, True)

    def __init__(self, op1_subset=full_subset, op2_subset=full_subset,
            result_subset=full_subset):
        """Construct a subset-able cross product.
        :param op1_subset: The subset of indices of operand 1 to be taken into
            account.  Given as a 3-sequence of bools.
        :param op2_subset: The subset of indices of operand 2 to be taken into
            account.  Given as a 3-sequence of bools.
        :param result_subset: The subset of indices of the result that are
            calculated.  Given as a 3-sequence of bools.
        """
        def subset_indices(subset):
            return [i for i, use_component in enumerate(subset)
                    if use_component]

        self.op1_subset = op1_subset
        self.op2_subset = op2_subset
        self.result_subset = result_subset

        import pymbolic
        op1 = pymbolic.var("x")
        op2 = pymbolic.var("y")

        self.functions = []
        self.component_lcjk = []
        for i, use_component in enumerate(result_subset):
            if use_component:
                this_expr = 0
                this_component = []
                for j, j_real in enumerate(subset_indices(op1_subset)):
                    for k, k_real in enumerate(subset_indices(op2_subset)):
                        lc = levi_civita((i, j_real, k_real))
                        if lc != 0:
                            this_expr += lc*op1.index(j)*op2.index(k)
                            this_component.append((lc, j, k))
                self.functions.append(pymbolic.compile(this_expr,
                    variables=[op1, op2]))
                self.component_lcjk.append(this_component)

    def __call__(self, x, y, three_mult=None):
        """Compute the subsetted cross product on the indexables *x* and *y*.
        :param three_mult: a function of three arguments *sign, xj, yk*
          used in place of the product *sign*xj*yk*. Defaults to just this
          product if not given.
        """
        from pytools.obj_array import flat_obj_array
        if three_mult is None:
            return flat_obj_array(*[f(x, y) for f in self.functions])
        else:
            return flat_obj_array(
                    *[sum(three_mult(lc, x[j], y[k]) for lc, j, k in lcjk)
                    for lcjk in self.component_lcjk])


cross = SubsettableCrossProduct()

# }}}


def count_subset(subset):
    from pytools import len_iterable
    return len_iterable(uc for uc in subset if uc)


def partial_to_all_subset_indices(subsets, base=0):
    """Takes a sequence of bools and generates it into an array of indices
    to be used to insert the subset into the full set.
    Example:
    >>> list(partial_to_all_subset_indices([[False, True, True], [True,False,True]]))
    [array([0 1]), array([2 3]
    """

    idx = base
    for subset in subsets:
        result = []
        for is_in in subset:
            if is_in:
                result.append(idx)
                idx += 1

        yield np.array(result, dtype=np.intp)


# {{{ OrderedSet

# Source: https://code.activestate.com/recipes/576694-orderedset/
# Author: Raymond Hettinger
# License: MIT

try:
    from collections.abc import MutableSet
except ImportError:
    from collections import MutableSet


class OrderedSet(MutableSet):

    def __init__(self, iterable=None):
        self.end = end = []
        end += [None, end, end]         # sentinel node for doubly linked list
        self.map = {}                   # key --> [key, prev, next]
        if iterable is not None:
            self |= iterable

    def __len__(self):
        return len(self.map)

    def __contains__(self, key):
        return key in self.map

    def add(self, key):
        if key not in self.map:
            end = self.end
            curr = end[1]
            curr[2] = end[1] = self.map[key] = [key, curr, end]

    def discard(self, key):
        if key in self.map:
            key, prev, next = self.map.pop(key)
            prev[2] = next
            next[1] = prev

    def __iter__(self):
        end = self.end
        curr = end[2]
        while curr is not end:
            yield curr[0]
            curr = curr[2]

    def __reversed__(self):
        end = self.end
        curr = end[1]
        while curr is not end:
            yield curr[0]
            curr = curr[1]

    def pop(self, last=True):
        if not self:
            raise KeyError("set is empty")
        key = self.end[1][0] if last else self.end[2][0]
        self.discard(key)
        return key

    def __repr__(self):
        if not self:
            return f"{self.__class__.__name__}()"
        return "{}({!r})".format(self.__class__.__name__, list(self))

    def __eq__(self, other):
        if isinstance(other, OrderedSet):
            return len(self) == len(other) and list(self) == list(other)
        return set(self) == set(other)

# }}}
