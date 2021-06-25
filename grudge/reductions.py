"""
.. currentmodule:: grudge.op

Nodal Reductions
----------------

.. note::

    In a distributed-memory setting, these reductions automatically
    reduce over all ranks involved, and return the same value on
    all ranks, in the manner of an MPI ``allreduce``.

.. autofunction:: norm
.. autofunction:: nodal_sum
.. autofunction:: nodal_min
.. autofunction:: nodal_max
.. autofunction:: integral

Rank-local reductions
----------------------

.. autofunction:: nodal_sum_loc
.. autofunction:: nodal_min_loc
.. autofunction:: nodal_max_loc

Elementwise reductions
----------------------

.. autofunction:: elementwise_sum
.. autofunction:: elementwise_integral
"""

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


from numbers import Number
from functools import reduce

from arraycontext import (
    ArrayContext,
    make_loopy_program
)

from grudge.discretization import DiscretizationCollection

from pytools import memoize_in
from pytools.obj_array import obj_array_vectorize

from meshmode.dof_array import DOFArray

import numpy as np
import grudge.dof_desc as dof_desc


# {{{ Nodal reductions

def _norm(dcoll: DiscretizationCollection, vec, p, dd):
    if isinstance(vec, Number):
        return np.fabs(vec)
    if p == 2:
        from grudge.op import _apply_mass_operator
        return np.real_if_close(np.sqrt(
            nodal_sum(
                dcoll,
                dd,
                vec.conj() * _apply_mass_operator(dcoll, dd, dd, vec)
            )
        ))
    elif p == np.inf:
        return nodal_max(dcoll, dd, abs(vec))
    else:
        raise NotImplementedError("Unsupported value of p")


def norm(dcoll: DiscretizationCollection, vec, p, dd=None) -> float:
    r"""Return the vector p-norm of a function represented
    by its vector of degrees of freedom *vec*.

    :arg vec: a :class:`~meshmode.dof_array.DOFArray` or an object array of
        a :class:`~meshmode.dof_array.DOFArray`\ s,
        where the last axis of the array must have length
        matching the volume dimension.
    :arg p: an integer denoting the order of the integral norm. Currently,
        only values of 2 or `numpy.inf` are supported.
    :arg dd: a :class:`~grudge.dof_desc.DOFDesc`, or a value convertible to one.
        Defaults to the base volume discretization if not provided.
    :returns: a nonegative scalar denoting the norm.
    """
    if dd is None:
        dd = dof_desc.DD_VOLUME

    dd = dof_desc.as_dofdesc(dd)

    if isinstance(vec, np.ndarray):
        if p == 2:
            return sum(
                norm(dcoll, vec[idx], p, dd=dd)**2
                for idx in np.ndindex(vec.shape)
            )**0.5
        elif p == np.inf:
            return max(
                norm(dcoll, vec[idx], np.inf, dd=dd)
                for idx in np.ndindex(vec.shape)
            )
        else:
            raise ValueError("unsupported norm order")

    return _norm(dcoll, vec, p, dd)


def nodal_sum(dcoll: DiscretizationCollection, dd, vec) -> float:
    r"""Return the nodal sum of a vector of degrees of freedom *vec*.

    :arg dd: a :class:`~grudge.dof_desc.DOFDesc`, or a value
        convertible to one.
    :arg vec: a :class:`~meshmode.dof_array.DOFArray`.
    :returns: a scalar denoting the nodal sum.
    """
    comm = dcoll.mpi_communicator
    if comm is None:
        return nodal_sum_loc(dcoll, dd, vec)

    # NOTE: Don't move this
    from mpi4py import MPI
    actx = vec.array_context

    return comm.allreduce(actx.to_numpy(nodal_sum_loc(dcoll, dd, vec)), op=MPI.SUM)


def nodal_sum_loc(dcoll: DiscretizationCollection, dd, vec) -> float:
    r"""Return the rank-local nodal sum of a vector of degrees of freedom *vec*.

    :arg dd: a :class:`~grudge.dof_desc.DOFDesc`, or a value
        convertible to one.
    :arg vec: a :class:`~meshmode.dof_array.DOFArray`.
    :returns: a scalar denoting the rank-local nodal sum.
    """
    actx = vec.array_context
    return sum([actx.np.sum(grp_ary) for grp_ary in vec])


def nodal_min(dcoll: DiscretizationCollection, dd, vec) -> float:
    r"""Return the nodal minimum of a vector of degrees of freedom *vec*.

    :arg dd: a :class:`~grudge.dof_desc.DOFDesc`, or a value
        convertible to one.
    :arg vec: a :class:`~meshmode.dof_array.DOFArray`.
    :returns: a scalar denoting the nodal minimum.
    """
    comm = dcoll.mpi_communicator
    if comm is None:
        return nodal_min_loc(dcoll, dd, vec)

    # NOTE: Don't move this
    from mpi4py import MPI
    actx = vec.array_context

    return comm.allreduce(actx.to_numpy(nodal_min_loc(dcoll, dd, vec)), op=MPI.MIN)


def nodal_min_loc(dcoll: DiscretizationCollection, dd, vec) -> float:
    r"""Return the rank-local nodal minimum of a vector of degrees
    of freedom *vec*.

    :arg dd: a :class:`~grudge.dof_desc.DOFDesc`, or a value
        convertible to one.
    :arg vec: a :class:`~meshmode.dof_array.DOFArray`.
    :returns: a scalar denoting the rank-local nodal minimum.
    """
    actx = vec.array_context
    return reduce(lambda acc, grp_ary: actx.np.minimum(acc, actx.np.min(grp_ary)),
                  vec, np.inf)


def nodal_max(dcoll: DiscretizationCollection, dd, vec) -> float:
    r"""Return the nodal maximum of a vector of degrees of freedom *vec*.

    :arg dd: a :class:`~grudge.dof_desc.DOFDesc`, or a value
        convertible to one.
    :arg vec: a :class:`~meshmode.dof_array.DOFArray`.
    :returns: a scalar denoting the nodal maximum.
    """
    comm = dcoll.mpi_communicator
    if comm is None:
        return nodal_max_loc(dcoll, dd, vec)

    # NOTE: Don't move this
    from mpi4py import MPI
    actx = vec.array_context

    return comm.allreduce(actx.to_numpy(nodal_max_loc(dcoll, dd, vec)), op=MPI.MAX)


def nodal_max_loc(dcoll: DiscretizationCollection, dd, vec) -> float:
    r"""Return the rank-local nodal maximum of a vector of degrees
    of freedom *vec*.

    :arg dd: a :class:`~grudge.dof_desc.DOFDesc`, or a value
        convertible to one.
    :arg vec: a :class:`~meshmode.dof_array.DOFArray`.
    :returns: a scalar denoting the rank-local nodal maximum.
    """
    actx = vec.array_context
    return reduce(lambda acc, grp_ary: actx.np.maximum(acc, actx.np.max(grp_ary)),
                  vec, -np.inf)


def integral(dcoll: DiscretizationCollection, dd, vec) -> float:
    """Numerically integrates a function represented by a
    :class:`~meshmode.dof_array.DOFArray` of degrees of freedom.

    :arg dd: a :class:`~grudge.dof_desc.DOFDesc`, or a value convertible to one.
    :arg vec: a :class:`~meshmode.dof_array.DOFArray`
    :returns: a scalar denoting the evaluated integral.
    """
    from grudge.op import _apply_mass_operator

    dd = dof_desc.as_dofdesc(dd)

    ones = dcoll.discr_from_dd(dd).zeros(vec.array_context) + 1.0
    return nodal_sum(
        dcoll, dd, vec * _apply_mass_operator(dcoll, dd, dd, ones)
    )

# }}}


# {{{  Elementwise reductions

def _map_elementwise_reduction(actx: ArrayContext, op_name):
    @memoize_in(actx, (_map_elementwise_reduction,
                       "elementwise_%s_prg" % op_name))
    def prg():
        return make_loopy_program(
            [
                "{[iel]: 0 <= iel < nelements}",
                "{[idof, jdof]: 0 <= idof, jdof < ndofs}"
            ],
            """
                result[iel, idof] = %s(jdof, operand[iel, jdof])
            """ % op_name,
            name="grudge_elementwise_%s_knl" % op_name
        )
    return prg()


def elementwise_sum(dcoll: DiscretizationCollection, *args) -> DOFArray:
    r"""Returns a vector of DOFs with all entries on each element set
    to the sum of DOFs on that element.

    May be called with ``(dcoll, vec)`` or ``(dcoll, dd, vec)``.

    :arg dd: a :class:`~grudge.dof_desc.DOFDesc`, or a value convertible to one.
        Defaults to the base volume discretization if not provided.
    :arg vec: a :class:`~meshmode.dof_array.DOFArray`
    :returns: a :class:`~meshmode.dof_array.DOFArray` whose entries
        denote the element-wise sum of *vec*.
    """

    if len(args) == 1:
        vec, = args
        dd = dof_desc.DOFDesc("vol", dof_desc.DISCR_TAG_BASE)
    elif len(args) == 2:
        dd, vec = args
    else:
        raise TypeError("invalid number of arguments")

    dd = dof_desc.as_dofdesc(dd)

    if isinstance(vec, np.ndarray):
        return obj_array_vectorize(
            lambda vi: elementwise_sum(dcoll, dd, vi), vec
        )

    actx = vec.array_context

    return DOFArray(
        actx,
        data=tuple(
            actx.call_loopy(
                _map_elementwise_reduction(actx, "sum"),
                operand=vec_i
            )["result"]
            for vec_i in vec
        )
    )


def elementwise_integral(dcoll: DiscretizationCollection, dd, vec) -> DOFArray:
    """Numerically integrates a function represented by a
    :class:`~meshmode.dof_array.DOFArray` of degrees of freedom in
    each element of a discretization, given by *dd*.

    :arg dd: a :class:`~grudge.dof_desc.DOFDesc`, or a value convertible to one.
    :arg vec: a :class:`~meshmode.dof_array.DOFArray`
    :returns: a :class:`~meshmode.dof_array.DOFArray` containing the
        elementwise integral if *vec*.
    """
    from grudge.op import _apply_mass_operator

    dd = dof_desc.as_dofdesc(dd)

    ones = dcoll.discr_from_dd(dd).zeros(vec.array_context) + 1.0
    return elementwise_sum(
        dcoll, dd, vec * _apply_mass_operator(dcoll, dd, dd, ones)
    )

# }}}


# vim: foldmethod=marker
