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
.. autofunction:: elementwise_max
.. autofunction:: elementwise_min
.. autofunction:: elementwise_integral
"""

from __future__ import annotations


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

from functools import partial, reduce

import numpy as np

from arraycontext import (
    ArrayOrContainer,
    Scalar,
    make_loopy_program,
    map_array_container,
    serialize_container,
    tag_axes,
)
from meshmode.dof_array import DOFArray
from meshmode.transform_metadata import (
    DiscretizationDOFAxisTag,
    DiscretizationElementAxisTag,
)
from pymbolic import Number, RealNumber
from pytools import memoize_in

import grudge.dof_desc as dof_desc
from grudge.array_context import MPIBasedArrayContext
from grudge.discretization import DiscretizationCollection


# {{{ Nodal reductions

def norm(dcoll: DiscretizationCollection, vec, p, dd=None) -> RealNumber:
    r"""Return the vector p-norm of a function represented
    by its vector of degrees of freedom *vec*.

    :arg vec: a :class:`~meshmode.dof_array.DOFArray` or an
        :class:`~arraycontext.ArrayContainer` of them.
    :arg p: an integer denoting the order of the integral norm. Currently,
        only values of 2 or `numpy.inf` are supported.
    :arg dd: a :class:`~grudge.dof_desc.DOFDesc`, or a value convertible to one.
        Defaults to the base volume discretization if not provided.
    :returns: a nonegative scalar denoting the norm.
    """
    if dd is None:
        dd = dof_desc.DD_VOLUME_ALL

    from arraycontext import get_container_context_recursively
    actx = get_container_context_recursively(vec)

    assert actx is not None

    dd = dof_desc.as_dofdesc(dd)

    if p == 2:
        from grudge.op import _apply_mass_operator
        return actx.np.sqrt(
            actx.np.abs(
                nodal_sum(
                    dcoll, dd,
                    actx.np.conjugate(vec)
                    * _apply_mass_operator(dcoll, dd, dd, vec))))
    elif p == np.inf:
        return nodal_max(dcoll, dd, actx.np.abs(vec), initial=0.)
    else:
        raise ValueError("unsupported norm order")


def nodal_sum(dcoll: DiscretizationCollection, dd, vec) -> Number:
    r"""Return the nodal sum of a vector of degrees of freedom *vec*.

    :arg dd: a :class:`~grudge.dof_desc.DOFDesc`, or a value
        convertible to one.
    :arg vec: a :class:`~meshmode.dof_array.DOFArray` or an
        :class:`~arraycontext.ArrayContainer`.
    :returns: a device scalar denoting the nodal sum.
    """
    from arraycontext import get_container_context_recursively
    actx = get_container_context_recursively(vec)

    if not isinstance(actx, MPIBasedArrayContext):
        return nodal_sum_loc(dcoll, dd, vec)

    comm = actx.mpi_communicator

    # NOTE: Do not move, we do not want to import mpi4py in single-rank computations
    from mpi4py import MPI

    return actx.from_numpy(
        comm.allreduce(actx.to_numpy(nodal_sum_loc(dcoll, dd, vec)), op=MPI.SUM))


def nodal_sum_loc(dcoll: DiscretizationCollection, dd, vec) -> Number:
    r"""Return the rank-local nodal sum of a vector of degrees of freedom *vec*.

    :arg dd: a :class:`~grudge.dof_desc.DOFDesc`, or a value
        convertible to one.
    :arg vec: a :class:`~meshmode.dof_array.DOFArray` or an
        :class:`~arraycontext.ArrayContainer` of them.
    :returns: a scalar denoting the rank-local nodal sum.
    """
    if not isinstance(vec, DOFArray):
        return sum(
            nodal_sum_loc(dcoll, dd, comp)
            for _, comp in serialize_container(vec)
        )

    actx = vec.array_context

    return sum(
        actx.np.sum(grp_ary) if grp_ary.size else actx.from_numpy(np.array(0.))
        for grp_ary in vec)


def nodal_min(dcoll: DiscretizationCollection, dd, vec, *, initial=None) -> RealNumber:
    r"""Return the nodal minimum of a vector of degrees of freedom *vec*.

    :arg dd: a :class:`~grudge.dof_desc.DOFDesc`, or a value
        convertible to one.
    :arg vec: a :class:`~meshmode.dof_array.DOFArray` or an
        :class:`~arraycontext.ArrayContainer` of them.
    :arg initial: an optional initial value. Defaults to `numpy.inf`.
    :returns: a device scalar denoting the nodal minimum.
    """
    from arraycontext import get_container_context_recursively
    actx = get_container_context_recursively(vec)

    if not isinstance(actx, MPIBasedArrayContext):
        return nodal_min_loc(dcoll, dd, vec, initial=initial)

    comm = actx.mpi_communicator

    # NOTE: Do not move, we do not want to import mpi4py in single-rank computations
    from mpi4py import MPI

    return actx.from_numpy(
        comm.allreduce(
            actx.to_numpy(nodal_min_loc(dcoll, dd, vec, initial=initial)),
            op=MPI.MIN))


def nodal_min_loc(
        dcoll: DiscretizationCollection, dd, vec, *, initial=None) -> RealNumber:
    r"""Return the rank-local nodal minimum of a vector of degrees
    of freedom *vec*.

    :arg dd: a :class:`~grudge.dof_desc.DOFDesc`, or a value
        convertible to one.
    :arg vec: a :class:`~meshmode.dof_array.DOFArray` or an
        :class:`~arraycontext.ArrayContainer` of them.
    :arg initial: an optional initial value. Defaults to `numpy.inf`.
    :returns: a scalar denoting the rank-local nodal minimum.
    """
    if not isinstance(vec, DOFArray):
        return np.min([
            nodal_min_loc(dcoll, dd, comp, initial=initial)
            for _, comp in serialize_container(vec)
        ])

    actx = vec.array_context

    if initial is None:
        initial = np.inf

    if np.isscalar(initial):
        initial = actx.from_numpy(np.array(initial))

    return reduce(
            lambda acc, grp_ary: actx.np.minimum(
                acc,
                actx.np.min(grp_ary) if grp_ary.size else initial),
            vec, initial)


def nodal_max(dcoll: DiscretizationCollection, dd, vec, *, initial=None) -> RealNumber:
    r"""Return the nodal maximum of a vector of degrees of freedom *vec*.

    :arg dd: a :class:`~grudge.dof_desc.DOFDesc`, or a value
        convertible to one.
    :arg vec: a :class:`~meshmode.dof_array.DOFArray` or an
        :class:`~arraycontext.ArrayContainer` of them.
    :arg initial: an optional initial value. Defaults to `-numpy.inf`.
    :returns: a device scalar denoting the nodal maximum.
    """
    from arraycontext import get_container_context_recursively
    actx = get_container_context_recursively(vec)

    if not isinstance(actx, MPIBasedArrayContext):
        return nodal_max_loc(dcoll, dd, vec, initial=initial)

    comm = actx.mpi_communicator

    # NOTE: Do not move, we do not want to import mpi4py in single-rank computations
    from mpi4py import MPI

    return actx.from_numpy(
        comm.allreduce(
            actx.to_numpy(nodal_max_loc(dcoll, dd, vec, initial=initial)),
            op=MPI.MAX))


def nodal_max_loc(
        dcoll: DiscretizationCollection, dd, vec, *, initial=None) -> RealNumber:
    r"""Return the rank-local nodal maximum of a vector of degrees
    of freedom *vec*.

    :arg dd: a :class:`~grudge.dof_desc.DOFDesc`, or a value
        convertible to one.
    :arg vec: a :class:`~meshmode.dof_array.DOFArray` or an
        :class:`~arraycontext.ArrayContainer`.
    :arg initial: an optional initial value. Defaults to `-numpy.inf`.
    :returns: a scalar denoting the rank-local nodal maximum.
    """
    if not isinstance(vec, DOFArray):
        return np.max([
            nodal_max_loc(dcoll, dd, comp, initial=initial)
            for _, comp in serialize_container(vec)
        ])

    actx = vec.array_context

    if initial is None:
        initial = -np.inf

    if np.isscalar(initial):
        initial = actx.from_numpy(np.array(initial))

    return reduce(
            lambda acc, grp_ary: actx.np.maximum(
                acc,
                actx.np.max(grp_ary) if grp_ary.size > 0 else initial),
            vec, initial)


def integral(dcoll: DiscretizationCollection, dd, vec) -> Scalar:
    """Numerically integrates a function represented by a
    :class:`~meshmode.dof_array.DOFArray` of degrees of freedom.

    :arg dd: a :class:`~grudge.dof_desc.DOFDesc`, or a value convertible to one.
    :arg vec: a :class:`~meshmode.dof_array.DOFArray` or an
        :class:`~arraycontext.ArrayContainer` of them.
    :returns: a device scalar denoting the evaluated integral.
    """
    dd = dof_desc.as_dofdesc(dd)
    discr = dcoll.discr_from_dd(dd)

    from grudge.geometry import area_element
    actx = vec.array_context
    area_elements = area_element(
        actx, dcoll, dd=dd,
        _use_geoderiv_connection=actx.supports_nonscalar_broadcasting)
    qwts = DOFArray(
        actx,
        data=tuple(
            actx.from_numpy(
                np.tile(grp.quadrature_rule().weights,
                        (ae_i.shape[0], 1)))*ae_i
            for grp, ae_i in zip(discr.groups, area_elements))
    )
    return nodal_sum(dcoll, dd, vec*qwts)

# }}}


# {{{  Elementwise reductions

def _apply_elementwise_reduction(
        op_name: str, dcoll: DiscretizationCollection,
        *args) -> ArrayOrContainer:
    r"""Returns an array container whose entries contain
    the elementwise reductions in each cell.

    May be called with ``(vec)`` or ``(dd, vec)``.

    Note that for array contexts which support nonscalar broadcasting
    (e.g. :class:`meshmode.array_context.PytatoPyOpenCLArrayContext`),
    the size of each component vector will be of shape ``(nelements, 1)``.
    Otherwise, the scalar value of the reduction will be repeated for each
    degree of freedom.

    :arg dd: a :class:`~grudge.dof_desc.DOFDesc`, or a value convertible to one.
        Defaults to the base volume discretization if not provided.
    :arg vec: a :class:`~meshmode.dof_array.DOFArray` or an
        :class:`~arraycontext.ArrayContainer`.
    :returns: a :class:`~meshmode.dof_array.DOFArray` or an
        :class:`~arraycontext.ArrayContainer`.
    """
    if len(args) == 1:
        vec, = args
        dd = dof_desc.DD_VOLUME_ALL
    elif len(args) == 2:
        dd, vec = args
    else:
        raise TypeError("invalid number of arguments")

    dd = dof_desc.as_dofdesc(dd)

    if not isinstance(vec, DOFArray):
        return map_array_container(
            partial(_apply_elementwise_reduction, op_name, dcoll, dd), vec
        )

    actx = vec.array_context

    if actx.supports_nonscalar_broadcasting:
        return DOFArray(
            actx,
            data=tuple(
                tag_axes(actx, {
                        0: DiscretizationElementAxisTag(),
                        1: DiscretizationDOFAxisTag()},
                    getattr(actx.np, op_name)(vec_i, axis=1).reshape(-1, 1))
                for vec_i in vec
            )
        )
    else:
        @memoize_in(actx, (_apply_elementwise_reduction, dd,
                        f"elementwise_{op_name}_prg"))
        def elementwise_prg():
            # FIXME: This computes the reduction value redundantly for each
            # output DOF.
            t_unit = make_loopy_program(
                [
                    "{[iel]: 0 <= iel < nelements}",
                    "{[idof, jdof]: 0 <= idof, jdof < ndofs}"
                ],
                f"""
                    result[iel, idof] = {op_name}(jdof, operand[iel, jdof])
                """,
                name=f"grudge_elementwise_{op_name}_knl"
            )
            import loopy as lp
            from meshmode.transform_metadata import (
                ConcurrentDOFInameTag,
                ConcurrentElementInameTag,
            )
            return lp.tag_inames(t_unit, {
                "iel": ConcurrentElementInameTag(),
                "idof": ConcurrentDOFInameTag()})

        return actx.tag_axis(1, DiscretizationDOFAxisTag(),
                DOFArray(
                    actx,
                    data=tuple(
                        actx.call_loopy(elementwise_prg(), operand=vec_i)["result"]
                        for vec_i in vec)))


def elementwise_sum(
        dcoll: DiscretizationCollection, *args) -> ArrayOrContainer:
    r"""Returns a vector of DOFs with all entries on each element set
    to the sum of DOFs on that element.

    May be called with ``(vec)`` or ``(dd, vec)``.

    The input *vec* can either be a :class:`~meshmode.dof_array.DOFArray` or
    an :class:`~arraycontext.ArrayContainer` with
    :class:`~meshmode.dof_array.DOFArray` entries. If the underlying
    array context (see :class:`arraycontext.ArrayContext`) for *vec*
    supports nonscalar broadcasting, all :class:`~meshmode.dof_array.DOFArray`
    entries will contain a single value for each element. Otherwise, the
    entries will have the same number of degrees of freedom as *vec*, but
    set to the same value.

    :arg dd: a :class:`~grudge.dof_desc.DOFDesc`, or a value convertible to one.
        Defaults to the base volume discretization if not provided.
    :arg vec: a :class:`~meshmode.dof_array.DOFArray` or an
        :class:`~arraycontext.ArrayContainer` of them
    :returns: a :class:`~meshmode.dof_array.DOFArray` or an
        :class:`~arraycontext.ArrayContainer` like *vec* whose entries
        denote the element-wise sum of *vec*.
    """
    return _apply_elementwise_reduction("sum", dcoll, *args)


def elementwise_max(
        dcoll: DiscretizationCollection, *args) -> ArrayOrContainer:
    r"""Returns a vector of DOFs with all entries on each element set
    to the maximum over all DOFs on that element.

    May be called with ``(vec)`` or ``(dd, vec)``.

    The input *vec* can either be a :class:`~meshmode.dof_array.DOFArray` or
    an :class:`~arraycontext.ArrayContainer` with
    :class:`~meshmode.dof_array.DOFArray` entries. If the underlying
    array context (see :class:`arraycontext.ArrayContext`) for *vec*
    supports nonscalar broadcasting, all :class:`~meshmode.dof_array.DOFArray`
    entries will contain a single value for each element. Otherwise, the
    entries will have the same number of degrees of freedom as *vec*, but
    set to the same value.

    :arg dcoll: a :class:`grudge.discretization.DiscretizationCollection`.
    :arg dd: a :class:`~grudge.dof_desc.DOFDesc`, or a value convertible to one.
        Defaults to the base volume discretization if not provided.
    :arg vec: a :class:`~meshmode.dof_array.DOFArray` or an
        :class:`~arraycontext.ArrayContainer`.
    :returns: a :class:`~meshmode.dof_array.DOFArray` or an
        :class:`~arraycontext.ArrayContainer` like *vec* whose entries
        denote the element-wise max of *vec*.
    """
    return _apply_elementwise_reduction("max", dcoll, *args)


def elementwise_min(
        dcoll: DiscretizationCollection, *args) -> ArrayOrContainer:
    r"""Returns a vector of DOFs with all entries on each element set
    to the minimum over all DOFs on that element.

    May be called with ``(vec)`` or ``(dd, vec)``.

    The input *vec* can either be a :class:`~meshmode.dof_array.DOFArray` or
    an :class:`~arraycontext.ArrayContainer` with
    :class:`~meshmode.dof_array.DOFArray` entries. If the underlying
    array context (see :class:`arraycontext.ArrayContext`) for *vec*
    supports nonscalar broadcasting, all :class:`~meshmode.dof_array.DOFArray`
    entries will contain a single value for each element. Otherwise, the
    entries will have the same number of degrees of freedom as *vec*, but
    set to the same value.

    :arg dcoll: a :class:`grudge.discretization.DiscretizationCollection`.
    :arg dd: a :class:`~grudge.dof_desc.DOFDesc`, or a value convertible to one.
        Defaults to the base volume discretization if not provided.
    :arg vec: a :class:`~meshmode.dof_array.DOFArray` or an
        :class:`~arraycontext.ArrayContainer` of them.
    :returns: a :class:`~meshmode.dof_array.DOFArray` or an
        :class:`~arraycontext.ArrayContainer` like *vec* whose entries
        denote the element-wise min of *vec*.
    """
    return _apply_elementwise_reduction("min", dcoll, *args)


def elementwise_integral(
        dcoll: DiscretizationCollection, *args) -> ArrayOrContainer:
    """Numerically integrates a function represented by a
    :class:`~meshmode.dof_array.DOFArray` of degrees of freedom in
    each element of a discretization, given by *dd*.

    May be called with ``(vec)`` or ``(dd, vec)``.

    The input *vec* can either be a :class:`~meshmode.dof_array.DOFArray` or
    an :class:`~arraycontext.ArrayContainer` with
    :class:`~meshmode.dof_array.DOFArray` entries. If the underlying
    array context (see :class:`arraycontext.ArrayContext`) for *vec*
    supports nonscalar broadcasting, all :class:`~meshmode.dof_array.DOFArray`
    entries will contain a single value for each element. Otherwise, the
    entries will have the same number of degrees of freedom as *vec*, but
    set to the same value.

    :arg dcoll: a :class:`grudge.discretization.DiscretizationCollection`.
    :arg dd: a :class:`~grudge.dof_desc.DOFDesc`, or a value convertible to one.
        Defaults to the base volume discretization if not provided.
    :arg vec: a :class:`~meshmode.dof_array.DOFArray` or an
        :class:`~arraycontext.ArrayContainer` of them.
    :returns: a :class:`~meshmode.dof_array.DOFArray` or an
        :class:`~arraycontext.ArrayContainer` like *vec* containing the
        elementwise integral if *vec*.
    """
    if len(args) == 1:
        vec, = args
        dd = dof_desc.DD_VOLUME_ALL
    elif len(args) == 2:
        dd, vec = args
    else:
        raise TypeError("invalid number of arguments")

    dd = dof_desc.as_dofdesc(dd)
    discr = dcoll.discr_from_dd(dd)

    from grudge.geometry import area_element
    actx = vec.array_context
    area_elements = area_element(
        actx, dcoll, dd=dd,
        _use_geoderiv_connection=actx.supports_nonscalar_broadcasting)
    qwts = DOFArray(
        actx,
        data=tuple(
            actx.from_numpy(
                np.tile(grp.quadrature_rule().weights,
                        (ae_i.shape[0], 1)))*ae_i
            for grp, ae_i in zip(discr.groups, area_elements))
    )
    return elementwise_sum(dcoll, dd, vec*qwts)

# }}}


# vim: foldmethod=marker
