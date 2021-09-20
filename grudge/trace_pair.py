"""
Trace Pairs
^^^^^^^^^^^

Container class
---------------

.. autoclass:: TracePair

Boundary trace functions
------------------------

.. autofunction:: bdry_trace_pair
.. autofunction:: bv_trace_pair

Interior and cross-rank trace functions
---------------------------------------

.. autofunction:: interior_trace_pairs
.. autofunction:: cross_rank_trace_pairs
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


from arraycontext import (
    ArrayContainer,
    is_array_container,
    with_container_arithmetic,
    dataclass_array_container,
    get_container_context_recursively,
    serialize_container,
    deserialize_container
)

from dataclasses import dataclass

from numbers import Number

from pytools import memoize_on_first_arg
from pytools.obj_array import obj_array_vectorize, make_obj_array

from grudge.discretization import DiscretizationCollection
from grudge.projection import project

from meshmode.dof_array import flatten_to_numpy, unflatten_from_numpy, DOFArray
from meshmode.mesh import BTAG_PARTITION

import numpy as np
import grudge.dof_desc as dof_desc


# {{{ Trace pair container class

@with_container_arithmetic(
    bcast_obj_array=False, eq_comparison=False, rel_comparison=False
)
@dataclass_array_container
@dataclass(init=False, frozen=True)
class TracePair:
    """A container class for data (both interior and exterior restrictions)
    on the boundaries of mesh elements.

    .. attribute:: dd

        an instance of :class:`grudge.dof_desc.DOFDesc` describing the
        discretization on which :attr:`int` and :attr:`ext` live.

    .. autoattribute:: int
    .. autoattribute:: ext
    .. autoattribute:: avg
    .. autoattribute:: diff

    .. automethod:: __getattr__
    .. automethod:: __getitem__
    .. automethod:: __len__

    .. note::

        :class:`TracePair` is currently used both by the symbolic (deprecated)
        and the current interfaces, with symbolic information or concrete data.
    """

    dd: dof_desc.DOFDesc
    interior: ArrayContainer
    exterior: ArrayContainer

    def __init__(self, dd, *, interior, exterior):
        object.__setattr__(self, "dd", dof_desc.as_dofdesc(dd))
        object.__setattr__(self, "interior", interior)
        object.__setattr__(self, "exterior", exterior)

    def __getattr__(self, name):
        """Return a new :class:`TracePair` resulting from executing attribute
        lookup with *name* on :attr:`int` and :attr:`ext`.
        """
        return TracePair(self.dd,
                         interior=getattr(self.interior, name),
                         exterior=getattr(self.exterior, name))

    def __getitem__(self, index):
        """Return a new :class:`TracePair` resulting from executing
        subscripting with *index* on :attr:`int` and :attr:`ext`.
        """
        return TracePair(self.dd,
                         interior=self.interior[index],
                         exterior=self.exterior[index])

    def __len__(self):
        """Return the total number of arrays associated with the
        :attr:`int` and :attr:`ext` restrictions of the :class:`TracePair`.
        Note that both must be the same.
        """
        assert len(self.exterior) == len(self.interior)
        return len(self.exterior)

    @property
    def int(self):
        """A class:`~meshmode.dof_array.DOFArray` or
        :class:`~arraycontext.ArrayContainer` of them representing the
        interior value to be used for the flux.
        """
        return self.interior

    @property
    def ext(self):
        """A class:`~meshmode.dof_array.DOFArray` or
        :class:`~arraycontext.ArrayContainer` of them representing the
        exterior value to be used for the flux.
        """
        return self.exterior

    @property
    def avg(self):
        """A class:`~meshmode.dof_array.DOFArray` or
        :class:`~arraycontext.ArrayContainer` of them representing the
        average of the interior and exterior values.
        """
        return 0.5 * (self.int + self.ext)

    @property
    def diff(self):
        """A class:`~meshmode.dof_array.DOFArray` or
        :class:`~arraycontext.ArrayContainer` of them representing the
        difference (exterior - interior) of the pair values.
        """
        return self.ext - self.int

# }}}


# {{{ Boundary trace pairs

def bdry_trace_pair(
        dcoll: DiscretizationCollection, dd, interior, exterior) -> TracePair:
    """Returns a trace pair defined on the exterior boundary. Input arguments
    are assumed to already be defined on the boundary denoted by *dd*.

    :arg dd: a :class:`~grudge.dof_desc.DOFDesc`, or a value convertible to one,
        which describes the boundary discretization.
    :arg interior: a :class:`~meshmode.dof_array.DOFArray` that contains data
        already on the boundary representing the interior value to be used
        for the flux.
    :arg exterior: a :class:`~meshmode.dof_array.DOFArray` that contains data
        that already lives on the boundary representing the exterior value to
        be used for the flux.
    :returns: a :class:`TracePair` on the boundary.
    """
    return TracePair(dd, interior=interior, exterior=exterior)


def bv_trace_pair(
        dcoll: DiscretizationCollection, dd, interior, exterior) -> TracePair:
    """Returns a trace pair defined on the exterior boundary. The interior
    argument is assumed to be defined on the volume discretization, and will
    therefore be restricted to the boundary *dd* prior to creating a
    :class:`TracePair`.

    :arg dd: a :class:`~grudge.dof_desc.DOFDesc`, or a value convertible to one,
        which describes the boundary discretization.
    :arg interior: a :class:`~meshmode.dof_array.DOFArray` that contains data
        defined in the volume, which will be restricted to the boundary denoted
        by *dd*. The result will be used as the interior value
        for the flux.
    :arg exterior: a :class:`~meshmode.dof_array.DOFArray` that contains data
        that already lives on the boundary representing the exterior value to
        be used for the flux.
    :returns: a :class:`TracePair` on the boundary.
    """
    return bdry_trace_pair(
        dcoll, dd, project(dcoll, "vol", dd, interior), exterior
    )

# }}}


# {{{ Interior trace pairs

def _interior_trace_pair(dcoll: DiscretizationCollection, vec) -> TracePair:
    r"""Return a :class:`TracePair` for the interior faces of
    *dcoll* with a discretization tag specified by *discr_tag*.
    This does not include interior faces on different MPI ranks.

    :arg vec: a :class:`~meshmode.dof_array.DOFArray` or object array of
        :class:`~meshmode.dof_array.DOFArray`\ s.
    :returns: a :class:`TracePair` object.
    """
    i = project(dcoll, "vol", "int_faces", vec)

    def get_opposite_face(el):
        if isinstance(el, Number):
            return el
        else:
            return dcoll.opposite_face_connection()(el)

    e = obj_array_vectorize(get_opposite_face, i)

    return TracePair("int_faces", interior=i, exterior=e)


def interior_trace_pairs(dcoll: DiscretizationCollection, vec) -> list:
    r"""Return a :class:`list` of :class:`TracePair` objects
    defined on the interior faces of *dcoll* and any faces connected to a
    parallel boundary.

    :arg vec: a :class:`~meshmode.dof_array.DOFArray` or object array of
        :class:`~meshmode.dof_array.DOFArray`\ s.
    :returns: a :class:`list` of :class:`TracePair` objects.
    """
    return (
        [_interior_trace_pair(dcoll, vec)]
        + cross_rank_trace_pairs(dcoll, vec)
    )


def interior_trace_pair(dcoll: DiscretizationCollection, vec) -> TracePair:
    from warnings import warn
    warn("`grudge.op.interior_trace_pair` is deprecated and will be dropped "
         "in version 2022.x. Use `grudge.trace_pair.interior_trace_pairs` "
         "instead, which includes contributions from different MPI ranks.",
         DeprecationWarning, stacklevel=2)
    return _interior_trace_pair(dcoll, vec)

# }}}


# {{{ Distributed-memory functionality

@memoize_on_first_arg
def connected_ranks(dcoll: DiscretizationCollection):
    from meshmode.distributed import get_connected_partitions
    return get_connected_partitions(dcoll._volume_discr.mesh)


def flatten_to_numpy_from_container(actx, ary):
    """
    """
    if is_array_container(ary):
        containerized_data_template = {}
        flat_data = []
        for key, value in serialize_container(ary):
            # FIXME: Need to special case nested object arrays
            if isinstance(value, np.ndarray) and value.dtype == "O":
                numpy_values = flatten_to_numpy(actx, value)
                flat_data.append(np.concatenate(numpy_values))
            else:
                numpy_values = actx.to_numpy(value)
                flat_data.append(numpy_values)
            containerized_data_template[key] = numpy_values
        return np.concatenate(flat_data), containerized_data_template
    else:
        # No data template for arys with only a single state
        # e.g. a single DOFArray
        return flatten_to_numpy(actx, ary), None


def unflatten_from_numpy_to_container(
        actx, numpy_ary, ary_container_data, discr, template):
    """
    """
    if not isinstance(template, DOFArray):
        remote_container_data = {}
        offset = 0
        for key, _ in serialize_container(template):
            _value = ary_container_data[key]
            # FIXME: Need to inspect value before it was flattened to determine
            # the right offsets
            if isinstance(_value, np.ndarray) and _value.dtype == "O":
                remote_comps = []
                for _subary_idx in range(len(_value)):
                    _subary = _value[_subary_idx]
                    _subary_ndofs, = _subary.shape
                    remote_comps.append(
                        unflatten_from_numpy(
                            actx, discr,
                            numpy_ary[offset:offset+_subary_ndofs]
                        )
                    )
                    offset += _subary_ndofs
                remote_data = make_obj_array(remote_comps)
            else:
                _ndofs, = _value.shape
                remote_data = unflatten_from_numpy(
                    actx, discr, numpy_ary[offset:offset+_ndofs]
                )
                offset += _ndofs
            remote_container_data[key] = remote_data
        return deserialize_container(
            template,
            remote_container_data.items()
        )
    else:
        return unflatten_from_numpy(actx, discr, numpy_ary)


class _RankBoundaryCommunication:
    base_tag = 1273

    def __init__(self, dcoll: DiscretizationCollection,
                 remote_rank, array_container, tag=None):
        self.tag = self.base_tag
        if tag is not None:
            self.tag += tag

        self.dcoll = dcoll
        self.array_context = get_container_context_recursively(array_container)
        self.remote_btag = BTAG_PARTITION(remote_rank)
        self.bdry_discr = dcoll.discr_from_dd(self.remote_btag)

        self.local_array_ctr = \
            project(dcoll, "vol", self.remote_btag, array_container)

        numpy_data, container_data = \
            flatten_to_numpy_from_container(self.array_context,
                                            self.local_array_ctr)

        # NOTE: Storing container data so we can use this information to
        # re-containerize the exchanged result
        self.container_data = container_data

        comm = self.dcoll.mpi_communicator

        self.send_req = comm.Isend(numpy_data, remote_rank, tag=self.tag)
        self.remote_data_host_numpy = np.empty_like(numpy_data)
        self.recv_req = comm.Irecv(
            self.remote_data_host_numpy, remote_rank, self.tag
        )

    def finish(self):
        self.recv_req.Wait()

        # Re-containerize the remote data
        remote_array_ctr = unflatten_from_numpy_to_container(
            self.array_context,
            self.remote_data_host_numpy,
            self.container_data,
            self.bdry_discr,
            self.local_array_ctr
        )

        bdry_conn = self.dcoll.distributed_boundary_swap_connection(
            dof_desc.as_dofdesc(dof_desc.DTAG_BOUNDARY(self.remote_btag))
        )
        swapped_remote_array_ctr = bdry_conn(remote_array_ctr)

        self.send_req.Wait()

        return TracePair(self.remote_btag,
                         interior=self.local_array_ctr,
                         exterior=swapped_remote_array_ctr)


def cross_rank_trace_pairs(
        dcoll: DiscretizationCollection, ary, tag=None) -> list:
    r"""Get a :class:`list` of *ary* trace pairs for each partition boundary.

    For each partition boundary, the field data values in *ary* are
    communicated to/from the neighboring partition. Presumably, this
    communication is MPI (but strictly speaking, may not be, and this
    routine is agnostic to the underlying communication).

    For each face on each partition boundary, a
    :class:`TracePair` is created with the locally, and
    remotely owned partition boundary face data as the `internal`, and `external`
    components, respectively. Each of the TracePair components are structured
    like *ary*.

    :arg ary: a single :class:`~meshmode.dof_array.DOFArray`, or an object
        array of :class:`~meshmode.dof_array.DOFArray`\ s
        of arbitrary shape.
    :returns: a :class:`list` of :class:`TracePair` objects.
    """
    if isinstance(ary, Number):
        return [TracePair(BTAG_PARTITION(remote_rank), interior=ary, exterior=ary)
                for remote_rank in connected_ranks(dcoll)]
    else:
        return [_RankBoundaryCommunication(dcoll, remote_rank, ary, tag=tag).finish()
                for remote_rank in connected_ranks(dcoll)]

# }}}


# vim: foldmethod=marker
