"""
Trace Pairs
^^^^^^^^^^^

Container class and auxiliary functionality
-------------------------------------------

.. autoclass:: TracePair

.. currentmodule:: grudge.op

.. autoclass:: project_tracepair
.. autoclass:: tracepair_with_discr_tag

Boundary trace functions
------------------------

.. autofunction:: bdry_trace_pair
.. autofunction:: bv_trace_pair

Interior, cross-rank, and inter-volume traces
---------------------------------------------

.. autofunction:: interior_trace_pairs
.. autofunction:: local_interior_trace_pair
.. autofunction:: inter_volume_trace_pairs
.. autofunction:: local_inter_volume_trace_pairs
.. autofunction:: cross_rank_trace_pairs
.. autofunction:: cross_rank_inter_volume_trace_pairs
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


from warnings import warn
from typing import List, Hashable, Optional, Type, Any, Sequence

from pytools.persistent_dict import KeyBuilder

from arraycontext import (
    ArrayContainer,
    ArrayContext,
    with_container_arithmetic,
    dataclass_array_container,
    get_container_context_recursively,
    flatten, to_numpy,
    unflatten, from_numpy,
    flat_size_and_dtype,
    ArrayOrContainer
)

from dataclasses import dataclass

from numbers import Number

from pytools import memoize_on_first_arg
from pytools.obj_array import obj_array_vectorize

from grudge.discretization import DiscretizationCollection
from grudge.projection import project

from meshmode.mesh import BTAG_PARTITION, PartitionID

import numpy as np

import grudge.dof_desc as dof_desc
from grudge.dof_desc import (
        DOFDesc, DD_VOLUME_ALL, FACE_RESTR_INTERIOR, DISCR_TAG_BASE,
        VolumeTag, VolumeDomainTag, BoundaryDomainTag,
        ConvertibleToDOFDesc,
        )


# {{{ trace pair container class

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
    """

    dd: DOFDesc
    interior: ArrayContainer
    exterior: ArrayContainer

    def __init__(self, dd: DOFDesc, *,
            interior: ArrayOrContainer,
            exterior: ArrayOrContainer):
        if not isinstance(dd, DOFDesc):
            warn("Constructing a TracePair with a first argument that is not "
                    "exactly a DOFDesc (but convertible to one) is deprecated. "
                    "This will stop working in July 2022. "
                    "Pass an actual DOFDesc instead.",
                    DeprecationWarning, stacklevel=2)
            dd = dof_desc.as_dofdesc(dd)

        object.__setattr__(self, "dd", dd)
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
        """A :class:`~meshmode.dof_array.DOFArray` or
        :class:`~arraycontext.ArrayContainer` of them representing the
        interior value to be used for the flux.
        """
        return self.interior

    @property
    def ext(self):
        """A :class:`~meshmode.dof_array.DOFArray` or
        :class:`~arraycontext.ArrayContainer` of them representing the
        exterior value to be used for the flux.
        """
        return self.exterior

    @property
    def avg(self):
        """A :class:`~meshmode.dof_array.DOFArray` or
        :class:`~arraycontext.ArrayContainer` of them representing the
        average of the interior and exterior values.
        """
        return 0.5 * (self.int + self.ext)

    @property
    def diff(self):
        """A :class:`~meshmode.dof_array.DOFArray` or
        :class:`~arraycontext.ArrayContainer` of them representing the
        difference (exterior - interior) of the pair values.
        """
        return self.ext - self.int

# }}}


# {{{ boundary trace pairs

def bdry_trace_pair(
        dcoll: DiscretizationCollection, dd: "ConvertibleToDOFDesc",
        interior, exterior) -> TracePair:
    """Returns a trace pair defined on the exterior boundary. Input arguments
    are assumed to already be defined on the boundary denoted by *dd*.
    If the input arguments *interior* and *exterior* are
    :class:`~arraycontext.container.ArrayContainer` objects, they must both
    have the same internal structure.

    :arg dd: a :class:`~grudge.dof_desc.DOFDesc`, or a value convertible to one,
        which describes the boundary discretization.
    :arg interior: a :class:`~meshmode.dof_array.DOFArray` or an
        :class:`~arraycontext.container.ArrayContainer` of them that contains data
        already on the boundary representing the interior value to be used
        for the flux.
    :arg exterior: a :class:`~meshmode.dof_array.DOFArray` or an
        :class:`~arraycontext.container.ArrayContainer` of them that contains data
        that already lives on the boundary representing the exterior value to
        be used for the flux.
    :returns: a :class:`TracePair` on the boundary.
    """
    if not isinstance(dd, DOFDesc):
        warn("Calling  bdry_trace_pair with a first argument that is not "
                "exactly a DOFDesc (but convertible to one) is deprecated. "
                "This will stop working in July 2022. "
                "Pass an actual DOFDesc instead.",
                DeprecationWarning, stacklevel=2)
        dd = dof_desc.as_dofdesc(dd)
    return TracePair(dd, interior=interior, exterior=exterior)


def bv_trace_pair(
        dcoll: DiscretizationCollection, dd: "ConvertibleToDOFDesc",
        interior, exterior) -> TracePair:
    """Returns a trace pair defined on the exterior boundary. The interior
    argument is assumed to be defined on the volume discretization, and will
    therefore be restricted to the boundary *dd* prior to creating a
    :class:`TracePair`.
    If the input arguments *interior* and *exterior* are
    :class:`~arraycontext.container.ArrayContainer` objects, they must both
    have the same internal structure.

    :arg dd: a :class:`~grudge.dof_desc.DOFDesc`, or a value convertible to one,
        which describes the boundary discretization.
    :arg interior: a :class:`~meshmode.dof_array.DOFArray` or an
        :class:`~arraycontext.container.ArrayContainer` that contains data
        defined in the volume, which will be restricted to the boundary denoted
        by *dd*. The result will be used as the interior value
        for the flux.
    :arg exterior: a :class:`~meshmode.dof_array.DOFArray` or an
        :class:`~arraycontext.container.ArrayContainer` that contains data
        that already lives on the boundary representing the exterior value to
        be used for the flux.
    :returns: a :class:`TracePair` on the boundary.
    """
    if not isinstance(dd, DOFDesc):
        warn("Calling  bv_trace_pair with a first argument that is not "
                "exactly a DOFDesc (but convertible to one) is deprecated. "
                "This will stop working in July 2022. "
                "Pass an actual DOFDesc instead.",
                DeprecationWarning, stacklevel=2)
        dd = dof_desc.as_dofdesc(dd)
    return bdry_trace_pair(
        dcoll, dd, project(dcoll, "vol", dd, interior), exterior)

# }}}


# {{{ interior trace pairs

def local_interior_trace_pair(
        dcoll: DiscretizationCollection, vec, *,
        volume_dd: Optional[DOFDesc] = None,
        ) -> TracePair:
    r"""Return a :class:`TracePair` for the interior faces of
    *dcoll* with a discretization tag specified by *discr_tag*.
    This does not include interior faces on different MPI ranks.

    :arg vec: a :class:`~meshmode.dof_array.DOFArray` or an
        :class:`~arraycontext.container.ArrayContainer` of them.

    For certain applications, it may be useful to distinguish between
    rank-local and cross-rank trace pairs. For example, avoiding unnecessary
    communication of derived quantities (i.e. temperature) on partition
    boundaries by computing them directly. Having the ability for
    user applications to distinguish between rank-local and cross-rank
    contributions can also help enable overlapping communication with
    computation.
    :returns: a :class:`TracePair` object.
    """
    if volume_dd is None:
        volume_dd = DD_VOLUME_ALL

    assert isinstance(volume_dd.domain_tag, VolumeDomainTag)
    trace_dd = volume_dd.trace(FACE_RESTR_INTERIOR)

    interior = project(dcoll, volume_dd, trace_dd, vec)

    def get_opposite_trace(el):
        if isinstance(el, Number):
            return el
        else:
            assert isinstance(trace_dd.domain_tag, BoundaryDomainTag)
            return dcoll.opposite_face_connection(trace_dd.domain_tag)(el)

    e = obj_array_vectorize(get_opposite_trace, interior)

    return TracePair(trace_dd, interior=interior, exterior=e)


def interior_trace_pair(dcoll: DiscretizationCollection, vec) -> TracePair:
    warn("`grudge.op.interior_trace_pair` is deprecated and will be dropped "
         "in version 2022.x. Use `local_interior_trace_pair` "
         "instead, or `interior_trace_pairs` which also includes contributions "
         "from different MPI ranks.",
         DeprecationWarning, stacklevel=2)
    return local_interior_trace_pair(dcoll, vec)


def interior_trace_pairs(dcoll: DiscretizationCollection, vec, *,
        comm_tag: Hashable = None, tag: Hashable = None,
        volume_dd: Optional[DOFDesc] = None) -> List[TracePair]:
    r"""Return a :class:`list` of :class:`TracePair` objects
    defined on the interior faces of *dcoll* and any faces connected to a
    parallel boundary.

    Note that :func:`local_interior_trace_pair` provides the rank-local contributions
    if those are needed in isolation. Similarly, :func:`cross_rank_trace_pairs`
    provides only the trace pairs defined on cross-rank boundaries.

    :arg vec: a :class:`~meshmode.dof_array.DOFArray` or an
        :class:`~arraycontext.container.ArrayContainer` of them.
    :arg comm_tag: a hashable object used to match sent and received data
        across ranks. Communication will only match if both endpoints specify
        objects that compare equal. A generalization of MPI communication
        tags to arbitary, potentially composite objects.
    :returns: a :class:`list` of :class:`TracePair` objects.
    """

    if tag is not None:
        warn("Specifying 'tag' is deprecated and will stop working in July of 2022. "
                "Specify 'comm_tag' instead.", DeprecationWarning, stacklevel=2)
        if comm_tag is not None:
            raise TypeError("may only specify one of 'tag' and 'comm_tag'")
        else:
            comm_tag = tag
    del tag

    if volume_dd is None:
        volume_dd = DD_VOLUME_ALL

    return (
        [local_interior_trace_pair(
            dcoll, vec, volume_dd=volume_dd)]
        + cross_rank_trace_pairs(
            dcoll, vec, comm_tag=comm_tag, volume_dd=volume_dd)
    )

# }}}


# {{{ inter-volume trace pairs

def local_inter_volume_trace_pairs(
        dcoll: DiscretizationCollection,
        self_volume_dd: DOFDesc, self_ary: ArrayOrContainer,
        other_volume_dd: DOFDesc, other_ary: ArrayOrContainer,
        ) -> ArrayOrContainer:
    if not isinstance(self_volume_dd.domain_tag, VolumeDomainTag):
        raise ValueError("self_volume_dd must describe a volume")
    if not isinstance(other_volume_dd.domain_tag, VolumeDomainTag):
        raise ValueError("other_volume_dd must describe a volume")
    if self_volume_dd.discretization_tag != DISCR_TAG_BASE:
        raise TypeError(
            f"expected a base-discretized self DOFDesc, got '{self_volume_dd}'")
    if other_volume_dd.discretization_tag != DISCR_TAG_BASE:
        raise TypeError(
            f"expected a base-discretized other DOFDesc, got '{other_volume_dd}'")

    rank = (
        dcoll.mpi_communicator.Get_rank()
        if dcoll.mpi_communicator is not None
        else None)

    self_part_id = dcoll._part_id_helper.make(rank, self_volume_dd.domain_tag.tag)
    other_part_id = dcoll._part_id_helper.make(rank, other_volume_dd.domain_tag.tag)

    self_trace_dd = self_volume_dd.trace(BTAG_PARTITION(other_part_id))
    other_trace_dd = other_volume_dd.trace(BTAG_PARTITION(self_part_id))

    # FIXME: In all likelihood, these traces will be reevaluated from
    # the other side, which is hard to prevent given the interface we
    # have. Lazy eval will hopefully collapse those redundant evaluations...
    self_trace = project(
            dcoll, self_volume_dd, self_trace_dd, self_ary)
    other_trace = project(
            dcoll, other_volume_dd, other_trace_dd, other_ary)

    other_to_self = dcoll._inter_partition_connections[
            other_part_id, self_part_id]

    def get_opposite_trace(el):
        if isinstance(el, Number):
            return el
        else:
            return other_to_self(el)

    return TracePair(
            self_trace_dd,
            interior=self_trace,
            exterior=obj_array_vectorize(get_opposite_trace, other_trace))


def inter_volume_trace_pairs(dcoll: DiscretizationCollection,
        self_volume_dd: DOFDesc, self_ary: ArrayOrContainer,
        other_volume_dd: DOFDesc, other_ary: ArrayOrContainer,
        comm_tag: Hashable = None) -> List[ArrayOrContainer]:
    """
    Note that :func:`local_inter_volume_trace_pairs` provides the rank-local
    contributions if those are needed in isolation. Similarly,
    :func:`cross_rank_inter_volume_trace_pairs` provides only the trace pairs
    defined on cross-rank boundaries.
    """
    # TODO documentation

    return (
        [local_inter_volume_trace_pairs(dcoll,
            self_volume_dd, self_ary, other_volume_dd, other_ary)]
        + cross_rank_inter_volume_trace_pairs(dcoll,
            self_volume_dd, self_ary, other_volume_dd, other_ary,
            comm_tag=comm_tag)
    )

# }}}


# {{{ distributed: helper functions

class _TagKeyBuilder(KeyBuilder):
    def update_for_type(self, key_hash, key: Type[Any]):
        self.rec(key_hash, (key.__module__, key.__name__, key.__name__,))


@memoize_on_first_arg
def _connected_partitions(
        dcoll: DiscretizationCollection,
        self_volume_tag: VolumeTag,
        other_volume_tag: VolumeTag
        ) -> Sequence[PartitionID]:
    result: List[PartitionID] = [
        connected_part_id
        for connected_part_id, part_id in dcoll._inter_partition_connections.keys()
        if (
            dcoll._part_id_helper.get_volume(part_id) == self_volume_tag
            and (
                dcoll._part_id_helper.get_volume(connected_part_id)
                == other_volume_tag))]

    return result


def _sym_tag_to_num_tag(comm_tag: Optional[Hashable]) -> Optional[int]:
    if comm_tag is None:
        return comm_tag

    if isinstance(comm_tag, int):
        return comm_tag

    # FIXME: This isn't guaranteed to be correct.
    # See here for discussion:
    # - https://github.com/illinois-ceesd/mirgecom/issues/617#issuecomment-1057082716  # noqa
    # - https://github.com/inducer/grudge/pull/222

    from mpi4py import MPI
    tag_ub = MPI.COMM_WORLD.Get_attr(MPI.TAG_UB)
    key_builder = _TagKeyBuilder()
    digest = key_builder(comm_tag)

    num_tag = sum(ord(ch) << i for i, ch in enumerate(digest)) % tag_ub

    warn("Encountered unknown symbolic tag "
            f"'{comm_tag}', assigning a value of '{num_tag}'. "
            "This is a temporary workaround, please ensure that "
            "tags are sufficiently distinct for your use case.")

    return num_tag

# }}}


# {{{ eager rank-boundary communication

class _RankBoundaryCommunicationEager:
    base_comm_tag = 1273

    def __init__(self,
            actx: ArrayContext,
            dcoll: DiscretizationCollection,
            *,
            local_part_id: PartitionID,
            remote_part_id: PartitionID,
            local_bdry_data: ArrayOrContainer,
            send_data: ArrayOrContainer,
            comm_tag: Optional[Hashable] = None):

        comm = dcoll.mpi_communicator
        assert comm is not None

        remote_rank = dcoll._part_id_helper.get_mpi_rank(remote_part_id)
        assert remote_rank is not None

        self.dcoll = dcoll
        self.array_context = actx
        self.local_part_id = local_part_id
        self.remote_part_id = remote_part_id
        self.bdry_discr = dcoll.discr_from_dd(
            BoundaryDomainTag(BTAG_PARTITION(remote_part_id)))
        self.local_bdry_data = local_bdry_data

        self.comm_tag = self.base_comm_tag
        comm_tag = _sym_tag_to_num_tag(comm_tag)
        if comm_tag is not None:
            self.comm_tag += comm_tag
        del comm_tag

        # NOTE: mpi4py currently (2021-11-03) holds a reference to the send
        # memory buffer for (i.e. `self.local_bdry_data_np`) until the send
        # requests is complete, however it is not clear that this is documented
        # behavior. We hold on to the buffer (via the instance attribute)
        # as well, just in case.
        self.send_data_np = to_numpy(flatten(send_data, actx), actx)
        self.send_req = comm.Isend(self.send_data_np,
                                   remote_rank,
                                   tag=self.comm_tag)

        recv_size, recv_dtype = flat_size_and_dtype(local_bdry_data)
        self.recv_data_np = np.empty(recv_size, recv_dtype)
        self.recv_req = comm.Irecv(self.recv_data_np, remote_rank, tag=self.comm_tag)

    def finish(self):
        # Wait for the nonblocking receive request to complete before
        # accessing the data
        self.recv_req.Wait()

        recv_data_flat = from_numpy(
                self.recv_data_np, self.array_context)
        unswapped_remote_bdry_data = unflatten(self.local_bdry_data,
                                     recv_data_flat, self.array_context)
        bdry_conn = self.dcoll._inter_partition_connections[
            self.remote_part_id, self.local_part_id]
        remote_bdry_data = bdry_conn(unswapped_remote_bdry_data)

        # Complete the nonblocking send request associated with communicating
        # `self.local_bdry_data_np`
        self.send_req.Wait()

        return TracePair(
                DOFDesc(
                    BoundaryDomainTag(BTAG_PARTITION(self.remote_part_id)),
                    DISCR_TAG_BASE),
                interior=self.local_bdry_data,
                exterior=remote_bdry_data)

# }}}


# {{{ lazy rank-boundary communication

class _RankBoundaryCommunicationLazy:
    def __init__(self,
            actx: ArrayContext,
            dcoll: DiscretizationCollection,
            *,
            local_part_id: PartitionID,
            remote_part_id: PartitionID,
            local_bdry_data: ArrayOrContainer,
            send_data: ArrayOrContainer,
            comm_tag: Optional[Hashable] = None) -> None:

        if comm_tag is None:
            raise ValueError("lazy communication requires 'comm_tag' to be supplied")

        self.dcoll = dcoll
        self.array_context = actx
        self.bdry_discr = dcoll.discr_from_dd(
            BoundaryDomainTag(BTAG_PARTITION(remote_part_id)))
        self.local_part_id = local_part_id
        self.remote_part_id = remote_part_id

        remote_rank = dcoll._part_id_helper.get_mpi_rank(remote_part_id)
        assert remote_rank is not None

        self.local_bdry_data = local_bdry_data

        from arraycontext.container.traversal import rec_keyed_map_array_container

        key_to_send_subary = {}

        def store_send_subary(key, send_subary):
            key_to_send_subary[key] = send_subary
            return send_subary
        rec_keyed_map_array_container(store_send_subary, send_data)

        from pytato import make_distributed_recv, staple_distributed_send

        def communicate_single_array(key, local_bdry_subary):
            ary_tag = (comm_tag, key)
            return staple_distributed_send(
                    key_to_send_subary[key], dest_rank=remote_rank, comm_tag=ary_tag,
                    stapled_to=make_distributed_recv(
                        src_rank=remote_rank, comm_tag=ary_tag,
                        shape=local_bdry_subary.shape,
                        dtype=local_bdry_subary.dtype,
                        axes=local_bdry_subary.axes))

        self.remote_data = rec_keyed_map_array_container(
                communicate_single_array, self.local_bdry_data)

    def finish(self):
        bdry_conn = self.dcoll._inter_partition_connections[
            self.remote_part_id, self.local_part_id]

        return TracePair(
                DOFDesc(
                    BoundaryDomainTag(BTAG_PARTITION(self.remote_part_id)),
                    DISCR_TAG_BASE),
                interior=self.local_bdry_data,
                exterior=bdry_conn(self.remote_data))

# }}}


# {{{ cross_rank_trace_pairs

def cross_rank_trace_pairs(
        dcoll: DiscretizationCollection, ary: ArrayOrContainer,
        tag: Hashable = None,
        *, comm_tag: Hashable = None,
        volume_dd: Optional[DOFDesc] = None) -> List[TracePair]:
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

    If *ary* is a number, rather than a
    :class:`~meshmode.dof_array.DOFArray` or an
    :class:`~arraycontext.container.ArrayContainer` of them, it is assumed
    that the same number is being communicated on every rank.

    :arg ary: a :class:`~meshmode.dof_array.DOFArray` or an
        :class:`~arraycontext.container.ArrayContainer` of them.

    :arg comm_tag: a hashable object used to match sent and received data
        across ranks. Communication will only match if both endpoints specify
        objects that compare equal. A generalization of MPI communication
        tags to arbitary, potentially composite objects.

    :returns: a :class:`list` of :class:`TracePair` objects.
    """
    # {{{ process arguments

    if volume_dd is None:
        volume_dd = DD_VOLUME_ALL

    if not isinstance(volume_dd.domain_tag, VolumeDomainTag):
        raise TypeError(f"expected a volume DOFDesc, got '{volume_dd}'")
    if volume_dd.discretization_tag != DISCR_TAG_BASE:
        raise TypeError(f"expected a base-discretized DOFDesc, got '{volume_dd}'")

    if tag is not None:
        warn("Specifying 'tag' is deprecated and will stop working in July of 2022. "
                "Specify 'comm_tag' (keyword-only) instead.",
                DeprecationWarning, stacklevel=2)
        if comm_tag is not None:
            raise TypeError("may only specify one of 'tag' and 'comm_tag'")
        else:
            comm_tag = tag
    del tag

    # }}}

    if dcoll.mpi_communicator is None:
        return []

    rank = dcoll.mpi_communicator.Get_rank()

    local_part_id = dcoll._part_id_helper.make(rank, volume_dd.domain_tag.tag)

    connected_part_ids = _connected_partitions(
            dcoll, self_volume_tag=volume_dd.domain_tag.tag,
            other_volume_tag=volume_dd.domain_tag.tag)

    remote_part_ids = [
        part_id
        for part_id in connected_part_ids
        if dcoll._part_id_helper.get_mpi_rank(part_id) != rank]

    # This asserts that there is only one data exchange per rank, so that
    # there is no risk of mismatched data reaching the wrong recipient.
    # (Since we have only a single tag.)
    assert len(remote_part_ids) == len(
        {dcoll._part_id_helper.get_mpi_rank(part_id) for part_id in remote_part_ids})

    if isinstance(ary, Number):
        # NOTE: Assumes that the same number is passed on every rank
        return [
            TracePair(
                DOFDesc(
                    BoundaryDomainTag(BTAG_PARTITION(remote_part_id)),
                    DISCR_TAG_BASE),
                interior=ary, exterior=ary)
            for remote_part_id in remote_part_ids]

    actx = get_container_context_recursively(ary)
    assert actx is not None

    from grudge.array_context import MPIPytatoArrayContextBase

    if isinstance(actx, MPIPytatoArrayContextBase):
        rbc = _RankBoundaryCommunicationLazy
    else:
        rbc = _RankBoundaryCommunicationEager

    def start_comm(remote_part_id):
        bdtag = BoundaryDomainTag(BTAG_PARTITION(remote_part_id))

        local_bdry_data = project(dcoll, volume_dd, bdtag, ary)

        return rbc(actx, dcoll,
            local_part_id=local_part_id,
            remote_part_id=remote_part_id,
            local_bdry_data=local_bdry_data,
            send_data=local_bdry_data,
            comm_tag=comm_tag)

    rank_bdry_communcators = [
        start_comm(remote_part_id)
        for remote_part_id in remote_part_ids]
    return [rc.finish() for rc in rank_bdry_communcators]

# }}}


# {{{ cross_rank_inter_volume_trace_pairs

def cross_rank_inter_volume_trace_pairs(
        dcoll: DiscretizationCollection,
        self_volume_dd: DOFDesc, self_ary: ArrayOrContainer,
        other_volume_dd: DOFDesc, other_ary: ArrayOrContainer,
        *, comm_tag: Hashable = None,
        ) -> List[TracePair]:
    # FIXME: Should this interface take in boundary data instead?
    # TODO: Docs
    r"""Get a :class:`list` of *ary* trace pairs for each partition boundary.

    :arg comm_tag: a hashable object used to match sent and received data
        across ranks. Communication will only match if both endpoints specify
        objects that compare equal. A generalization of MPI communication
        tags to arbitary, potentially composite objects.

    :returns: a :class:`list` of :class:`TracePair` objects.
    """
    # {{{ process arguments

    if not isinstance(self_volume_dd.domain_tag, VolumeDomainTag):
        raise ValueError("self_volume_dd must describe a volume")
    if not isinstance(other_volume_dd.domain_tag, VolumeDomainTag):
        raise ValueError("other_volume_dd must describe a volume")
    if self_volume_dd.discretization_tag != DISCR_TAG_BASE:
        raise TypeError(
            f"expected a base-discretized self DOFDesc, got '{self_volume_dd}'")
    if other_volume_dd.discretization_tag != DISCR_TAG_BASE:
        raise TypeError(
            f"expected a base-discretized other DOFDesc, got '{other_volume_dd}'")

    # }}}

    if dcoll.mpi_communicator is None:
        return []

    rank = dcoll.mpi_communicator.Get_rank()

    local_part_id = dcoll._part_id_helper.make(rank, self_volume_dd.domain_tag.tag)

    connected_part_ids = _connected_partitions(
            dcoll, self_volume_tag=self_volume_dd.domain_tag.tag,
            other_volume_tag=other_volume_dd.domain_tag.tag)

    remote_part_ids = [
        part_id
        for part_id in connected_part_ids
        if dcoll._part_id_helper.get_mpi_rank(part_id) != rank]

    # This asserts that there is only one data exchange per rank, so that
    # there is no risk of mismatched data reaching the wrong recipient.
    # (Since we have only a single tag.)
    assert len(remote_part_ids) == len(
        {dcoll._part_id_helper.get_mpi_rank(part_id) for part_id in remote_part_ids})

    actx = get_container_context_recursively(self_ary)
    assert actx is not None

    from grudge.array_context import MPIPytatoArrayContextBase

    if isinstance(actx, MPIPytatoArrayContextBase):
        rbc = _RankBoundaryCommunicationLazy
    else:
        rbc = _RankBoundaryCommunicationEager

    def start_comm(remote_part_id):
        bdtag = BoundaryDomainTag(BTAG_PARTITION(remote_part_id))

        local_bdry_data = project(dcoll, self_volume_dd, bdtag, self_ary)
        send_data = project(dcoll, other_volume_dd,
                BTAG_PARTITION(local_part_id), other_ary)

        return rbc(actx, dcoll,
                local_part_id=local_part_id,
                remote_part_id=remote_part_id,
                local_bdry_data=local_bdry_data,
                send_data=send_data,
                comm_tag=comm_tag)

    rank_bdry_communcators = [
        start_comm(remote_part_id)
        for remote_part_id in remote_part_ids]
    return [rc.finish() for rc in rank_bdry_communcators]

# }}}


# {{{ project_tracepair

def project_tracepair(
        dcoll: DiscretizationCollection, new_dd: dof_desc.DOFDesc,
        tpair: TracePair) -> TracePair:
    r"""Return a new :class:`TracePair` :func:`~grudge.op.project`\ 'ed
    onto *new_dd*.
    """
    return TracePair(
        new_dd,
        interior=project(dcoll, tpair.dd, new_dd, tpair.int),
        exterior=project(dcoll, tpair.dd, new_dd, tpair.ext)
    )


def tracepair_with_discr_tag(dcoll, discr_tag, tpair: TracePair) -> TracePair:
    r"""Return a new :class:`TracePair` :func:`~grudge.op.project`\ 'ed
    onto a :class:`~grudge.dof_desc.DOFDesc` with discretization tag *discr_tag*.
    """
    return project_tracepair(dcoll, tpair.dd.with_discr_tag(discr_tag), tpair)

# }}}

# vim: foldmethod=marker
