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

Interior and cross-rank trace functions
---------------------------------------

.. autofunction:: interior_trace_pairs
.. autofunction:: local_interior_trace_pair
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


from warnings import warn
from typing import List, Hashable, Optional, Type, Any

from pytools.persistent_dict import KeyBuilder

from arraycontext import (
    ArrayContainer,
    with_container_arithmetic,
    dataclass_array_container,
    get_container_context_recursively,
    flatten, to_numpy,
    unflatten, from_numpy,
    ArrayOrContainer
)

from dataclasses import dataclass

from numbers import Number

from pytools import memoize_on_first_arg

from grudge.discretization import DiscretizationCollection
from grudge.projection import project

from meshmode.mesh import BTAG_PARTITION

import numpy as np

import grudge.dof_desc as dof_desc
from grudge.dof_desc import (
        DOFDesc, DD_VOLUME_ALL, FACE_RESTR_INTERIOR, DISCR_TAG_BASE,
        VolumeDomainTag,
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
    :class:`~arraycontext.ArrayContainer` objects, they must both
    have the same internal structure.

    :arg dd: a :class:`~grudge.dof_desc.DOFDesc`, or a value convertible to one,
        which describes the boundary discretization.
    :arg interior: a :class:`~meshmode.dof_array.DOFArray` or an
        :class:`~arraycontext.ArrayContainer` of them that contains data
        already on the boundary representing the interior value to be used
        for the flux.
    :arg exterior: a :class:`~meshmode.dof_array.DOFArray` or an
        :class:`~arraycontext.ArrayContainer` of them that contains data
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
    :class:`~arraycontext.ArrayContainer` objects, they must both
    have the same internal structure.

    :arg dd: a :class:`~grudge.dof_desc.DOFDesc`, or a value convertible to one,
        which describes the boundary discretization.
    :arg interior: a :class:`~meshmode.dof_array.DOFArray` or an
        :class:`~arraycontext.ArrayContainer` that contains data
        defined in the volume, which will be restricted to the boundary denoted
        by *dd*. The result will be used as the interior value
        for the flux.
    :arg exterior: a :class:`~meshmode.dof_array.DOFArray` or an
        :class:`~arraycontext.ArrayContainer` that contains data
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
        dcoll, dd, project(dcoll, dd.domain_tag.volume_tag, dd, interior), exterior)

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
        :class:`~arraycontext.ArrayContainer` of them.

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

    opposite_face_conn = dcoll.opposite_face_connection(trace_dd.domain_tag)

    def get_opposite_trace(ary):
        if isinstance(ary, Number):
            return ary
        else:
            return opposite_face_conn(ary)

    from arraycontext import rec_map_array_container
    from meshmode.dof_array import DOFArray
    exterior = rec_map_array_container(
        get_opposite_trace,
        interior,
        leaf_class=DOFArray)

    return TracePair(trace_dd, interior=interior, exterior=exterior)


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
        :class:`~arraycontext.ArrayContainer` of them.
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


# {{{ distributed: helper functions

class _TagKeyBuilder(KeyBuilder):
    def update_for_type(self, key_hash, key: Type[Any]):
        self.rec(key_hash, (key.__module__, key.__name__, key.__name__,))


@memoize_on_first_arg
def connected_ranks(
        dcoll: DiscretizationCollection,
        volume_dd: Optional[DOFDesc] = None):
    if volume_dd is None:
        volume_dd = DD_VOLUME_ALL

    from meshmode.distributed import get_connected_partitions
    return get_connected_partitions(
        dcoll._volume_discrs[volume_dd.domain_tag.tag].mesh)


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
                 dcoll: DiscretizationCollection,
                 array_container: ArrayOrContainer,
                 remote_rank, comm_tag: Optional[int] = None,
                 volume_dd=DD_VOLUME_ALL):
        actx = get_container_context_recursively(array_container)
        bdry_dd = volume_dd.trace(BTAG_PARTITION(remote_rank))

        local_bdry_data = project(dcoll, volume_dd, bdry_dd, array_container)
        comm = dcoll.mpi_communicator
        assert comm is not None

        self.dcoll = dcoll
        self.array_context = actx
        self.remote_bdry_dd = bdry_dd
        self.bdry_discr = dcoll.discr_from_dd(bdry_dd)
        self.local_bdry_data = local_bdry_data
        self.local_bdry_data_np = \
            to_numpy(flatten(self.local_bdry_data, actx), actx)

        self.comm_tag = self.base_comm_tag
        comm_tag = _sym_tag_to_num_tag(comm_tag)
        if comm_tag is not None:
            self.comm_tag += comm_tag
        del comm_tag

        # Here, we initialize both send and recieve operations through
        # mpi4py `Request` (MPI_Request) instances for comm.Isend (MPI_Isend)
        # and comm.Irecv (MPI_Irecv) respectively. These initiate non-blocking
        # point-to-point communication requests and require explicit management
        # via the use of wait (MPI_Wait, MPI_Waitall, MPI_Waitany, MPI_Waitsome),
        # test (MPI_Test, MPI_Testall, MPI_Testany, MPI_Testsome), and cancel
        # (MPI_Cancel). The rank-local data `self.local_bdry_data_np` will have its
        # associated memory buffer sent across connected ranks and must not be
        # modified at the Python level during this process. Completion of the
        # requests is handled in :meth:`finish`.
        #
        # For more details on the mpi4py semantics, see:
        # https://mpi4py.readthedocs.io/en/stable/overview.html#nonblocking-communications
        #
        # NOTE: mpi4py currently (2021-11-03) holds a reference to the send
        # memory buffer for (i.e. `self.local_bdry_data_np`) until the send
        # requests is complete, however it is not clear that this is documented
        # behavior. We hold on to the buffer (via the instance attribute)
        # as well, just in case.
        self.send_req = comm.Isend(self.local_bdry_data_np,
                                   remote_rank,
                                   tag=self.comm_tag)
        self.remote_data_host_numpy = np.empty_like(self.local_bdry_data_np)
        self.recv_req = comm.Irecv(self.remote_data_host_numpy,
                                   remote_rank,
                                   tag=self.comm_tag)

    def finish(self):
        # Wait for the nonblocking receive request to complete before
        # accessing the data
        self.recv_req.Wait()

        # Nonblocking receive is complete, we can now access the data and apply
        # the boundary-swap connection
        actx = self.array_context
        remote_bdry_data_flat = from_numpy(self.remote_data_host_numpy, actx)
        remote_bdry_data = unflatten(self.local_bdry_data,
                                     remote_bdry_data_flat, actx)
        bdry_conn = self.dcoll.distributed_boundary_swap_connection(
            self.remote_bdry_dd)
        swapped_remote_bdry_data = bdry_conn(remote_bdry_data)

        # Complete the nonblocking send request associated with communicating
        # `self.local_bdry_data_np`
        self.send_req.Wait()

        return TracePair(self.remote_bdry_dd,
                         interior=self.local_bdry_data,
                         exterior=swapped_remote_bdry_data)

# }}}


# {{{ lazy rank-boundary communication

class _RankBoundaryCommunicationLazy:
    def __init__(self,
                 dcoll: DiscretizationCollection,
                 array_container: ArrayOrContainer,
                 remote_rank: int, comm_tag: Hashable,
                 volume_dd=DD_VOLUME_ALL):
        if comm_tag is None:
            raise ValueError("lazy communication requires 'tag' to be supplied")

        bdry_dd = volume_dd.trace(BTAG_PARTITION(remote_rank))

        self.dcoll = dcoll
        self.array_context = get_container_context_recursively(array_container)
        self.remote_bdry_dd = bdry_dd
        self.bdry_discr = dcoll.discr_from_dd(self.remote_bdry_dd)

        self.local_bdry_data = project(
            dcoll, volume_dd, bdry_dd, array_container)

        from pytato import make_distributed_recv, staple_distributed_send

        def communicate_single_array(key, local_bdry_ary):
            ary_tag = (comm_tag, key)
            return staple_distributed_send(
                    local_bdry_ary, dest_rank=remote_rank, comm_tag=ary_tag,
                    stapled_to=make_distributed_recv(
                        src_rank=remote_rank, comm_tag=ary_tag,
                        shape=local_bdry_ary.shape, dtype=local_bdry_ary.dtype,
                        axes=local_bdry_ary.axes))

        from arraycontext.container.traversal import rec_keyed_map_array_container
        self.remote_data = rec_keyed_map_array_container(
                communicate_single_array, self.local_bdry_data)

    def finish(self):
        bdry_conn = self.dcoll.distributed_boundary_swap_connection(
            self.remote_bdry_dd)

        return TracePair(self.remote_bdry_dd,
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
    :class:`~arraycontext.ArrayContainer` of them, it is assumed
    that the same number is being communicated on every rank.

    :arg ary: a :class:`~meshmode.dof_array.DOFArray` or an
        :class:`~arraycontext.ArrayContainer` of them.

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

    if isinstance(ary, Number):
        # NOTE: Assumed that the same number is passed on every rank
        return [TracePair(
                volume_dd.trace(BTAG_PARTITION(remote_rank)),
                interior=ary, exterior=ary)
            for remote_rank in connected_ranks(dcoll, volume_dd=volume_dd)]

    actx = get_container_context_recursively(ary)

    from grudge.array_context import MPIPytatoArrayContextBase

    if isinstance(actx, MPIPytatoArrayContextBase):
        rbc_class = _RankBoundaryCommunicationLazy
    else:
        rbc_class = _RankBoundaryCommunicationEager

    # Initialize and post all sends/receives
    rank_bdry_communcators = [
        rbc_class(dcoll, ary, remote_rank, comm_tag=comm_tag, volume_dd=volume_dd)
        for remote_rank in connected_ranks(dcoll, volume_dd=volume_dd)
    ]

    # Complete send/receives and return communicated data
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
