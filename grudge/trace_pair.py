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

References
----------
.. class:: DiscretizationTag

    See :class:`grudge.dof_desc.DiscretizationTag`.

.. currentmodule:: arraycontext.typing

.. class:: ArithArrayContainerT

    See :attr:`arraycontext.typing.ArithArrayContainerT`.
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

from dataclasses import dataclass
from numbers import Number
from typing import TYPE_CHECKING, ClassVar, Generic, cast
from warnings import warn

import numpy as np

from arraycontext import (
    ArithArrayContainer,
    ArithArrayContainerT,
    ArrayContext,
    ArrayOrContainerOrScalar,
    dataclass_array_container,
    flatten,
    get_container_context_recursively,
    unflatten,
    with_container_arithmetic,
)
from arraycontext.typing import is_scalar_like
from meshmode.mesh import BTAG_PARTITION, PartID
from pytools import memoize_on_first_arg
from pytools.persistent_dict import KeyBuilder

import grudge.dof_desc as dof_desc
from grudge.array_context import MPIBasedArrayContext
from grudge.dof_desc import (
    DD_VOLUME_ALL,
    DISCR_TAG_BASE,
    FACE_RESTR_INTERIOR,
    BoundaryDomainTag,
    DiscretizationTag,
    DOFDesc,
    ScalarDomainTag,
    ToDOFDescConvertible,
    VolumeDomainTag,
    as_dofdesc,
)
from grudge.projection import project


if TYPE_CHECKING:
    from collections.abc import Hashable, Sequence, Sized

    from _typeshed import SupportsGetItem

    from meshmode.discretization import Discretization

    from grudge.discretization import DiscretizationCollection


# {{{ trace pair container class

@with_container_arithmetic(bcasts_across_obj_array=False,
                           eq_comparison=False,
                           rel_comparison=False,
                           )
@dataclass_array_container
@dataclass(init=False, frozen=True)
class TracePair(Generic[ArithArrayContainerT]):
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
    interior: ArithArrayContainerT
    exterior: ArithArrayContainerT

    # NOTE: disable numpy doing any array math
    __array_ufunc__: ClassVar[None] = None

    def __init__(self, dd: DOFDesc, *,
            interior: ArithArrayContainerT,
            exterior: ArithArrayContainerT):
        if not isinstance(dd, DOFDesc):
            warn("Constructing a TracePair with a first argument that is not "
                    "exactly a DOFDesc (but convertible to one) is deprecated. "
                    "This will stop working in December 2022. "
                    "Pass an actual DOFDesc instead.",
                    DeprecationWarning, stacklevel=2)
            dd = dof_desc.as_dofdesc(dd)

        object.__setattr__(self, "dd", dd)
        object.__setattr__(self, "interior", interior)
        object.__setattr__(self, "exterior", exterior)

    def __getattr__(self, name: str) -> TracePair[ArithArrayContainer]:
        """Return a new :class:`TracePair` resulting from executing attribute
        lookup with *name* on :attr:`int` and :attr:`ext`.
        """
        return TracePair(self.dd,
                         interior=getattr(self.interior, name),
                         exterior=getattr(self.exterior, name))

    def __getitem__(self, index: int | slice) -> TracePair[ArithArrayContainer]:
        """Return a new :class:`TracePair` resulting from executing
        subscripting with *index* on :attr:`int` and :attr:`ext`.
        """
        return TracePair(self.dd,
                         interior=cast(
                            "SupportsGetItem[int | slice, ArithArrayContainer]",
                            self.interior)[index],
                         exterior=cast(
                            "SupportsGetItem[int | slice, ArithArrayContainer]",
                            self.exterior)[index]
                     )

    def __len__(self):
        """Return the total number of arrays associated with the
        :attr:`int` and :attr:`ext` restrictions of the :class:`TracePair`.
        Note that both must be the same.
        """
        len_ext = len(cast("Sized", self.exterior))
        len_int = len(cast("Sized", self.interior))
        assert len_int == len_ext
        return len_ext

    @property
    def int(self) -> ArithArrayContainerT:
        """A :class:`~meshmode.dof_array.DOFArray` or
        :class:`~arraycontext.ArrayContainer` of them representing the
        interior value to be used for the flux.
        """
        return self.interior

    @property
    def ext(self) -> ArithArrayContainerT:
        """A :class:`~meshmode.dof_array.DOFArray` or
        :class:`~arraycontext.ArrayContainer` of them representing the
        exterior value to be used for the flux.
        """
        return self.exterior

    @property
    def avg(self) -> ArithArrayContainerT:
        """A :class:`~meshmode.dof_array.DOFArray` or
        :class:`~arraycontext.ArrayContainer` of them representing the
        average of the interior and exterior values.
        """
        return 0.5 * (self.int + self.ext)

    @property
    def diff(self) -> ArithArrayContainerT:
        """A :class:`~meshmode.dof_array.DOFArray` or
        :class:`~arraycontext.ArrayContainer` of them representing the
        difference (exterior - interior) of the pair values.
        """
        return self.ext - self.int

# }}}


# {{{ boundary trace pairs

def bdry_trace_pair(
            dcoll: DiscretizationCollection,
            dd: ToDOFDescConvertible,
            interior: ArithArrayContainerT,
            exterior: ArithArrayContainerT,
        ) -> TracePair[ArithArrayContainerT]:
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
                "This will stop working in December 2022. "
                "Pass an actual DOFDesc instead.",
                DeprecationWarning, stacklevel=2)
        dd = dof_desc.as_dofdesc(dd)
    return TracePair(dd, interior=interior, exterior=exterior)


def bv_trace_pair(
            dcoll: DiscretizationCollection,
            dd: ToDOFDescConvertible,
            interior: ArithArrayContainerT,
            exterior: ArithArrayContainerT,
        ) -> TracePair[ArithArrayContainerT]:
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
                "This will stop working in December 2022. "
                "Pass an actual DOFDesc instead.",
                DeprecationWarning, stacklevel=2)
        dd = dof_desc.as_dofdesc(dd)
    dd = as_dofdesc(dd)
    return bdry_trace_pair(
        dcoll, dd, project(dcoll, dd.domain_tag.volume_tag, dd, interior), exterior)

# }}}


# {{{ interior trace pairs

def local_interior_trace_pair(
            dcoll: DiscretizationCollection,
            vec: ArithArrayContainerT,
            *, volume_dd: DOFDesc | None = None
        ) -> TracePair[ArithArrayContainerT]:
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

    assert isinstance(trace_dd.domain_tag, BoundaryDomainTag)
    opposite_face_conn = dcoll.opposite_face_connection(trace_dd.domain_tag)

    def get_opposite_trace(
                ary: ArrayOrContainerOrScalar
            ) -> ArrayOrContainerOrScalar:
        if is_scalar_like(ary):
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


def interior_trace_pair(
            dcoll: DiscretizationCollection,
            vec: ArithArrayContainerT
        ) -> TracePair[ArithArrayContainerT]:
    warn("`grudge.op.interior_trace_pair` is deprecated and will be dropped "
         "in version 2022.x. Use `local_interior_trace_pair` "
         "instead, or `interior_trace_pairs` which also includes contributions "
         "from different MPI ranks.",
         DeprecationWarning, stacklevel=2)
    return local_interior_trace_pair(dcoll, vec)


def interior_trace_pairs(
            dcoll: DiscretizationCollection,
            vec: ArithArrayContainerT,
            *,
            comm_tag: Hashable | None = None,
            volume_dd: DOFDesc | None = None
        ) -> list[TracePair[ArithArrayContainerT]]:
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
        tags to arbitrary, potentially composite objects.
    :returns: a :class:`list` of :class:`TracePair` objects.
    """

    if volume_dd is None:
        volume_dd = DD_VOLUME_ALL

    return [local_interior_trace_pair(dcoll, vec, volume_dd=volume_dd),
            *cross_rank_trace_pairs(dcoll, vec, comm_tag=comm_tag, volume_dd=volume_dd)]

# }}}


# {{{ distributed: helper functions

@memoize_on_first_arg
def connected_parts(
        dcoll: DiscretizationCollection,
        volume_dd: DOFDesc | None = None) -> Sequence[PartID]:
    if volume_dd is None:
        volume_dd = DD_VOLUME_ALL

    if isinstance(volume_dd.domain_tag, ScalarDomainTag):
        return []
    from meshmode.distributed import get_connected_parts
    return get_connected_parts(
        dcoll._volume_discrs[volume_dd.domain_tag.tag].mesh)


def _sym_tag_to_num_tag(comm_tag: Hashable | None, base_tag: int) -> int:
    if comm_tag is None:
        return base_tag

    if isinstance(comm_tag, int):
        return comm_tag + base_tag

    # FIXME: This isn't guaranteed to be correct.
    # See here for discussion:
    # - https://github.com/illinois-ceesd/mirgecom/issues/617#issuecomment-1057082716  # noqa
    # - https://github.com/inducer/grudge/pull/222
    # Since only 1 communication can be pending for a given tag at a time,
    # this does not matter currently. See https://github.com/inducer/grudge/issues/223

    from mpi4py import MPI
    tag_ub = MPI.COMM_WORLD.Get_attr(MPI.TAG_UB)
    assert tag_ub is not None
    key_builder = KeyBuilder()
    digest = key_builder(comm_tag)

    num_tag = sum(ord(ch) << i for i, ch in enumerate(digest)) % tag_ub

    warn("Encountered unknown symbolic tag "
            f"'{comm_tag}', assigning a value of '{num_tag+base_tag}'. "
            "This is a temporary workaround, please ensure that "
            "tags are sufficiently distinct for your use case.",
            stacklevel=1)

    return num_tag + base_tag

# }}}


# {{{ eager rank-boundary communication

class _RankBoundaryCommunicationEager:
    base_comm_tag: ClassVar[int] = 1273

    def __init__(self,
                 actx: MPIBasedArrayContext,
                 dcoll: DiscretizationCollection,
                 array_container: ArithArrayContainer,
                 remote_rank, comm_tag: Hashable = None,
                 volume_dd=DD_VOLUME_ALL):
        bdry_dd = volume_dd.trace(BTAG_PARTITION(remote_rank))

        local_bdry_data = project(dcoll, volume_dd, bdry_dd, array_container)
        comm = actx.mpi_communicator
        assert comm is not None

        self.dcoll = dcoll
        self.array_context = actx
        self.remote_bdry_dd = bdry_dd
        self.bdry_discr = dcoll.discr_from_dd(bdry_dd)
        self.local_bdry_data = local_bdry_data
        self.local_bdry_data_np = \
            actx.to_numpy(flatten(self.local_bdry_data, actx))

        self.comm_tag = _sym_tag_to_num_tag(comm_tag, self.base_comm_tag)

        # Here, we initialize both send and receive operations through
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
        remote_bdry_data_flat = actx.from_numpy(self.remote_data_host_numpy)
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
    dcoll: DiscretizationCollection
    array_context: ArrayContext
    bdry_discr: Discretization
    remote_bdry_dd: DOFDesc

    def __init__(self,
                 actx: MPIBasedArrayContext,
                 dcoll: DiscretizationCollection,
                 array_container: ArithArrayContainer,
                 remote_rank: int, comm_tag: Hashable,
                 volume_dd: ToDOFDescConvertible = DD_VOLUME_ALL):
        if comm_tag is None:
            raise ValueError("lazy communication requires 'comm_tag' to be supplied")

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
            dcoll: DiscretizationCollection,
            ary: ArithArrayContainerT,
            *, comm_tag: Hashable = None,
            volume_dd: ToDOFDescConvertible = None
        ) -> list[TracePair[ArithArrayContainerT]]:
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
        tags to arbitrary, potentially composite objects.

    :returns: a :class:`list` of :class:`TracePair` objects.
    """
    # {{{ process arguments

    if volume_dd is None:
        volume_dd = DD_VOLUME_ALL

    if not isinstance(volume_dd.domain_tag, VolumeDomainTag):
        raise TypeError(f"expected a volume DOFDesc, got '{volume_dd}'")
    if volume_dd.discretization_tag != DISCR_TAG_BASE:
        raise TypeError(f"expected a base-discretized DOFDesc, got '{volume_dd}'")

    # }}}

    if isinstance(ary, Number):
        # NOTE: Assumed that the same number is passed on every rank
        return [TracePair(
                volume_dd.trace(BTAG_PARTITION(remote_rank)),
                interior=ary, exterior=ary)
            for remote_rank in connected_parts(dcoll, volume_dd=volume_dd)]

    actx = get_container_context_recursively(ary)

    from grudge.array_context import MPIBasePytatoPyOpenCLArrayContext

    if isinstance(actx, MPIBasePytatoPyOpenCLArrayContext):
        rbc_class: type[
            _RankBoundaryCommunicationEager | _RankBoundaryCommunicationLazy
        ] = _RankBoundaryCommunicationLazy
    else:
        rbc_class = _RankBoundaryCommunicationEager

    cparts = connected_parts(dcoll, volume_dd=volume_dd)

    if not cparts:
        return []
    assert isinstance(actx, MPIBasedArrayContext)

    # Initialize and post all sends/receives
    rank_bdry_communicators = [
        rbc_class(actx, dcoll, ary,
                  # FIXME: This is a casualty of incomplete multi-volume support
                  # for now.
                  cast("int", remote_rank),
                  comm_tag=comm_tag, volume_dd=volume_dd)
        for remote_rank in cparts
    ]

    # Complete send/receives and return communicated data
    return cast("list[TracePair[ArithArrayContainerT]]",
                [rc.finish() for rc in rank_bdry_communicators])

# }}}


# {{{ project_tracepair

def project_tracepair(
            dcoll: DiscretizationCollection, new_dd: dof_desc.DOFDesc,
            tpair: TracePair[ArithArrayContainerT]
        ) -> TracePair[ArithArrayContainerT]:
    r"""Return a new :class:`TracePair` :func:`~grudge.op.project`\ 'ed
    onto *new_dd*.
    """
    return TracePair(
        new_dd,
        interior=project(dcoll, tpair.dd, new_dd, tpair.int),
        exterior=project(dcoll, tpair.dd, new_dd, tpair.ext)
    )


def tracepair_with_discr_tag(
            dcoll: DiscretizationCollection,
            discr_tag: DiscretizationTag,
            tpair: TracePair[ArithArrayContainerT]
        ) -> TracePair[ArithArrayContainerT]:
    r"""Return a new :class:`TracePair` :func:`~grudge.op.project`\ 'ed
    onto a :class:`~grudge.dof_desc.DOFDesc` with discretization tag *discr_tag*.
    """
    return project_tracepair(dcoll, tpair.dd.with_discr_tag(discr_tag), tpair)

# }}}

# vim: foldmethod=marker
