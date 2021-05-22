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

from pytools import memoize_on_first_arg
from pytools.obj_array import obj_array_vectorize, make_obj_array

from meshmode.dof_array import flatten, unflatten
from meshmode.mesh import BTAG_PARTITION

import numpy as np
import grudge.dof_desc as dof_desc


__doc__ = """
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


# {{{ Trace pair container class

class TracePair:
    """A container class for data (both interior and exterior restrictions)
    on the boundaries of mesh elements.

    .. attribute:: dd

        an instance of :class:`grudge.dof_desc.DOFDesc` describing the
        discretization on which :attr:`int` and :attr:`ext` live.

    .. autoattribute:: int
    .. autoattribute:: ext
    .. autoattribute:: avg

    .. note::

        :class:`TracePair` is currently used both by the symbolic (deprecated)
        and the current interfaces, with symbolic information or concrete data.
    """
    def __init__(self, dd, *, interior, exterior):
        self.dd = dof_desc.as_dofdesc(dd)
        self.interior = interior
        self.exterior = exterior

    def __getitem__(self, index):
        return TracePair(self.dd,
                         interior=self.interior[index],
                         exterior=self.exterior[index])

    def __len__(self):
        assert len(self.exterior) == len(self.interior)
        return len(self.exterior)

    @property
    def int(self):
        """A value (symbolic expression or :class:`~meshmode.dof_array.DOFArray`
        or object array of either) representing the interior value to
        be used for the flux.
        """
        return self.interior

    @property
    def ext(self):
        """A value (symbolic expression or :class:`~meshmode.dof_array.DOFArray`
        or object array of either) representing the exterior value to
        be used for the flux.
        """
        return self.exterior

    @property
    def avg(self):
        """A value (symbolic expression or :class:`~meshmode.dof_array.DOFArray`
        or object array of either) representing the average of the interior
        and exterior values.
        """
        return 0.5 * (self.int + self.ext)

# }}}


# {{{ Boundary trace pairs

def bdry_trace_pair(dcoll, dd, interior, exterior):
    """Returns a trace pair defined on the exterior boundary. Input arguments
    are assumed to already be defined on the boundary.

    :arg dcoll: a :class:`grudge.discretization.DiscretizationCollection`.
    :arg dd: a :class:`~grudge.dof_desc.DOFDesc`, or a value convertible to one,
        which describes the boundary discretization.
    :arg interior: a :class:`~meshmode.dof_array.DOFArray` that contains data
        already on the boundary representing the interior value to be used
        for the flux.
    :arg exterior: a :class:`~meshmode.dof_array.DOFArray` that contains data
        that already lives on the boundary representing the exterior value to
        be used for the flux.
    :returns: a :class:`grudge.trace_pair.TracePair` on the boundary.
    """
    return TracePair(dd, interior=interior, exterior=exterior)


def bv_trace_pair(dcoll, dd, interior, exterior):
    """Returns a trace pair defined on the exterior boundary. The interior
    argument is assumed to be defined on the volume discretization, and will
    therefore be restricted to the boundary prior to creating a
    :class:`grudge.trace_pair.TracePair`.

    :arg dcoll: a :class:`grudge.discretization.DiscretizationCollection`.
    :arg dd: a :class:`~grudge.dof_desc.DOFDesc`, or a value convertible to one,
        which describes the boundary discretization.
    :arg interior: a :class:`~meshmode.dof_array.DOFArray` that contains data
        defined in the volume, which will be restricted to the boundary denoted
        by *dd*. The result will be used as the interior value
        for the flux.
    :arg exterior: a :class:`~meshmode.dof_array.DOFArray` that contains data
        that already lives on the boundary representing the exterior value to
        be used for the flux.
    :returns: a :class:`grudge.trace_pair.TracePair` on the boundary.
    """
    from grudge.op import project

    interior = project(dcoll, "vol", dd, interior)
    return bdry_trace_pair(dcoll, dd, interior, exterior)

# }}}


# {{{ Interior trace pairs

def _interior_trace_pair(dcoll, vec):
    r"""Return a :class:`grudge.trace_pair.TracePair` for the interior faces of
    *dcoll* with a discretization tag specified by *discr_tag*.
    This does not include interior faces on different MPI ranks.

    :arg dcoll: a :class:`grudge.discretization.DiscretizationCollection`.
    :arg vec: a :class:`~meshmode.dof_array.DOFArray` or object array of
        :class:`~meshmode.dof_array.DOFArray`\ s.
    :returns: a :class:`grudge.trace_pair.TracePair` object.
    """
    from grudge.op import project

    i = project(dcoll, "vol", "int_faces", vec)

    def get_opposite_face(el):
        if isinstance(el, Number):
            return el
        else:
            return dcoll.opposite_face_connection()(el)

    e = obj_array_vectorize(get_opposite_face, i)

    return TracePair("int_faces", interior=i, exterior=e)


def interior_trace_pairs(dcoll, vec):
    r"""Return a :class:`list` of :class:`grudge.trace_pair.TracePair` objects
    defined on the interior faces of *dcoll* and any faces connected to a
    parallel boundary.

    :arg dcoll: a :class:`grudge.discretization.DiscretizationCollection`.
    :arg vec: a :class:`~meshmode.dof_array.DOFArray` or object array of
        :class:`~meshmode.dof_array.DOFArray`\ s.
    :returns: a :class:`list` of :class:`grudge.trace_pair.TracePair` objects.
    """
    return (
        [_interior_trace_pair(dcoll, vec)]
        + cross_rank_trace_pairs(dcoll, vec)
    )


def interior_trace_pair(dcoll, vec):
    from warnings import warn
    warn("`grudge.op.interior_trace_pair` is deprecated and will be dropped "
         "in version 2022.x. Use `grudge.trace_pair.interior_trace_pairs` "
         "instead, which includes contributions from different MPI ranks.",
         DeprecationWarning, stacklevel=2)
    return _interior_trace_pair(dcoll, vec)

# }}}


# {{{ Distributed-memory functionality

@memoize_on_first_arg
def connected_ranks(dcoll):
    from meshmode.distributed import get_connected_partitions
    return get_connected_partitions(dcoll._volume_discr.mesh)


class _RankBoundaryCommunication:
    base_tag = 1273

    def __init__(self, dcoll, remote_rank, vol_field, tag=None):
        self.tag = self.base_tag
        if tag is not None:
            self.tag += tag

        self.dcoll = dcoll
        self.array_context = vol_field.array_context
        self.remote_btag = BTAG_PARTITION(remote_rank)
        self.bdry_discr = dcoll.discr_from_dd(self.remote_btag)

        from grudge.op import project

        self.local_dof_array = project(dcoll, "vol", self.remote_btag, vol_field)

        local_data = self.array_context.to_numpy(flatten(self.local_dof_array))
        comm = self.dcoll.mpi_communicator

        self.send_req = comm.Isend(local_data, remote_rank, tag=self.tag)
        self.remote_data_host = np.empty_like(local_data)
        self.recv_req = comm.Irecv(self.remote_data_host, remote_rank, self.tag)

    def finish(self):
        self.recv_req.Wait()

        actx = self.array_context
        remote_dof_array = unflatten(
            self.array_context, self.bdry_discr,
            actx.from_numpy(self.remote_data_host)
        )

        bdry_conn = self.dcoll.get_distributed_boundary_swap_connection(
            dof_desc.as_dofdesc(dof_desc.DTAG_BOUNDARY(self.remote_btag))
        )
        swapped_remote_dof_array = bdry_conn(remote_dof_array)

        self.send_req.Wait()

        return TracePair(self.remote_btag,
                         interior=self.local_dof_array,
                         exterior=swapped_remote_dof_array)


def _cross_rank_trace_pairs_scalar_field(dcoll, vec, tag=None):
    if isinstance(vec, Number):
        return [TracePair(BTAG_PARTITION(remote_rank), interior=vec, exterior=vec)
                for remote_rank in connected_ranks(dcoll)]
    else:
        rbcomms = [_RankBoundaryCommunication(dcoll, remote_rank, vec, tag=tag)
                   for remote_rank in connected_ranks(dcoll)]
        return [rbcomm.finish() for rbcomm in rbcomms]


def cross_rank_trace_pairs(dcoll, ary, tag=None):
    r"""Get a :class:`list` of *ary* trace pairs for each partition boundary.

    For each partition boundary, the field data values in *ary* are
    communicated to/from the neighboring partition. Presumably, this
    communication is MPI (but strictly speaking, may not be, and this
    routine is agnostic to the underlying communication).

    For each face on each partition boundary, a
    :class:`grudge.trace_pair.TracePair` is created with the locally, and
    remotely owned partition boundary face data as the `internal`, and `external`
    components, respectively. Each of the TracePair components are structured
    like *ary*.

    :arg dcoll: a :class:`grudge.discretization.DiscretizationCollection`.
    :arg ary: a single :class:`~meshmode.dof_array.DOFArray`, or an object
        array of :class:`~meshmode.dof_array.DOFArray`\ s
        of arbitrary shape.
    :returns: a :class:`list` of :class:`grudge.trace_pair.TracePair` objects.
    """
    if isinstance(ary, np.ndarray):
        oshape = ary.shape
        comm_vec = ary.flatten()

        n, = comm_vec.shape
        result = {}
        # FIXME: Batch this communication rather than
        # doing it in sequence.
        for ivec in range(n):
            for rank_tpair in _cross_rank_trace_pairs_scalar_field(
                    dcoll, comm_vec[ivec]):
                assert isinstance(rank_tpair.dd.domain_tag, dof_desc.DTAG_BOUNDARY)
                assert isinstance(rank_tpair.dd.domain_tag.tag, BTAG_PARTITION)
                result[rank_tpair.dd.domain_tag.tag.part_nr, ivec] = rank_tpair

        return [
            TracePair(
                dd=dof_desc.as_dofdesc(
                    dof_desc.DTAG_BOUNDARY(BTAG_PARTITION(remote_rank))),
                interior=make_obj_array([
                    result[remote_rank, i].int for i in range(n)]).reshape(oshape),
                exterior=make_obj_array([
                    result[remote_rank, i].ext for i in range(n)]).reshape(oshape)
            ) for remote_rank in connected_ranks(dcoll)
        ]
    else:
        return _cross_rank_trace_pairs_scalar_field(dcoll, ary, tag=tag)

# }}}


# vim: foldmethod=marker