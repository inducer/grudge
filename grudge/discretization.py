from __future__ import division, absolute_import

__copyright__ = "Copyright (C) 2015-2017 Andreas Kloeckner, Bogdan Enache"

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


import six
from pytools import memoize_method
import pyopencl as cl
from grudge import sym
import numpy as np


# FIXME Naming not ideal
class DiscretizationBase(object):
    pass


class DGDiscretizationWithBoundaries(DiscretizationBase):
    """
    .. automethod :: discr_from_dd
    .. automethod :: connection_from_dds

    .. autoattribute :: cl_context
    .. autoattribute :: dim
    .. autoattribute :: ambient_dim
    .. autoattribute :: mesh

    .. automethod :: empty
    .. automethod :: zeros
    """

    def __init__(self, cl_ctx, mesh, order, quad_tag_to_group_factory=None,
            mpi_communicator=None):
        """
        :param quad_tag_to_group_factory: A mapping from quadrature tags (typically
            strings--but may be any hashable/comparable object) to a
            :class:`meshmode.discretization.ElementGroupFactory` indicating with
            which quadrature discretization the operations are to be carried out,
            or *None* to indicate that operations with this quadrature tag should
            be carried out with the standard volume discretization.
        """

        if quad_tag_to_group_factory is None:
            quad_tag_to_group_factory = {}

        self.order = order
        self.quad_tag_to_group_factory = quad_tag_to_group_factory

        from meshmode.discretization import Discretization

        self._volume_discr = Discretization(cl_ctx, mesh,
                self.group_factory_for_quadrature_tag(sym.QTAG_NONE))

        # {{{ management of discretization-scoped common subexpressions

        from pytools import UniqueNameGenerator
        self._discr_scoped_name_gen = UniqueNameGenerator()

        self._discr_scoped_subexpr_to_name = {}
        self._discr_scoped_subexpr_name_to_value = {}

        # }}}

        with cl.CommandQueue(cl_ctx) as queue:
            self._dist_boundary_connections = \
                    self._set_up_distributed_communication(mpi_communicator, queue)

        self.mpi_communicator = mpi_communicator

    def get_management_rank_index(self):
        return 0

    def is_management_rank(self):
        if self.mpi_communicator is None:
            return True
        else:
            return self.mpi_communicator.Get_rank() \
                    == self._get_management_rank_index()

    def _set_up_distributed_communication(self, mpi_communicator, queue):
        from_dd = sym.DOFDesc("vol", sym.QTAG_NONE)

        from meshmode.distributed import get_connected_partitions
        connected_parts = get_connected_partitions(self._volume_discr.mesh)

        if mpi_communicator is None and connected_parts:
            raise RuntimeError("must supply an MPI communicator when using a "
                    "distributed mesh")

        grp_factory = self.group_factory_for_quadrature_tag(sym.QTAG_NONE)

        setup_helpers = {}
        boundary_connections = {}

        from meshmode.distributed import MPIBoundaryCommSetupHelper
        for i_remote_part in connected_parts:
            conn = self.connection_from_dds(
                    from_dd,
                    sym.DOFDesc(sym.BTAG_PARTITION(i_remote_part), sym.QTAG_NONE))
            setup_helper = setup_helpers[i_remote_part] = MPIBoundaryCommSetupHelper(
                    mpi_communicator, queue, conn, i_remote_part, grp_factory)
            setup_helper.post_sends()

        for i_remote_part, setup_helper in six.iteritems(setup_helpers):
            boundary_connections[i_remote_part] = setup_helper.complete_setup()

        return boundary_connections

    def get_distributed_boundary_swap_connection(self, dd):
        if dd.quadrature_tag != sym.QTAG_NONE:
            # FIXME
            raise NotImplementedError("Distributed communication with quadrature")

        assert isinstance(dd.domain_tag, sym.DTAG_BOUNDARY)
        assert isinstance(dd.domain_tag.tag, sym.BTAG_PARTITION)

        return self._dist_boundary_connections[dd.domain_tag.tag.part_nr]

    @memoize_method
    def discr_from_dd(self, dd):
        dd = sym.as_dofdesc(dd)

        qtag = dd.quadrature_tag

        if dd.is_volume():
            if qtag is not sym.QTAG_NONE:
                return self._quad_volume_discr(qtag)
            return self._volume_discr

        if qtag is not sym.QTAG_NONE:
            no_quad_discr = self.discr_from_dd(sym.DOFDesc(dd.domain_tag))

            from meshmode.discretization import Discretization
            return Discretization(
                    self._volume_discr.cl_context,
                    no_quad_discr.mesh,
                    self.group_factory_for_quadrature_tag(qtag))

        assert qtag is sym.QTAG_NONE

        if dd.domain_tag is sym.FACE_RESTR_ALL:
            return self._all_faces_volume_connection().to_discr
        elif dd.domain_tag is sym.FACE_RESTR_INTERIOR:
            return self._interior_faces_connection().to_discr
        elif dd.is_boundary_or_partition_interface():
            return self._boundary_connection(dd.domain_tag.tag).to_discr
        else:
            raise ValueError("DOF desc tag not understood: " + str(dd))

    @memoize_method
    def connection_from_dds(self, from_dd, to_dd):
        from_dd = sym.as_dofdesc(from_dd)
        to_dd = sym.as_dofdesc(to_dd)

        to_qtag = to_dd.quadrature_tag

        if (
                not from_dd.is_volume()
                and from_dd.quadrature_tag == to_dd.quadrature_tag
                and to_dd.domain_tag is sym.FACE_RESTR_ALL):
            faces_conn = self.connection_from_dds(
                    sym.DOFDesc("vol"),
                    sym.DOFDesc(from_dd.domain_tag))

            from meshmode.discretization.connection import \
                    make_face_to_all_faces_embedding

            return make_face_to_all_faces_embedding(
                    faces_conn, self.discr_from_dd(to_dd),
                    self.discr_from_dd(from_dd))

        # {{{ simplify domain + qtag change into chained

        if (
                from_dd.domain_tag != to_dd.domain_tag
                and from_dd.quadrature_tag is sym.QTAG_NONE
                and to_dd.quadrature_tag is not sym.QTAG_NONE):

            from meshmode.discretization.connection import \
                    ChainedDiscretizationConnection
            intermediate_dd = sym.DOFDesc(to_dd.domain_tag)
            return ChainedDiscretizationConnection(
                    [
                        # first change domain
                        self.connection_from_dds(
                            from_dd,
                            intermediate_dd),

                        # then go to quad grid
                        self.connection_from_dds(
                            intermediate_dd,
                            to_dd
                            )])

        # }}}

        # {{{ generic to-quad

        if (
                from_dd.domain_tag == to_dd.domain_tag
                and from_dd.quadrature_tag is sym.QTAG_NONE
                and to_dd.quadrature_tag is not sym.QTAG_NONE):
            from meshmode.discretization.connection.same_mesh import \
                    make_same_mesh_connection

            return make_same_mesh_connection(
                    self.discr_from_dd(to_dd),
                    self.discr_from_dd(from_dd))

        # }}}

        if from_dd.quadrature_tag is not sym.QTAG_NONE:
            raise ValueError("cannot interpolate *from* a "
                    "(non-interpolatory) quadrature grid")

        assert to_qtag is sym.QTAG_NONE

        if from_dd.is_volume():
            if to_dd.domain_tag is sym.FACE_RESTR_ALL:
                return self._all_faces_volume_connection()
            if to_dd.domain_tag is sym.FACE_RESTR_INTERIOR:
                return self._interior_faces_connection()
            elif to_dd.is_boundary_or_partition_interface():
                assert from_dd.quadrature_tag is sym.QTAG_NONE
                return self._boundary_connection(to_dd.domain_tag.tag)
            elif to_dd.is_volume():
                from meshmode.discretization.connection import \
                        make_same_mesh_connection
                to_discr = self._quad_volume_discr(to_dd.quadrature_tag)
                from_discr = self._volume_discr
                return make_same_mesh_connection(to_discr, from_discr)

            else:
                raise ValueError("cannot interpolate from volume to: " + str(to_dd))

        else:
            raise ValueError("cannot interpolate from: " + str(from_dd))

    def group_factory_for_quadrature_tag(self, quadrature_tag):
        """
        OK to override in user code to control mode/node choice.
        """

        if quadrature_tag is None:
            quadrature_tag = sym.QTAG_NONE

        from meshmode.discretization.poly_element import \
                PolynomialWarpAndBlendGroupFactory

        if quadrature_tag is not sym.QTAG_NONE:
            return self.quad_tag_to_group_factory[quadrature_tag]
        else:
            return PolynomialWarpAndBlendGroupFactory(order=self.order)

    @memoize_method
    def _quad_volume_discr(self, quadrature_tag):
        from meshmode.discretization import Discretization

        return Discretization(self._volume_discr.cl_context, self._volume_discr.mesh,
                self.group_factory_for_quadrature_tag(quadrature_tag))

    # {{{ boundary

    @memoize_method
    def _boundary_connection(self, boundary_tag):
        from meshmode.discretization.connection import make_face_restriction
        return make_face_restriction(
                        self._volume_discr,
                        self.group_factory_for_quadrature_tag(sym.QTAG_NONE),
                        boundary_tag=boundary_tag)

    # }}}

    # {{{ interior faces

    @memoize_method
    def _interior_faces_connection(self):
        from meshmode.discretization.connection import (
                make_face_restriction, FACE_RESTR_INTERIOR)
        return make_face_restriction(
                        self._volume_discr,
                        self.group_factory_for_quadrature_tag(sym.QTAG_NONE),
                        FACE_RESTR_INTERIOR,

                        # FIXME: This will need to change as soon as we support
                        # pyramids or other elements with non-identical face
                        # types.
                        per_face_groups=False)

    @memoize_method
    def opposite_face_connection(self):
        from meshmode.discretization.connection import \
                make_opposite_face_connection

        return make_opposite_face_connection(self._interior_faces_connection())

    # }}}

    # {{{ all-faces

    @memoize_method
    def _all_faces_volume_connection(self):
        from meshmode.discretization.connection import (
                make_face_restriction, FACE_RESTR_ALL)
        return make_face_restriction(
                        self._volume_discr,
                        self.group_factory_for_quadrature_tag(sym.QTAG_NONE),
                        FACE_RESTR_ALL,

                        # FIXME: This will need to change as soon as we support
                        # pyramids or other elements with non-identical face
                        # types.
                        per_face_groups=False)

    # }}}

    @property
    def cl_context(self):
        return self._volume_discr.cl_context

    @property
    def dim(self):
        return self._volume_discr.dim

    @property
    def ambient_dim(self):
        return self._volume_discr.ambient_dim

    @property
    def real_dtype(self):
        return self._volume_discr.real_dtype

    @property
    def complex_dtype(self):
        return self._volume_discr.complex_dtype

    @property
    def mesh(self):
        return self._volume_discr.mesh

    def empty(self, queue=None, dtype=None, extra_dims=None, allocator=None):
        return self._volume_discr.empty(queue, dtype, extra_dims=extra_dims,
                allocator=allocator)

    def zeros(self, queue, dtype=None, extra_dims=None, allocator=None):
        return self._volume_discr.zeros(queue, dtype, extra_dims=extra_dims,
                allocator=allocator)

    def is_volume_where(self, where):
        from grudge import sym
        return (
                where is None
                or where == sym.VTAG_ALL)


class PointsDiscretization(DiscretizationBase):
    """Implements just enough of the discretization interface to be
    able to smuggle some points into :func:`bind`.
    """

    def __init__(self, nodes):
        self._nodes = nodes
        self.real_dtype = np.dtype(np.float64)
        self.complex_dtype = np.dtype({
                np.float32: np.complex64,
                np.float64: np.complex128
        }[self.real_dtype.type])

        self.mpi_communicator = None

    def ambient_dim(self):
        return self._nodes.shape[0]

    @property
    def mesh(self):
        return self

    @property
    def nnodes(self):
        return self._nodes.shape[-1]

    def nodes(self):
        return self._nodes

    @property
    def facial_adjacency_groups(self):
        return []

    def discr_from_dd(self, dd):
        dd = sym.as_dofdesc(dd)

        if dd.quadrature_tag is not sym.QTAG_NONE:
            raise ValueError("quadrature discretization requested from "
                    "PointsDiscretization")
        if dd.domain_tag is not sym.DTAG_VOLUME_ALL:
            raise ValueError("non-volume discretization requested from "
                    "PointsDiscretization")

        return self

    @property
    def quad_tag_to_group_factory(self):
        return {}


# vim: foldmethod=marker
