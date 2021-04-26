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

from pytools import memoize_method
from grudge.dof_desc import \
    QTAG_NONE, QTAG_MODAL, DTAG_BOUNDARY, DOFDesc, as_dofdesc
import numpy as np  # noqa: F401
from meshmode.array_context import ArrayContext
from meshmode.discretization.connection import \
    FACE_RESTR_INTERIOR, FACE_RESTR_ALL, make_face_restriction
from meshmode.mesh import BTAG_PARTITION


__doc__ = """
.. autoclass:: DiscretizationCollection
"""


class DiscretizationCollection:
    """
    .. automethod :: discr_from_dd
    .. automethod :: connection_from_dds

    .. autoattribute :: dim
    .. autoattribute :: ambient_dim
    .. autoattribute :: mesh

    .. automethod :: empty
    .. automethod :: zeros
    """

    def __init__(self, array_context, mesh, order=None,
            quad_tag_to_group_factory=None, mpi_communicator=None):
        """
        :param quad_tag_to_group_factory: A mapping from quadrature tags (typically
            strings--but may be any hashable/comparable object) to a
            :class:`~meshmode.discretization.poly_element.ElementGroupFactory`
            indicating with which quadrature discretization the operations are
            to be carried out, or *None* to indicate that operations with this
            quadrature tag should be carried out with the standard volume
            discretization.
        """

        self._setup_actx = array_context

        from meshmode.discretization.poly_element import \
                PolynomialWarpAndBlendGroupFactory

        if quad_tag_to_group_factory is None:
            if order is None:
                raise TypeError("one of 'order' and "
                        "'quad_tag_to_group_factory' must be given")

            quad_tag_to_group_factory = {
                    QTAG_NONE: PolynomialWarpAndBlendGroupFactory(order=order)}
        else:
            if order is not None:
                quad_tag_to_group_factory = quad_tag_to_group_factory.copy()
                if QTAG_NONE in quad_tag_to_group_factory:
                    raise ValueError("if 'order' is given, "
                            "'quad_tag_to_group_factory' must not have a "
                            "key of QTAG_NONE")

                quad_tag_to_group_factory[QTAG_NONE] = \
                        PolynomialWarpAndBlendGroupFactory(order=order)

        # Modal discr should always comes from the base discretization
        quad_tag_to_group_factory[QTAG_MODAL] = \
            _generate_modal_group_factory(
                quad_tag_to_group_factory[QTAG_NONE]
            )

        self.quad_tag_to_group_factory = quad_tag_to_group_factory

        from meshmode.discretization import Discretization

        self._volume_discr = Discretization(array_context, mesh,
                self.group_factory_for_quadrature_tag(QTAG_NONE))

        # {{{ management of discretization-scoped common subexpressions

        from pytools import UniqueNameGenerator
        self._discr_scoped_name_gen = UniqueNameGenerator()

        self._discr_scoped_subexpr_to_name = {}
        self._discr_scoped_subexpr_name_to_value = {}

        # }}}

        self._dist_boundary_connections = \
                self._set_up_distributed_communication(
                        mpi_communicator, array_context)

        self.mpi_communicator = mpi_communicator

    def get_management_rank_index(self):
        return 0

    def is_management_rank(self):
        if self.mpi_communicator is None:
            return True
        else:
            return self.mpi_communicator.Get_rank() \
                    == self.get_management_rank_index()

    def _set_up_distributed_communication(self, mpi_communicator, array_context):
        from_dd = DOFDesc("vol", QTAG_NONE)

        boundary_connections = {}

        from meshmode.distributed import get_connected_partitions
        connected_parts = get_connected_partitions(self._volume_discr.mesh)

        if connected_parts:
            if mpi_communicator is None:
                raise RuntimeError("must supply an MPI communicator when using a "
                    "distributed mesh")

            grp_factory = self.group_factory_for_quadrature_tag(QTAG_NONE)

            local_boundary_connections = {}
            for i_remote_part in connected_parts:
                local_boundary_connections[i_remote_part] = self.connection_from_dds(
                        from_dd, DOFDesc(BTAG_PARTITION(i_remote_part),
                        QTAG_NONE))

            from meshmode.distributed import MPIBoundaryCommSetupHelper
            with MPIBoundaryCommSetupHelper(mpi_communicator, array_context,
                    local_boundary_connections, grp_factory) as bdry_setup_helper:
                while True:
                    conns = bdry_setup_helper.complete_some()
                    if not conns:
                        break
                    for i_remote_part, conn in conns.items():
                        boundary_connections[i_remote_part] = conn

        return boundary_connections

    def get_distributed_boundary_swap_connection(self, dd):
        if dd.quadrature_tag != QTAG_NONE:
            # FIXME
            raise NotImplementedError("Distributed communication with quadrature")

        assert isinstance(dd.domain_tag, DTAG_BOUNDARY)
        assert isinstance(dd.domain_tag.tag, BTAG_PARTITION)

        return self._dist_boundary_connections[dd.domain_tag.tag.part_nr]

    @memoize_method
    def discr_from_dd(self, dd):
        dd = as_dofdesc(dd)

        qtag = dd.quadrature_tag

        if qtag is QTAG_MODAL:
            return self._modal_discr(dd)

        if dd.is_volume():
            if qtag is not QTAG_NONE:
                return self._quad_volume_discr(qtag)
            return self._volume_discr

        if qtag is not QTAG_NONE:
            no_quad_discr = self.discr_from_dd(DOFDesc(dd.domain_tag))

            from meshmode.discretization import Discretization
            return Discretization(
                    self._setup_actx,
                    no_quad_discr.mesh,
                    self.group_factory_for_quadrature_tag(qtag))

        assert qtag is QTAG_NONE

        if dd.domain_tag is FACE_RESTR_ALL:
            return self._all_faces_volume_connection().to_discr
        elif dd.domain_tag is FACE_RESTR_INTERIOR:
            return self._interior_faces_connection().to_discr
        elif dd.is_boundary_or_partition_interface():
            return self._boundary_connection(dd.domain_tag.tag).to_discr
        else:
            raise ValueError("DOF desc tag not understood: " + str(dd))

    @memoize_method
    def connection_from_dds(self, from_dd, to_dd):
        from_dd = as_dofdesc(from_dd)
        to_dd = as_dofdesc(to_dd)

        to_qtag = to_dd.quadrature_tag
        from_qtag = from_dd.quadrature_tag

        # {{{ mapping between modal and nodal representations

        if (from_qtag is QTAG_MODAL and to_qtag is not QTAG_MODAL):
            return self._modal_to_nodal_connection(to_dd)

        if (to_qtag is QTAG_MODAL and from_qtag is not QTAG_MODAL):
            return self._nodal_to_modal_connection(from_dd)

        # }}}

        assert (to_qtag is not QTAG_MODAL and from_qtag is not QTAG_MODAL)

        if (
                not from_dd.is_volume()
                and from_qtag == to_qtag
                and to_dd.domain_tag is FACE_RESTR_ALL):
            faces_conn = self.connection_from_dds(
                    DOFDesc("vol"),
                    DOFDesc(from_dd.domain_tag))

            from meshmode.discretization.connection import \
                    make_face_to_all_faces_embedding

            return make_face_to_all_faces_embedding(
                    self._setup_actx,
                    faces_conn, self.discr_from_dd(to_dd),
                    self.discr_from_dd(from_dd))

        # {{{ simplify domain + qtag change into chained

        if (from_dd.domain_tag != to_dd.domain_tag
                and from_qtag is QTAG_NONE
                and to_qtag is not QTAG_NONE):

            from meshmode.discretization.connection import \
                    ChainedDiscretizationConnection
            intermediate_dd = DOFDesc(to_dd.domain_tag)
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

        if (from_dd.domain_tag == to_dd.domain_tag
                and from_qtag is QTAG_NONE
                and to_qtag is not QTAG_NONE):

            from meshmode.discretization.connection.same_mesh import \
                    make_same_mesh_connection

            return make_same_mesh_connection(
                    self._setup_actx,
                    self.discr_from_dd(to_dd),
                    self.discr_from_dd(from_dd))

        # }}}

        if from_qtag is not QTAG_NONE:
            raise ValueError("cannot interpolate *from* a "
                    "(non-interpolatory) quadrature grid")

        assert to_qtag is QTAG_NONE

        if from_dd.is_volume():
            if to_dd.domain_tag is FACE_RESTR_ALL:
                return self._all_faces_volume_connection()
            if to_dd.domain_tag is FACE_RESTR_INTERIOR:
                return self._interior_faces_connection()
            elif to_dd.is_boundary_or_partition_interface():
                assert from_qtag is QTAG_NONE
                return self._boundary_connection(to_dd.domain_tag.tag)
            elif to_dd.is_volume():
                from meshmode.discretization.connection import \
                        make_same_mesh_connection
                to_discr = self._quad_volume_discr(to_qtag)
                from_discr = self._volume_discr
                return make_same_mesh_connection(self._setup_actx, to_discr,
                            from_discr)

            else:
                raise ValueError("cannot interpolate from volume to: " + str(to_dd))

        else:
            raise ValueError("cannot interpolate from: " + str(from_dd))

    def group_factory_for_quadrature_tag(self, quadrature_tag):
        """
        OK to override in user code to control mode/node choice.
        """

        if quadrature_tag is None:
            quadrature_tag = QTAG_NONE

        return self.quad_tag_to_group_factory[quadrature_tag]

    @memoize_method
    def _quad_volume_discr(self, quadrature_tag):
        from meshmode.discretization import Discretization

        return Discretization(self._setup_actx, self._volume_discr.mesh,
                self.group_factory_for_quadrature_tag(quadrature_tag))

    # {{{ modal to nodal connections

    @memoize_method
    def _modal_discr(self, domain_tag):
        from meshmode.discretization import Discretization

        discr_base = self.discr_from_dd(DOFDesc(domain_tag, QTAG_NONE))
        return Discretization(
            self._setup_actx, discr_base.mesh,
            self.group_factory_for_quadrature_tag(QTAG_MODAL)
        )

    @memoize_method
    def _modal_to_nodal_connection(self, to_dd):
        """
        :arg to_dd: a :class:`grudge.dof_desc.DOFDesc`
            describing the dofs corresponding to the
            *to_discr*
        """
        from meshmode.discretization.connection import \
            ModalToNodalDiscretizationConnection

        return ModalToNodalDiscretizationConnection(
            from_discr=self._modal_discr(to_dd.domain_tag),
            to_discr=self.discr_from_dd(to_dd)
        )

    @memoize_method
    def _nodal_to_modal_connection(self, from_dd):
        """
        :arg from_dd: a :class:`grudge.dof_desc.DOFDesc`
            describing the dofs corresponding to the
            *from_discr*
        """
        from meshmode.discretization.connection import \
            NodalToModalDiscretizationConnection

        return NodalToModalDiscretizationConnection(
            from_discr=self.discr_from_dd(from_dd),
            to_discr=self._modal_discr(from_dd.domain_tag)
        )

    # }}}

    # {{{ boundary

    @memoize_method
    def _boundary_connection(self, boundary_tag):
        return make_face_restriction(
                self._setup_actx,
                self._volume_discr,
                self.group_factory_for_quadrature_tag(QTAG_NONE),
                boundary_tag=boundary_tag)

    # }}}

    # {{{ interior faces

    @memoize_method
    def _interior_faces_connection(self):
        return make_face_restriction(
                self._setup_actx,
                self._volume_discr,
                self.group_factory_for_quadrature_tag(QTAG_NONE),
                FACE_RESTR_INTERIOR,

                # FIXME: This will need to change as soon as we support
                # pyramids or other elements with non-identical face
                # types.
                per_face_groups=False)

    @memoize_method
    def opposite_face_connection(self):
        from meshmode.discretization.connection import \
                make_opposite_face_connection

        return make_opposite_face_connection(
                self._setup_actx,
                self._interior_faces_connection())

    # }}}

    # {{{ all-faces

    @memoize_method
    def _all_faces_volume_connection(self):
        return make_face_restriction(
                self._setup_actx,
                self._volume_discr,
                self.group_factory_for_quadrature_tag(QTAG_NONE),
                FACE_RESTR_ALL,

                # FIXME: This will need to change as soon as we support
                # pyramids or other elements with non-identical face
                # types.
                per_face_groups=False)

    # }}}

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

    def empty(self, array_context: ArrayContext, dtype=None):
        return self._volume_discr.empty(array_context, dtype)

    def zeros(self, array_context: ArrayContext, dtype=None):
        return self._volume_discr.zeros(array_context, dtype)

    def is_volume_where(self, where):
        return where is None or as_dofdesc(where).is_volume()

    @property
    def order(self):
        from warnings import warn
        warn("DiscretizationCollection.order is deprecated, "
                "consider using the orders of element groups instead. "
                "'order' will go away in 2021.",
                DeprecationWarning, stacklevel=2)

        from pytools import single_valued
        return single_valued(egrp.order for egrp in self._volume_discr.groups)


class DGDiscretizationWithBoundaries(DiscretizationCollection):
    def __init__(self, *args, **kwargs):
        from warnings import warn
        warn("DGDiscretizationWithBoundaries is deprecated and will go away "
                "in 2022. Use DiscretizationCollection instead.",
                DeprecationWarning, stacklevel=2)

        super().__init__(*args, **kwargs)


def _generate_modal_group_factory(nodal_group_factory):
    from meshmode.discretization.poly_element import (
        ModalSimplexGroupFactory,
        ModalTensorProductGroupFactory
    )
    from meshmode.mesh import SimplexElementGroup, TensorProductElementGroup

    order = nodal_group_factory.order
    mesh_group_cls = nodal_group_factory.mesh_group_class

    if mesh_group_cls is SimplexElementGroup:
        return ModalSimplexGroupFactory(order=order)
    elif mesh_group_cls is TensorProductElementGroup:
        return ModalTensorProductGroupFactory(order=order)
    else:
        raise ValueError(
            f"Unknown mesh element group: {mesh_group_cls}"
        )

# vim: foldmethod=marker
