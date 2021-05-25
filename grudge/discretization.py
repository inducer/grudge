"""
.. currentmodule:: grudge

.. autoclass:: DiscretizationCollection
"""

__copyright__ = """
Copyright (C) 2015-2017 Andreas Kloeckner, Bogdan Enache
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

from pytools import memoize_method

from grudge.dof_desc import (
    DISCR_TAG_BASE, DISCR_TAG_MODAL,
    DTAG_BOUNDARY, DOFDesc, as_dofdesc
)

import numpy as np  # noqa: F401

from arraycontext.context import ArrayContext

from meshmode.discretization.connection import (
    FACE_RESTR_INTERIOR,
    FACE_RESTR_ALL,
    make_face_restriction
)
from meshmode.mesh import BTAG_PARTITION

from warnings import warn


class DiscretizationCollection:
    """A collection of discretizations on various mesh entities
    (volume, interior facets, boundaries) and associated element
    groups.

    .. automethod:: __init__

    .. autoattribute:: dim
    .. autoattribute:: ambient_dim
    .. autoattribute:: mesh
    .. autoattribute:: real_dtype
    .. autoattribute:: complex_dtype

    .. automethod:: discr_from_dd
    .. automethod:: connection_from_dds

    .. automethod:: empty
    .. automethod:: zeros
    """

    def __init__(self, array_context, mesh, order=None,
            discr_tag_to_group_factory=None, mpi_communicator=None,
            # FIXME: `quad_tag_to_group_factory` is deprecated
            quad_tag_to_group_factory=None):
        """Constructor for the :class:`DiscretizationCollection` object.

        :arg actx: an :class:`~arraycontext.context.ArrayContext`.
        :arg mesh: a :class:`~meshmode.mesh.Mesh` object.
        :arg discr_tag_to_group_factory: A mapping from discretization tags
            (typically one of: :class:`grudge.dof_desc.DISCR_TAG_BASE`,
            :class:`grudge.dof_desc.DISCR_TAG_MODAL`, or
            :class:`grudge.dof_desc.DISCR_TAG_QUAD`) to a
            :class:`~meshmode.discretization.poly_element.ElementGroupFactory`
            indicating with which type of discretization the operations are
            to be carried out, or *None* to indicate that operations with this
            discretization tag should be carried out with the standard volume
            discretization.
        """

        if (quad_tag_to_group_factory is not None
                and discr_tag_to_group_factory is not None):
            raise ValueError(
                "Both `quad_tag_to_group_factory` and `discr_tag_to_group_factory` "
                "are specified. Use `discr_tag_to_group_factory` instead."
            )

        # FIXME: `quad_tag_to_group_factory` is deprecated
        if (quad_tag_to_group_factory is not None
                and discr_tag_to_group_factory is None):
            warn("`quad_tag_to_group_factory` is a deprecated kwarg and will "
                 "be dropped in version 2022.x. Use `discr_tag_to_group_factory` "
                 "instead.",
                 DeprecationWarning, stacklevel=2)
            discr_tag_to_group_factory = quad_tag_to_group_factory

        self._setup_actx = array_context

        from meshmode.discretization.poly_element import \
                PolynomialWarpAndBlendGroupFactory

        if discr_tag_to_group_factory is None:
            if order is None:
                raise TypeError(
                    "one of 'order' and 'discr_tag_to_group_factory' must be given"
                )

            discr_tag_to_group_factory = {
                    DISCR_TAG_BASE: PolynomialWarpAndBlendGroupFactory(order=order)}
        else:
            if order is not None:
                discr_tag_to_group_factory = discr_tag_to_group_factory.copy()
                if DISCR_TAG_BASE in discr_tag_to_group_factory:
                    raise ValueError(
                        "if 'order' is given, 'discr_tag_to_group_factory' must "
                        "not have a key of DISCR_TAG_BASE"
                    )

                discr_tag_to_group_factory[DISCR_TAG_BASE] = \
                        PolynomialWarpAndBlendGroupFactory(order=order)

        # Modal discr should always comes from the base discretization
        discr_tag_to_group_factory[DISCR_TAG_MODAL] = \
            _generate_modal_group_factory(
                discr_tag_to_group_factory[DISCR_TAG_BASE]
            )

        self.discr_tag_to_group_factory = discr_tag_to_group_factory

        from meshmode.discretization import Discretization

        self._volume_discr = Discretization(
            array_context, mesh,
            self.group_factory_for_discretization_tag(DISCR_TAG_BASE)
        )

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

    @property
    def quad_tag_to_group_factory(self):
        warn("`DiscretizationCollection.quad_tag_to_group_factory` "
             "is deprecated and will go away in 2022. Use "
             "`DiscretizationCollection.discr_tag_to_group_factory` "
             "instead.",
             DeprecationWarning, stacklevel=2)

        return self.discr_tag_to_group_factory

    def get_management_rank_index(self):
        return 0

    def is_management_rank(self):
        if self.mpi_communicator is None:
            return True
        else:
            return self.mpi_communicator.Get_rank() \
                    == self.get_management_rank_index()

    def _set_up_distributed_communication(self, mpi_communicator, array_context):
        from_dd = DOFDesc("vol", DISCR_TAG_BASE)

        boundary_connections = {}

        from meshmode.distributed import get_connected_partitions
        connected_parts = get_connected_partitions(self._volume_discr.mesh)

        if connected_parts:
            if mpi_communicator is None:
                raise RuntimeError("must supply an MPI communicator when using a "
                    "distributed mesh")

            grp_factory = \
                self.group_factory_for_discretization_tag(DISCR_TAG_BASE)

            local_boundary_connections = {}
            for i_remote_part in connected_parts:
                local_boundary_connections[i_remote_part] = self.connection_from_dds(
                        from_dd, DOFDesc(BTAG_PARTITION(i_remote_part),
                        DISCR_TAG_BASE))

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
        """Provides a mapping from the base volume discretization
        to the exterior boundary restriction on a parallel boundary
        partition described by *dd*. This connection is used to
        communicate across element boundaries in different parallel
        partitions during distributed runs.

        :arg dd: a :class:`~grudge.dof_desc.DOFDesc`, or a value
            convertible to one. The domain tag must be a subclass
            of :class:`grudge.dof_desc.DTAG_BOUNDARY` with an
            associated :class:`meshmode.mesh.BTAG_PARTITION`
            corresponding to a particular communication rank.
        """
        if dd.discretization_tag is not DISCR_TAG_BASE:
            # FIXME
            raise NotImplementedError(
                "Distributed communication with discretization tag "
                f"{dd.discretization_tag} is not implemented."
            )

        assert isinstance(dd.domain_tag, DTAG_BOUNDARY)
        assert isinstance(dd.domain_tag.tag, BTAG_PARTITION)

        return self._dist_boundary_connections[dd.domain_tag.tag.part_nr]

    @memoize_method
    def discr_from_dd(self, dd):
        """Provides a :class:`meshmode.discretization.Discretization`
        object from *dd*.

        :arg dd: a :class:`~grudge.dof_desc.DOFDesc`, or a value
            convertible to one.
        """
        dd = as_dofdesc(dd)

        discr_tag = dd.discretization_tag

        if discr_tag is DISCR_TAG_MODAL:
            return self._modal_discr(dd.domain_tag)

        if dd.is_volume():
            if discr_tag is not DISCR_TAG_BASE:
                return self._discr_tag_volume_discr(discr_tag)
            return self._volume_discr

        if discr_tag is not DISCR_TAG_BASE:
            no_quad_discr = self.discr_from_dd(DOFDesc(dd.domain_tag))

            from meshmode.discretization import Discretization
            return Discretization(
                self._setup_actx,
                no_quad_discr.mesh,
                self.group_factory_for_discretization_tag(discr_tag)
            )

        assert discr_tag is DISCR_TAG_BASE

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
        """Provides a mapping (connection) from one discretization to
        another, e.g. from the volume to the boundary, or from the
        base to the an overintegrated quadrature discretization, or from
        a nodal representation to a modal representation.

        :arg from_dd: a :class:`~grudge.dof_desc.DOFDesc`, or a value
            convertible to one.
        :arg to_dd: a :class:`~grudge.dof_desc.DOFDesc`, or a value
            convertible to one.
        """
        from_dd = as_dofdesc(from_dd)
        to_dd = as_dofdesc(to_dd)

        to_discr_tag = to_dd.discretization_tag
        from_discr_tag = from_dd.discretization_tag

        # {{{ mapping between modal and nodal representations

        if (from_discr_tag is DISCR_TAG_MODAL
                and to_discr_tag is not DISCR_TAG_MODAL):
            return self._modal_to_nodal_connection(to_dd)

        if (to_discr_tag is DISCR_TAG_MODAL
                and from_discr_tag is not DISCR_TAG_MODAL):
            return self._nodal_to_modal_connection(from_dd)

        # }}}

        assert (to_discr_tag is not DISCR_TAG_MODAL
                    and from_discr_tag is not DISCR_TAG_MODAL)

        if (not from_dd.is_volume()
                and from_discr_tag == to_discr_tag
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

        # {{{ simplify domain + discr_tag change into chained

        if (from_dd.domain_tag != to_dd.domain_tag
                and from_discr_tag is DISCR_TAG_BASE
                and to_discr_tag is not DISCR_TAG_BASE):

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

        # DISCR_TAG_MODAL is handled above
        if (from_dd.domain_tag == to_dd.domain_tag
                and from_discr_tag is DISCR_TAG_BASE
                and to_discr_tag is not DISCR_TAG_BASE):

            from meshmode.discretization.connection.same_mesh import \
                    make_same_mesh_connection

            return make_same_mesh_connection(
                    self._setup_actx,
                    self.discr_from_dd(to_dd),
                    self.discr_from_dd(from_dd))

        # }}}

        if from_discr_tag is not DISCR_TAG_BASE:
            raise ValueError("cannot interpolate *from* a "
                    "(non-interpolatory) quadrature grid")

        assert to_discr_tag is DISCR_TAG_BASE

        if from_dd.is_volume():
            if to_dd.domain_tag is FACE_RESTR_ALL:
                return self._all_faces_volume_connection()
            if to_dd.domain_tag is FACE_RESTR_INTERIOR:
                return self._interior_faces_connection()
            elif to_dd.is_boundary_or_partition_interface():
                assert from_discr_tag is DISCR_TAG_BASE
                return self._boundary_connection(to_dd.domain_tag.tag)
            elif to_dd.is_volume():
                from meshmode.discretization.connection import \
                        make_same_mesh_connection
                to_discr = self._discr_tag_volume_discr(to_discr_tag)
                from_discr = self._volume_discr
                return make_same_mesh_connection(self._setup_actx, to_discr,
                            from_discr)

            else:
                raise ValueError("cannot interpolate from volume to: " + str(to_dd))

        else:
            raise ValueError("cannot interpolate from: " + str(from_dd))

    def group_factory_for_quadrature_tag(self, discretization_tag):
        warn("`DiscretizationCollection.group_factory_for_quadrature_tag` "
             "is deprecated and will go away in 2022. Use "
             "`DiscretizationCollection.group_factory_for_discretization_tag` "
             "instead.",
             DeprecationWarning, stacklevel=2)

        return self.group_factory_for_discretization_tag(discretization_tag)

    def group_factory_for_discretization_tag(self, discretization_tag):
        """
        OK to override in user code to control mode/node choice.
        """
        if discretization_tag is None:
            discretization_tag = DISCR_TAG_BASE

        return self.discr_tag_to_group_factory[discretization_tag]

    @memoize_method
    def _discr_tag_volume_discr(self, discretization_tag):
        from meshmode.discretization import Discretization

        return Discretization(
            self._setup_actx, self._volume_discr.mesh,
            self.group_factory_for_discretization_tag(discretization_tag)
        )

    # {{{ modal to nodal connections

    @memoize_method
    def _modal_discr(self, domain_tag):
        from meshmode.discretization import Discretization

        discr_base = self.discr_from_dd(DOFDesc(domain_tag, DISCR_TAG_BASE))
        return Discretization(
            self._setup_actx, discr_base.mesh,
            self.group_factory_for_discretization_tag(DISCR_TAG_MODAL)
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
            self.group_factory_for_discretization_tag(DISCR_TAG_BASE),
            boundary_tag=boundary_tag
        )

    # }}}

    # {{{ interior faces

    @memoize_method
    def _interior_faces_connection(self):
        return make_face_restriction(
            self._setup_actx,
            self._volume_discr,
            self.group_factory_for_discretization_tag(DISCR_TAG_BASE),
            FACE_RESTR_INTERIOR,

            # FIXME: This will need to change as soon as we support
            # pyramids or other elements with non-identical face
            # types.
            per_face_groups=False
        )

    @memoize_method
    def opposite_face_connection(self):
        """Provides a mapping from the base volume discretization
        to the exterior boundary restriction on a neighboring element.
        This does not take into account parallel partitions.
        """
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
            self.group_factory_for_discretization_tag(DISCR_TAG_BASE),
            FACE_RESTR_ALL,

            # FIXME: This will need to change as soon as we support
            # pyramids or other elements with non-identical face
            # types.
            per_face_groups=False
        )

    # }}}

    @property
    def dim(self):
        """Return the topological dimension."""
        return self._volume_discr.dim

    @property
    def ambient_dim(self):
        """Return the dimension of the ambient space."""
        return self._volume_discr.ambient_dim

    @property
    def real_dtype(self):
        """Return the data type used for real-valued arithmetic."""
        return self._volume_discr.real_dtype

    @property
    def complex_dtype(self):
        """Return the data type used for complex-valued arithmetic."""
        return self._volume_discr.complex_dtype

    @property
    def mesh(self):
        """Return a :class:`meshmode.mesh.Mesh` over which the discretization
        collection is built.
        """
        return self._volume_discr.mesh

    def empty(self, array_context: ArrayContext, dtype=None):
        """Return an empty :class:`~meshmode.dof_array.DOFArray`.

        :arg array_context: an :class:`~arraycontext.context.ArrayContext`.
        :arg dtype: type special value 'c' will result in a
            vector of dtype :attr:`complex_dtype`. If
            *None* (the default), a real vector will be returned.
        """
        return self._volume_discr.empty(array_context, dtype)

    def zeros(self, array_context: ArrayContext, dtype=None):
        """Return a zero-initialized :class:`~meshmode.dof_array.DOFArray`.

        :arg array_context: an :class:`~arraycontext.context.ArrayContext`.
        :arg dtype: type special value 'c' will result in a
            vector of dtype :attr:`complex_dtype`. If
            *None* (the default), a real vector will be returned.
        """
        return self._volume_discr.zeros(array_context, dtype)

    def is_volume_where(self, where):
        return where is None or as_dofdesc(where).is_volume()

    @property
    def order(self):
        warn("DiscretizationCollection.order is deprecated, "
                "consider using the orders of element groups instead. "
                "'order' will go away in 2021.",
                DeprecationWarning, stacklevel=2)

        from pytools import single_valued
        return single_valued(egrp.order for egrp in self._volume_discr.groups)


class DGDiscretizationWithBoundaries(DiscretizationCollection):
    def __init__(self, *args, **kwargs):
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
