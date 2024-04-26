"""
.. autoclass:: DiscretizationTag

.. currentmodule:: grudge
.. autoclass:: DiscretizationCollection
.. autofunction:: make_discretization_collection

.. currentmodule:: grudge.discretization
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

from typing import Mapping, Optional, Union, TYPE_CHECKING, Any
from meshmode.discretization.poly_element import (
    InterpolatoryEdgeClusteredGroupFactory, ModalGroupFactory)

from pytools import memoize_method, single_valued

from grudge.dof_desc import (
    VTAG_ALL,
    DD_VOLUME_ALL,
    DISCR_TAG_BASE,
    DISCR_TAG_MODAL,
    VolumeDomainTag, BoundaryDomainTag,
    DOFDesc,
    VolumeTag, DomainTag,
    DiscretizationTag,
    as_dofdesc,
    ConvertibleToDOFDesc
)

import numpy as np  # noqa: F401

from arraycontext import ArrayContext

from meshmode.discretization import ElementGroupFactory, Discretization
from meshmode.discretization.connection import (
    FACE_RESTR_INTERIOR,
    FACE_RESTR_ALL,
    make_face_restriction,
    DiscretizationConnection
)
from meshmode.mesh import Mesh, BTAG_PARTITION
from meshmode.dof_array import DOFArray

from warnings import warn

if TYPE_CHECKING:
    import mpi4py.MPI


# {{{ discr_tag_to_group_factory normalization

def _normalize_discr_tag_to_group_factory(
        dim: int,
        discr_tag_to_group_factory: Optional[
            Mapping[DiscretizationTag, ElementGroupFactory]],
        order: Optional[int]
        ) -> Mapping[DiscretizationTag, ElementGroupFactory]:
    if discr_tag_to_group_factory is None:
        if order is None:
            raise TypeError(
                "one of 'order' and 'discr_tag_to_group_factory' must be given"
            )

        discr_tag_to_group_factory = {
                DISCR_TAG_BASE: InterpolatoryEdgeClusteredGroupFactory(order=order)}
    else:
        discr_tag_to_group_factory = dict(discr_tag_to_group_factory)

        if order is not None:
            if DISCR_TAG_BASE in discr_tag_to_group_factory:
                raise ValueError(
                    "if 'order' is given, 'discr_tag_to_group_factory' must "
                    "not have a key of DISCR_TAG_BASE"
                )

            discr_tag_to_group_factory[DISCR_TAG_BASE] = \
                    InterpolatoryEdgeClusteredGroupFactory(order)

    assert discr_tag_to_group_factory is not None

    # Modal discr should always come from the base discretization
    if DISCR_TAG_MODAL not in discr_tag_to_group_factory and order is not None:
        discr_tag_to_group_factory[DISCR_TAG_MODAL] = ModalGroupFactory(order)

    return discr_tag_to_group_factory

# }}}


class DiscretizationCollection:
    """A collection of discretizations, defined on the same underlying
    :class:`~meshmode.mesh.Mesh`, corresponding to various mesh entities
    (volume, interior facets, boundaries) and associated element
    groups.

    .. note::

        Do not call the constructor directly. Use
        :func:`make_discretization_collection` instead.

    .. autoattribute:: dim
    .. autoattribute:: ambient_dim
    .. autoattribute:: real_dtype
    .. autoattribute:: complex_dtype

    .. automethod:: discr_from_dd
    .. automethod:: connection_from_dds

    .. automethod:: empty
    .. automethod:: zeros

    .. automethod:: nodes
    .. automethod:: normal

    .. rubric:: Internal functionality

    .. automethod:: _base_to_geoderiv_connection
    """

    # {{{ constructor

    def __init__(self, array_context: ArrayContext,
            volume_discrs: Union[Mesh, Mapping[VolumeTag, Discretization]],
            order: Optional[int] = None,
            discr_tag_to_group_factory: Optional[
                Mapping[DiscretizationTag, ElementGroupFactory]] = None,
            mpi_communicator: Optional["mpi4py.MPI.Intracomm"] = None,
            ) -> None:
        """
        :arg discr_tag_to_group_factory: A mapping from discretization tags
            (typically one of: :class:`grudge.dof_desc.DISCR_TAG_BASE`,
            :class:`grudge.dof_desc.DISCR_TAG_MODAL`, or
            :class:`grudge.dof_desc.DISCR_TAG_QUAD`) to a
            :class:`~meshmode.discretization.ElementGroupFactory`
            indicating with which type of discretization the operations are
            to be carried out, or *None* to indicate that operations with this
            discretization tag should be carried out with the standard volume
            discretization.
        """

        self._setup_actx = array_context.clone()

        # {{{ process mpi_communicator argument

        if mpi_communicator is not None:
            warn("Passing 'mpi_communicator' is deprecated. This will stop working "
                    "in 2023. Instead, pass an MPIBasedArrayContext.",
                    DeprecationWarning, stacklevel=2)

            from grudge.array_context import MPIBasedArrayContext
            if (isinstance(array_context, MPIBasedArrayContext)
                    and mpi_communicator is not array_context.mpi_communicator):
                raise ValueError("mpi_communicator passed to "
                        "DiscretizationCollection and the MPI communicator "
                        "used to created the MPIBasedArrayContext must be "
                        "idetical, which they aren't.")
        else:
            from grudge.array_context import MPIBasedArrayContext
            if isinstance(self._setup_actx, MPIBasedArrayContext):
                mpi_communicator = self._setup_actx.mpi_communicator

        self._mpi_communicator = mpi_communicator

        # }}}

        from meshmode.discretization import Discretization

        if isinstance(volume_discrs, Mesh):
            # {{{ deprecated backward compatibility yuck

            warn("Calling the DiscretizationCollection constructor directly "
                    "is deprecated, call make_discretization_collection "
                    "instead. This will stop working in 2023.",
                    DeprecationWarning, stacklevel=2)

            mesh = volume_discrs

            discr_tag_to_group_factory = _normalize_discr_tag_to_group_factory(
                    dim=mesh.dim,
                    discr_tag_to_group_factory=discr_tag_to_group_factory,
                    order=order)
            self._discr_tag_to_group_factory = discr_tag_to_group_factory

            volume_discr = Discretization(
                        array_context, mesh,
                        self.group_factory_for_discretization_tag(DISCR_TAG_BASE))
            volume_discrs = {VTAG_ALL: volume_discr}

            del mesh

            # }}}
        else:
            assert discr_tag_to_group_factory is not None
            self._discr_tag_to_group_factory = discr_tag_to_group_factory

        self._volume_discrs = volume_discrs

        self._dist_boundary_connections = {
            vtag: self._set_up_distributed_communication(
                        vtag, mpi_communicator, array_context)
            for vtag in self._volume_discrs.keys()}

    # }}}

    @property
    def mpi_communicator(self):
        warn("Accessing DiscretizationCollection.mpi_communicator is deprecated. "
                "This will stop working in 2023. "
                "Instead, use an MPIBasedArrayContext, and obtain the communicator "
                "from that.", DeprecationWarning, stacklevel=2)

        return self._mpi_communicator

    def get_management_rank_index(self):
        return 0

    def is_management_rank(self):
        if self.mpi_communicator is None:
            return True
        else:
            return self.mpi_communicator.Get_rank() \
                    == self.get_management_rank_index()

    # {{{ distributed

    def _set_up_distributed_communication(
            self, vtag, mpi_communicator, array_context):
        from_dd = DOFDesc(VolumeDomainTag(vtag), DISCR_TAG_BASE)

        boundary_connections = {}

        from meshmode.distributed import get_connected_partitions
        connected_parts = get_connected_partitions(self._volume_discrs[vtag].mesh)

        if connected_parts:
            if mpi_communicator is None:
                raise RuntimeError("must supply an MPI communicator when using a "
                    "distributed mesh")

            grp_factory = \
                self.group_factory_for_discretization_tag(DISCR_TAG_BASE)

            local_boundary_connections = {}
            for i_remote_part in connected_parts:
                local_boundary_connections[i_remote_part] = self.connection_from_dds(
                        from_dd, from_dd.trace(BTAG_PARTITION(i_remote_part)))

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

    def distributed_boundary_swap_connection(self, dd):
        """Provides a mapping from the base volume discretization
        to the exterior boundary restriction on a parallel boundary
        partition described by *dd*. This connection is used to
        communicate across element boundaries in different parallel
        partitions during distributed runs.

        :arg dd: a :class:`~grudge.dof_desc.DOFDesc`, or a value
            convertible to one. The domain tag must be a subclass
            of :class:`grudge.dof_desc.BoundaryDomainTag` with an
            associated :class:`meshmode.mesh.BTAG_PARTITION`
            corresponding to a particular communication rank.
        """
        if dd.discretization_tag is not DISCR_TAG_BASE:
            # FIXME
            raise NotImplementedError(
                "Distributed communication with discretization tag "
                f"{dd.discretization_tag} is not implemented."
            )

        assert isinstance(dd.domain_tag, BoundaryDomainTag)
        assert isinstance(dd.domain_tag.tag, BTAG_PARTITION)

        vtag = dd.domain_tag.volume_tag

        return self._dist_boundary_connections[vtag][dd.domain_tag.tag.part_id]

    # }}}

    # {{{ discr_from_dd

    @memoize_method
    def discr_from_dd(self, dd: "ConvertibleToDOFDesc") -> Discretization:
        """Provides a :class:`meshmode.discretization.Discretization`
        object from *dd*.
        """
        dd = as_dofdesc(dd)

        discr_tag = dd.discretization_tag

        if discr_tag is DISCR_TAG_MODAL:
            return self._modal_discr(dd.domain_tag)

        if dd.is_volume():
            return self._volume_discr_from_dd(dd)

        if discr_tag is not DISCR_TAG_BASE:
            base_discr = self.discr_from_dd(dd.with_discr_tag(DISCR_TAG_BASE))

            from meshmode.discretization import Discretization
            return Discretization(
                self._setup_actx,
                base_discr.mesh,
                self.group_factory_for_discretization_tag(discr_tag)
            )

        assert discr_tag is DISCR_TAG_BASE

        if isinstance(dd.domain_tag, BoundaryDomainTag):
            if dd.domain_tag.tag in [FACE_RESTR_ALL, FACE_RESTR_INTERIOR]:
                return self._faces_connection(dd.domain_tag).to_discr
            else:
                return self._boundary_connection(dd.domain_tag).to_discr
        else:
            raise ValueError(f"DOF desc not understood: {dd}")

    # }}}

    # {{{ _base_to_geoderiv_connection

    @memoize_method
    def _has_affine_groups(self, domain_tag: DomainTag) -> bool:
        from modepy.shapes import Simplex
        discr = self.discr_from_dd(DOFDesc(domain_tag, DISCR_TAG_BASE))
        return any(
                megrp.is_affine
                and issubclass(megrp._modepy_shape_cls, Simplex)
                for megrp in discr.mesh.groups)

    @memoize_method
    def _base_to_geoderiv_connection(self, dd: DOFDesc) -> DiscretizationConnection:
        r"""The "geometry derivatives" discretization for a given *dd* is
        typically identical to the one returned by :meth:`discr_from_dd`,
        however for affinely-mapped simplicial elements, it will use a
        :math:`P^0` discretization having a single DOF per element.
        As a result, :class:`~meshmode.dof_array.DOFArray`\ s on this
        are broadcast-compatible with the discretizations returned by
        :meth:`discr_from_dd`.

        This is an internal function, not intended for use outside
        :mod:`grudge`.
        """
        base_discr = self.discr_from_dd(dd)
        if not self._has_affine_groups(dd.domain_tag):
            # no benefit to having another discretization that takes
            # advantage of affine-ness
            from meshmode.discretization.connection import \
                    IdentityDiscretizationConnection
            return IdentityDiscretizationConnection(base_discr)

        base_group_factory = self.group_factory_for_discretization_tag(
                dd.discretization_tag)

        def geo_group_factory(megrp):
            from modepy.shapes import Simplex
            from meshmode.discretization.poly_element import \
                    PolynomialEquidistantSimplexElementGroup
            if megrp.is_affine and issubclass(megrp._modepy_shape_cls, Simplex):
                return PolynomialEquidistantSimplexElementGroup(
                        megrp, order=0)
            else:
                return base_group_factory(megrp)

        from meshmode.discretization import Discretization
        geo_deriv_discr = Discretization(
            self._setup_actx, base_discr.mesh,
            geo_group_factory)

        from meshmode.discretization.connection.same_mesh import \
                make_same_mesh_connection
        return make_same_mesh_connection(
                self._setup_actx,
                to_discr=geo_deriv_discr,
                from_discr=base_discr)

    # }}}

    # {{{ connection_from_dds

    @memoize_method
    def connection_from_dds(
            self, from_dd: "ConvertibleToDOFDesc", to_dd: "ConvertibleToDOFDesc"
            ) -> DiscretizationConnection:
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

        if (isinstance(from_dd.domain_tag, BoundaryDomainTag)
                and from_discr_tag == to_discr_tag
                and isinstance(to_dd.domain_tag, BoundaryDomainTag)
                and to_dd.domain_tag.tag is FACE_RESTR_ALL):
            faces_conn = self.connection_from_dds(
                    DOFDesc(
                        VolumeDomainTag(from_dd.domain_tag.volume_tag),
                        DISCR_TAG_BASE),
                    from_dd.with_discr_tag(DISCR_TAG_BASE))

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
            intermediate_dd = to_dd.with_discr_tag(DISCR_TAG_BASE)
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
            raise ValueError("cannot get a connection *from* a "
                    f"(non-interpolatory) quadrature grid: '{from_dd}'")

        assert to_discr_tag is DISCR_TAG_BASE

        if isinstance(from_dd.domain_tag, VolumeDomainTag):
            if isinstance(to_dd.domain_tag, BoundaryDomainTag):
                if to_dd.domain_tag.volume_tag != from_dd.domain_tag.tag:
                    raise ValueError("cannot get a connection from one volume "
                            f"('{from_dd.domain_tag.tag}') "
                            "to the boundary of another volume "
                            f"('{to_dd.domain_tag.volume_tag}') ")
                if to_dd.domain_tag.tag in [FACE_RESTR_ALL, FACE_RESTR_INTERIOR]:
                    return self._faces_connection(to_dd.domain_tag)
                elif isinstance(to_dd.domain_tag, BoundaryDomainTag):
                    assert from_discr_tag is DISCR_TAG_BASE
                    return self._boundary_connection(to_dd.domain_tag)
            elif to_dd.is_volume():
                if to_dd.domain_tag != from_dd.domain_tag:
                    raise ValueError("cannot get a connection between "
                            "volumes of different tags: requested "
                            f"'{from_dd.domain_tag}' -> '{to_dd.domain_tag}'")

                from meshmode.discretization.connection import \
                        make_same_mesh_connection
                return make_same_mesh_connection(
                        self._setup_actx,
                        self._volume_discr_from_dd(to_dd),
                        self._volume_discr_from_dd(from_dd))

            else:
                raise ValueError(
                        f"cannot get a connection from volume to: '{to_dd}'")

        else:
            raise ValueError(f"cannot get a connection from: '{from_dd}'")

    # }}}

    # {{{ group_factory_for_discretization_tag

    def group_factory_for_discretization_tag(self, discretization_tag):
        if discretization_tag is None:
            discretization_tag = DISCR_TAG_BASE

        return self._discr_tag_to_group_factory[discretization_tag]

    # }}}

    # {{{ (internal) discretization getters

    @memoize_method
    def _volume_discr_from_dd(self, dd: DOFDesc) -> Discretization:
        assert isinstance(dd.domain_tag, VolumeDomainTag)

        try:
            base_volume_discr = self._volume_discrs[dd.domain_tag.tag]
        except KeyError:
            raise ValueError("a volume discretization with volume tag "
                    f"'{dd.domain_tag.tag}' is not known")

        # Refuse to re-make the volume discretization
        if dd.discretization_tag is DISCR_TAG_BASE:
            return base_volume_discr

        from meshmode.discretization import Discretization
        return Discretization(
            self._setup_actx, base_volume_discr.mesh,
            self.group_factory_for_discretization_tag(dd.discretization_tag)
        )

    @memoize_method
    def _modal_discr(self, domain_tag) -> Discretization:
        from meshmode.discretization import Discretization

        discr_base = self.discr_from_dd(DOFDesc(domain_tag, DISCR_TAG_BASE))
        return Discretization(
            self._setup_actx, discr_base.mesh,
            self.group_factory_for_discretization_tag(DISCR_TAG_MODAL)
        )

    # }}}

    # {{{ connection factories: modal<->nodal

    @memoize_method
    def _modal_to_nodal_connection(self, to_dd: DOFDesc) -> DiscretizationConnection:
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
    def _nodal_to_modal_connection(
            self, from_dd: DOFDesc) -> DiscretizationConnection:
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

    # {{{ connection factories: boundary

    @memoize_method
    def _boundary_connection(
            self, domain_tag: BoundaryDomainTag) -> DiscretizationConnection:
        return make_face_restriction(
                self._setup_actx,
                self._volume_discr_from_dd(
                    DOFDesc(VolumeDomainTag(domain_tag.volume_tag), DISCR_TAG_BASE)),
                self.group_factory_for_discretization_tag(DISCR_TAG_BASE),
                boundary_tag=domain_tag.tag)

    # }}}

    # {{{ connection factories: faces

    @memoize_method
    def _faces_connection(
            self, domain_tag: BoundaryDomainTag) -> DiscretizationConnection:
        assert domain_tag.tag in [FACE_RESTR_INTERIOR, FACE_RESTR_ALL]

        return make_face_restriction(
            self._setup_actx,
            self._volume_discr_from_dd(
                DOFDesc(
                    VolumeDomainTag(domain_tag.volume_tag), DISCR_TAG_BASE)),
            self.group_factory_for_discretization_tag(DISCR_TAG_BASE),
            domain_tag.tag,

            # FIXME: This will need to change as soon as we support
            # pyramids or other elements with non-identical face
            # types.
            per_face_groups=False
        )

    @memoize_method
    def opposite_face_connection(
            self, domain_tag: BoundaryDomainTag) -> DiscretizationConnection:
        """Provides a mapping from the base volume discretization
        to the exterior boundary restriction on a neighboring element.
        This does not take into account parallel partitions.
        """
        from meshmode.discretization.connection import \
                make_opposite_face_connection

        assert domain_tag.tag is FACE_RESTR_INTERIOR

        return make_opposite_face_connection(
                self._setup_actx,
                self._faces_connection(domain_tag))

    # }}}

    # {{{ properties

    @property
    def dim(self) -> int:
        """Return the topological dimension."""
        return single_valued(discr.dim for discr in self._volume_discrs.values())

    @property
    def ambient_dim(self) -> int:
        """Return the dimension of the ambient space."""
        return single_valued(
                discr.ambient_dim for discr in self._volume_discrs.values())

    @property
    def real_dtype(self) -> "np.dtype[Any]":
        """Return the data type used for real-valued arithmetic."""
        return single_valued(
                discr.real_dtype for discr in self._volume_discrs.values())

    @property
    def complex_dtype(self) -> "np.dtype[Any]":
        """Return the data type used for complex-valued arithmetic."""
        return single_valued(
                discr.complex_dtype for discr in self._volume_discrs.values())

    # }}}

    # {{{ array creators

    def empty(self, array_context: ArrayContext, dtype=None,
            *, dd: Optional[DOFDesc] = None) -> DOFArray:
        """Return an empty :class:`~meshmode.dof_array.DOFArray` defined at
        the volume nodes: :class:`grudge.dof_desc.DD_VOLUME_ALL`.

        :arg array_context: an :class:`~arraycontext.context.ArrayContext`.
        :arg dtype: type special value 'c' will result in a
            vector of dtype :attr:`complex_dtype`. If
            *None* (the default), a real vector will be returned.
        """
        if dd is None:
            dd = DD_VOLUME_ALL
        return self.discr_from_dd(dd).empty(array_context, dtype)

    def zeros(self, array_context: ArrayContext, dtype=None,
            *, dd: Optional[DOFDesc] = None) -> DOFArray:
        """Return a zero-initialized :class:`~meshmode.dof_array.DOFArray`
        defined at the volume nodes, :class:`grudge.dof_desc.DD_VOLUME_ALL`.

        :arg array_context: an :class:`~arraycontext.context.ArrayContext`.
        :arg dtype: type special value 'c' will result in a
            vector of dtype :attr:`complex_dtype`. If
            *None* (the default), a real vector will be returned.
        """
        if dd is None:
            dd = DD_VOLUME_ALL

        return self.discr_from_dd(dd).zeros(array_context, dtype)

    def is_volume_where(self, where):
        return where is None or as_dofdesc(where).is_volume()

    # }}}

    # {{{ discretization-specific geometric fields

    def nodes(self, dd=None):
        r"""Return the nodes of a discretization specified by *dd*.

        :arg dd: a :class:`~grudge.dof_desc.DOFDesc`, or a value convertible to one.
            Defaults to the base volume discretization.
        :returns: an object array of frozen :class:`~meshmode.dof_array.DOFArray`\ s
        """
        if dd is None:
            dd = DD_VOLUME_ALL
        return self.discr_from_dd(dd).nodes()

    def normal(self, dd):
        r"""Get the unit normal to the specified surface discretization, *dd*.

        :arg dd: a :class:`~grudge.dof_desc.DOFDesc` as the surface discretization.
        :returns: an object array of frozen :class:`~meshmode.dof_array.DOFArray`\ s.
        """
        warn("DiscretizationCollection.normal is deprecated and "
             "will stop working in 2024. "
             "Use grudge.geometry.normal instead. "
             "DiscretizationCollection.normal may provide non-P^0 normals "
             "even when this would otherwise be possible, see "
             "https://github.com/inducer/grudge/issues/314 for details.",
             DeprecationWarning, stacklevel=2)

        from grudge.geometry import normal
        return self._setup_actx.freeze(normal(self._setup_actx, self, dd))

    # }}}


# {{{ make_discretization_collection

MeshOrDiscr = Union[Mesh, Discretization]


def make_discretization_collection(
        array_context: ArrayContext,
        volumes: Union[
            MeshOrDiscr,
            Mapping[VolumeTag, MeshOrDiscr]],
        order: Optional[int] = None,
        discr_tag_to_group_factory: Optional[
            Mapping[DiscretizationTag, ElementGroupFactory]] = None,
        ) -> DiscretizationCollection:
    """
    :arg discr_tag_to_group_factory: A mapping from discretization tags
        (typically one of: :class:`~grudge.dof_desc.DISCR_TAG_BASE`,
        :class:`~grudge.dof_desc.DISCR_TAG_MODAL`, or
        :class:`~grudge.dof_desc.DISCR_TAG_QUAD`) to a
        :class:`~meshmode.discretization.ElementGroupFactory`
        indicating with which type of discretization the operations are
        to be carried out, or *None* to indicate that operations with this
        discretization tag should be carried out with the standard volume
        discretization.

    .. note::

        If passing a :class:`~meshmode.discretization.Discretization` in
        *volumes*, it must be nodal and unisolvent, consistent with
        :class:`~grudge.dof_desc.DISCR_TAG_BASE`.

    .. note::

        To use the resulting :class:`DiscretizationCollection` in a
        distributed-memory manner, the *array_context* passed in
        must be one of the distributed-memory array contexts
        from :mod:`grudge.array_context`. Unlike the (now-deprecated,
        for direct use) constructor of :class:`DiscretizationCollection`,
        this function no longer accepts a separate MPI communicator.

    .. note::

        If the resulting :class:`DiscretizationCollection` is distributed
        across multiple ranks, then this is an MPI-collective operation,
        i.e. all ranks in the communicator must enter this function at the same
        time.
    """
    if isinstance(volumes, (Mesh, Discretization)):
        volumes = {VTAG_ALL: volumes}

    from pytools import is_single_valued

    assert len(volumes) > 0
    assert is_single_valued(mesh_or_discr.ambient_dim
            for mesh_or_discr in volumes.values())

    discr_tag_to_group_factory = _normalize_discr_tag_to_group_factory(
            dim=single_valued(
                mesh_or_discr.dim for mesh_or_discr in volumes.values()),
            discr_tag_to_group_factory=discr_tag_to_group_factory,
            order=order)

    del order

    if any(
            isinstance(mesh_or_discr, Discretization)
            for mesh_or_discr in volumes.values()):
        raise NotImplementedError("Doesn't work at the moment")

    volume_discrs = {
        vtag: Discretization(
            array_context,
            mesh,
            discr_tag_to_group_factory[DISCR_TAG_BASE])
        for vtag, mesh in volumes.items()}

    return DiscretizationCollection(
            array_context=array_context,
            volume_discrs=volume_discrs,
            discr_tag_to_group_factory=discr_tag_to_group_factory)

# }}}


# vim: foldmethod=marker
