"""
Volume tags
-----------

.. autoclass:: VolumeTag
.. autoclass:: VTAG_ALL

:mod:`grudge`-specific boundary tags
------------------------------------

Domain tags
-----------

A domain tag identifies a geometric part (or whole) of the domain described
by a :class:`grudge.DiscretizationCollection`. This can be a volume or a boundary.

.. autoclass:: DTAG_SCALAR
.. autoclass:: DTAG_VOLUME_ALL
.. autoclass:: VolumeDomainTag
.. autoclass:: BoundaryDomainTag

Discretization tags
-------------------

A discretization tag serves as a symbolic identifier of the manner in which
meaning is assigned to degrees of freedom.

.. autoclass:: DISCR_TAG_BASE
.. autoclass:: DISCR_TAG_QUAD
.. autoclass:: DISCR_TAG_MODAL

DOF Descriptor
--------------

.. autoclass:: DOFDesc
.. autofunction:: as_dofdesc

Shortcuts
---------

.. data:: DD_SCALAR
.. data:: DD_VOLUME_ALL
.. data:: DD_VOLUME_ALL_MODAL

Internal things that are visble due to type annotations
-------------------------------------------------------

.. class:: _DiscretizationTag
.. class:: ConvertibleToDOFDesc

    Anything that is convertible to a :class:`DOFDesc` via :func:`as_dofdesc`.
"""

__copyright__ = """
Copyright (C) 2008 Andreas Kloeckner
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

from dataclasses import dataclass, replace
from typing import Any, Hashable, Optional, Tuple, Type, Union
from warnings import warn

from meshmode.discretization.connection import FACE_RESTR_ALL, FACE_RESTR_INTERIOR
from meshmode.mesh import (
    BTAG_ALL,
    BTAG_NONE,
    BTAG_PARTITION,
    BTAG_REALLY_ALL,
    BoundaryTag,
)
from pytools import to_identifier


# {{{ volume tags

class VTAG_ALL:  # noqa: N801
    pass


VolumeTag = Hashable

# }}}


# {{{ domain tag

@dataclass(frozen=True, eq=True)
class ScalarDomainTag:
    """A domain tag denoting scalar values."""


DTAG_SCALAR = ScalarDomainTag()


@dataclass(frozen=True, eq=True, init=True)
class VolumeDomainTag:
    """A domain tag referring to a volume identified by the volume tag :attr:`tag`.

    .. attribute:: tag

    .. automethod:: __init__
    """
    tag: VolumeTag


DTAG_VOLUME_ALL = VolumeDomainTag(VTAG_ALL)


@dataclass(frozen=True, eq=True, init=True)
class BoundaryDomainTag:
    """A domain tag referring to a boundary identified by the
    boundary tag :attr:`tag`.

    .. attribute:: tag
    .. attribute:: volume_tag

    .. automethod:: __init__
    """
    tag: BoundaryTag
    volume_tag: VolumeTag = VTAG_ALL


DomainTag = Union[ScalarDomainTag, VolumeDomainTag, BoundaryDomainTag]

# }}}


# {{{ discretization tag

class _DiscretizationTag:
    pass


DiscretizationTag = Type[_DiscretizationTag]


class DISCR_TAG_BASE(_DiscretizationTag):  # noqa: N801
    """A discretization tag indicating the use of a
    nodal and unisolvent discretization. This tag is used
    to distinguish the base discretization from quadrature
    (e.g. overintegration) or modal (:class:`DISCR_TAG_MODAL`)
    discretizations.
    """


class DISCR_TAG_QUAD(_DiscretizationTag):  # noqa: N801
    """A discretization tag indicating the use of a quadrature discretization
    grid, which typically affords higher quadrature accuracy (e.g. for
    nonlinear terms) at the expense of unisolvency. This tag is used to
    distinguish the quadrature discretization (e.g. overintegration) from modal
    (:class:`DISCR_TAG_MODAL`) or base (:class:`DISCR_TAG_BASE`)
    discretizations.

    For working with multiple quadrature grids, it is
    recommended to create appropriate subclasses of
    :class:`DISCR_TAG_QUAD` and define appropriate
    :class:`DOFDesc` objects corresponding to each
    subclass. For example:

    .. code-block:: python

        class CustomQuadTag(DISCR_TAG_QUAD):
            "A custom quadrature discretization tag."

        dd = DOFDesc(DTAG_VOLUME_ALL, CustomQuadTag)
    """


class DISCR_TAG_MODAL(_DiscretizationTag):  # noqa: N801
    """A discretization tag indicating the use of unisolvent modal degrees of
    freedom. This tag is used to distinguish the modal discretization from the
    base (nodal) discretization (e.g.  :class:`DISCR_TAG_BASE`) or
    discretizations on quadrature grids (:class:`DISCR_TAG_QUAD`).
    """

# }}}


# {{{ DOF descriptor

@dataclass(frozen=True, eq=True)
class DOFDesc:
    """Describes the meaning of degrees of freedom.

    .. attribute:: domain_tag
    .. attribute:: discretization_tag

    .. automethod:: __init__

    .. automethod:: is_scalar
    .. automethod:: is_discretized
    .. automethod:: is_volume
    .. automethod:: is_boundary_or_partition_interface
    .. automethod:: is_trace

    .. automethod:: uses_quadrature

    .. automethod:: with_domain_tag
    .. automethod:: with_discr_tag
    .. automethod:: trace
    .. automethod:: untrace
    .. automethod:: with_boundary_tag

    .. automethod:: __eq__
    .. automethod:: __ne__
    .. automethod:: __hash__
    .. automethod:: as_identifier
    """

    domain_tag: DomainTag
    discretization_tag: DiscretizationTag

    def __init__(self,
            domain_tag: Any,
            discretization_tag: Optional[Type[DiscretizationTag]] = None):

        if (
                not (isinstance(domain_tag,
                    (ScalarDomainTag, BoundaryDomainTag, VolumeDomainTag)))
                or discretization_tag is None
                or (
                    not isinstance(discretization_tag, type)
                    or not issubclass(discretization_tag, _DiscretizationTag))):
            warn("Sloppy construction of DOFDesc is deprecated. "
                    "This will stop working in 2023. "
                    "Call as_dofdesc instead, with the same arguments. ",
                    DeprecationWarning, stacklevel=2)

            domain_tag, discretization_tag = _normalize_domain_and_discr_tag(
                    domain_tag, discretization_tag)

        object.__setattr__(self, "domain_tag", domain_tag)
        object.__setattr__(self, "discretization_tag", discretization_tag)

    def is_scalar(self) -> bool:
        return isinstance(self.domain_tag, ScalarDomainTag)

    def is_discretized(self) -> bool:
        return not self.is_scalar()

    def is_volume(self) -> bool:
        return isinstance(self.domain_tag, VolumeDomainTag)

    def is_boundary_or_partition_interface(self) -> bool:
        return (isinstance(self.domain_tag, BoundaryDomainTag)
                and self.domain_tag.tag not in [
                    FACE_RESTR_ALL,
                    FACE_RESTR_INTERIOR])

    def is_trace(self) -> bool:
        return isinstance(self.domain_tag, BoundaryDomainTag)

    def uses_quadrature(self) -> bool:
        # FIXME: String tags are deprecated
        if isinstance(self.discretization_tag, str):
            # All strings are interpreted as quadrature-related tags
            return True
        elif isinstance(self.discretization_tag, type):
            if issubclass(self.discretization_tag, DISCR_TAG_QUAD):
                return True
            elif issubclass(self.discretization_tag,
                            (DISCR_TAG_BASE, DISCR_TAG_MODAL)):
                return False

        raise ValueError(
            f"Invalid discretization tag: {self.discretization_tag}")

    def with_dtag(self, dtag) -> "DOFDesc":
        from warnings import warn
        warn("'with_dtag' is deprecated. Use 'with_domain_tag' instead. "
                "This will stop working in 2023",
                DeprecationWarning, stacklevel=2)
        return replace(self, domain_tag=dtag)

    def with_domain_tag(self, dtag) -> "DOFDesc":
        return replace(self, domain_tag=dtag)

    def with_discr_tag(self, discr_tag) -> "DOFDesc":
        return replace(self, discretization_tag=discr_tag)

    def trace(self, btag: BoundaryTag) -> "DOFDesc":
        """Return a :class:`DOFDesc` for the restriction of the volume
        descriptor *self* to the boundary named by *btag*.

        An error is raised if this method is called on a non-volume instance of
        :class:`DOFDesc`.
        """
        if not isinstance(self.domain_tag, VolumeDomainTag):
            raise ValueError(f"must originate on volume, got '{self.domain_tag}'")
        return replace(self,
                domain_tag=BoundaryDomainTag(btag, volume_tag=self.domain_tag.tag))

    def untrace(self) -> "DOFDesc":
        """Return a :class:`DOFDesc` for the volume associated with the boundary
        descriptor *self*.

        An error is raised if this method is called on a non-boundary instance of
        :class:`DOFDesc`.
        """
        if not isinstance(self.domain_tag, BoundaryDomainTag):
            raise ValueError(f"must originate on boundary, got '{self.domain_tag}'")
        return replace(self,
                domain_tag=VolumeDomainTag(self.domain_tag.volume_tag))

    def with_boundary_tag(self, btag: BoundaryTag) -> "DOFDesc":
        """Return a :class:`DOFDesc` representing a boundary named by *btag*
        on the same volume as *self*.

        An error is raised if this method is called on a non-boundary instance of
        :class:`DOFDesc`.
        """
        if not isinstance(self.domain_tag, BoundaryDomainTag):
            raise ValueError(f"must originate on boundary, got '{self.domain_tag}'")
        return replace(self,
                domain_tag=replace(self.domain_tag, tag=btag))

    def as_identifier(self) -> str:
        """Returns a descriptive string for this :class:`DOFDesc` that is usable
        in Python identifiers.
        """

        if self.domain_tag is DTAG_SCALAR:
            dom_id = "sc"
        elif self.domain_tag is DTAG_VOLUME_ALL:
            dom_id = "vol"
        elif self.domain_tag is FACE_RESTR_ALL:
            dom_id = "f_all"
        elif self.domain_tag is FACE_RESTR_INTERIOR:
            dom_id = "f_int"
        elif isinstance(self.domain_tag, VolumeDomainTag):
            vtag = self.domain_tag.tag
            if isinstance(vtag, type):
                vtag = vtag.__name__.replace("VTAG_", "").lower()
            elif isinstance(vtag, str):
                vtag = to_identifier(vtag)
            else:
                vtag = to_identifier(str(vtag))
            dom_id = f"v_{vtag}"
        elif isinstance(self.domain_tag, BoundaryDomainTag):
            btag = self.domain_tag.tag
            if isinstance(btag, type):
                btag = btag.__name__.replace("BTAG_", "").lower()
            elif isinstance(btag, str):
                btag = to_identifier(btag)
            else:
                btag = to_identifier(str(btag))
            dom_id = f"b_{btag}"
        else:
            raise ValueError(f"unexpected domain tag: '{self.domain_tag}'")

        if isinstance(self.discretization_tag, str):
            discr_id = to_identifier(self.discretization_tag)
        elif issubclass(self.discretization_tag, DISCR_TAG_QUAD):
            discr_id = "_quad"
        elif self.discretization_tag is DISCR_TAG_BASE:
            discr_id = ""
        elif self.discretization_tag is DISCR_TAG_MODAL:
            discr_id = "_modal"
        else:
            raise ValueError(
                f"Unexpected discretization tag: {self.discretization_tag}"
            )

        return f"{dom_id}{discr_id}"


DD_SCALAR = DOFDesc(DTAG_SCALAR, DISCR_TAG_BASE)
DD_VOLUME_ALL = DOFDesc(DTAG_VOLUME_ALL, DISCR_TAG_BASE)
DD_VOLUME_ALL_MODAL = DOFDesc(DTAG_VOLUME_ALL, DISCR_TAG_MODAL)


def _normalize_domain_and_discr_tag(
        domain: Any,
        discretization_tag: Optional[DiscretizationTag] = None,
        *, _contextual_volume_tag: Optional[VolumeTag] = None
        ) -> Tuple[DomainTag, DiscretizationTag]:

    if _contextual_volume_tag is None:
        _contextual_volume_tag = VTAG_ALL

    if domain == "scalar":
        domain = DTAG_SCALAR
    elif isinstance(domain, (ScalarDomainTag, BoundaryDomainTag, VolumeDomainTag)):
        pass
    elif domain in [VTAG_ALL, "vol"]:
        domain = DTAG_VOLUME_ALL
    elif domain in [FACE_RESTR_ALL, "all_faces"]:
        domain = BoundaryDomainTag(FACE_RESTR_ALL, _contextual_volume_tag)
    elif domain in [FACE_RESTR_INTERIOR, "int_faces"]:
        domain = BoundaryDomainTag(FACE_RESTR_INTERIOR, _contextual_volume_tag)
    elif isinstance(domain, BTAG_PARTITION):
        domain = BoundaryDomainTag(domain, _contextual_volume_tag)
    elif domain in [BTAG_ALL, BTAG_REALLY_ALL, BTAG_NONE]:
        domain = BoundaryDomainTag(domain, _contextual_volume_tag)
    else:
        raise ValueError(f"domain tag not understood: {domain}")

    if domain is DTAG_SCALAR and discretization_tag is not None:
        raise ValueError("cannot have nontrivial discretization tag on scalar")

    if discretization_tag is None:
        discretization_tag = DISCR_TAG_BASE

    return domain, discretization_tag


ConvertibleToDOFDesc = Any


def as_dofdesc(
        domain: "ConvertibleToDOFDesc",
        discretization_tag: Optional[DiscretizationTag] = None,
        *, _contextual_volume_tag: Optional[VolumeTag] = None) -> DOFDesc:
    """
    :arg domain_tag: One of the following:
        :class:`DTAG_SCALAR` (or the string ``"scalar"``),
        :class:`DTAG_VOLUME_ALL` (or the string ``"vol"``)
        for the default volume discretization,
        :data:`~meshmode.discretization.connection.FACE_RESTR_ALL`
        (or the string ``"all_faces"``), or
        :data:`~meshmode.discretization.connection.FACE_RESTR_INTERIOR`
        (or the string ``"int_faces"``), or one of
        :class:`~meshmode.mesh.BTAG_ALL`,
        :class:`~meshmode.mesh.BTAG_NONE`,
        :class:`~meshmode.mesh.BTAG_REALLY_ALL`,
        :class:`~meshmode.mesh.BTAG_PARTITION`,
        or *None* to indicate that the geometry is not yet known.

    :arg discretization_tag:
        *None* or :class:`DISCR_TAG_BASE` to indicate the use of the basic
        discretization grid, :class:`DISCR_TAG_MODAL` to indicate a
        modal discretization, or :class:`DISCR_TAG_QUAD` to indicate
        the use of a quadrature grid.
    """

    if isinstance(domain, DOFDesc):
        return domain

    domain, discretization_tag = _normalize_domain_and_discr_tag(
            domain, discretization_tag,
            _contextual_volume_tag=_contextual_volume_tag)

    return DOFDesc(domain, discretization_tag)

# }}}


# {{{ deprecations

_deprecated_name_to_new_name = {
        "DTAG_VOLUME": "VolumeDomainTag",
        "DTAG_BOUNDARY": "BoundaryDomainTag",
        "DD_VOLUME": "DD_VOLUME_ALL",
        "DD_VOLUME_MODAL": "DD_VOLUME_ALL_MODAL"
        }


def __getattr__(name):
    if name in _deprecated_name_to_new_name:
        warn(f"'{name}' is deprecated and will be dropped "
             f"in version 2023.x. Use '{_deprecated_name_to_new_name[name]}' "
             "instead.",
             DeprecationWarning, stacklevel=2)
        return globals()[_deprecated_name_to_new_name[name]]

    raise AttributeError(f"module {__name__} has no attribute {name}")

# }}}


# vim: foldmethod=marker
