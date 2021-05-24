"""Degree of freedom (DOF) descriptions"""

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

from meshmode.discretization.connection import \
    FACE_RESTR_INTERIOR, FACE_RESTR_ALL
from meshmode.mesh import \
    BTAG_PARTITION, BTAG_ALL, BTAG_REALLY_ALL, BTAG_NONE
from warnings import warn
import sys


__doc__ = """
.. autoclass:: DTAG_SCALAR
.. autoclass:: DTAG_VOLUME_ALL
.. autoclass:: DTAG_BOUNDARY

.. autoclass:: DISCR_TAG_BASE
.. autoclass:: DISCR_TAG_QUAD
.. autoclass:: DISCR_TAG_MODAL

.. autoclass:: DOFDesc
.. autofunction:: as_dofdesc

.. data:: DD_SCALAR
.. data:: DD_VOLUME
.. data:: DD_VOLUME_MODAL
"""


# {{{ DOF description

class DTAG_SCALAR:  # noqa: N801
    """A domain tag denoting scalar values."""


class DTAG_VOLUME_ALL:  # noqa: N801
    """
    A domain tag denoting values defined
    in all cell volumes.
    """


class DTAG_BOUNDARY:  # noqa: N801
    """A domain tag describing the values on element
    boundaries which are adjacent to elements
    of another :class:`~meshmode.mesh.Mesh`.

    .. attribute:: tag

    .. automethod:: __init__
    .. automethod:: __eq__
    .. automethod:: __ne__
    .. automethod:: __hash__
    """

    def __init__(self, tag):
        """
        :arg tag: One of the following:
            :class:`~meshmode.mesh.BTAG_ALL`,
            :class:`~meshmode.mesh.BTAG_NONE`,
            :class:`~meshmode.mesh.BTAG_REALLY_ALL`,
            :class:`~meshmode.mesh.BTAG_PARTITION`.
        """
        self.tag = tag

    def __eq__(self, other):
        return isinstance(other, DTAG_BOUNDARY) and self.tag == other.tag

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash(type(self)) ^ hash(self.tag)

    def __repr__(self):
        return "<{}({})>".format(type(self).__name__, repr(self.tag))


class DISCR_TAG_BASE:  # noqa: N801
    """A discretization tag indicating the use of a
    basic discretization grid. This tag is used
    to distinguish the base discretization from quadrature
    (e.g. overintegration) or modal (:class:`DISCR_TAG_MODAL`)
    discretizations.
    """


class DISCR_TAG_QUAD:  # noqa: N801
    """A discretization tag indicating the use of a
    quadrature discretization grid. This tag is used
    to distinguish the quadrature discretization
    (e.g. overintegration) from modal (:class:`DISCR_TAG_MODAL`)
    or base (:class:`DISCR_TAG_BASE`) discretizations.

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


class DISCR_TAG_MODAL:  # noqa: N801
    """A discretization tag indicating the use of a
    basic discretization grid with modal degrees of
    freedom. This tag is used to distinguish the
    modal discretization from the base (nodal)
    discretization (e.g. :class:`DISCR_TAG_BASE`) or
    discretizations on quadrature grids (:class:`DISCR_TAG_QUAD`).
    """


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

    .. automethod:: with_discr_tag
    .. automethod:: with_dtag

    .. automethod:: __eq__
    .. automethod:: __ne__
    .. automethod:: __hash__
    """

    def __init__(self, domain_tag, discretization_tag=None,
                 # FIXME: `quadrature_tag` is deprecated
                 quadrature_tag=None):
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

        if domain_tag is None:
            pass
        elif domain_tag in [DTAG_SCALAR, "scalar"]:
            domain_tag = DTAG_SCALAR
        elif domain_tag in [DTAG_VOLUME_ALL, "vol"]:
            domain_tag = DTAG_VOLUME_ALL
        elif domain_tag in [FACE_RESTR_ALL, "all_faces"]:
            domain_tag = FACE_RESTR_ALL
        elif domain_tag in [FACE_RESTR_INTERIOR, "int_faces"]:
            domain_tag = FACE_RESTR_INTERIOR
        elif isinstance(domain_tag, BTAG_PARTITION):
            domain_tag = DTAG_BOUNDARY(domain_tag)
        elif domain_tag in [BTAG_ALL, BTAG_REALLY_ALL, BTAG_NONE]:
            domain_tag = DTAG_BOUNDARY(domain_tag)
        elif isinstance(domain_tag, DTAG_BOUNDARY):
            pass
        else:
            raise ValueError("domain tag not understood: %s" % domain_tag)

        if (quadrature_tag is not None and discretization_tag is not None):
            raise ValueError(
                "Both `quadrature_tag` and `discretization_tag` are specified. "
                "Use `discretization_tag` instead."
            )

        # FIXME: `quadrature_tag` is deprecated
        if (quadrature_tag is not None and discretization_tag is None):
            warn("`quadrature_tag` is a deprecated kwarg and will be dropped "
                 "in version 2022.x. Use `discretization_tag` instead.",
                 DeprecationWarning, stacklevel=2)
            discretization_tag = quadrature_tag

        if domain_tag is DTAG_SCALAR and discretization_tag is not None:
            raise ValueError("cannot have nontrivial discretization tag on scalar")

        if discretization_tag is None:
            discretization_tag = DISCR_TAG_BASE

        # FIXME: String tags are deprecated
        if isinstance(discretization_tag, str):
            warn("Support for string values of `discretization_tag` will "
                 "be dropped in version 2022.x. Use one of the `DISCR_TAG_` "
                 "tags instead.",
                 DeprecationWarning, stacklevel=2)

        self.domain_tag = domain_tag
        self.discretization_tag = discretization_tag

    @property
    def quadrature_tag(self):
        warn("`DOFDesc.quadrature_tag` is deprecated and will be dropped "
             "in version 2022.x. Use `DOFDesc.discretization_tag` instead.",
             DeprecationWarning, stacklevel=2)
        return self.discretization_tag

    def is_scalar(self):
        return self.domain_tag is DTAG_SCALAR

    def is_discretized(self):
        return not self.is_scalar()

    def is_volume(self):
        return self.domain_tag is DTAG_VOLUME_ALL

    def is_boundary_or_partition_interface(self):
        return isinstance(self.domain_tag, DTAG_BOUNDARY)

    def is_trace(self):
        return (self.is_boundary_or_partition_interface()
                or self.domain_tag in [
                    FACE_RESTR_ALL,
                    FACE_RESTR_INTERIOR])

    def uses_quadrature(self):
        # FIXME: String tags are deprecated
        # Check for string first, otherwise
        # `issubclass` will raise an exception whenever
        # its first argument is not a class.
        # This can go away once support for strings is dropped
        # completely.
        if isinstance(self.discretization_tag, str):
            # All strings are interpreted as quadrature-related tags
            return True
        elif issubclass(self.discretization_tag, DISCR_TAG_QUAD):
            return True
        elif issubclass(self.discretization_tag,
                        (DISCR_TAG_BASE, DISCR_TAG_MODAL)):
            return False
        else:
            raise ValueError(
                f"Unsure how to interpret tag: {self.discretization_tag}"
            )

    def with_qtag(self, discr_tag):
        warn("`DOFDesc.with_qtag` is deprecated and will be dropped "
             "in version 2022.x. Use `DOFDesc.with_discr_tag` instead.",
             DeprecationWarning, stacklevel=2)
        return self.with_discr_tag(discr_tag)

    def with_discr_tag(self, discr_tag):
        return type(self)(domain_tag=self.domain_tag,
                          discretization_tag=discr_tag)

    def with_dtag(self, dtag):
        return type(self)(domain_tag=dtag,
                          discretization_tag=self.discretization_tag)

    def __eq__(self, other):
        return (type(self) == type(other)
                and self.domain_tag == other.domain_tag
                and self.discretization_tag == other.discretization_tag)

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash((type(self), self.domain_tag, self.discretization_tag))

    def __repr__(self):
        def fmt(s):
            if isinstance(s, type):
                return s.__name__
            else:
                return repr(s)

        return "DOFDesc({}, {})".format(
                fmt(self.domain_tag),
                fmt(self.discretization_tag))


DD_SCALAR = DOFDesc(DTAG_SCALAR, None)

DD_VOLUME = DOFDesc(DTAG_VOLUME_ALL, None)

DD_VOLUME_MODAL = DOFDesc(DTAG_VOLUME_ALL, DISCR_TAG_MODAL)


def as_dofdesc(dd):
    if isinstance(dd, DOFDesc):
        return dd
    return DOFDesc(dd, discretization_tag=None)

# }}}


# {{{ Deprecated tags

_deprecated_name_to_new_name = {"QTAG_NONE": "DISCR_TAG_BASE",
                                "QTAG_MODAL": "DISCR_TAG_MODAL"}


def __getattr__(name):
    if name in _deprecated_name_to_new_name:
        warn(f"'{name}' is deprecated and will be dropped "
             f"in version 2022.x. Use '{_deprecated_name_to_new_name[name]}' "
             "instead.",
             DeprecationWarning, stacklevel=2)
        return globals()[_deprecated_name_to_new_name[name]]

    raise AttributeError(f"module {__name__} has no attribute {name}")


if sys.version_info < (3, 7):
    for name in _deprecated_name_to_new_name:
        globals()[name] = globals()[_deprecated_name_to_new_name[name]]

# }}}


# vim: foldmethod=marker
