"""Degree of freedom (DOF) descriptions"""

__copyright__ = """
Copyright (C) 2008 Andreas Kloeckner
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

__doc__ = """
DOF description
^^^^^^^^^^^^^^^

.. autoclass:: DTAG_SCALAR
.. autoclass:: DTAG_VOLUME_ALL
.. autoclass:: DTAG_BOUNDARY
.. autoclass:: QTAG_NONE

.. autoclass:: DOFDesc
.. autofunction:: as_dofdesc

.. data:: DD_SCALAR
.. data:: DD_VOLUME
"""


# {{{ DOF description

class DTAG_SCALAR:          # noqa: N801
    pass


class DTAG_VOLUME_ALL:      # noqa: N801
    pass


class DTAG_BOUNDARY:        # noqa: N801
    def __init__(self, tag):
        self.tag = tag

    def __eq__(self, other):
        return isinstance(other, DTAG_BOUNDARY) and self.tag == other.tag

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash(type(self)) ^ hash(self.tag)

    def __repr__(self):
        return "<{}({})>".format(type(self).__name__, repr(self.tag))


class QTAG_NONE:            # noqa: N801
    pass


class DOFDesc:
    """Describes the meaning of degrees of freedom.

    .. attribute:: domain_tag
    .. attribute:: quadrature_tag

    .. automethod:: is_scalar
    .. automethod:: is_discretized
    .. automethod:: is_volume
    .. automethod:: is_boundary_or_partition_interface
    .. automethod:: is_trace

    .. automethod:: uses_quadrature

    .. automethod:: with_qtag
    .. automethod:: with_dtag

    .. automethod:: __eq__
    .. automethod:: __ne__
    .. automethod:: __hash__
    """

    def __init__(self, domain_tag, quadrature_tag=None):
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

        :arg quadrature_tag:
            *None* to indicate that the quadrature grid is not known, or
            :class:`QTAG_NONE` to indicate the use of the basic discretization
            grid, or a string to indicate the use of the thus-tagged quadratue
            grid.
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

        if domain_tag is DTAG_SCALAR and quadrature_tag is not None:
            raise ValueError("cannot have nontrivial quadrature tag on scalar")

        if quadrature_tag is None:
            quadrature_tag = QTAG_NONE

        self.domain_tag = domain_tag
        self.quadrature_tag = quadrature_tag

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
        if self.quadrature_tag is None:
            return False
        if self.quadrature_tag is QTAG_NONE:
            return False

        return True

    def with_qtag(self, qtag):
        return type(self)(domain_tag=self.domain_tag, quadrature_tag=qtag)

    def with_dtag(self, dtag):
        return type(self)(domain_tag=dtag, quadrature_tag=self.quadrature_tag)

    def __eq__(self, other):
        return (type(self) == type(other)
                and self.domain_tag == other.domain_tag
                and self.quadrature_tag == other.quadrature_tag)

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash((type(self), self.domain_tag, self.quadrature_tag))

    def __repr__(self):
        def fmt(s):
            if isinstance(s, type):
                return s.__name__
            else:
                return repr(s)

        return "DOFDesc({}, {})".format(
                fmt(self.domain_tag),
                fmt(self.quadrature_tag))


DD_SCALAR = DOFDesc(DTAG_SCALAR, None)

DD_VOLUME = DOFDesc(DTAG_VOLUME_ALL, None)


def as_dofdesc(dd):
    if isinstance(dd, DOFDesc):
        return dd
    return DOFDesc(dd, quadrature_tag=None)

# }}}


# vim: foldmethod=marker
