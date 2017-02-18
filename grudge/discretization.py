from __future__ import division, absolute_import

__copyright__ = "Copyright (C) 2015 Andreas Kloeckner"

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
from grudge import sym


class Discretization(object):
    """
    .. attribute :: volume_discr

    .. automethod :: boundary_connection
    .. automethod :: boundary_discr

    .. automethod :: interior_faces_connection
    .. automethod :: interior_faces_discr

    .. automethod :: all_faces_connection
    .. automethod :: all_faces_discr

    .. autoattribute :: cl_context
    .. autoattribute :: dim
    .. autoattribute :: ambient_dim
    .. autoattribute :: mesh

    .. automethod :: empty
    .. automethod :: zeros
    """

    def __init__(self, cl_ctx, mesh, order, quad_min_degrees=None):
        """
        :param quad_min_degrees: A mapping from quadrature tags to the degrees
            to which the desired quadrature is supposed to be exact.
        """

        if quad_min_degrees is None:
            quad_min_degrees = {}

        from meshmode.discretization import Discretization
        from meshmode.discretization.poly_element import \
                PolynomialWarpAndBlendGroupFactory

        self.volume_discr = Discretization(cl_ctx, mesh,
                PolynomialWarpAndBlendGroupFactory(order=order))

        self.order = order
        self.quad_min_degrees = quad_min_degrees

    # {{{ boundary

    @memoize_method
    def boundary_connection(self, boundary_tag, quadrature_tag=None):
        from meshmode.discretization.poly_element import \
                PolynomialWarpAndBlendGroupFactory

        if quadrature_tag is not sym.QTAG_NONE:
            # FIXME
            raise NotImplementedError("quadrature")

        from meshmode.discretization.connection import make_face_restriction
        return make_face_restriction(
                        self.volume_discr,
                        PolynomialWarpAndBlendGroupFactory(order=self.order),
                        boundary_tag=boundary_tag)

    def boundary_discr(self, boundary_tag, quadrature_tag=None):
        return self.boundary_connection(boundary_tag, quadrature_tag).to_discr

    # }}}

    # {{{ interior faces

    @memoize_method
    def interior_faces_connection(self, quadrature_tag=None):
        from meshmode.discretization.poly_element import \
                PolynomialWarpAndBlendGroupFactory

        if quadrature_tag is not sym.QTAG_NONE:
            # FIXME
            raise NotImplementedError("quadrature")

        from meshmode.discretization.connection import (
                make_face_restriction, FRESTR_INTERIOR_FACES)
        return make_face_restriction(
                        self.volume_discr,
                        PolynomialWarpAndBlendGroupFactory(order=self.order),
                        FRESTR_INTERIOR_FACES,

                        # FIXME: This will need to change as soon as we support
                        # pyramids or other elements with non-identical face
                        # types.
                        per_face_groups=False)

    def interior_faces_discr(self, quadrature_tag=None):
        return self.interior_faces_connection(quadrature_tag).to_discr

    @memoize_method
    def opposite_face_connection(self, quadrature_tag):
        if quadrature_tag is not sym.QTAG_NONE:
            # FIXME
            raise NotImplementedError("quadrature")

        from meshmode.discretization.connection import \
                make_opposite_face_connection

        return make_opposite_face_connection(
                self.interior_faces_connection(quadrature_tag))

    # }}}

    # {{{ all-faces

    @memoize_method
    def all_faces_volume_connection(self, quadrature_tag=None):
        if quadrature_tag is not sym.QTAG_NONE:
            # FIXME
            raise NotImplementedError("quadrature")

        from meshmode.discretization.poly_element import \
                PolynomialWarpAndBlendGroupFactory
        from meshmode.discretization.connection import (
                make_face_restriction, FRESTR_ALL_FACES)
        return make_face_restriction(
                        self.volume_discr,
                        PolynomialWarpAndBlendGroupFactory(order=self.order),
                        FRESTR_ALL_FACES,

                        # FIXME: This will need to change as soon as we support
                        # pyramids or other elements with non-identical face
                        # types.
                        per_face_groups=False)

    def all_faces_discr(self, quadrature_tag=None):
        return self.all_faces_volume_connection(quadrature_tag).to_discr

    @memoize_method
    def all_faces_connection(self, boundary_tag, quadrature_tag=None):
        """Return a
        :class:`meshmode.discretization.connection.DiscretizationConnection`
        that goes from either
        :meth:`interior_faces_discr` (if *boundary_tag* is None)
        or
        :meth:`boundary_discr` (if *boundary_tag* is not None)
        to a discretization containing all the faces of the volume
        discretization.
        """
        from meshmode.discretization.connection import \
                make_face_to_all_faces_embedding

        if boundary_tag is None:
            faces_conn = self.interior_faces_connection(quadrature_tag)
        else:
            faces_conn = self.boundary_connection(boundary_tag, quadrature_tag)

        return make_face_to_all_faces_embedding(
                faces_conn, self.all_faces_discr(quadrature_tag))

    # }}}

    @property
    def cl_context(self):
        return self.volume_discr.cl_context

    @property
    def dim(self):
        return self.volume_discr.dim

    @property
    def ambient_dim(self):
        return self.volume_discr.ambient_dim

    @property
    def mesh(self):
        return self.volume_discr.mesh

    def empty(self, queue=None, dtype=None, extra_dims=None, allocator=None):
        return self.volume_discr.empty(queue, dtype, extra_dims=extra_dims,
                allocator=allocator)

    def zeros(self, queue, dtype=None, extra_dims=None, allocator=None):
        return self.volume_discr.zeros(queue, dtype, extra_dims=extra_dims,
                allocator=allocator)

    def is_volume_where(self, where):
        from grudge import sym
        return (
                where is None
                or where == sym.VTAG_ALL)


# vim: foldmethod=marker
