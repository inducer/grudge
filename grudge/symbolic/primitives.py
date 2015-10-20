"""Operator template language: primitives."""

from __future__ import division, absolute_import

__copyright__ = "Copyright (C) 2008 Andreas Kloeckner"

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

from six.moves import range, intern

import numpy as np
import pymbolic.primitives
from meshmode.mesh import BTAG_ALL, BTAG_NONE  # noqa

from pymbolic.primitives import (  # noqa
        cse_scope as cse_scope_base,
        make_common_subexpression as cse, If, Comparison)
from pymbolic.geometric_algebra import MultiVector
from pytools.obj_array import join_fields, make_obj_array  # noqa


class LeafBase(pymbolic.primitives.AlgebraicLeaf):
    def stringifier(self):
        from grudge.symbolic.mappers import StringifyMapper
        return StringifyMapper


class VTAG_ALL:
    """This is used in a 'where' field to denote the volume discretization."""
    pass


__doc__ = """
Symbols
^^^^^^^

.. autoclass:: Field
.. autoclass:: make_sym_vector
.. autoclass:: make_sym_array
.. autoclass:: ScalarParameter
.. autoclass:: CFunction

Helpers
^^^^^^^

.. autoclass:: OperatorBinding
.. autoclass:: PrioritizedSubexpression
.. autoclass:: BoundaryPair
.. autoclass:: Ones

Geometry data
^^^^^^^^^^^^^
.. autoclass:: NodeCoordinateComponent
.. autofunction:: nodes
.. autofunction:: mv_nodes
.. autofunction:: forward_metric_derivative
.. autofunction:: inverse_metric_derivative
.. autofunction:: forward_metric_derivative_mat
.. autofunction:: inverse_metric_derivative_mat
.. autofunction:: pseudoscalar
.. autofunction:: area_element
.. autofunction:: mv_normal
.. autofunction:: normal
"""


# {{{ variables

class cse_scope(cse_scope_base):  # noqa
    DISCRETIZATION = "grudge_discretization"


Field = pymbolic.primitives.Variable

make_sym_vector = pymbolic.primitives.make_sym_vector
make_sym_array = pymbolic.primitives.make_sym_array


def make_field(var_or_string):
    if not isinstance(var_or_string, pymbolic.primitives.Expression):
        return Field(var_or_string)
    else:
        return var_or_string


class ScalarParameter(pymbolic.primitives.Variable):
    """A placeholder for a user-supplied scalar variable."""

    def stringifier(self):
        from grudge.symbolic.mappers import StringifyMapper
        return StringifyMapper

    mapper_method = intern("map_scalar_parameter")


class CFunction(pymbolic.primitives.Variable):
    """A symbol representing a C-level function, to be used as the function
    argument of :class:`pymbolic.primitives.Call`.
    """
    def stringifier(self):
        from grudge.symbolic.mappers import StringifyMapper
        return StringifyMapper

    def __call__(self, expr):
        from pytools.obj_array import with_object_array_or_scalar
        from functools import partial
        return with_object_array_or_scalar(
                partial(pymbolic.primitives.Expression.__call__, self),
                expr)

    mapper_method = "map_c_function"


sqrt = CFunction("sqrt")
exp = CFunction("exp")
sin = CFunction("sin")
cos = CFunction("cos")

# }}}


# {{{ technical helpers

class OperatorBinding(LeafBase):
    def __init__(self, op, field):
        self.op = op
        self.field = field

    mapper_method = intern("map_operator_binding")

    def __getinitargs__(self):
        return self.op, self.field

    def is_equal(self, other):
        from pytools.obj_array import obj_array_equal
        return (other.__class__ == self.__class__
                and other.op == self.op
                and obj_array_equal(other.field, self.field))

    def get_hash(self):
        from pytools.obj_array import obj_array_to_hashable
        return hash((self.__class__, self.op, obj_array_to_hashable(self.field)))


class PrioritizedSubexpression(pymbolic.primitives.CommonSubexpression):
    """When the optemplate-to-code transformation is performed,
    prioritized subexpressions  work like common subexpression in
    that they are assigned their own separate identifier/register
    location. In addition to this behavior, prioritized subexpressions
    are evaluated with a settable priority, allowing the user to
    expedite or delay the evaluation of the subexpression.
    """

    def __init__(self, child, priority=0):
        pymbolic.primitives.CommonSubexpression.__init__(self, child)
        self.priority = priority

    def __getinitargs__(self):
        return (self.child, self.priority)

    def get_extra_properties(self):
        return {"priority": self.priority}


class BoundaryPair(LeafBase):
    """Represents a pairing of a volume and a boundary field, used for the
    application of boundary fluxes.
    """

    def __init__(self, field, bfield, tag=BTAG_ALL):
        self.field = field
        self.bfield = bfield
        self.tag = tag

    mapper_method = intern("map_boundary_pair")

    def __getinitargs__(self):
        return (self.field, self.bfield, self.tag)

    def get_hash(self):
        from pytools.obj_array import obj_array_to_hashable

        return hash((self.__class__,
            obj_array_to_hashable(self.field),
            obj_array_to_hashable(self.bfield),
            self.tag))

    def is_equal(self, other):
        from pytools.obj_array import obj_array_equal
        return (self.__class__ == other.__class__
                and obj_array_equal(other.field,  self.field)
                and obj_array_equal(other.bfield, self.bfield)
                and other.tag == self.tag)

# }}}


class Ones(LeafBase):
    def __init__(self, quadrature_tag=None, where=None):
        self.where = where
        self.quadrature_tag = quadrature_tag

    def __getinitargs__(self):
        return (self.where, self.quadrature_tag,)

    mapper_method = intern("map_ones")


# {{{ geometry data

class DiscretizationProperty(LeafBase):
    """
    .. attribute:: where

        *None* for the default volume discretization or a boundary
        tag for an operation on the denoted part of the boundary.

    .. attribute:: quadrature_tag

        quadrature tag for the grid on
        which this geometric factor is needed, or None for
        nodal representation.
    """

    def __init__(self, quadrature_tag, where=None):
        self.quadrature_tag = quadrature_tag
        self.where = where

    def __getinitargs__(self):
        return (self.quadrature_tag, self.where)


class NodeCoordinateComponent(DiscretizationProperty):

    def __init__(self, axis, quadrature_tag=None, where=None):
        super(NodeCoordinateComponent, self).__init__(quadrature_tag, where)
        self.axis = axis

    def __getinitargs__(self):
        return (self.axis, self.quadrature_tag)

    mapper_method = intern("map_node_coordinate_component")


def nodes(ambient_dim, quadrature_tag=None, where=None):
    return np.array([NodeCoordinateComponent(i, quadrature_tag, where)
        for i in range(ambient_dim)], dtype=object)


def mv_nodes(ambient_dim, quadrature_tag=None, where=None):
    return MultiVector(
            nodes(ambient_dim, quadrature_tag=quadrature_tag, where=where))


def forward_metric_derivative(xyz_axis, rst_axis, where=None,
        quadrature_tag=None):
    r"""
    Pointwise metric derivatives representing

    .. math::

        \frac{d x_{\mathtt{xyz\_axis}} }{d r_{\mathtt{rst\_axis}} }
    """
    from grudge.symbolic.operators import (
            ReferenceDifferentiationOperator, QuadratureGridUpsampler)
    diff_op = ReferenceDifferentiationOperator(
            rst_axis, where=where)

    result = diff_op(NodeCoordinateComponent(xyz_axis, where=where))

    if quadrature_tag is not None:
        result = QuadratureGridUpsampler(quadrature_tag, where)(result)

    return cse(
            result,
            prefix="dx%d_dr%d" % (xyz_axis, rst_axis),
            scope=cse_scope.DISCRETIZATION)


def forward_metric_derivative_vector(ambient_dim, rst_axis, where=None,
        quadrature_tag=None):
    return make_obj_array([
        forward_metric_derivative(
            i, rst_axis, where=where, quadrature_tag=quadrature_tag)
        for i in range(ambient_dim)])


def forward_metric_derivative_mv(ambient_dim, rst_axis, where=None,
        quadrature_tag=None):
    return MultiVector(
        forward_metric_derivative_vector(
            ambient_dim, rst_axis, where=where, quadrature_tag=quadrature_tag))


def parametrization_derivative(ambient_dim, dim=None, where=None,
        quadrature_tag=None):
    if dim is None:
        dim = ambient_dim

    from pytools import product
    return product(
        forward_metric_derivative_mv(
            ambient_dim, rst_axis, where, quadrature_tag)
        for rst_axis in range(dim))


def inverse_metric_derivative(rst_axis, xyz_axis, ambient_dim, dim=None,
        where=None, quadrature_tag=None):
    if dim is None:
        dim = ambient_dim

    if dim != ambient_dim:
        raise ValueError("not clear what inverse_metric_derivative means if "
                "the derivative matrix is not square")

    par_vecs = [
        forward_metric_derivative_mv(
            ambient_dim, rst, where, quadrature_tag)
        for rst in range(dim)]

    # Yay Cramer's rule! (o rly?)
    from functools import reduce, partial
    from operator import xor as outerprod_op
    outerprod = partial(reduce, outerprod_op)

    def outprod_with_unit(i, at):
        unit_vec = np.zeros(dim)
        unit_vec[i] = 1

        vecs = par_vecs[:]
        vecs[at] = MultiVector(unit_vec)

        return outerprod(vecs)

    volume_pseudoscalar_inv = cse(outerprod(
        forward_metric_derivative_mv(
            ambient_dim, rst_axis, where, quadrature_tag)
        for rst_axis in range(dim)).inv())

    return (outprod_with_unit(xyz_axis, rst_axis)
            * volume_pseudoscalar_inv
            ).as_scalar()


def forward_metric_derivative_mat(ambient_dim, dim=None,
        where=None, quadrature_tag=None):
    if dim is None:
        dim = ambient_dim

    result = np.zeros((ambient_dim, dim), dtype=np.object)
    for j in range(dim):
        result[:, j] = forward_metric_derivative_vector(ambient_dim, j,
                where=where, quadrature_tag=quadrature_tag)
    return result


def inverse_metric_derivative_mat(ambient_dim, dim=None,
        where=None, quadrature_tag=None):
    if dim is None:
        dim = ambient_dim

    result = np.zeros((ambient_dim, dim), dtype=np.object)
    for i in range(dim):
        for j in range(ambient_dim):
            result[i, j] = inverse_metric_derivative(
                    i, j, ambient_dim, dim,
                    where=where, quadrature_tag=quadrature_tag)

    return result


def pseudoscalar(ambient_dim, dim=None, where=None, quadrature_tag=None):
    if dim is None:
        dim = ambient_dim

    return cse(
        parametrization_derivative(ambient_dim, dim, where=where,
            quadrature_tag=quadrature_tag)
        .project_max_grade(),
        "pseudoscalar", cse_scope.DISCRETIZATION)


def area_element(ambient_dim, dim=None, where=None, quadrature_tag=None):
    return cse(
            sqrt(
                pseudoscalar(ambient_dim, dim, where, quadrature_tag=quadrature_tag)
                .norm_squared()),
            "area_element", cse_scope.DISCRETIZATION)


def mv_normal(tag, ambient_dim, dim=None, quadrature_tag=None):
    """Exterior unit normal as a MultiVector."""

    if dim is None:
        dim = ambient_dim - 1

    # Don't be tempted to add a sign here. As it is, it produces
    # exterior normals for positively oriented curves.

    pder = (
            pseudoscalar(ambient_dim, dim, tag, quadrature_tag=quadrature_tag)
            /
            area_element(ambient_dim, dim, tag, quadrature_tag=quadrature_tag))
    return cse(pder.I | pder, "normal",
            cse_scope.DISCRETIZATION)


def normal(tag, ambient_dim, dim=None, quadrature_tag=None):
    return mv_normal(tag, ambient_dim, dim,
            quadrature_tag=quadrature_tag).as_vector()

# }}}


# vim: foldmethod=marker
