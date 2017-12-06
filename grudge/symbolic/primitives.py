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
from meshmode.mesh import BTAG_ALL, BTAG_REALLY_ALL, BTAG_NONE  # noqa
from meshmode.discretization.connection import (  # noqa
        FACE_RESTR_ALL, FACE_RESTR_INTERIOR)

from pymbolic.primitives import (  # noqa
        cse_scope as cse_scope_base,
        make_common_subexpression as cse, If, Comparison)
from pymbolic.geometric_algebra import MultiVector
from pytools.obj_array import join_fields, make_obj_array  # noqa


class ExpressionBase(pymbolic.primitives.Expression):
    def stringifier(self):
        from grudge.symbolic.mappers import StringifyMapper
        return StringifyMapper


def _sym():
    # A hack to make referring to grudge.sym possible without
    # circular imports and tons of typing.

    from grudge import sym
    return sym


__doc__ = """

.. currentmodule:: grudge.sym

.. autoclass:: If

DOF description
^^^^^^^^^^^^^^^

.. autoclass:: DTAG_SCALAR
.. autoclass:: DTAG_VOLUME_ALL
.. autoclass:: QTAG_NONE
.. autoclass:: DOFDesc
.. data:: DD_SCALAR
.. data:: DD_VOLUME

Symbols
^^^^^^^

.. autoclass:: Variable
.. autoclass:: ScalarVariable
.. autoclass:: make_sym_array
.. autoclass:: make_sym_mv
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


# {{{ DOF description

class DTAG_SCALAR:  # noqa
    pass


class DTAG_VOLUME_ALL:  # noqa
    pass


class DTAG_BOUNDARY:  # noqa
    def __init__(self, tag):
        self.tag = tag


class QTAG_NONE:  # noqa
    pass


class DOFDesc(object):
    """Describes the meaning of degrees of freedom.

    .. attribute:: domain_tag
    .. attribute:: quadrature_tag
    .. automethod:: is_scalar
    .. automethod:: is_discretized
    .. automethod:: is_volume
    .. automethod:: is_boundary
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
            :class:`grudge.sym.DTAG_SCALAR` (or the string ``"scalar"``),
            :class:`grudge.sym.DTAG_VOLUME_ALL` (or the string ``"vol"``)
            for the default volume discretization,
            :class:`meshmode.discretization.connection.FACE_RESTR_ALL`
            (or the string ``"all_faces"``),
            or
            :class:`meshmode.discretization.connection.FACE_RESTR_INTERIOR`
            (or the string ``"int_faces"``),
            or one of
            :class:`meshmode.discretization.BTAG_ALL`,
            :class:`meshmode.discretization.BTAG_NONE`,
            :class:`meshmode.discretization.BTAG_REALLY_ALL`,
            or :class
            or *None* to indicate that the geometry is not yet known.

        :arg quadrature_tag:
            *None* to indicate that the quadrature grid is not known,or
            :class:`QTAG_NONE` to indicate the use of the basic discretization
            grid, or a string to indicate the use of the thus-tagged quadratue
            grid.
        """
        if domain_tag == "scalar":
            domain_tag = DTAG_SCALAR
        elif domain_tag is DTAG_SCALAR:
            domain_tag = DTAG_SCALAR
        elif domain_tag == "vol":
            domain_tag = DTAG_VOLUME_ALL
        elif domain_tag is DTAG_VOLUME_ALL:
            pass
        elif domain_tag == "all_faces":
            domain_tag = FACE_RESTR_ALL
        elif domain_tag is FACE_RESTR_ALL:
            pass
        elif domain_tag == "int_faces":
            domain_tag = FACE_RESTR_INTERIOR
        elif domain_tag is FACE_RESTR_INTERIOR:
            pass
        elif domain_tag is None:
            pass
        elif domain_tag in [BTAG_ALL, BTAG_REALLY_ALL, BTAG_NONE]:
            # FIXME: Should wrap these in DTAG_BOUNDARY
            pass
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

    def is_boundary(self):
        return (
                self.domain_tag in [
                    BTAG_ALL, BTAG_NONE, BTAG_REALLY_ALL]
                or isinstance(self.domain_tag, DTAG_BOUNDARY))

    def is_trace(self):
        return (self.is_boundary()
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

        return "DOFDesc(%s, %s)" % (fmt(self.domain_tag), fmt(self.quadrature_tag))


DD_SCALAR = DOFDesc(DTAG_SCALAR, None)

DD_VOLUME = DOFDesc(DTAG_VOLUME_ALL, None)


def as_dofdesc(dd):
    if isinstance(dd, DOFDesc):
        return dd
    else:
        return DOFDesc(dd, None)

# }}}


# {{{ has-dof-desc mix-in

class HasDOFDesc(object):
    """
    .. attribute:: dd

        an instance of :class:`grudge.sym.DOFDesc` describing the
        discretization on which this property is given.
    """

    def __init__(self, dd):
        self.dd = dd

    def __getinitargs__(self):
        return (self.dd,)

    def with_dd(self, dd):
        """Return a copy of *self*, modified to the given DOF descriptor.
        """
        return type(self)(
                *self.__getinitargs__()[:-1],
                dd=dd or self.dd)

# }}}


# {{{ variables

class cse_scope(cse_scope_base):  # noqa
    DISCRETIZATION = "grudge_discretization"


class Variable(HasDOFDesc, ExpressionBase, pymbolic.primitives.Variable):
    """A user-supplied input variable with a known :class:`DOFDesc`.
    """

    def __init__(self, name, dd=None):
        if dd is None:
            dd = DD_VOLUME

        HasDOFDesc.__init__(self, dd)
        pymbolic.primitives.Variable.__init__(self, name)

    def __getinitargs__(self):
        return (self.name, self.dd,)

    mapper_method = "map_grudge_variable"


var = Variable


class ScalarVariable(Variable):
    def __init__(self, name):
        super(ScalarVariable, self).__init__(name, DD_SCALAR)


def make_sym_array(name, shape, dd=None):
    def var_factory(name):
        return Variable(name, dd)

    return pymbolic.primitives.make_sym_array(name, shape, var_factory)


def make_sym_mv(name, dim, var_factory=None):
    return MultiVector(
            make_sym_array(name, dim, var_factory))


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

class OperatorBinding(ExpressionBase):
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

# }}}


class Ones(ExpressionBase, HasDOFDesc):
    mapper_method = intern("map_ones")


# {{{ geometry data

class DiscretizationProperty(ExpressionBase, HasDOFDesc):
    pass


class NodeCoordinateComponent(DiscretizationProperty):
    def __init__(self, axis, dd=None):
        if not dd.is_discretized():
            raise ValueError("dd must be a discretization for "
                    "NodeCoordinateComponent")

        super(NodeCoordinateComponent, self).__init__(dd)
        self.axis = axis

        assert dd.domain_tag is not None

    def __getinitargs__(self):
        return (self.axis, self.dd)

    mapper_method = intern("map_node_coordinate_component")


def nodes(ambient_dim, dd=None):
    if dd is None:
        dd = DD_VOLUME

    dd = as_dofdesc(dd)

    return np.array([NodeCoordinateComponent(i, dd)
        for i in range(ambient_dim)], dtype=object)


def mv_nodes(ambient_dim, dd=None):
    return MultiVector(nodes(ambient_dim, dd))


def forward_metric_derivative(xyz_axis, rst_axis, dd=None):
    r"""
    Pointwise metric derivatives representing

    .. math::

        \frac{d x_{\mathtt{xyz\_axis}} }{d r_{\mathtt{rst\_axis}} }
    """
    if dd is None:
        dd = DD_VOLUME

    inner_dd = dd.with_qtag(QTAG_NONE)

    diff_op = _sym().RefDiffOperator(rst_axis, inner_dd)

    result = diff_op(NodeCoordinateComponent(xyz_axis, inner_dd))

    if dd.uses_quadrature():
        result = _sym().interp(inner_dd, dd)(result)

    return cse(
            result,
            prefix="dx%d_dr%d" % (xyz_axis, rst_axis),
            scope=cse_scope.DISCRETIZATION)


def forward_metric_derivative_vector(ambient_dim, rst_axis, dd=None):
    return make_obj_array([
        forward_metric_derivative(i, rst_axis, dd=dd)
        for i in range(ambient_dim)])


def forward_metric_derivative_mv(ambient_dim, rst_axis, dd=None):
    return MultiVector(
        forward_metric_derivative_vector(ambient_dim, rst_axis, dd=dd))


def parametrization_derivative(ambient_dim, dim=None, dd=None):
    if dim is None:
        dim = ambient_dim

    from pytools import product
    return product(
        forward_metric_derivative_mv(ambient_dim, rst_axis, dd)
        for rst_axis in range(dim))


def inverse_metric_derivative(rst_axis, xyz_axis, ambient_dim, dim=None,
        dd=None):
    if dim is None:
        dim = ambient_dim

    if dim != ambient_dim:
        raise ValueError("not clear what inverse_metric_derivative means if "
                "the derivative matrix is not square")

    par_vecs = [
        forward_metric_derivative_mv(ambient_dim, rst, dd)
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
            ambient_dim, rst_axis, dd=dd)
        for rst_axis in range(dim)).inv())

    return cse(
            (outprod_with_unit(xyz_axis, rst_axis)
                * volume_pseudoscalar_inv).as_scalar(),
            prefix="dr%d_dx%d" % (rst_axis, xyz_axis),
            scope=cse_scope.DISCRETIZATION)


def forward_metric_derivative_mat(ambient_dim, dim=None, dd=None):
    if dim is None:
        dim = ambient_dim

    result = np.zeros((ambient_dim, dim), dtype=np.object)
    for j in range(dim):
        result[:, j] = forward_metric_derivative_vector(ambient_dim, j, dd=dd)
    return result


def inverse_metric_derivative_mat(ambient_dim, dim=None, dd=None):
    if dim is None:
        dim = ambient_dim

    result = np.zeros((ambient_dim, dim), dtype=np.object)
    for i in range(dim):
        for j in range(ambient_dim):
            result[i, j] = inverse_metric_derivative(
                    i, j, ambient_dim, dim, dd=dd)

    return result


def pseudoscalar(ambient_dim, dim=None, dd=None):
    if dim is None:
        dim = ambient_dim

    return cse(
        parametrization_derivative(ambient_dim, dim, dd=dd)
        .project_max_grade(),
        "pseudoscalar", cse_scope.DISCRETIZATION)


def area_element(ambient_dim, dim=None, dd=None):
    return cse(
            sqrt(
                pseudoscalar(ambient_dim, dim, dd=dd)
                .norm_squared()),
            "area_element", cse_scope.DISCRETIZATION)


def mv_normal(dd, ambient_dim, dim=None):
    """Exterior unit normal as a MultiVector."""

    dd = as_dofdesc(dd)

    if not dd.is_trace():
        raise ValueError("may only request normals on boundaries")

    if dim is None:
        dim = ambient_dim - 1

    # Don't be tempted to add a sign here. As it is, it produces
    # exterior normals for positively oriented curves.

    pder = (
            pseudoscalar(ambient_dim, dim, dd=dd)
            /
            area_element(ambient_dim, dim, dd=dd))
    return cse(
            # Dorst Section 3.7.2
            pder << pder.I.inv(),
            "normal", cse_scope.DISCRETIZATION)


def normal(dd, ambient_dim, dim=None, quadrature_tag=None):
    return mv_normal(dd, ambient_dim, dim).as_vector()

# }}}


# {{{ trace pair

class TracePair:
    """
    .. attribute:: dd

        an instance of :class:`grudge.sym.DOFDesc` describing the
        discretization on which :attr:`interior` and :attr:`exterior`
        live.

    .. attribute:: interior

        an expression representing the interior value to
        be used for the flux.

    .. attribute:: exterior

        an expression representing the exterior value to
        be used for the flux.
    """
    def __init__(self, dd, interior, exterior):
        """
        """

        self.dd = as_dofdesc(dd)
        self.interior = interior
        self.exterior = exterior

    def __getitem__(self, index):
        return TracePair(
                self.dd,
                self.exterior[index],
                self.interior[index])

    @property
    def int(self):
        return self.interior

    @property
    def ext(self):
        return self.exterior

    @property
    def avg(self):
        return 0.5*(self.int + self.ext)


def int_tpair(expression, qtag=None):
    i = _sym().interp("vol", "int_faces")(expression)
    e = cse(_sym().OppositeInteriorFaceSwap()(i))

    if qtag is not None and qtag != _sym().QTAG_NONE:
        q_dd = _sym().DOFDesc("int_faces", qtag)
        i = cse(_sym().interp("int_faces", q_dd)(i))
        e = cse(_sym().interp("int_faces", q_dd)(e))
    else:
        q_dd = "int_faces"

    return TracePair(q_dd, i, e)

    #i = cse(_sym().interp("vol", "int_faces")(expression))
    #e = cse(_sym().OppositeInteriorFaceSwap()(i))
    #return TracePair("int_faces", i, e)


def bdry_tpair(dd, interior, exterior):
    """
    :arg interior: an expression that already lives on the boundary
        representing the interior value to be used
        for the flux.
    :arg exterior: an expression that already lives on the boundary
        representing the exterior value to be used
        for the flux.
    """
    return TracePair(dd, interior, exterior)


def bv_tpair(dd, interior, exterior):
    """
    :arg interior: an expression that lives in the volume
        and will be restricted to the boundary identified by
        *tag* before use.
    :arg exterior: an expression that already lives on the boundary
        representing the exterior value to be used
        for the flux.
    """
    interior = _sym().cse(_sym().interp("vol", dd)(interior))
    return TracePair(dd, interior, exterior)

# }}}


# vim: foldmethod=marker
