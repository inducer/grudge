"""Operator template language: primitives."""

__copyright__ = "Copyright (C) 2008-2017 Andreas Kloeckner, Bogdan Enache"

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

from sys import intern

import numpy as np
from pytools.obj_array import make_obj_array

from meshmode.mesh import (
        BTAG_ALL,
        BTAG_REALLY_ALL,
        BTAG_NONE,
        BTAG_PARTITION)
from meshmode.discretization.connection import (
        FACE_RESTR_ALL,
        FACE_RESTR_INTERIOR)

import pymbolic.primitives as prim
from pymbolic.primitives import (
        Variable as VariableBase,
        CommonSubexpression as CommonSubexpressionBase,
        cse_scope as cse_scope_base,
        make_common_subexpression as cse)
from pymbolic.geometric_algebra import MultiVector


class ExpressionBase(prim.Expression):
    def make_stringifier(self, originating_stringifier=None):
        from grudge.symbolic.mappers import StringifyMapper
        return StringifyMapper()


__doc__ = """

.. currentmodule:: grudge.sym

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

Symbols
^^^^^^^

.. autoclass:: Variable
.. autoclass:: ScalarVariable
.. autoclass:: FunctionSymbol

.. autofunction:: make_sym_array
.. autofunction:: make_sym_mv

.. function :: fabs(arg)
.. function :: sqrt(arg)
.. function :: exp(arg)
.. function :: sin(arg)
.. function :: cos(arg)
.. function :: besesl_j(n, arg)
.. function :: besesl_y(n, arg)

Helpers
^^^^^^^

.. autoclass:: OperatorBinding
.. autoclass:: PrioritizedSubexpression
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
.. autofunction:: first_fundamental_form
.. autofunction:: inverse_first_fundamental_form
.. autofunction:: inverse_surface_metric_derivative
.. autofunction:: second_fundamental_form
.. autofunction:: shape_operator
.. autofunction:: pseudoscalar
.. autofunction:: area_element
.. autofunction:: mv_normal
.. autofunction:: normal
.. autofunction:: surface_normal
.. autofunction:: summed_curvature
.. autofunction:: mean_curvature

Trace Pair
^^^^^^^^^^

.. autoclass:: TracePair
.. autofunction:: int_tpair
.. autofunction:: bv_tpair
.. autofunction:: bdry_tpair
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


# {{{ has-dof-desc mix-in

class HasDOFDesc:
    """
    .. attribute:: dd

        an instance of :class:`grudge.sym.DOFDesc` describing the
        discretization on which this property is given.
    """

    def __init__(self, *args, **kwargs):
        # The remaining arguments are passed to the chained superclass.

        if "dd" in kwargs:
            dd = kwargs.pop("dd")
        else:
            dd = args[-1]
            args = args[:-1]

        super().__init__(*args, **kwargs)
        self.dd = dd

    def __getinitargs__(self):
        return (self.dd,)

    def with_dd(self, dd):
        """Return a copy of *self*, modified to the given DOF descriptor.
        """
        return type(self)(*self.__getinitargs__())

# }}}


# {{{ variables

class cse_scope(cse_scope_base):        # noqa: N801
    DISCRETIZATION = "grudge_discretization"


class Variable(HasDOFDesc, ExpressionBase, VariableBase):
    """A user-supplied input variable with a known :class:`DOFDesc`.
    """
    init_arg_names = ("name", "dd")

    def __init__(self, name, dd=None):
        if dd is None:
            dd = DD_VOLUME

        super().__init__(name, dd)

    def __getinitargs__(self):
        return (self.name, self.dd,)

    mapper_method = "map_grudge_variable"


var = Variable


class ScalarVariable(Variable):
    def __init__(self, name):
        super().__init__(name, DD_SCALAR)


def make_sym_array(name, shape, dd=None):
    def var_factory(name):
        return Variable(name, dd)

    return prim.make_sym_array(name, shape, var_factory)


def make_sym_mv(name, dim, var_factory=None):
    return MultiVector(
            make_sym_array(name, dim, var_factory))


class FunctionSymbol(ExpressionBase, VariableBase):
    """A symbol to be used as the function argument of
    :class:`~pymbolic.primitives.Call`.
    """

    def __call__(self, *exprs):
        from pytools.obj_array import obj_array_vectorize_n_args
        return obj_array_vectorize_n_args(
                super().__call__, *exprs)

    mapper_method = "map_function_symbol"


fabs = FunctionSymbol("fabs")
sqrt = FunctionSymbol("sqrt")
exp = FunctionSymbol("exp")
sin = FunctionSymbol("sin")
cos = FunctionSymbol("cos")
atan2 = FunctionSymbol("atan2")
bessel_j = FunctionSymbol("bessel_j")
bessel_y = FunctionSymbol("bessel_y")

# }}}


# {{{ technical helpers

class OperatorBinding(ExpressionBase):
    init_arg_names = ("op", "field")

    def __init__(self, op, field):
        self.op = op
        self.field = field

    mapper_method = intern("map_operator_binding")

    def __getinitargs__(self):
        return self.op, self.field

    def is_equal(self, other):
        return (other.__class__ == self.__class__
                and other.op == self.op
                and np.array_equal(other.field, self.field))

    def get_hash(self):
        from pytools.obj_array import obj_array_to_hashable
        return hash((self.__class__, self.op, obj_array_to_hashable(self.field)))


class PrioritizedSubexpression(CommonSubexpressionBase):
    """When the optemplate-to-code transformation is performed,
    prioritized subexpressions  work like common subexpression in
    that they are assigned their own separate identifier/register
    location. In addition to this behavior, prioritized subexpressions
    are evaluated with a settable priority, allowing the user to
    expedite or delay the evaluation of the subexpression.
    """

    def __init__(self, child, priority=0):
        super().__init__(child)
        self.priority = priority

    def __getinitargs__(self):
        return (self.child, self.priority)

    def get_extra_properties(self):
        return {"priority": self.priority}

# }}}


class Ones(HasDOFDesc, ExpressionBase):
    mapper_method = intern("map_ones")


class _SignedFaceOnes(HasDOFDesc, ExpressionBase):
    """Produces DoFs on a face that are :math:`-1` if their corresponding
    face number is odd and :math:`+1` if it is even.
    *dd* must refer to a 0D (point-shaped) trace domain.
    This is based on the face order of
    :meth:`meshmode.mesh.MeshElementGroup.face_vertex_indices`.

   .. note::

       This is used as a hack to generate normals with the correct orientation
       in 1D problems, and so far only intended for this particular use cases.
       (If you can think of a better way, please speak up!)
    """

    def __init__(self, dd):
        dd = as_dofdesc(dd)
        assert dd.is_trace()
        super().__init__(dd)

    mapper_method = intern("map_signed_face_ones")


# {{{ geometry data

class DiscretizationProperty(HasDOFDesc, ExpressionBase):
    pass


class NodeCoordinateComponent(DiscretizationProperty):
    def __init__(self, axis, dd=None):
        if not dd.is_discretized():
            raise ValueError("dd must be a discretization for "
                    "NodeCoordinateComponent")

        super().__init__(dd)
        self.axis = axis

        assert dd.domain_tag is not None

    init_arg_names = ("axis", "dd")

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


def forward_metric_nth_derivative(xyz_axis, ref_axes, dd=None):
    r"""
    Pointwise metric derivatives representing repeated derivatives

    .. math::

        \frac{\partial^n x_{\mathrm{xyz\_axis}} }{\partial r_{\mathrm{ref\_axes}}}

    where *ref_axes* is a multi-index description.

    :arg ref_axes: a :class:`tuple` of tuples indicating indices of
        coordinate axes of the reference element to the number of derivatives
        which will be taken. For example, the value ``((0, 2), (1, 1))``
        indicates taking the second derivative with respect to the first
        axis and the first derivative with respect to the second
        axis. Each axis must occur only once and the tuple must be sorted
        by the axis index.

        May also be a singile integer *i*, which is viewed as equivalent
        to ``((i, 1),)``.
    """

    if isinstance(ref_axes, int):
        ref_axes = ((ref_axes, 1),)

    if not isinstance(ref_axes, tuple):
        raise ValueError("ref_axes must be a tuple")

    if tuple(sorted(ref_axes)) != ref_axes:
        raise ValueError("ref_axes must be sorted")

    if len(dict(ref_axes)) != len(ref_axes):
        raise ValueError("ref_axes must not contain an axis more than once")

    if dd is None:
        dd = DD_VOLUME
    inner_dd = dd.with_qtag(QTAG_NONE)

    from pytools import flatten
    flat_ref_axes = flatten([rst_axis] * n for rst_axis, n in ref_axes)

    from grudge.symbolic.operators import RefDiffOperator
    result = NodeCoordinateComponent(xyz_axis, inner_dd)
    for rst_axis in flat_ref_axes:
        result = RefDiffOperator(rst_axis, inner_dd)(result)

    if dd.uses_quadrature():
        from grudge.symbolic.operators import project
        result = project(inner_dd, dd)(result)

    prefix = "dx%d_%s" % (
            xyz_axis,
            "_".join("%sr%d" % ("d" * n, rst_axis) for rst_axis, n in ref_axes))

    return cse(result, prefix, cse_scope.DISCRETIZATION)


def forward_metric_derivative(xyz_axis, rst_axis, dd=None):
    r"""
    Pointwise metric derivatives representing

    .. math::

        \frac{\partial x_{\mathrm{xyz\_axis}} }{\partial r_{\mathrm{rst\_axis}}}
    """

    return forward_metric_nth_derivative(xyz_axis, rst_axis, dd=dd)


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

    if dim == 0:
        from pymbolic.geometric_algebra import get_euclidean_space
        return MultiVector(_SignedFaceOnes(dd),
                space=get_euclidean_space(ambient_dim))

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


def first_fundamental_form(ambient_dim, dim=None, dd=None):
    if dim is None:
        dim = ambient_dim - 1

    mder = forward_metric_derivative_mat(ambient_dim, dim=dim, dd=dd)
    return cse(mder.T.dot(mder), "form1_mat", cse_scope.DISCRETIZATION)


def inverse_first_fundamental_form(ambient_dim, dim=None, dd=None):
    if dim is None:
        dim = ambient_dim - 1

    if ambient_dim == dim:
        inv_mder = inverse_metric_derivative_mat(ambient_dim, dim=dim, dd=dd)
        inv_form1 = inv_mder.dot(inv_mder.T)
    else:
        form1 = first_fundamental_form(ambient_dim, dim, dd)

        if dim == 1:
            inv_form1 = np.array([[1.0/form1[0, 0]]], dtype=np.object)
        elif dim == 2:
            (E, F), (_, G) = form1      # noqa: N806
            inv_form1 = 1.0 / (E * G - F * F) * np.array([
                [G, -F], [-F, E]
                ], dtype=np.object)
        else:
            raise ValueError("%dD surfaces not supported" % dim)

    return cse(inv_form1, "inv_form1_mat", cse_scope.DISCRETIZATION)


def inverse_surface_metric_derivative(rst_axis, xyz_axis,
        ambient_dim, dim=None, dd=None):
    if dim is None:
        dim = ambient_dim - 1

    if ambient_dim == dim:
        return inverse_metric_derivative(rst_axis, xyz_axis,
                ambient_dim, dim=dim, dd=dd)
    else:
        inv_form1 = inverse_first_fundamental_form(ambient_dim, dim=dim, dd=dd)
        imd = sum(
                inv_form1[rst_axis, d]*forward_metric_derivative(xyz_axis, d, dd=dd)
                for d in range(dim))

        return cse(imd,
                prefix="ds%d_dx%d" % (rst_axis, xyz_axis),
                scope=cse_scope.DISCRETIZATION)


def second_fundamental_form(ambient_dim, dim=None, dd=None):
    if dim is None:
        dim = ambient_dim - 1

    normal = surface_normal(ambient_dim, dim=dim, dd=dd).as_vector()
    if dim == 1:
        second_ref_axes = [((0, 2),)]
    elif dim == 2:
        second_ref_axes = [((0, 2),), ((0, 1), (1, 1)), ((1, 2),)]
    else:
        raise ValueError("%dD surfaces not supported" % dim)

    from pytools import flatten
    form2 = np.empty((dim, dim), dtype=np.object)
    for ref_axes in second_ref_axes:
        i, j = flatten([rst_axis] * n for rst_axis, n in ref_axes)

        ruv = np.array([
            forward_metric_nth_derivative(xyz_axis, ref_axes, dd=dd)
            for xyz_axis in range(ambient_dim)])
        form2[i, j] = form2[j, i] = normal.dot(ruv)

    return cse(form2, "form2_mat", cse_scope.DISCRETIZATION)


def shape_operator(ambient_dim, dim=None, dd=None):
    if dim is None:
        dim = ambient_dim - 1

    inv_form1 = inverse_first_fundamental_form(ambient_dim, dim=dim, dd=dd)
    form2 = second_fundamental_form(ambient_dim, dim=dim, dd=dd)

    return cse(-form2.dot(inv_form1), "shape_operator", cse_scope.DISCRETIZATION)


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


def surface_normal(ambient_dim, dim=None, dd=None):
    dd = as_dofdesc(dd)
    if dim is None:
        dim = ambient_dim - 1

    # NOTE: Don't be tempted to add a sign here. As it is, it produces
    # exterior normals for positively oriented curves.

    pder = pseudoscalar(ambient_dim, dim, dd=dd) \
            / area_element(ambient_dim, dim, dd=dd)

    # Dorst Section 3.7.2
    return cse(pder << pder.I.inv(),
            "surface_normal",
            cse_scope.DISCRETIZATION)


def mv_normal(dd, ambient_dim, dim=None):
    """Exterior unit normal as a :class:`~pymbolic.geometric_algebra.MultiVector`."""
    dd = as_dofdesc(dd)
    if not dd.is_trace():
        raise ValueError("may only request normals on boundaries")

    if dim is None:
        dim = ambient_dim - 1

    if dim == ambient_dim - 1:
        return surface_normal(ambient_dim, dim=dim, dd=dd)

    # NOTE: In the case of (d - 2)-dimensional curves, we don't really have
    # enough information on the face to decide what an "exterior face normal"
    # is (e.g the "normal" to a 1D curve in 3D space is actually a
    # "normal plane")
    #
    # The trick done here is that we take the surface normal, move it to the
    # face and then take a cross product with the face tangent to get the
    # correct exterior face normal vector.
    assert dim == ambient_dim - 2

    from grudge.symbolic.operators import project
    volm_normal = (
            surface_normal(ambient_dim, dim=dim + 1, dd=DD_VOLUME)
            .map(project(DD_VOLUME, dd)))
    pder = pseudoscalar(ambient_dim, dim, dd=dd)

    mv = cse(-(volm_normal ^ pder) << volm_normal.I.inv(),
            "face_normal",
            cse_scope.DISCRETIZATION)

    return cse(mv / sqrt(mv.norm_squared()),
            "unit_face_normal",
            cse_scope.DISCRETIZATION)


def normal(dd, ambient_dim, dim=None):
    return mv_normal(dd, ambient_dim, dim).as_vector()


def summed_curvature(ambient_dim, dim=None, dd=None):
    """Sum of the principal curvatures"""

    if dim is None:
        dim = ambient_dim - 1

    if ambient_dim == 1:
        return 0.0

    if ambient_dim == dim:
        return 0.0

    op = shape_operator(ambient_dim, dim=dim, dd=dd)
    return cse(np.trace(op), "summed_curvature", cse_scope.DISCRETIZATION)


def mean_curvature(ambient_dim, dim=None, dd=None):
    """Averaged (by dimension) sum of the principal curvatures."""
    return 1.0 / (ambient_dim-1.0) * summed_curvature(ambient_dim, dim=dim, dd=dd)

# }}}


# {{{ trace pair

class TracePair:
    """
    .. attribute:: dd

        an instance of :class:`grudge.sym.DOFDesc` describing the
        discretization on which :attr:`interior` and :attr:`exterior`
        live.

    .. attribute:: interior

        a value (symbolic expression or :class:`~meshmode.dof_array.DOFArray`
        or object array of either) representing the interior value to
        be used for the flux.

    .. attribute:: exterior

        a value (symbolic expression or :class:`~meshmode.dof_array.DOFArray`
        or object array of either) representing the exterior value to
        be used for the flux.

    .. note::

        :class:`TracePair` is used both by the symbolic and the eager interface,
        with symbolic information or concrete data.
    """
    def __init__(self, dd, *, interior, exterior):
        """
        """

        self.dd = as_dofdesc(dd)
        self.interior = interior
        self.exterior = exterior

    def __getitem__(self, index):
        return TracePair(
                self.dd,
                interior=self.interior[index],
                exterior=self.exterior[index])

    def __len__(self):
        assert len(self.exterior) == len(self.interior)
        return len(self.exterior)

    @property
    def int(self):
        return self.interior

    @property
    def ext(self):
        return self.exterior

    @property
    def avg(self):
        return 0.5*(self.int + self.ext)


def int_tpair(expression, qtag=None, from_dd=None):
    from grudge.symbolic.operators import project, OppositeInteriorFaceSwap

    if from_dd is None:
        from_dd = DD_VOLUME
    assert not from_dd.uses_quadrature()

    trace_dd = DOFDesc(FACE_RESTR_INTERIOR, qtag)
    if from_dd.domain_tag == trace_dd.domain_tag:
        i = expression
    else:
        i = project(from_dd, trace_dd.with_qtag(None))(expression)
    e = cse(OppositeInteriorFaceSwap()(i))

    if trace_dd.uses_quadrature():
        i = cse(project(trace_dd.with_qtag(None), trace_dd)(i))
        e = cse(project(trace_dd.with_qtag(None), trace_dd)(e))

    return TracePair(trace_dd, interior=i, exterior=e)


def bdry_tpair(dd, interior, exterior):
    """
    :arg interior: an expression that already lives on the boundary
        representing the interior value to be used
        for the flux.
    :arg exterior: an expression that already lives on the boundary
        representing the exterior value to be used
        for the flux.
    """
    return TracePair(dd, interior=interior, exterior=exterior)


def bv_tpair(dd, interior, exterior):
    """
    :arg interior: an expression that lives in the volume
        and will be restricted to the boundary identified by
        *tag* before use.
    :arg exterior: an expression that already lives on the boundary
        representing the exterior value to be used
        for the flux.
    """
    from grudge.symbolic.operators import project
    interior = cse(project("vol", dd)(interior))
    return TracePair(dd, interior=interior, exterior=exterior)

# }}}


# vim: foldmethod=marker
