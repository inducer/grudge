"""Mappers to transform symbolic operators."""

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

import numpy as np
import pymbolic.primitives
import pymbolic.mapper.stringifier
import pymbolic.mapper.evaluator
import pymbolic.mapper.dependency
import pymbolic.mapper.substitutor
import pymbolic.mapper.constant_folder
import pymbolic.mapper.constant_converter
import pymbolic.mapper.flop_counter
from pymbolic.mapper import CSECachingMapperMixin

from grudge import sym
import grudge.symbolic.operators as op
from grudge.tools import OrderedSet


# {{{ mixins

class LocalOpReducerMixin:
    """Reduces calls to mapper methods for all local differentiation
    operators to a single mapper method, and likewise for mass
    operators.
    """
    # {{{ global differentiation
    def map_diff(self, expr, *args, **kwargs):
        return self.map_diff_base(expr, *args, **kwargs)

    def map_minv_st(self, expr, *args, **kwargs):
        return self.map_diff_base(expr, *args, **kwargs)

    def map_stiffness(self, expr, *args, **kwargs):
        return self.map_diff_base(expr, *args, **kwargs)

    def map_stiffness_t(self, expr, *args, **kwargs):
        return self.map_diff_base(expr, *args, **kwargs)

    def map_quad_stiffness_t(self, expr, *args, **kwargs):
        return self.map_diff_base(expr, *args, **kwargs)
    # }}}

    # {{{ global mass
    def map_mass_base(self, expr, *args, **kwargs):
        return self.map_elementwise_linear(expr, *args, **kwargs)

    def map_mass(self, expr, *args, **kwargs):
        return self.map_mass_base(expr, *args, **kwargs)

    def map_inverse_mass(self, expr, *args, **kwargs):
        return self.map_mass_base(expr, *args, **kwargs)

    def map_quad_mass(self, expr, *args, **kwargs):
        return self.map_mass_base(expr, *args, **kwargs)
    # }}}

    # {{{ reference differentiation
    def map_ref_diff(self, expr, *args, **kwargs):
        return self.map_ref_diff_base(expr, *args, **kwargs)

    def map_ref_stiffness_t(self, expr, *args, **kwargs):
        return self.map_ref_diff_base(expr, *args, **kwargs)

    def map_ref_quad_stiffness_t(self, expr, *args, **kwargs):
        return self.map_ref_diff_base(expr, *args, **kwargs)
    # }}}

    # {{{ reference mass
    def map_ref_mass_base(self, expr, *args, **kwargs):
        return self.map_elementwise_linear(expr, *args, **kwargs)

    def map_ref_mass(self, expr, *args, **kwargs):
        return self.map_ref_mass_base(expr, *args, **kwargs)

    def map_ref_inverse_mass(self, expr, *args, **kwargs):
        return self.map_ref_mass_base(expr, *args, **kwargs)

    def map_ref_quad_mass(self, expr, *args, **kwargs):
        return self.map_ref_mass_base(expr, *args, **kwargs)
    # }}}


class FluxOpReducerMixin:
    """Reduces calls to mapper methods for all flux
    operators to a smaller number of mapper methods.
    """
    def map_flux(self, expr, *args, **kwargs):
        return self.map_flux_base(expr, *args, **kwargs)

    def map_bdry_flux(self, expr, *args, **kwargs):
        return self.map_flux_base(expr, *args, **kwargs)

    def map_quad_flux(self, expr, *args, **kwargs):
        return self.map_flux_base(expr, *args, **kwargs)

    def map_quad_bdry_flux(self, expr, *args, **kwargs):
        return self.map_flux_base(expr, *args, **kwargs)


class OperatorReducerMixin(LocalOpReducerMixin, FluxOpReducerMixin):
    """Reduces calls to *any* operator mapping function to just one."""
    def _map_op_base(self, expr, *args, **kwargs):
        return self.map_operator(expr, *args, **kwargs)

    map_elementwise_linear = _map_op_base

    map_projection = _map_op_base

    map_elementwise_sum = _map_op_base
    map_elementwise_min = _map_op_base
    map_elementwise_max = _map_op_base

    map_nodal_sum = _map_op_base
    map_nodal_min = _map_op_base
    map_nodal_max = _map_op_base

    map_stiffness = _map_op_base
    map_diff = _map_op_base
    map_stiffness_t = _map_op_base

    map_ref_diff = _map_op_base
    map_ref_stiffness_t = _map_op_base

    map_mass = _map_op_base
    map_inverse_mass = _map_op_base
    map_ref_mass = _map_op_base
    map_ref_inverse_mass = _map_op_base

    map_opposite_partition_face_swap = _map_op_base
    map_opposite_interior_face_swap = _map_op_base
    map_face_mass_operator = _map_op_base
    map_ref_face_mass_operator = _map_op_base


class CombineMapperMixin:
    def map_operator_binding(self, expr):
        return self.combine([self.rec(expr.op), self.rec(expr.field)])


class IdentityMapperMixin(LocalOpReducerMixin, FluxOpReducerMixin):
    def map_operator_binding(self, expr, *args, **kwargs):
        assert not isinstance(self, BoundOpMapperMixin), \
                "IdentityMapper instances cannot be combined with " \
                "the BoundOpMapperMixin"

        return type(expr)(
                self.rec(expr.op, *args, **kwargs),
                self.rec(expr.field, *args, **kwargs))

    # {{{ operators

    def map_elementwise_linear(self, expr, *args, **kwargs):
        assert not isinstance(self, BoundOpMapperMixin), \
                "IdentityMapper instances cannot be combined with " \
                "the BoundOpMapperMixin"

        # it's a leaf--no changing children
        return expr

    map_projection = map_elementwise_linear

    map_elementwise_sum = map_elementwise_linear
    map_elementwise_min = map_elementwise_linear
    map_elementwise_max = map_elementwise_linear

    map_nodal_sum = map_elementwise_linear
    map_nodal_min = map_elementwise_linear
    map_nodal_max = map_elementwise_linear

    map_stiffness = map_elementwise_linear
    map_diff = map_elementwise_linear
    map_stiffness_t = map_elementwise_linear

    map_ref_diff = map_elementwise_linear
    map_ref_stiffness_t = map_elementwise_linear

    map_mass = map_elementwise_linear
    map_inverse_mass = map_elementwise_linear
    map_ref_mass = map_elementwise_linear
    map_ref_inverse_mass = map_elementwise_linear

    map_opposite_partition_face_swap = map_elementwise_linear
    map_opposite_interior_face_swap = map_elementwise_linear
    map_face_mass_operator = map_elementwise_linear
    map_ref_face_mass_operator = map_elementwise_linear

    # }}}

    # {{{ primitives

    def map_grudge_variable(self, expr, *args, **kwargs):
        # it's a leaf--no changing children
        return expr

    map_function_symbol = map_grudge_variable
    map_ones = map_grudge_variable
    map_signed_face_ones = map_grudge_variable
    map_node_coordinate_component = map_grudge_variable

    # }}}


class BoundOpMapperMixin:
    def map_operator_binding(self, expr, *args, **kwargs):
        return getattr(self, expr.op.mapper_method)(
                expr.op, expr.field, *args, **kwargs)

# }}}


# {{{ basic mappers

class CombineMapper(CombineMapperMixin, pymbolic.mapper.CombineMapper):
    pass


class DependencyMapper(
        CombineMapperMixin,
        pymbolic.mapper.dependency.DependencyMapper,
        OperatorReducerMixin):
    def __init__(self,
            include_operator_bindings=True,
            composite_leaves=None,
            **kwargs):
        if composite_leaves is False:
            include_operator_bindings = False
        if composite_leaves is True:
            include_operator_bindings = True

        pymbolic.mapper.dependency.DependencyMapper.__init__(self,
                composite_leaves=composite_leaves, **kwargs)

        self.include_operator_bindings = include_operator_bindings

    def map_operator_binding(self, expr):
        if self.include_operator_bindings:
            return {expr}
        else:
            return CombineMapperMixin.map_operator_binding(self, expr)

    def map_operator(self, expr):
        return set()

    def map_grudge_variable(self, expr):
        return {expr}

    def _map_leaf(self, expr):
        return set()

    map_ones = _map_leaf
    map_signed_face_ones = _map_leaf
    map_node_coordinate_component = _map_leaf


class FlopCounter(
        CombineMapperMixin,
        pymbolic.mapper.flop_counter.FlopCounter):
    def map_operator_binding(self, expr):
        return self.rec(expr.field)

    def map_grudge_variable(self, expr):
        return 0

    def map_function_symbol(self, expr):
        return 1

    def map_ones(self, expr):
        return 0

    def map_node_coordinate_component(self, expr):
        return 0


class IdentityMapper(
        IdentityMapperMixin,
        pymbolic.mapper.IdentityMapper):
    pass


class SubstitutionMapper(pymbolic.mapper.substitutor.SubstitutionMapper,
        IdentityMapperMixin):
    pass


class CSERemover(IdentityMapper):
    def map_common_subexpression(self, expr):
        return self.rec(expr.child)

# }}}


# {{{ operator binder

class OperatorBinder(CSECachingMapperMixin, IdentityMapper):
    map_common_subexpression_uncached = \
            IdentityMapper.map_common_subexpression

    def map_product(self, expr):
        if len(expr.children) == 0:
            return expr

        from pymbolic.primitives import flattened_product, Product

        first = expr.children[0]
        if isinstance(first, op.Operator):
            prod = flattened_product(expr.children[1:])
            if isinstance(prod, Product) and len(prod.children) > 1:
                from warnings import warn
                warn("Binding '%s' to more than one "
                        "operand in a product is ambiguous - "
                        "use the parenthesized form instead."
                        % first)
            return sym.OperatorBinding(first, self.rec(prod))
        else:
            return self.rec(first) * self.rec(flattened_product(expr.children[1:]))

# }}}


# {{{ dof desc (dd) replacement

class DOFDescReplacer(IdentityMapper):
    def __init__(self, prev_dd, new_dd):
        self.prev_dd = prev_dd
        self.new_dd = new_dd

    def map_operator_binding(self, expr):
        if (isinstance(expr.op, op.OppositeInteriorFaceSwap)
                    and expr.op.dd_in == self.prev_dd
                    and expr.op.dd_out == self.prev_dd):
            field = self.rec(expr.field)
            return op.OppositePartitionFaceSwap(dd_in=self.new_dd,
                                                dd_out=self.new_dd)(field)
        elif (isinstance(expr.op, op.ProjectionOperator)
                    and expr.op.dd_out == self.prev_dd):
            return op.ProjectionOperator(dd_in=expr.op.dd_in,
                                            dd_out=self.new_dd)(expr.field)
        elif (isinstance(expr.op, op.RefDiffOperatorBase)
                    and expr.op.dd_out == self.prev_dd
                    and expr.op.dd_in == self.prev_dd):
            return type(expr.op)(expr.op.rst_axis,
                                      dd_in=self.new_dd,
                                      dd_out=self.new_dd)(self.rec(expr.field))

    def map_node_coordinate_component(self, expr):
        if expr.dd == self.prev_dd:
            return type(expr)(expr.axis, self.new_dd)

# }}}


# {{{ mappers for distributed computation

class OppositeInteriorFaceSwapUniqueIDAssigner(
        CSECachingMapperMixin, IdentityMapper):
    map_common_subexpression_uncached = IdentityMapper.map_common_subexpression

    def __init__(self):
        super().__init__()
        self._next_id = 0
        self.seen_ids = set()

    def next_id(self):
        while self._next_id in self.seen_ids:
            self._next_id += 1

        result = self._next_id
        self._next_id += 1
        self.seen_ids.add(result)

        return result

    def map_opposite_interior_face_swap(self, expr):
        if expr.unique_id is not None:
            if expr.unique_id in self.seen_ids:
                raise ValueError("OppositeInteriorFaceSwap unique ID '%d' "
                        "is not unique" % expr.unique_id)

            self.seen_ids.add(expr.unique_id)
            return expr

        else:
            return type(expr)(expr.dd_in, expr.dd_out, self.next_id())


class DistributedMapper(CSECachingMapperMixin, IdentityMapper):
    map_common_subexpression_uncached = IdentityMapper.map_common_subexpression

    def __init__(self, connected_parts):
        self.connected_parts = connected_parts

    def map_operator_binding(self, expr):
        from meshmode.mesh import BTAG_PARTITION
        from meshmode.discretization.connection import (FACE_RESTR_ALL,
                                                        FACE_RESTR_INTERIOR)
        if (isinstance(expr.op, op.ProjectionOperator)
                and expr.op.dd_in.domain_tag is FACE_RESTR_INTERIOR
                and expr.op.dd_out.domain_tag is FACE_RESTR_ALL):
            distributed_work = 0
            for i_remote_part in self.connected_parts:
                mapped_field = RankGeometryChanger(i_remote_part)(expr.field)
                btag_part = BTAG_PARTITION(i_remote_part)
                distributed_work += op.ProjectionOperator(dd_in=btag_part,
                                             dd_out=expr.op.dd_out)(mapped_field)
            return expr + distributed_work
        else:
            return IdentityMapper.map_operator_binding(self, expr)


class RankGeometryChanger(CSECachingMapperMixin, IdentityMapper):
    map_common_subexpression_uncached = IdentityMapper.map_common_subexpression

    def __init__(self, i_remote_part):
        from meshmode.discretization.connection import FACE_RESTR_INTERIOR
        from meshmode.mesh import BTAG_PARTITION
        self.prev_dd = sym.as_dofdesc(FACE_RESTR_INTERIOR)
        self.new_dd = sym.as_dofdesc(BTAG_PARTITION(i_remote_part))

    def _raise_unable(self, expr):
        raise ValueError("encountered '%s' in updating subexpression for "
            "changed geometry (likely for distributed computation); "
            "unable to adapt from '%s' to '%s'"
            % (str(expr), self.prev_dd, self.new_dd))

    def map_operator_binding(self, expr):
        if (isinstance(expr.op, op.OppositeInteriorFaceSwap)
                    and expr.op.dd_in == self.prev_dd
                    and expr.op.dd_out == self.prev_dd):
            field = self.rec(expr.field)
            return op.OppositePartitionFaceSwap(
                    dd_in=self.new_dd,
                    dd_out=self.new_dd,
                    unique_id=expr.op.unique_id)(field)
        elif (isinstance(expr.op, op.ProjectionOperator)
                    and expr.op.dd_out == self.prev_dd):
            return op.ProjectionOperator(dd_in=expr.op.dd_in,
                                            dd_out=self.new_dd)(expr.field)
        elif (isinstance(expr.op, op.RefDiffOperator)
                    and expr.op.dd_out == self.prev_dd
                    and expr.op.dd_in == self.prev_dd):
            return op.RefDiffOperator(expr.op.rst_axis,
                                      dd_in=self.new_dd,
                                      dd_out=self.new_dd)(self.rec(expr.field))
        else:
            self._raise_unable(expr)

    def map_grudge_variable(self, expr):
        self._raise_unable(expr)

    def map_node_coordinate_component(self, expr):
        if expr.dd == self.prev_dd:
            return type(expr)(expr.axis, self.new_dd)
        else:
            self._raise_unable(expr)

# }}}


# {{{ operator specializer

class OperatorSpecializer(CSECachingMapperMixin, IdentityMapper):
    """Guided by a typedict obtained through type inference (i.e. by
    :class:`grudge.symbolic.mappers.type_inference.TypeInferrrer`),
    substitutes more specialized operators for generic ones.

    For example, if type inference has determined that a differentiation
    operator is applied to an argument on a quadrature grid, this
    differentiation operator is then swapped out for a *quadrature*
    differentiation operator.
    """

    def __init__(self, typedict):
        """
        :param typedict: generated by
        :class:`grudge.symbolic.mappers.type_inference.TypeInferrer`.
        """
        self.typedict = typedict

    map_common_subexpression_uncached = \
            IdentityMapper.map_common_subexpression

    def map_operator_binding(self, expr):
        from grudge.symbolic.primitives import BoundaryPair

        from grudge.symbolic.mappers.type_inference import (
                type_info, QuadratureRepresentation)

        # {{{ figure out field type
        try:
            field_type = self.typedict[expr.field]
        except TypeError:
            # numpy arrays are not hashable
            # has_quad_operand remains unset

            assert isinstance(expr.field, np.ndarray)
        else:
            try:
                field_repr_tag = field_type.repr_tag
            except AttributeError:
                # boundary pairs are not assigned types
                assert isinstance(expr.field, BoundaryPair)
                has_quad_operand = False
            else:
                has_quad_operand = isinstance(field_repr_tag,
                            QuadratureRepresentation)
        # }}}

        # Based on where this is run in the symbolic operator processing
        # pipeline, we may encounter both reference and non-reference
        # operators.

        # {{{ elementwise operators

        if isinstance(expr.op, op.MassOperator) and has_quad_operand:
            return op.QuadratureMassOperator(
                    field_repr_tag.quadrature_tag)(self.rec(expr.field))

        elif isinstance(expr.op, op.RefMassOperator) and has_quad_operand:
            return op.RefQuadratureMassOperator(
                    field_repr_tag.quadrature_tag)(self.rec(expr.field))

        elif (isinstance(expr.op, op.StiffnessTOperator) and has_quad_operand):
            return op.QuadratureStiffnessTOperator(
                    expr.op.xyz_axis, field_repr_tag.quadrature_tag)(
                            self.rec(expr.field))

        elif (isinstance(expr.op, op.RefStiffnessTOperator)
                and has_quad_operand):
            return op.RefQuadratureStiffnessTOperator(
                    expr.op.xyz_axis, field_repr_tag.quadrature_tag)(
                            self.rec(expr.field))

        elif (isinstance(expr.op, op.QuadratureGridUpsampler)
                and isinstance(field_type, type_info.BoundaryVectorBase)):
            # potential shortcut:
            # if (isinstance(expr.field, OperatorBinding)
            #        and isinstance(expr.field.op, RestrictToBoundary)):
            #    return QuadratureRestrictToBoundary(
            #            expr.field.op.tag, expr.op.quadrature_tag)(
            #                    self.rec(expr.field.field))

            return op.QuadratureBoundaryGridUpsampler(
                    expr.op.quadrature_tag, field_type.boundary_tag)(expr.field)
        # }}}

        elif isinstance(expr.op, op.RestrictToBoundary) and has_quad_operand:
            raise TypeError("RestrictToBoundary cannot be applied to "
                    "quadrature-based operands--use QuadUpsample(Boundarize(...))")

        else:
            return IdentityMapper.map_operator_binding(self, expr)

# }}}


# {{{ global-to-reference mapper

class GlobalToReferenceMapper(CSECachingMapperMixin, IdentityMapper):
    """Maps operators that apply on the global function space down to operators on
    reference elements, together with explicit multiplication by geometric factors.
    """

    def __init__(self, discrwb):
        CSECachingMapperMixin.__init__(self)
        IdentityMapper.__init__(self)

        self.ambient_dim = discrwb.ambient_dim
        self.dim = discrwb.dim

        volume_discr = discrwb.discr_from_dd(sym.DD_VOLUME)
        self.use_wadg = not all(grp.is_affine for grp in volume_discr.groups)

    map_common_subexpression_uncached = \
            IdentityMapper.map_common_subexpression

    def map_operator_binding(self, expr):
        # Global-to-reference is run after operator specialization, so
        # if we encounter non-quadrature operators here, we know they
        # must be nodal.

        dd_in = expr.op.dd_in
        dd_out = expr.op.dd_out

        if dd_in.is_volume():
            dim = self.dim
        else:
            dim = self.dim - 1

        jac_in = sym.area_element(self.ambient_dim, dim, dd=dd_in)
        jac_noquad = sym.area_element(self.ambient_dim, dim,
                dd=dd_in.with_qtag(sym.QTAG_NONE))

        def rewrite_derivative(ref_class, field,  dd_in, with_jacobian=True):
            def imd(rst):
                return sym.inverse_surface_metric_derivative(
                        rst, expr.op.xyz_axis,
                        ambient_dim=self.ambient_dim, dim=self.dim,
                        dd=dd_in)

            rec_field = self.rec(field)
            if with_jacobian:
                jac_tag = sym.area_element(self.ambient_dim, self.dim, dd=dd_in)
                rec_field = jac_tag * rec_field

                return sum(
                        ref_class(rst_axis, dd_in=dd_in)(rec_field * imd(rst_axis))
                        for rst_axis in range(self.dim))
            else:
                return sum(
                        ref_class(rst_axis, dd_in=dd_in)(rec_field) * imd(rst_axis)
                        for rst_axis in range(self.dim))

        if isinstance(expr.op, op.MassOperator):
            return op.RefMassOperator(dd_in, dd_out)(
                    jac_in * self.rec(expr.field))

        elif isinstance(expr.op, op.InverseMassOperator):
            if self.use_wadg:
                # based on https://arxiv.org/pdf/1608.03836.pdf
                return op.RefInverseMassOperator(dd_in, dd_out)(
                    op.RefMassOperator(dd_in, dd_out)(
                        1.0/jac_in * op.RefInverseMassOperator(dd_in, dd_out)(
                            self.rec(expr.field))
                            )
                    )
            else:
                return op.RefInverseMassOperator(dd_in, dd_out)(
                        1/jac_in * self.rec(expr.field))

        elif isinstance(expr.op, op.FaceMassOperator):
            jac_in_surf = sym.area_element(
                    self.ambient_dim, self.dim - 1, dd=dd_in)
            return op.RefFaceMassOperator(dd_in, dd_out)(
                    jac_in_surf * self.rec(expr.field))

        elif isinstance(expr.op, op.StiffnessOperator):
            return op.RefMassOperator(dd_in=dd_in, dd_out=dd_out)(
                    jac_noquad
                    * self.rec(
                        op.DiffOperator(expr.op.xyz_axis)(expr.field)))

        elif isinstance(expr.op, op.DiffOperator):
            return rewrite_derivative(
                    op.RefDiffOperator,
                    expr.field, dd_in=dd_in, with_jacobian=False)

        elif isinstance(expr.op, op.StiffnessTOperator):
            return rewrite_derivative(
                    op.RefStiffnessTOperator,
                    expr.field, dd_in=dd_in)

        elif isinstance(expr.op, op.MInvSTOperator):
            return self.rec(
                    op.InverseMassOperator()(
                        op.StiffnessTOperator(expr.op.xyz_axis)(
                            self.rec(expr.field))))

        else:
            return IdentityMapper.map_operator_binding(self, expr)

# }}}


# {{{ stringification ---------------------------------------------------------

class StringifyMapper(pymbolic.mapper.stringifier.StringifyMapper):
    def _format_dd(self, dd):
        def fmt(s):
            if isinstance(s, type):
                return s.__name__
            else:
                return repr(s)

        from meshmode.mesh import BTAG_PARTITION
        from meshmode.discretization.connection import (
                FACE_RESTR_ALL, FACE_RESTR_INTERIOR)
        if dd.domain_tag is None:
            result = "?"
        elif dd.domain_tag is sym.DTAG_VOLUME_ALL:
            result = "vol"
        elif dd.domain_tag is sym.DTAG_SCALAR:
            result = "scalar"
        elif dd.domain_tag is FACE_RESTR_ALL:
            result = "all_faces"
        elif dd.domain_tag is FACE_RESTR_INTERIOR:
            result = "int_faces"
        elif isinstance(dd.domain_tag, BTAG_PARTITION):
            result = "part%d_faces" % dd.domain_tag.part_nr
        else:
            result = fmt(dd.domain_tag)

        if dd.quadrature_tag is None:
            pass
        elif dd.quadrature_tag is sym.QTAG_NONE:
            result += "q"
        else:
            result += "Q"+fmt(dd.quadrature_tag)

        return result

    def _format_op_dd(self, op):
        return ":{}->{}".format(
                self._format_dd(op.dd_in),
                self._format_dd(op.dd_out))

    # {{{ elementwise ops

    def map_elementwise_sum(self, expr, enclosing_prec):
        return "ElementwiseSum" + self._format_op_dd(expr)

    def map_elementwise_max(self, expr, enclosing_prec):
        return "ElementwiseMax" + self._format_op_dd(expr)

    def map_elementwise_min(self, expr, enclosing_prec):
        return "ElementwiseMin" + self._format_op_dd(expr)

    # }}}

    # {{{ nodal ops

    def map_nodal_sum(self, expr, enclosing_prec):
        return "NodalSum" + self._format_op_dd(expr)

    def map_nodal_max(self, expr, enclosing_prec):
        return "NodalMax" + self._format_op_dd(expr)

    def map_nodal_min(self, expr, enclosing_prec):
        return "NodalMin" + self._format_op_dd(expr)

    # }}}

    # {{{ global differentiation

    def map_diff(self, expr, enclosing_prec):
        return "Diffx%d%s" % (expr.xyz_axis, self._format_op_dd(expr))

    def map_minv_st(self, expr, enclosing_prec):
        return "MInvSTx%d%s" % (expr.xyz_axis, self._format_op_dd(expr))

    def map_stiffness(self, expr, enclosing_prec):
        return "Stiffx%d%s" % (expr.xyz_axis, self._format_op_dd(expr))

    def map_stiffness_t(self, expr, enclosing_prec):
        return "StiffTx%d%s" % (expr.xyz_axis, self._format_op_dd(expr))

    # }}}

    # {{{ global mass

    def map_mass(self, expr, enclosing_prec):
        return "M"

    def map_inverse_mass(self, expr, enclosing_prec):
        return "InvM"

    # }}}

    # {{{ reference differentiation

    def map_ref_diff(self, expr, enclosing_prec):
        return "Diffr%d%s" % (expr.rst_axis, self._format_op_dd(expr))

    def map_ref_stiffness_t(self, expr, enclosing_prec):
        return "StiffTr%d%s" % (expr.rst_axis, self._format_op_dd(expr))

    # }}}

    # {{{ reference mass

    def map_ref_mass(self, expr, enclosing_prec):
        return "RefM" + self._format_op_dd(expr)

    def map_ref_inverse_mass(self, expr, enclosing_prec):
        return "RefInvM" + self._format_op_dd(expr)

    # }}}

    def map_elementwise_linear(self, expr, enclosing_prec):
        return "ElWLin:{}{}".format(
                expr.__class__.__name__,
                self._format_op_dd(expr))

    # {{{ flux

    def map_face_mass_operator(self, expr, enclosing_prec):
        return "FaceM" + self._format_op_dd(expr)

    def map_ref_face_mass_operator(self, expr, enclosing_prec):
        return "RefFaceM" + self._format_op_dd(expr)

    def map_opposite_partition_face_swap(self, expr, enclosing_prec):
        return "PartSwap" + self._format_op_dd(expr)

    def map_opposite_interior_face_swap(self, expr, enclosing_prec):
        return "OppSwap" + self._format_op_dd(expr)

    # }}}

    def map_ones(self, expr, enclosing_prec):
        return "Ones:" + self._format_dd(expr.dd)

    def map_signed_face_ones(self, expr, enclosing_prec):
        return "SignedOnes:" + self._format_dd(expr.dd)

    # {{{ geometry data

    def map_node_coordinate_component(self, expr, enclosing_prec):
        return "x[%d]@%s" % (expr.axis, self._format_dd(expr.dd))

    # }}}

    def map_operator_binding(self, expr, enclosing_prec):
        from pymbolic.mapper.stringifier import PREC_NONE
        return "<{}>({})".format(
                self.rec(expr.op, PREC_NONE),
                self.rec(expr.field, PREC_NONE))

    def map_grudge_variable(self, expr, enclosing_prec):
        return "{}:{}".format(expr.name, self._format_dd(expr.dd))

    def map_function_symbol(self, expr, enclosing_prec):
        return expr.name

    def map_projection(self, expr, enclosing_prec):
        return "Project" + self._format_op_dd(expr)


class PrettyStringifyMapper(
        pymbolic.mapper.stringifier.CSESplittingStringifyMapperMixin,
        StringifyMapper):
    pass


class NoCSEStringifyMapper(StringifyMapper):
    def map_common_subexpression(self, expr, enclosing_prec):
        return self.rec(expr.child, enclosing_prec)

# }}}


# {{{ quadrature support

class QuadratureCheckerAndRemover(CSECachingMapperMixin, IdentityMapper):
    """Checks whether all quadratu
    """

    def __init__(self, quad_tag_to_group_factory):
        IdentityMapper.__init__(self)
        CSECachingMapperMixin.__init__(self)
        self.quad_tag_to_group_factory = quad_tag_to_group_factory

    map_common_subexpression_uncached = \
            IdentityMapper.map_common_subexpression

    def _process_dd(self, dd, location_descr):
        from grudge.symbolic.primitives import DOFDesc, QTAG_NONE
        if dd.quadrature_tag is not QTAG_NONE:
            if dd.quadrature_tag not in self.quad_tag_to_group_factory:
                raise ValueError("found unknown quadrature tag '%s' in '%s'"
                        % (dd.quadrature_tag, location_descr))

            grp_factory = self.quad_tag_to_group_factory[dd.quadrature_tag]
            if grp_factory is None:
                dd = DOFDesc(dd.domain_tag, QTAG_NONE)

        return dd

    def map_operator_binding(self, expr):
        dd_in_orig = dd_in = expr.op.dd_in
        dd_out_orig = dd_out = expr.op.dd_out

        dd_in = self._process_dd(dd_in, "dd_in of %s" % type(expr.op).__name__)
        dd_out = self._process_dd(dd_out, "dd_out of %s" % type(expr.op).__name__)

        if dd_in_orig == dd_in and dd_out_orig == dd_out:
            # unchanged
            return IdentityMapper.map_operator_binding(self, expr)

        import grudge.symbolic.operators as op
        # changed

        if dd_in == dd_out and isinstance(expr.op, op.ProjectionOperator):
            # This used to be to-quad interpolation and has become a no-op.
            # Remove it.
            return self.rec(expr.field)

        if isinstance(expr.op, op.StiffnessTOperator):
            new_op = type(expr.op)(expr.op.xyz_axis, dd_in, dd_out)
        elif isinstance(expr.op, (op.FaceMassOperator, op.ProjectionOperator)):
            new_op = type(expr.op)(dd_in, dd_out)
        else:
            raise NotImplementedError("do not know how to modify dd_in and dd_out "
                    "in %s" % type(expr.op).__name__)

        return new_op(self.rec(expr.field))

    def map_ones(self, expr):
        dd = self._process_dd(expr.dd, location_descr=type(expr).__name__)
        return type(expr)(dd)

    def map_node_coordinate_component(self, expr):
        dd = self._process_dd(expr.dd, location_descr=type(expr).__name__)
        return type(expr)(expr.axis, dd)

# }}}


# {{{ simplification / optimization

class ConstantToNumpyConversionMapper(
        CSECachingMapperMixin,
        pymbolic.mapper.constant_converter.ConstantToNumpyConversionMapper,
        IdentityMapperMixin):
    map_common_subexpression_uncached = (
            pymbolic.mapper.constant_converter
            .ConstantToNumpyConversionMapper
            .map_common_subexpression)


class CommutativeConstantFoldingMapper(
        pymbolic.mapper.constant_folder.CommutativeConstantFoldingMapper,
        IdentityMapperMixin):

    def __init__(self):
        pymbolic.mapper.constant_folder\
                .CommutativeConstantFoldingMapper.__init__(self)
        self.dep_mapper = DependencyMapper()

    def is_constant(self, expr):
        return not bool(self.dep_mapper(expr))

    def map_operator_binding(self, expr):
        field = self.rec(expr.field)

        from grudge.tools import is_zero
        if is_zero(field):
            return 0

        return expr.op(field)


class EmptyFluxKiller(CSECachingMapperMixin, IdentityMapper):
    def __init__(self, mesh):
        IdentityMapper.__init__(self)
        self.mesh = mesh

    map_common_subexpression_uncached = \
            IdentityMapper.map_common_subexpression

    def map_operator_binding(self, expr):
        from meshmode.mesh import is_boundary_tag_empty
        if (isinstance(expr.op, sym.ProjectionOperator)
                and expr.op.dd_out.is_boundary_or_partition_interface()):
            domain_tag = expr.op.dd_out.domain_tag
            assert isinstance(domain_tag, sym.DTAG_BOUNDARY)
            if is_boundary_tag_empty(self.mesh, domain_tag.tag):
                return 0

        return IdentityMapper.map_operator_binding(self, expr)


class _InnerDerivativeJoiner(pymbolic.mapper.RecursiveMapper):
    def map_operator_binding(self, expr, derivatives):
        if isinstance(expr.op, op.DifferentiationOperator):
            derivatives.setdefault(expr.op, []).append(expr.field)
            return 0
        else:
            return DerivativeJoiner()(expr)

    def map_common_subexpression(self, expr, derivatives):
        # no use preserving these if we're moving derivatives around
        return self.rec(expr.child, derivatives)

    def map_sum(self, expr, derivatives):
        from pymbolic.primitives import flattened_sum
        return flattened_sum(tuple(
            self.rec(child, derivatives) for child in expr.children))

    def map_product(self, expr, derivatives):
        from grudge.symbolic.tools import is_scalar
        from pytools import partition
        scalars, nonscalars = partition(is_scalar, expr.children)

        if len(nonscalars) != 1:
            return DerivativeJoiner()(expr)
        else:
            from pymbolic import flattened_product
            factor = flattened_product(scalars)
            nonscalar, = nonscalars

            sub_derivatives = {}
            nonscalar = self.rec(nonscalar, sub_derivatives)

            def do_map(expr):
                if is_scalar(expr):
                    return expr
                else:
                    return self.rec(expr, derivatives)

            for operator, operands in sub_derivatives.items():
                for operand in operands:
                    derivatives.setdefault(operator, []).append(
                            factor*operand)

            return factor*nonscalar

    def map_constant(self, expr, *args):
        return DerivativeJoiner()(expr)

    def map_scalar_parameter(self, expr, *args):
        return DerivativeJoiner()(expr)

    def map_if(self, expr, *args):
        return DerivativeJoiner()(expr)

    def map_power(self, expr, *args):
        return DerivativeJoiner()(expr)

    # these two are necessary because they're forwarding targets
    def map_algebraic_leaf(self, expr, *args):
        return DerivativeJoiner()(expr)

    def map_quotient(self, expr, *args):
        return DerivativeJoiner()(expr)

    map_node_coordinate_component = map_algebraic_leaf


class DerivativeJoiner(CSECachingMapperMixin, IdentityMapper):
    r"""Joins derivatives:

    .. math::

        \frac{\partial A}{\partial x} + \frac{\partial B}{\partial x}
        \rightarrow
        \frac{\partial (A+B)}{\partial x}.
    """
    map_common_subexpression_uncached = \
            IdentityMapper.map_common_subexpression

    def map_sum(self, expr):
        idj = _InnerDerivativeJoiner()

        def invoke_idj(expr):
            sub_derivatives = {}
            result = idj(expr, sub_derivatives)
            if not sub_derivatives:
                return expr
            else:
                for operator, operands in sub_derivatives.items():
                    derivatives.setdefault(operator, []).extend(operands)

                return result

        derivatives = {}
        new_children = [invoke_idj(child)
                for child in expr.children]

        for operator, operands in derivatives.items():
            new_children.insert(0, operator(
                sum(self.rec(operand) for operand in operands)))

        from pymbolic.primitives import flattened_sum
        return flattened_sum(new_children)


class _InnerInverseMassContractor(pymbolic.mapper.RecursiveMapper):
    def __init__(self, outer_mass_contractor):
        self.outer_mass_contractor = outer_mass_contractor
        self.extra_operator_count = 0

    def map_constant(self, expr):
        from grudge.tools import is_zero

        if is_zero(expr):
            return 0
        else:
            return op.OperatorBinding(
                    op.InverseMassOperator(),
                    self.outer_mass_contractor(expr))

    def map_algebraic_leaf(self, expr):
        return op.OperatorBinding(
                op.InverseMassOperator(),
                self.outer_mass_contractor(expr))

    def map_operator_binding(self, binding):
        if isinstance(binding.op, op.MassOperator):
            return binding.field
        elif isinstance(binding.op, op.StiffnessOperator):
            return op.DifferentiationOperator(binding.op.xyz_axis)(
                    self.outer_mass_contractor(binding.field))
        elif isinstance(binding.op, op.StiffnessTOperator):
            return op.MInvSTOperator(binding.op.xyz_axis)(
                    self.outer_mass_contractor(binding.field))
        elif isinstance(binding.op, op.FluxOperator):
            assert not binding.op.is_lift

            return op.FluxOperator(binding.op.flux, is_lift=True)(
                    self.outer_mass_contractor(binding.field))
        elif isinstance(binding.op, op.BoundaryFluxOperator):
            assert not binding.op.is_lift

            return op.BoundaryFluxOperator(binding.op.flux,
                    binding.op.boundary_tag, is_lift=True)(
                        self.outer_mass_contractor(binding.field))
        else:
            self.extra_operator_count += 1
            return op.InverseMassOperator()(
                self.outer_mass_contractor(binding))

    def map_sum(self, expr):
        return expr.__class__(tuple(self.rec(child) for child in expr.children))

    def map_product(self, expr):
        def is_scalar(expr):
            return isinstance(expr, (int, float, complex, sym.ScalarParameter))

        from pytools import len_iterable
        nonscalar_count = len_iterable(ch
                for ch in expr.children
                if not is_scalar(ch))

        if nonscalar_count > 1:
            # too complicated, don't touch it
            self.extra_operator_count += 1
            return op.InverseMassOperator()(
                    self.outer_mass_contractor(expr))
        else:
            def do_map(expr):
                if is_scalar(expr):
                    return expr
                else:
                    return self.rec(expr)
            return expr.__class__(tuple(
                do_map(child) for child in expr.children))


class InverseMassContractor(CSECachingMapperMixin, IdentityMapper):
    # assumes all operators to be bound
    map_common_subexpression_uncached = \
            IdentityMapper.map_common_subexpression

    def map_operator_binding(self, binding):
        # we only care about bindings of inverse mass operators

        if isinstance(binding.op, op.InverseMassOperator):
            iimc = _InnerInverseMassContractor(self)
            proposed_result = iimc(binding.field)
            if iimc.extra_operator_count > 1:
                # We're introducing more work than we're saving.
                # Don't perform the simplification
                return binding.op(self.rec(binding.field))
            else:
                return proposed_result
        else:
            return binding.op(self.rec(binding.field))

# }}}


# {{{ error checker

class ErrorChecker(CSECachingMapperMixin, IdentityMapper):
    map_common_subexpression_uncached = \
            IdentityMapper.map_common_subexpression

    def __init__(self, mesh):
        self.mesh = mesh

    def map_operator_binding(self, expr):
        if isinstance(expr.op, op.DiffOperatorBase):
            if (self.mesh is not None
                    and expr.op.xyz_axis >= self.mesh.ambient_dim):
                raise ValueError("optemplate tries to differentiate along a "
                        "non-existent axis (e.g. Z in 2D)")

        # FIXME: Also check fluxes
        return IdentityMapper.map_operator_binding(self, expr)

    def map_normal(self, expr):
        if self.mesh is not None and expr.axis >= self.mesh.dimensions:
            raise ValueError("optemplate tries to differentiate along a "
                    "non-existent axis (e.g. Z in 2D)")

        return expr

# }}}


# {{{ collectors for various symbolic operator components

# To maintain deterministic output in code generation, these collectors return
# OrderedSets. (As an example for why this is useful, the order of collected
# values determines the names of intermediate variables. If variable names
# aren't deterministic, loopy's kernel caching doesn't help us much across
# runs.)

class CollectorMixin(OperatorReducerMixin, LocalOpReducerMixin, FluxOpReducerMixin):
    def combine(self, values):
        from pytools import flatten
        return OrderedSet(flatten(values))

    def map_constant(self, expr, *args, **kwargs):
        return OrderedSet()

    map_variable = map_constant
    map_grudge_variable = map_constant
    map_function_symbol = map_constant

    map_ones = map_grudge_variable
    map_signed_face_ones = map_grudge_variable
    map_node_coordinate_component = map_grudge_variable

    map_operator = map_grudge_variable


# I'm not sure this works still.
#class GeometricFactorCollector(CollectorMixin, CombineMapper):
#    pass


class BoundOperatorCollector(CSECachingMapperMixin, CollectorMixin, CombineMapper):
    def __init__(self, op_class):
        self.op_class = op_class

    map_common_subexpression_uncached = \
            CombineMapper.map_common_subexpression

    def map_operator_binding(self, expr):
        if isinstance(expr.op, self.op_class):
            result = OrderedSet([expr])
        else:
            result = OrderedSet()

        return result | CombineMapper.map_operator_binding(self, expr)


class FluxExchangeCollector(CSECachingMapperMixin, CollectorMixin, CombineMapper):
    map_common_subexpression_uncached = \
            CombineMapper.map_common_subexpression

    def map_flux_exchange(self, expr):
        return OrderedSet([expr])

# }}}


# {{{ evaluation

class Evaluator(pymbolic.mapper.evaluator.EvaluationMapper):
    pass


class SymbolicEvaluator(pymbolic.mapper.evaluator.EvaluationMapper):
    def map_operator_binding(self, expr, *args, **kwargs):
        return expr.op(self.rec(expr.field, *args, **kwargs))

    def map_node_coordinate_component(self, expr, *args, **kwargs):
        return expr

    def map_call(self, expr, *args, **kwargs):
        return type(expr)(
                expr.function,
                tuple(self.rec(child, *args, **kwargs)
                    for child in expr.parameters))

    def map_call_with_kwargs(self, expr, *args, **kwargs):
        return type(expr)(
                expr.function,
                tuple(self.rec(child, *args, **kwargs)
                    for child in expr.parameters),
                {
                    key: self.rec(val, *args, **kwargs)
                    for key, val in expr.kw_parameters.items()}
                    )

    def map_common_subexpression(self, expr):
        return type(expr)(self.rec(expr.child), expr.prefix, expr.scope)

# }}}


# vim: foldmethod=marker
