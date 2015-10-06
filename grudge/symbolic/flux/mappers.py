"""Building blocks for flux computation. Flux compilation."""

__copyright__ = "Copyright (C) 2007 Andreas Kloeckner"

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


import pymbolic.mapper.collector
import pymbolic.mapper.flattener
import pymbolic.mapper.substitutor
import pymbolic.mapper.constant_folder
import pymbolic.mapper.flop_counter


class FluxIdentityMapperMixin(object):
    def map_field_component(self, expr):
        return expr

    def map_normal(self, expr):
        # a leaf
        return expr

    map_element_jacobian = map_normal
    map_face_jacobian = map_normal
    map_element_order = map_normal
    map_local_mesh_size = map_normal
    map_c_function = map_normal

    def map_scalar_parameter(self, expr):
        return expr


class FluxIdentityMapper(
        pymbolic.mapper.IdentityMapper,
        FluxIdentityMapperMixin):
    pass


class FluxSubstitutionMapper(pymbolic.mapper.substitutor.SubstitutionMapper,
        FluxIdentityMapperMixin):
    def map_field_component(self, expr):
        result = self.subst_func(expr)
        if result is not None:
            return result
        else:
            return expr

    map_normal = map_field_component


class FluxStringifyMapper(pymbolic.mapper.stringifier.StringifyMapper):
    def map_field_component(self, expr, enclosing_prec):
        if expr.is_interior:
            return "Int[%d]" % expr.index
        else:
            return "Ext[%d]" % expr.index

    def map_normal(self, expr, enclosing_prec):
        return "Normal(%d)" % expr.axis

    def map_element_jacobian(self, expr, enclosing_prec):
        return "ElJac"

    def map_face_jacobian(self, expr, enclosing_prec):
        return "FJac"

    def map_element_order(self, expr, enclosing_prec):
        return "N"

    def map_local_mesh_size(self, expr, enclosing_prec):
        return "h"


class PrettyFluxStringifyMapper(
        pymbolic.mapper.stringifier.CSESplittingStringifyMapperMixin,
        FluxStringifyMapper):
    pass


class FluxFlattenMapper(pymbolic.mapper.flattener.FlattenMapper,
        FluxIdentityMapperMixin):
    pass


class FluxDependencyMapper(pymbolic.mapper.dependency.DependencyMapper):
    def map_field_component(self, expr):
        return set([expr])

    def map_normal(self, expr):
        return set()

    map_element_jacobian = map_normal
    map_face_jacobian = map_normal
    map_element_order = map_normal
    map_local_mesh_size = map_normal
    map_c_function = map_normal

    def map_scalar_parameter(self, expr):
        return set([expr])


class FluxTermCollector(pymbolic.mapper.collector.TermCollector,
        FluxIdentityMapperMixin):
    pass


class FluxAllDependencyMapper(FluxDependencyMapper):
    def map_normal(self, expr):
        return set([expr])

    def map_element_order(self, expr):
        return set([expr])

    def map_local_mesh_size(self, expr):
        return set([expr])


class FluxNormalizationMapper(pymbolic.mapper.collector.TermCollector,
        FluxIdentityMapperMixin):
    def get_dependencies(self, expr):
        return FluxAllDependencyMapper()(expr)

    def map_constant_flux(self, expr):
        if expr.local_c == expr.neighbor_c:
            return expr.local_c
        else:
            return expr


class FluxCCFMapper(pymbolic.mapper.constant_folder.CommutativeConstantFoldingMapper,
        FluxIdentityMapperMixin):
    def is_constant(self, expr):
        return not bool(FluxAllDependencyMapper()(expr))


class FluxFlipper(FluxIdentityMapper):
    def map_field_component(self, expr):
        return expr.__class__(expr.index, not expr.is_interior)

    def map_normal(self, expr):
        return -expr

    def map_element_jacobian(self, expr):
        return expr.__class__(not expr.is_interior)

    map_element_order = map_element_jacobian


class FluxFlopCounter(pymbolic.mapper.flop_counter.FlopCounter):
    def map_normal(self, expr):
        return 0

    map_element_jacobian = map_normal
    map_face_jacobian = map_normal
    map_element_order = map_normal
    map_local_mesh_size = map_normal

    def map_field_component(self, expr):
        return 0

    def map_function_symbol(self, expr):
        return 1

    def map_c_function(self, expr):
        return 1

    def map_scalar_parameter(self, expr):
        return 0

# vim: foldmethod=marker
