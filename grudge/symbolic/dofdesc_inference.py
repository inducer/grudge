__copyright__ = "Copyright (C) 2017 Andreas Kloeckner"

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


# This is purely leaves-to-roots. No need to propagate information in the
# opposite direction.


from pymbolic.mapper import RecursiveMapper, CSECachingMapperMixin
from grudge.symbolic.primitives import DOFDesc, DTAG_SCALAR


def unify_dofdescs(dd_a, dd_b, expr=None):
    if dd_a is None:
        assert dd_b is not None
        return dd_b

    if expr is not None:
        loc_str = "in expression %s" % str(expr)
    else:
        loc_str = ""

    from grudge.symbolic.primitives import DTAG_SCALAR
    if dd_a.domain_tag != dd_b.domain_tag:
        if dd_a.domain_tag == DTAG_SCALAR:
            return dd_b
        elif dd_b.domain_tag == DTAG_SCALAR:
            return dd_a
        else:
            raise ValueError("mismatched domain tags " + loc_str)

    # domain tags match
    if dd_a.quadrature_tag != dd_b.quadrature_tag:
        raise ValueError("mismatched quadrature tags " + loc_str)

    return dd_a


class InferrableMultiAssignment:
    """An assignemnt 'instruction' which may be used as part of type
    inference.

    .. method:: get_assignees(rec)

        :returns: a :class:`set` of names which are assigned values by
        this assignment.

    .. method:: infer_dofdescs(rec)

        :returns: a list of ``(name, :class:`grudge.symbolic.primitives.DOFDesc`)``
        tuples, each indicating the value type of the value with *name*.
    """

    # (not a base class--only documents the interface)


class DOFDescInferenceMapper(RecursiveMapper, CSECachingMapperMixin):
    def __init__(self, assignments, function_registry,
                name_to_dofdesc=None, check=True):
        """
        :arg assignments: a list of objects adhering to
            :class:`InferrableMultiAssignment`.
        :returns: an instance of :class:`DOFDescInferenceMapper`
        """

        self.check = check

        self.name_to_assignment = {
                name: a
                for a in assignments
                if not a.neglect_for_dofdesc_inference
                for name in a.get_assignees()}

        if name_to_dofdesc is None:
            name_to_dofdesc = {}
        else:
            name_to_dofdesc = name_to_dofdesc.copy()

        self.name_to_dofdesc = name_to_dofdesc

        self.function_registry = function_registry

    def infer_for_name(self, name):
        try:
            return self.name_to_dofdesc[name]
        except KeyError:
            a = self.name_to_assignment[name]

            inf_method = getattr(self, a.mapper_method)
            for r_name, r_dofdesc in inf_method(a):
                assert r_name not in self.name_to_dofdesc
                self.name_to_dofdesc[r_name] = r_dofdesc

            return self.name_to_dofdesc[name]

    # {{{ expression mappings

    def map_constant(self, expr):
        return DOFDesc(DTAG_SCALAR)

    def map_grudge_variable(self, expr):
        return expr.dd

    def map_variable(self, expr):
        return self.infer_for_name(expr.name)

    def map_subscript(self, expr):
        # FIXME: Subscript has same type as aggregate--a bit weird
        return self.rec(expr.aggregate)

    def map_multi_child(self, expr, children):
        dofdesc = None

        for ch in children:
            dofdesc = unify_dofdescs(dofdesc, self.rec(ch), expr)

        if dofdesc is None:
            raise ValueError("no DOFDesc found for expression %s" % expr)
        else:
            return dofdesc

    def map_sum(self, expr):
        return self.map_multi_child(expr, expr.children)

    map_product = map_sum
    map_max = map_sum
    map_min = map_sum

    def map_quotient(self, expr):
        return self.map_multi_child(expr, (expr.numerator, expr.denominator))

    def map_power(self, expr):
        return self.map_multi_child(expr, (expr.base, expr.exponent))

    def map_if(self, expr):
        return self.map_multi_child(expr, [expr.condition, expr.then, expr.else_])

    def map_comparison(self, expr):
        return self.map_multi_child(expr, [expr.left, expr.right])

    def map_nodal_sum(self, expr, enclosing_prec):
        return DOFDesc(DTAG_SCALAR)

    map_nodal_max = map_nodal_sum
    map_nodal_min = map_nodal_sum

    def map_operator_binding(self, expr):
        operator = expr.op

        if self.check:
            op_dd = self.rec(expr.field)
            if op_dd != operator.dd_in:
                raise ValueError("mismatched input to %s "
                        "(got: %s, expected: %s)"
                        " in '%s'"
                        % (
                            type(expr).__name__,
                            op_dd, expr.op.dd_in,
                            str(expr)))

        return operator.dd_out

    def map_ones(self, expr):
        return expr.dd

    map_node_coordinate_component = map_ones
    map_signed_face_ones = map_ones

    def map_call(self, expr):
        arg_dds = [
                self.rec(par)
                for par in expr.parameters]

        return (
                self.function_registry[expr.function.name]
                .get_result_dofdesc(arg_dds))

    # }}}

    # {{{ instruction mappings

    def map_insn_assign(self, insn):
        return [
                (name, self.rec(expr))
                for name, expr in zip(insn.names, insn.exprs)
                ]

    def map_insn_rank_data_swap(self, insn):
        return [(insn.name, insn.dd_out)]

    map_insn_assign_to_discr_scoped = map_insn_assign

    def map_insn_diff_batch_assign(self, insn):
        if self.check:
            repr_op = insn.operators[0]
            input_dd = self.rec(insn.field)
            if input_dd != repr_op.dd_in:
                raise ValueError("mismatched input to %s "
                        "(got: %s, expected: %s)"
                        % (
                            type(insn).__name__,
                            input_dd, repr_op.dd_in,
                            ))

        return [
                (name, op.dd_out)
                for name, op in zip(insn.names, insn.operators)]

    # }}}

# vim: foldmethod=marker
