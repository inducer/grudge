"""Building blocks and mappers for operator expression trees."""

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

from six.moves import intern

import numpy as np
import numpy.linalg as la  # noqa
import pymbolic.primitives


def _sym():
    # A hack to make referring to grudge.sym possible without
    # circular imports and tons of typing.

    from grudge import sym
    return sym


# {{{ base classes

class Operator(pymbolic.primitives.Expression):
    """
    .. attribute:: dd_in

        an instance of :class:`grudge.sym.DOFDesc` describing the
        input discretization.

    .. attribute:: dd_out

        an instance of :class:`grudge.sym.DOFDesc` describing the
        output discretization.
    """

    def __init__(self, dd_in, dd_out):
        self.dd_in = _sym().as_dofdesc(dd_in)
        self.dd_out = _sym().as_dofdesc(dd_out)

    def stringifier(self):
        from grudge.symbolic.mappers import StringifyMapper
        return StringifyMapper

    def __call__(self, expr):
        from pytools.obj_array import with_object_array_or_scalar
        from grudge.tools import is_zero

        def bind_one(subexpr):
            if is_zero(subexpr):
                return subexpr
            else:
                from grudge.symbolic.primitives import OperatorBinding
                return OperatorBinding(self, subexpr)

        return with_object_array_or_scalar(bind_one, expr)

    def with_dd(self, dd_in=None, dd_out=None):
        """Return a copy of *self*, modified to the given DOF descriptors.
        """
        return type(self)(
                *self.__getinitargs__()[:-2],
                dd_in=dd_in or self.dd_in,
                dd_out=dd_out or self.dd_out)

    def __getinitargs__(self):
        return (self.dd_in, self.dd_out,)

# }}}


class ElementwiseLinearOperator(Operator):
    def matrix(self, element_group):
        raise NotImplementedError

    mapper_method = intern("map_elementwise_linear")


class InterpolationOperator(Operator):
    mapper_method = intern("map_interpolation")


interp = InterpolationOperator


class ElementwiseMaxOperator(Operator):
    mapper_method = intern("map_elementwise_max")


# {{{ nodal reduction: sum, integral, max

class NodalReductionOperator(Operator):
    def __init__(self, dd_in, dd_out=None):
        if dd_out is None:
            dd_out = _sym().DD_SCALAR

        assert dd_out.is_scalar()

        super(NodalReductionOperator, self).__init__(dd_out=dd_out, dd_in=dd_in)


class NodalSum(NodalReductionOperator):
    mapper_method = intern("map_nodal_sum")


class NodalMax(NodalReductionOperator):
    mapper_method = intern("map_nodal_max")


class NodalMin(NodalReductionOperator):
    mapper_method = intern("map_nodal_min")

# }}}


# {{{ differentiation operators

# {{{ global differentiation

class DiffOperatorBase(Operator):
    def __init__(self, xyz_axis, dd_in=None, dd_out=None):
        if dd_in is None:
            dd_in = _sym().DD_VOLUME
        if dd_out is None:
            dd_out = dd_in.with_qtag(_sym().QTAG_NONE)
        if dd_out.uses_quadrature():
            raise ValueError("differentiation outputs are not on "
                    "quadrature grids")

        super(DiffOperatorBase, self).__init__(dd_in, dd_out)

        self.xyz_axis = xyz_axis

    def __getinitargs__(self):
        return (self.xyz_axis, self.dd_in, self.dd_out)

    def equal_except_for_axis(self, other):
        return (type(self) == type(other)
                # first argument is always the axis
                and self.__getinitargs__()[1:] == other.__getinitargs__()[1:])


class StrongFormDiffOperatorBase(DiffOperatorBase):
    pass


class WeakFormDiffOperatorBase(DiffOperatorBase):
    pass


class StiffnessOperator(StrongFormDiffOperatorBase):
    mapper_method = intern("map_stiffness")


class DiffOperator(StrongFormDiffOperatorBase):
    mapper_method = intern("map_diff")


class StiffnessTOperator(WeakFormDiffOperatorBase):
    mapper_method = intern("map_stiffness_t")


class MInvSTOperator(WeakFormDiffOperatorBase):
    mapper_method = intern("map_minv_st")

# }}}


# {{{ reference-element differentiation

class RefDiffOperatorBase(ElementwiseLinearOperator):
    def __init__(self, rst_axis, dd_in=None, dd_out=None):
        if dd_in is None:
            dd_in = _sym().DD_VOLUME
        if dd_out is None:
            dd_out = dd_in.with_qtag(_sym().QTAG_NONE)
        if dd_out.uses_quadrature():
            raise ValueError("differentiation outputs are not on "
                    "quadrature grids")

        super(RefDiffOperatorBase, self).__init__(dd_in, dd_out)

        self.rst_axis = rst_axis

    def __getinitargs__(self):
        return (self.rst_axis, self.dd_in, self.dd_out)

    def equal_except_for_axis(self, other):
        return (type(self) == type(other)
                # first argument is always the axis
                and self.__getinitargs__()[1:] == other.__getinitargs__()[1:])


class RefDiffOperator(RefDiffOperatorBase):
    mapper_method = intern("map_ref_diff")

    @staticmethod
    def matrices(element_group):
        return element_group.diff_matrices()


class RefStiffnessTOperator(RefDiffOperatorBase):
    mapper_method = intern("map_ref_stiffness_t")

    @staticmethod
    def matrices(element_group):
        assert element_group.is_orthogonal_basis()
        mmat = element_group.mass_matrix()
        return [dmat.T.dot(mmat.T) for dmat in element_group.diff_matrices()]

# }}}

# }}}


# {{{ various elementwise linear operators

class FilterOperator(ElementwiseLinearOperator):
    def __init__(self, mode_response_func, dd_in=None, dd_out=None):
        """
        :param mode_response_func: A function mapping
          ``(mode_tuple, local_discretization)`` to a float indicating the
          factor by which this mode is to be multiplied after filtering.
          (For example an instance of
          :class:`ExponentialFilterResponseFunction`.
        """
        if dd_in is None:
            dd_in = _sym().DD_VOLUME
        if dd_out is None:
            dd_out = dd_in

        if dd_in.uses_quadrature():
            raise ValueError("dd_in may not use quadrature")
        if dd_in != dd_out:
            raise ValueError("dd_in and dd_out must be identical")

        super(FilterOperator, self).__init__(dd_in, dd_out)

        self.mode_response_func = mode_response_func

    def __getinitargs__(self):
        return (self.mode_response_func, self.dd_in, self.dd_out)

    def matrix(self, eg):
        # FIXME
        raise NotImplementedError()

        ldis = eg.local_discretization

        filter_coeffs = [self.mode_response_func(mid, ldis)
            for mid in ldis.generate_mode_identifiers()]

        # build filter matrix
        vdm = ldis.vandermonde()
        from grudge.tools import leftsolve
        mat = np.asarray(
            leftsolve(vdm,
                np.dot(vdm, np.diag(filter_coeffs))),
            order="C")

        return mat


class AveragingOperator(ElementwiseLinearOperator):
    def __init__(self, dd_in=None, dd_out=None):
        if dd_in is None:
            dd_in = _sym().DD_VOLUME
        if dd_out is None:
            dd_out = dd_in

        if dd_in.uses_quadrature():
            raise ValueError("dd_in may not use quadrature")
        if dd_in != dd_out:
            raise ValueError("dd_in and dd_out must be identical")

        super(FilterOperator, self).__init__(dd_in, dd_out)

    def matrix(self, eg):
        # average matrix, so that AVE*fields = cellaverage(fields)
        # see Hesthaven and Warburton page 227

        # FIXME
        raise NotImplementedError()

        mmat = eg.local_discretization.mass_matrix()
        standard_el_vol = np.sum(np.dot(mmat, np.ones(mmat.shape[0])))
        avg_mat_row = np.sum(mmat, 0)/standard_el_vol

        avg_mat = np.zeros((np.size(avg_mat_row), np.size(avg_mat_row)))
        avg_mat[:] = avg_mat_row
        return avg_mat


class InverseVandermondeOperator(ElementwiseLinearOperator):
    def matrix(self, element_group):
        raise NotImplementedError()  # FIXME


class VandermondeOperator(ElementwiseLinearOperator):
    def matrix(self, element_group):
        raise NotImplementedError()  # FIXME

# }}}

# }}}


# {{{ mass operators

class MassOperatorBase(Operator):
    """
    :attr:`dd_in` and :attr:`dd_out` may be surface or volume discretizations.
    """

    def __init__(self, dd_in=None, dd_out=None):
        if dd_in is None:
            dd_in = _sym().DD_VOLUME
        if dd_out is None:
            dd_out = dd_in

        if dd_out != dd_in:
            raise ValueError("dd_out and dd_in must be identical")

        super(MassOperatorBase, self).__init__(dd_in, dd_out)


class MassOperator(MassOperatorBase):
    mapper_method = intern("map_mass")


class InverseMassOperator(MassOperatorBase):
    mapper_method = intern("map_inverse_mass")


class RefMassOperatorBase(ElementwiseLinearOperator):
    pass


class RefMassOperator(RefMassOperatorBase):
    @staticmethod
    def matrix(element_group):
        return element_group.mass_matrix()

    mapper_method = intern("map_ref_mass")


class RefInverseMassOperator(RefMassOperatorBase):
    @staticmethod
    def matrix(element_group):
        import modepy as mp
        return mp.inverse_mass_matrix(
                element_group.basis(),
                element_group.unit_nodes)

    mapper_method = intern("map_ref_inverse_mass")

# }}}


# {{{ boundary-related operators

class OppositeRankFaceSwap(Operator):
    def __init__(self, i_remote_rank, dd_in=None, dd_out=None):
        sym = _sym()

        if dd_in is None:
            # dd_in = sym.DOFDesc(sym.FRESTR_INTERIOR_FACES)
            dd_in = sym.DOFDesc(sym.BTAG_PARTITION)  # TODO: Throws an error later
        if dd_out is None:
            dd_out = dd_in

        # if dd_in.domain_tag is not sym.BTAG_PARTITION:
        #     raise ValueError("dd_in must be a rank boundary faces domain")
        # if dd_out != dd_in:
        #     raise ValueError("dd_out and dd_in must be identical")

        super(OppositeRankFaceSwap, self).__init__(dd_in, dd_out)
        self.i_remote_rank = i_remote_rank

    mapper_method = intern("map_opposite_rank_face_swap")


class OppositeInteriorFaceSwap(Operator):
    def __init__(self, dd_in=None, dd_out=None):
        sym = _sym()

        if dd_in is None:
            dd_in = sym.DOFDesc(sym.FRESTR_INTERIOR_FACES, None)
        if dd_out is None:
            dd_out = dd_in

        if dd_in.domain_tag is not sym.FRESTR_INTERIOR_FACES:
            raise ValueError("dd_in must be an interior faces domain")
        if dd_out != dd_in:
            raise ValueError("dd_out and dd_in must be identical")

        super(OppositeInteriorFaceSwap, self).__init__(dd_in, dd_out)

    mapper_method = intern("map_opposite_interior_face_swap")


class FaceMassOperatorBase(ElementwiseLinearOperator):
    def __init__(self, dd_in=None, dd_out=None):
        sym = _sym()

        if dd_in is None:
            dd_in = sym.DOFDesc(sym.FRESTR_ALL_FACES, None)

        if dd_out is None:
            dd_out = sym.DOFDesc("vol", sym.QTAG_NONE)
        if dd_out.uses_quadrature():
            raise ValueError("face mass operator outputs are not on "
                    "quadrature grids")

        if not dd_out.is_volume():
            raise ValueError("dd_out must be a volume domain")
        if dd_in.domain_tag is not sym.FRESTR_ALL_FACES:
            raise ValueError("dd_in must be an interior faces domain")

        super(FaceMassOperatorBase, self).__init__(dd_in, dd_out)


class FaceMassOperator(FaceMassOperatorBase):
    mapper_method = intern("map_face_mass_operator")


class RefFaceMassOperator(ElementwiseLinearOperator):
    def matrix(self, afgrp, volgrp, dtype):
        nfaces = volgrp.mesh_el_group.nfaces
        assert afgrp.nelements == nfaces * volgrp.nelements

        matrix = np.empty(
                (volgrp.nunit_nodes,
                    nfaces,
                    afgrp.nunit_nodes),
                dtype=dtype)

        from modepy.tools import UNIT_VERTICES
        import modepy as mp
        for iface, fvi in enumerate(
                volgrp.mesh_el_group.face_vertex_indices()):
            face_vertices = UNIT_VERTICES[volgrp.dim][np.array(fvi)].T
            matrix[:, iface, :] = mp.nodal_face_mass_matrix(
                    volgrp.basis(), volgrp.unit_nodes, afgrp.unit_nodes,
                    volgrp.order,
                    face_vertices)

        # np.set_printoptions(linewidth=200, precision=3)
        # matrix[np.abs(matrix) < 1e-13] = 0
        # print(matrix)
        # 1/0

        return matrix

    mapper_method = intern("map_ref_face_mass_operator")

# }}}


# {{{ convenience functions for operator creation

def nabla(dim):
    from pytools.obj_array import make_obj_array
    return make_obj_array(
            [DiffOperator(i) for i in range(dim)])


def minv_stiffness_t(dim):
    from pytools.obj_array import make_obj_array
    return make_obj_array(
        [MInvSTOperator(i) for i in range(dim)])


def stiffness(dim):
    from pytools.obj_array import make_obj_array
    return make_obj_array(
        [StiffnessOperator(i) for i in range(dim)])


def stiffness_t(dim):
    from pytools.obj_array import make_obj_array
    return make_obj_array(
        [StiffnessTOperator(i) for i in range(dim)])


def integral(arg, dd=None):
    sym = _sym()

    if dd is None:
        dd = sym.DD_VOLUME

    dd = sym.as_dofdesc(dd)

    return sym.NodalSum(dd)(
            arg * sym.cse(
                sym.MassOperator(dd_in=dd)(sym.Ones(dd)),
                "mass_quad_weights",
                sym.cse_scope.DISCRETIZATION))


def norm(p, arg, dd=None):
    """
    :arg arg: is assumed to be a vector, i.e. have shape ``(n,)``.
    """
    sym = _sym()

    if dd is None:
        dd = sym.DD_VOLUME

    dd = sym.as_dofdesc(dd)

    if p == 2:
        norm_squared = sym.NodalSum(dd_in=dd)(
                sym.CFunction("fabs")(
                    arg * sym.MassOperator()(arg)))

        if isinstance(norm_squared, np.ndarray):
            norm_squared = norm_squared.sum()

        return sym.CFunction("sqrt")(norm_squared)

    elif p == np.Inf:
        result = sym.NodalMax()(sym.CFunction("fabs")(arg))
        from pymbolic.primitives import Max

        if isinstance(norm_squared, np.ndarray):
            from functools import reduce
            result = reduce(Max, result)

        return result

    else:
        raise ValueError("unsupported value of p")

# }}}

# vim: foldmethod=marker
