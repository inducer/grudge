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
import pymbolic.primitives

from typing import Tuple

__doc__ = """

Building blocks and mappers for operator expression trees.

Basic Operators
^^^^^^^^^^^^^^^

.. autoclass:: Operator
.. autoclass:: ElementwiseLinearOperator
.. autoclass:: ProjectionOperator

.. data:: project

Reductions
^^^^^^^^^^

.. autoclass:: ElementwiseSumOperator
.. autoclass:: ElementwiseMinOperator
.. autoclass:: ElementwiseMaxOperator

.. autoclass:: NodalReductionOperator
.. autoclass:: NodalSum
.. autoclass:: NodalMax
.. autoclass:: NodalMin

Differentiation
^^^^^^^^^^^^^^^

.. autoclass:: StrongFormDiffOperatorBase
.. autoclass:: WeakFormDiffOperatorBase
.. autoclass:: StiffnessOperator
.. autoclass:: DiffOperator
.. autoclass:: StiffnessTOperator
.. autoclass:: MInvSTOperator

.. autoclass:: RefDiffOperator
.. autoclass:: RefStiffnessTOperator

.. autofunction:: nabla
.. autofunction:: minv_stiffness_t
.. autofunction:: stiffness
.. autofunction:: stiffness_t

Mass Operators
^^^^^^^^^^^^^^

.. autoclass:: MassOperatorBase

.. autoclass:: MassOperator
.. autoclass:: InverseMassOperator
.. autoclass:: FaceMassOperator

.. autoclass:: RefMassOperator
.. autoclass:: RefInverseMassOperator
.. autoclass:: RefFaceMassOperator

"""


# {{{ base classes

class Operator(pymbolic.primitives.Expression):
    """
    .. attribute:: dd_in

        an instance of :class:`~grudge.sym.DOFDesc` describing the
        input discretization.

    .. attribute:: dd_out

        an instance of :class:`~grudge.sym.DOFDesc` describing the
        output discretization.
    """

    def __init__(self, dd_in, dd_out):
        import grudge.symbolic.primitives as prim
        self.dd_in = prim.as_dofdesc(dd_in)
        self.dd_out = prim.as_dofdesc(dd_out)

    def stringifier(self):
        from grudge.symbolic.mappers import StringifyMapper
        return StringifyMapper

    def __call__(self, expr):
        from pytools.obj_array import obj_array_vectorize
        from grudge.tools import is_zero

        def bind_one(subexpr):
            if is_zero(subexpr):
                return subexpr
            else:
                from grudge.symbolic.primitives import OperatorBinding
                return OperatorBinding(self, subexpr)

        return obj_array_vectorize(bind_one, expr)

    def with_dd(self, dd_in=None, dd_out=None):
        """Return a copy of *self*, modified to the given DOF descriptors.
        """
        return type(self)(
                *self.__getinitargs__()[:-2],
                dd_in=dd_in or self.dd_in,
                dd_out=dd_out or self.dd_out)

    init_arg_names: Tuple[str, ...] = ("dd_in", "dd_out")

    def __getinitargs__(self):
        return (self.dd_in, self.dd_out,)

# }}}


class ElementwiseLinearOperator(Operator):
    def matrix(self, element_group):
        raise NotImplementedError

    mapper_method = intern("map_elementwise_linear")


class ProjectionOperator(Operator):
    def __init__(self, dd_in, dd_out):
        super().__init__(dd_in, dd_out)

    def __call__(self, expr):
        from pytools.obj_array import obj_array_vectorize

        def project_one(subexpr):
            from pymbolic.primitives import is_constant
            if self.dd_in == self.dd_out:
                # no-op projection, go away
                return subexpr
            elif is_constant(subexpr):
                return subexpr
            else:
                from grudge.symbolic.primitives import OperatorBinding
                return OperatorBinding(self, subexpr)

        return obj_array_vectorize(project_one, expr)

    mapper_method = intern("map_projection")


project = ProjectionOperator


class InterpolationOperator(ProjectionOperator):
    def __init__(self, dd_in, dd_out):
        from warnings import warn
        warn("'InterpolationOperator' is deprecated, "
                "use 'ProjectionOperator' instead.",
                DeprecationWarning, stacklevel=2)

        super().__init__(dd_in, dd_out)


def interp(dd_in, dd_out):
    from warnings import warn
    warn("using 'interp' is deprecated, use 'project' instead.",
            DeprecationWarning, stacklevel=2)

    return ProjectionOperator(dd_in, dd_out)


# {{{ element reduction: sum, min, max

class ElementwiseReductionOperator(Operator):
    def __init__(self, dd):
        super().__init__(dd_in=dd, dd_out=dd)


class ElementwiseSumOperator(ElementwiseReductionOperator):
    """Returns a vector of DOFs with all entries on each element set
    to the sum of DOFs on that element.
    """

    mapper_method = intern("map_elementwise_sum")


class ElementwiseMinOperator(ElementwiseReductionOperator):
    """Returns a vector of DOFs with all entries on each element set
    to the minimum of DOFs on that element.
    """

    mapper_method = intern("map_elementwise_min")


class ElementwiseMaxOperator(ElementwiseReductionOperator):
    """Returns a vector of DOFs with all entries on each element set
    to the maximum of DOFs on that element.
    """

    mapper_method = intern("map_elementwise_max")

# }}}


# {{{ nodal reduction: sum, integral, max

class NodalReductionOperator(Operator):
    def __init__(self, dd_in, dd_out=None):
        if dd_out is None:
            import grudge.symbolic.primitives as prim
            dd_out = prim.DD_SCALAR

        assert dd_out.is_scalar()

        super().__init__(dd_out=dd_out, dd_in=dd_in)


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
        import grudge.symbolic.primitives as prim
        if dd_in is None:
            dd_in = prim.DD_VOLUME

        if dd_out is None:
            dd_out = dd_in.with_qtag(prim.QTAG_NONE)
        else:
            dd_out = prim.as_dofdesc(dd_out)

        if dd_out.uses_quadrature():
            raise ValueError("differentiation outputs are not on "
                    "quadrature grids")

        super().__init__(dd_in, dd_out)

        self.xyz_axis = xyz_axis

    init_arg_names = ("xyz_axis", "dd_in", "dd_out")

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
        import grudge.symbolic.primitives as prim
        if dd_in is None:
            dd_in = prim.DD_VOLUME

        if dd_out is None:
            dd_out = dd_in.with_qtag(prim.QTAG_NONE)

        if dd_out.uses_quadrature():
            raise ValueError("differentiation outputs are not on "
                    "quadrature grids")

        super().__init__(dd_in, dd_out)

        self.rst_axis = rst_axis

    init_arg_names = ("rst_axis", "dd_in", "dd_out")

    def __getinitargs__(self):
        return (self.rst_axis, self.dd_in, self.dd_out)

    def equal_except_for_axis(self, other):
        return (type(self) == type(other)
                # first argument is always the axis
                and self.__getinitargs__()[1:] == other.__getinitargs__()[1:])


class RefDiffOperator(RefDiffOperatorBase):
    mapper_method = intern("map_ref_diff")

    @staticmethod
    def matrices(out_element_group, in_element_group):
        assert in_element_group == out_element_group
        return in_element_group.diff_matrices()


class RefStiffnessTOperator(RefDiffOperatorBase):
    mapper_method = intern("map_ref_stiffness_t")

    @staticmethod
    def matrices(out_elem_grp, in_elem_grp):
        if in_elem_grp == out_elem_grp:
            assert in_elem_grp.is_orthogonal_basis()
            mmat = in_elem_grp.mass_matrix()
            return [dmat.T.dot(mmat.T) for dmat in in_elem_grp.diff_matrices()]

        from modepy import vandermonde
        vand = vandermonde(out_elem_grp.basis(), out_elem_grp.unit_nodes)
        grad_vand = vandermonde(out_elem_grp.grad_basis(), in_elem_grp.unit_nodes)
        vand_inv_t = np.linalg.inv(vand).T

        if not isinstance(grad_vand, tuple):
            # NOTE: special case for 1d
            grad_vand = (grad_vand,)

        weights = in_elem_grp.weights
        return np.einsum("c,bz,acz->abc", weights, vand_inv_t, grad_vand)


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
        import grudge.symbolic.primitives as prim
        if dd_in is None:
            dd_in = prim.DD_VOLUME

        if dd_out is None:
            dd_out = dd_in

        if dd_in.uses_quadrature():
            raise ValueError("dd_in may not use quadrature")
        if dd_in != dd_out:
            raise ValueError("dd_in and dd_out must be identical")

        super().__init__(dd_in, dd_out)

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
        import grudge.symbolic.primitives as prim
        if dd_in is None:
            dd_in = prim.DD_VOLUME

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


# {{{ mass operators

class MassOperatorBase(Operator):
    """
    Inherits from :class:`Operator`.

    :attr:`~Operator.dd_in` and :attr:`~Operator.dd_out` may be surface or volume
    discretizations.
    """

    def __init__(self, dd_in=None, dd_out=None):
        import grudge.symbolic.primitives as prim
        if dd_in is None:
            dd_in = prim.DD_VOLUME
        if dd_out is None:
            dd_out = prim.DD_VOLUME

        super().__init__(dd_in, dd_out)


class MassOperator(MassOperatorBase):
    mapper_method = intern("map_mass")


class InverseMassOperator(MassOperatorBase):
    mapper_method = intern("map_inverse_mass")


class RefMassOperatorBase(ElementwiseLinearOperator):
    pass


class RefMassOperator(RefMassOperatorBase):
    @staticmethod
    def matrix(out_element_group, in_element_group):
        if out_element_group == in_element_group:
            return in_element_group.mass_matrix()

        from modepy import vandermonde
        vand = vandermonde(out_element_group.basis(), out_element_group.unit_nodes)
        o_vand = vandermonde(out_element_group.basis(), in_element_group.unit_nodes)
        vand_inv_t = np.linalg.inv(vand).T

        weights = in_element_group.weights
        return np.einsum("j,ik,jk->ij", weights, vand_inv_t, o_vand)

    mapper_method = intern("map_ref_mass")


class RefInverseMassOperator(RefMassOperatorBase):
    @staticmethod
    def matrix(in_element_group, out_element_group):
        assert in_element_group == out_element_group
        import modepy as mp
        return mp.inverse_mass_matrix(
                in_element_group.basis(),
                in_element_group.unit_nodes)

    mapper_method = intern("map_ref_inverse_mass")

# }}}


# {{{ boundary-related operators

class OppositeInteriorFaceSwap(Operator):
    """
    .. attribute:: unique_id

        An integer identifying this specific instances of
        :class:`OppositePartitionFaceSwap` within an entire bound symbolic
        operator. Is assigned automatically by :func:`grudge.bind`
        if not already set by the user. This will become
        :class:`OppositePartitionFaceSwap.unique_id` in distributed
        runs.
    """

    def __init__(self, dd_in=None, dd_out=None, unique_id=None):
        import grudge.symbolic.primitives as prim
        if dd_in is None:
            dd_in = prim.DOFDesc(prim.FACE_RESTR_INTERIOR, None)
        if dd_out is None:
            dd_out = dd_in

        super().__init__(dd_in, dd_out)
        if self.dd_in.domain_tag is not prim.FACE_RESTR_INTERIOR:
            raise ValueError("dd_in must be an interior faces domain")
        if self.dd_out != self.dd_in:
            raise ValueError("dd_out and dd_in must be identical")

        assert unique_id is None or isinstance(unique_id, int)
        self.unique_id = unique_id

    init_arg_names = ("dd_in", "dd_out", "unique_id")

    def __getinitargs__(self):
        return (self.dd_in, self.dd_out, self.unique_id)

    mapper_method = intern("map_opposite_interior_face_swap")


class OppositePartitionFaceSwap(Operator):
    """
    .. attribute:: unique_id

        An integer corresponding to the :attr:`OppositeInteriorFaceSwap.unique_id`
        which led to the creation of this object. This integer is used as an
        MPI tag offset to keep different subexpressions apart in MPI traffic.
    """
    def __init__(self, dd_in=None, dd_out=None, unique_id=None):
        import grudge.symbolic.primitives as prim

        if dd_in is None and dd_out is None:
            raise ValueError("dd_in or dd_out must be specified")
        elif dd_in is None:
            dd_in = dd_out
        elif dd_out is None:
            dd_out = dd_in

        super().__init__(dd_in, dd_out)
        if not (isinstance(self.dd_in.domain_tag, prim.DTAG_BOUNDARY)
                and isinstance(self.dd_in.domain_tag.tag, prim.BTAG_PARTITION)):
            raise ValueError(
                    "dd_in must be a partition boundary faces domain, not '%s'"
                    % self.dd_in.domain_tag)
        if self.dd_out != self.dd_in:
            raise ValueError("dd_out and dd_in must be identical")

        self.i_remote_part = self.dd_in.domain_tag.tag.part_nr

        assert unique_id is None or isinstance(unique_id, int)
        self.unique_id = unique_id

    init_arg_names = ("dd_in", "dd_out", "unique_id")

    def __getinitargs__(self):
        return (self.dd_in, self.dd_out, self.unique_id)

    mapper_method = intern("map_opposite_partition_face_swap")


class FaceMassOperatorBase(ElementwiseLinearOperator):
    def __init__(self, dd_in=None, dd_out=None):
        import grudge.symbolic.primitives as prim
        if dd_in is None:
            dd_in = prim.DOFDesc(prim.FACE_RESTR_ALL, None)

        if dd_out is None or dd_out == "vol":
            dd_out = prim.DOFDesc("vol", prim.QTAG_NONE)

        if dd_out.uses_quadrature():
            raise ValueError("face mass operator outputs are not on "
                    "quadrature grids")

        if not dd_out.is_volume():
            raise ValueError("dd_out must be a volume domain")
        if dd_in.domain_tag is not prim.FACE_RESTR_ALL:
            raise ValueError("dd_in must be an interior faces domain")

        super().__init__(dd_in, dd_out)


class FaceMassOperator(FaceMassOperatorBase):
    mapper_method = intern("map_face_mass_operator")


class RefFaceMassOperator(ElementwiseLinearOperator):
    def matrix(self, afgrp, volgrp, dtype):
        nfaces = volgrp.mesh_el_group.nfaces
        assert afgrp.nelements == nfaces * volgrp.nelements

        matrix = np.empty(
                (volgrp.nunit_dofs,
                    nfaces,
                    afgrp.nunit_dofs),
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


def stiffness_t(dim, dd_in=None, dd_out=None):
    from pytools.obj_array import make_obj_array
    return make_obj_array(
        [StiffnessTOperator(i, dd_in, dd_out) for i in range(dim)])


def integral(arg, dd=None):
    import grudge.symbolic.primitives as prim

    if dd is None:
        dd = prim.DD_VOLUME
    dd = prim.as_dofdesc(dd)

    return NodalSum(dd)(
            arg * prim.cse(
                MassOperator(dd_in=dd, dd_out=dd)(prim.Ones(dd)),
                "mass_quad_weights",
                prim.cse_scope.DISCRETIZATION))


def norm(p, arg, dd=None):
    """
    :arg arg: is assumed to be a vector, i.e. have shape ``(n,)``.
    """
    import grudge.symbolic.primitives as prim

    if dd is None:
        dd = prim.DD_VOLUME
    dd = prim.as_dofdesc(dd)

    if p == 2:
        norm_squared = NodalSum(dd_in=dd)(
                arg * MassOperator()(arg))

        if isinstance(norm_squared, np.ndarray):
            if len(norm_squared.shape) != 1:
                raise NotImplementedError("can only take the norm of vectors")

            norm_squared = norm_squared.sum()

        return prim.sqrt(norm_squared)

    elif p == np.inf:
        result = NodalMax(dd_in=dd)(prim.fabs(arg))

        if isinstance(result, np.ndarray):
            if len(result.shape) != 1:
                raise NotImplementedError("can only take the norm of vectors")

            from pymbolic.primitives import Max
            result = Max(result)

        return result

    else:
        raise ValueError("unsupported value of p")


def h_max_from_volume(ambient_dim, dim=None, dd=None):
    """Defines a characteristic length based on the volume of the elements.
    This length may not be representative if the elements have very high
    aspect ratios.
    """

    import grudge.symbolic.primitives as prim
    if dd is None:
        dd = prim.DD_VOLUME
    dd = prim.as_dofdesc(dd)

    if dim is None:
        dim = ambient_dim

    return NodalMax(dd_in=dd)(
            ElementwiseSumOperator(dd)(
                MassOperator(dd_in=dd, dd_out=dd)(prim.Ones(dd))
                )
            )**(1.0/dim)


def h_min_from_volume(ambient_dim, dim=None, dd=None):
    """Defines a characteristic length based on the volume of the elements.
    This length may not be representative if the elements have very high
    aspect ratios.
    """

    import grudge.symbolic.primitives as prim
    if dd is None:
        dd = prim.DD_VOLUME
    dd = prim.as_dofdesc(dd)

    if dim is None:
        dim = ambient_dim

    return NodalMin(dd_in=dd)(
            ElementwiseSumOperator(dd)(
                MassOperator(dd_in=dd, dd_out=dd)(prim.Ones(dd))
                )
            )**(1.0/dim)

# }}}

# vim: foldmethod=marker
