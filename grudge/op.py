"""
Core DG routines
^^^^^^^^^^^^^^^^

Elementwise differentiation
---------------------------

.. autofunction:: local_grad
.. autofunction:: local_d_dx
.. autofunction:: local_div

Weak derivative operators
-------------------------

.. autofunction:: weak_local_grad
.. autofunction:: weak_local_d_dx
.. autofunction:: weak_local_div

Mass, inverse mass, and face mass operators
-------------------------------------------

.. autofunction:: mass
.. autofunction:: inverse_mass
.. autofunction:: face_mass
"""

__copyright__ = """
Copyright (C) 2021 Andreas Kloeckner
Copyright (C) 2021 University of Illinois Board of Trustees
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


from arraycontext import ArrayContext, map_array_container
from arraycontext.container import ArrayOrContainerT

from functools import partial

from meshmode.dof_array import DOFArray
from meshmode.transform_metadata import FirstAxisIsElementsTag

from grudge.discretization import DiscretizationCollection

from pytools import keyed_memoize_in
from pytools.obj_array import obj_array_vectorize, make_obj_array

import numpy as np

import grudge.dof_desc as dof_desc

from grudge.interpolation import interp  # noqa: F401
from grudge.projection import project  # noqa: F401

from grudge.reductions import (  # noqa: F401
    norm,
    nodal_sum,
    nodal_min,
    nodal_max,
    nodal_sum_loc,
    nodal_min_loc,
    nodal_max_loc,
    integral,
    elementwise_sum,
    elementwise_max,
    elementwise_min,
    elementwise_integral,
)

from grudge.trace_pair import (  # noqa: F401
    interior_trace_pair,
    interior_trace_pairs,
    connected_ranks,
    cross_rank_trace_pairs,
    bdry_trace_pair,
    bv_trace_pair
)


# {{{ common derivative "kernels"

def _single_axis_derivative_kernel(
        actx, out_discr, in_discr, get_diff_mat, inv_jac_mat, xyz_axis, vec,
        *, metric_in_matvec):
    # This gets used from both the strong and the weak derivative. These differ
    # in three ways:
    # - which differentiation matrix gets used,
    # - whether inv_jac_mat is pre-multiplied by a factor that includes the
    #   area element, and
    # - whether the chain rule terms ("inv_jac_mat") sit outside (strong)
    #   or inside (weak) the matrix-vector product that carries out the
    #   derivative, cf. "metric_in_matvec".
    return DOFArray(
        actx,
        data=tuple(
            # r for rst axis
            actx.einsum("rej,rij,ej->ei" if metric_in_matvec else "rei,rij,ej->ei",
                        ijm_i[xyz_axis],
                        get_diff_mat(
                            actx,
                            out_element_group=out_grp,
                            in_element_group=in_grp
                        ),
                        vec_i,
                        arg_names=("inv_jac_t", "ref_stiffT_mat", "vec", ),
                        tagged=(FirstAxisIsElementsTag(),))

            for out_grp, in_grp, vec_i, ijm_i in zip(
                out_discr.groups, in_discr.groups, vec,
                inv_jac_mat)))


def _gradient_kernel(actx, out_discr, in_discr, get_diff_mat, inv_jac_mat, vec,
        *, metric_in_matvec):
    # See _single_axis_derivative_kernel for comments on the usage scenarios
    # (both strong and weak derivative) and their differences.
    per_group_grads = [
        # r for rst axis
        # x for xyz axis
        actx.einsum("xrej,rij,ej->xei" if metric_in_matvec else "xrei,rij,ej->xei",
                    ijm_i,
                    get_diff_mat(
                        actx,
                        out_element_group=out_grp,
                        in_element_group=in_grp
                    ),
                    vec_i,
                    arg_names=("inv_jac_t", "ref_stiffT_mat", "vec"),
                    tagged=(FirstAxisIsElementsTag(),))
        for out_grp, in_grp, vec_i, ijm_i in zip(
            out_discr.groups, in_discr.groups, vec,
            inv_jac_mat)]

    return make_obj_array([
            DOFArray(
                actx, data=tuple([pgg_i[xyz_axis] for pgg_i in per_group_grads]))
            for xyz_axis in range(out_discr.ambient_dim)])

# }}}


# {{{ Derivative operators

def reference_derivative_matrices(actx: ArrayContext,
        out_element_group, in_element_group):
    # We're accepting in_element_group for interface consistency with
    # reference_stiffness_transpose_matrix.
    assert out_element_group is in_element_group

    @keyed_memoize_in(
        actx, reference_derivative_matrices,
        lambda grp: grp.discretization_key())
    def get_ref_derivative_mats(grp):
        from meshmode.discretization.poly_element import diff_matrices

        return actx.freeze(
            actx.from_numpy(
                np.asarray(
                    [dfmat for dfmat in diff_matrices(grp)]
                )
            )
        )
    return get_ref_derivative_mats(out_element_group)


def local_grad(
        dcoll: DiscretizationCollection, vec, *, nested=False) -> np.ndarray:
    r"""Return the element-local gradient of a function :math:`f` represented
    by *vec*:

    .. math::

        \nabla|_E f = \left(
            \partial_x|_E f, \partial_y|_E f, \partial_z|_E f \right)

    :arg vec: a :class:`~meshmode.dof_array.DOFArray` or an
        :class:`~arraycontext.container.ArrayContainer` of them.
    :arg nested: return nested object arrays instead of a single multidimensional
        array if *vec* is non-scalar.
    :returns: an object array (possibly nested) of
        :class:`~meshmode.dof_array.DOFArray`\ s or
        :class:`~arraycontext.container.ArrayContainer`\ s.
    """
    if isinstance(vec, np.ndarray):
        grad = obj_array_vectorize(
                lambda el: local_grad(dcoll, el, nested=nested), vec)
        if nested:
            return grad
        else:
            return np.stack(grad, axis=0)

    from grudge.geometry import inverse_surface_metric_derivative_mat

    discr = dcoll.discr_from_dd(dof_desc.DD_VOLUME)
    actx = vec.array_context

    inverse_jac_mat = inverse_surface_metric_derivative_mat(actx, dcoll,
            _use_geoderiv_connection=actx.supports_nonscalar_broadcasting)
    return _gradient_kernel(
        actx,
        discr,
        discr,
        reference_derivative_matrices,
        inverse_jac_mat,
        vec,
        metric_in_matvec=False
    )


def local_d_dx(
        dcoll: DiscretizationCollection, xyz_axis, vec) -> ArrayOrContainerT:
    r"""Return the element-local derivative along axis *xyz_axis* of a
    function :math:`f` represented by *vec*:

    .. math::

        \frac{\partial f}{\partial \lbrace x,y,z\rbrace}\Big|_E

    :arg xyz_axis: an integer indicating the axis along which the derivative
        is taken.
    :arg vec: a :class:`~meshmode.dof_array.DOFArray` or an
        :class:`~arraycontext.container.ArrayContainer` of them.
    :returns: a :class:`~meshmode.dof_array.DOFArray` or an
        :class:`~arraycontext.container.ArrayContainer` of them.
    """
    discr = dcoll.discr_from_dd(dof_desc.DD_VOLUME)
    actx = vec.array_context

    from grudge.geometry import inverse_surface_metric_derivative_mat
    inverse_jac_mat = inverse_surface_metric_derivative_mat(actx, dcoll,
            _use_geoderiv_connection=actx.supports_nonscalar_broadcasting)

    return _single_axis_derivative_kernel(
        actx,
        discr,
        discr,
        reference_derivative_matrices,
        inverse_jac_mat,
        xyz_axis,
        vec,
        metric_in_matvec=False
    )


def _div_helper(dcoll: DiscretizationCollection, diff_func, vecs):
    if not isinstance(vecs, np.ndarray):
        raise TypeError("argument must be an object array")
    assert vecs.dtype == object

    if isinstance(vecs[(0,)*vecs.ndim], np.ndarray):
        div_shape = vecs.shape
    else:
        if vecs.shape[-1] != dcoll.ambient_dim:
            raise ValueError("last dimension of *vecs* argument doesn't match "
                    "ambient dimension")
        div_shape = vecs.shape[:-1]

    if len(div_shape) == 0:
        return sum(diff_func(i, vec_i) for i, vec_i in enumerate(vecs))
    else:
        result = np.zeros(div_shape, dtype=object)
        for idx in np.ndindex(div_shape):
            result[idx] = sum(
                    diff_func(i, vec_i) for i, vec_i in enumerate(vecs[idx]))
        return result


def local_div(dcoll: DiscretizationCollection, vecs) -> ArrayOrContainerT:
    r"""Return the element-local divergence of the vector function
    :math:`\mathbf{f}` represented by *vecs*:

    .. math::

        \nabla|_E \cdot \mathbf{f} = \sum_{i=1}^d \partial_{x_i}|_E \mathbf{f}_i

    :arg vecs: an object array of
        :class:`~meshmode.dof_array.DOFArray`\ s or
        :class:`~arraycontext.container.ArrayContainer` objects,
        where the last axis of the array must have length
        matching the volume dimension.
    :returns: a :class:`~meshmode.dof_array.DOFArray` or an
        :class:`~arraycontext.container.ArrayContainer` of them.
    """

    return _div_helper(
        dcoll,
        lambda i, subvec: local_d_dx(dcoll, i, subvec),
        vecs
    )

# }}}


# {{{ Weak derivative operators

def reference_stiffness_transpose_matrix(
        actx: ArrayContext, out_element_group, in_element_group):
    @keyed_memoize_in(
        actx, reference_stiffness_transpose_matrix,
        lambda out_grp, in_grp: (out_grp.discretization_key(),
                                 in_grp.discretization_key()))
    def get_ref_stiffness_transpose_mat(out_grp, in_grp):
        if in_grp == out_grp:
            from meshmode.discretization.poly_element import \
                mass_matrix, diff_matrices

            mmat = mass_matrix(out_grp)
            return actx.freeze(
                actx.from_numpy(
                    np.asarray(
                        [dmat.T @ mmat.T for dmat in diff_matrices(out_grp)]
                    )
                )
            )

        from modepy import vandermonde
        basis = out_grp.basis_obj()
        vand = vandermonde(basis.functions, out_grp.unit_nodes)
        grad_vand = vandermonde(basis.gradients, in_grp.unit_nodes)
        vand_inv_t = np.linalg.inv(vand).T

        if not isinstance(grad_vand, tuple):
            # NOTE: special case for 1d
            grad_vand = (grad_vand,)

        weights = in_grp.quadrature_rule().weights
        return actx.freeze(
            actx.from_numpy(
                np.einsum(
                    "c,bz,acz->abc",
                    weights,
                    vand_inv_t,
                    grad_vand
                ).copy()  # contigify the array
            )
        )
    return get_ref_stiffness_transpose_mat(out_element_group,
                                           in_element_group)


def weak_local_grad(
        dcoll: DiscretizationCollection, *args, nested=False) -> np.ndarray:
    r"""Return the element-local weak gradient of the volume function
    represented by *vec*.

    May be called with ``(vec)`` or ``(dd_in, vec)``.

    Specifically, the function returns an object array where the :math:`i`-th
    component is the weak derivative with respect to the :math:`i`-th coordinate
    of a scalar function :math:`f`. See :func:`weak_local_d_dx` for further
    information. For non-scalar :math:`f`, the function will return a nested object
    array containing the component-wise weak derivatives.

    :arg dd_in: a :class:`~grudge.dof_desc.DOFDesc`, or a value convertible to one,
        describing the type of quadrature discretization to be used in the volume.
        Defaults to the base volume discretization if not provided.
    :arg vec: a :class:`~meshmode.dof_array.DOFArray` or an
        :class:`~arraycontext.container.ArrayContainer` of them.
    :arg nested: return nested object arrays instead of a single multidimensional
        array if *vec* is non-scalar
    :returns: an object array (possibly nested) of
        :class:`~meshmode.dof_array.DOFArray`\ s or
        :class:`~arraycontext.container.ArrayContainer`\ s.
    """
    if len(args) == 1:
        vec, = args
        dd_in = dof_desc.DOFDesc("vol", dof_desc.DISCR_TAG_BASE)
    elif len(args) == 2:
        dd_in, vec = args
    else:
        raise TypeError("invalid number of arguments")

    if isinstance(vec, np.ndarray):
        grad = obj_array_vectorize(
                lambda el: weak_local_grad(dcoll, dd_in, el, nested=nested), vec)
        if nested:
            return grad
        else:
            return np.stack(grad, axis=0)

    from grudge.geometry import inverse_surface_metric_derivative_mat

    in_discr = dcoll.discr_from_dd(dd_in)
    out_discr = dcoll.discr_from_dd(dof_desc.DD_VOLUME)

    actx = vec.array_context
    inverse_jac_mat = inverse_surface_metric_derivative_mat(actx, dcoll, dd=dd_in,
            times_area_element=True,
            _use_geoderiv_connection=actx.supports_nonscalar_broadcasting)

    return _gradient_kernel(
        actx,
        out_discr,
        in_discr,
        reference_stiffness_transpose_matrix,
        inverse_jac_mat,
        vec,
        metric_in_matvec=True
    )


def weak_local_d_dx(dcoll: DiscretizationCollection, *args) -> ArrayOrContainerT:
    r"""Return the element-local weak derivative along axis *xyz_axis* of the
    volume function represented by *vec*.

    May be called with ``(xyz_axis, vec)`` or ``(dd_in, xyz_axis, vec)``.

    Specifically, this function computes the volume contribution of the
    weak derivative in the :math:`i`-th component (specified by *xyz_axis*)
    of a function :math:`f`, in each element :math:`E`, with respect to polynomial
    test functions :math:`\phi`:

    .. math::

        \int_E \partial_i\phi\,f\,\mathrm{d}x \sim
        \mathbf{D}_{E,i}^T \mathbf{M}_{E}^T\mathbf{f}|_E,

    where :math:`\mathbf{D}_{E,i}` is the polynomial differentiation matrix on
    an :math:`E` for the :math:`i`-th spatial coordinate, :math:`\mathbf{M}_E`
    is the elemental mass matrix (see :func:`mass` for more information), and
    :math:`\mathbf{f}|_E` is a vector of coefficients for :math:`f` on :math:`E`.

    :arg dd_in: a :class:`~grudge.dof_desc.DOFDesc`, or a value convertible to one,
        describing the type of quadrature discretization to be used in the volume.
        Defaults to the base volume discretization if not provided.
    :arg xyz_axis: an integer indicating the axis along which the derivative
        is taken.
    :arg vec: a :class:`~meshmode.dof_array.DOFArray` or an
        :class:`~arraycontext.container.ArrayContainer` of them.
    :returns: a :class:`~meshmode.dof_array.DOFArray` or an
        :class:`~arraycontext.container.ArrayContainer` of them.
    """
    if len(args) == 2:
        xyz_axis, vec = args
        dd_in = dof_desc.DOFDesc("vol", dof_desc.DISCR_TAG_BASE)
    elif len(args) == 3:
        dd_in, xyz_axis, vec = args
    else:
        raise TypeError("invalid number of arguments")

    from grudge.geometry import inverse_surface_metric_derivative_mat

    in_discr = dcoll.discr_from_dd(dd_in)
    out_discr = dcoll.discr_from_dd(dof_desc.DD_VOLUME)

    actx = vec.array_context
    inverse_jac_mat = inverse_surface_metric_derivative_mat(actx, dcoll, dd=dd_in,
            times_area_element=True,
            _use_geoderiv_connection=actx.supports_nonscalar_broadcasting)

    return _single_axis_derivative_kernel(
        actx,
        out_discr,
        in_discr,
        reference_stiffness_transpose_matrix,
        inverse_jac_mat,
        xyz_axis,
        vec,
        metric_in_matvec=True
    )


def weak_local_div(dcoll: DiscretizationCollection, *args) -> ArrayOrContainerT:
    r"""Return the element-local weak divergence of the vector volume function
    represented by *vecs*.

    May be called with ``(vecs)`` or ``(dd_in, vecs)``.

    Specifically, this function computes the volume contribution of the
    weak divergence of a vector function :math:`\mathbf{f}`, in each element
    :math:`E`, with respect to polynomial test functions :math:`\phi`:

    .. math::

        \int_E \nabla \phi \cdot \mathbf{f}\,\mathrm{d}x \sim
        \sum_{i=1}^d \mathbf{D}_{E,i}^T \mathbf{M}_{E}^T\mathbf{f}_i|_E,

    where :math:`\mathbf{D}_{E,i}` is the polynomial differentiation matrix on
    an :math:`E` for the :math:`i`-th spatial coordinate, and :math:`\mathbf{M}_E`
    is the elemental mass matrix (see :func:`mass` for more information).

    :arg dd_in: a :class:`~grudge.dof_desc.DOFDesc`, or a value convertible to one,
        describing the type of quadrature discretization to be used in the volume.
        Defaults to the base volume discretization if not provided.
    :arg vecs: an object array of
        :class:`~meshmode.dof_array.DOFArray`\s or
        :class:`~arraycontext.container.ArrayContainer` objects,
        where the last axis of the array must have length
        matching the volume dimension.
    :returns: a :class:`~meshmode.dof_array.DOFArray` or an
        :class:`~arraycontext.container.ArrayContainer` of them.
    """
    if len(args) == 1:
        vecs, = args
        dd_in = dof_desc.DOFDesc("vol", dof_desc.DISCR_TAG_BASE)
    elif len(args) == 2:
        dd_in, vecs = args
    else:
        raise TypeError("invalid number of arguments")

    return _div_helper(
        dcoll,
        lambda i, subvec: weak_local_d_dx(dcoll, dd_in, i, subvec),
        vecs
    )

# }}}


# {{{ Mass operator

def reference_mass_matrix(actx: ArrayContext, out_element_group, in_element_group):
    @keyed_memoize_in(
        actx, reference_mass_matrix,
        lambda out_grp, in_grp: (out_grp.discretization_key(),
                                 in_grp.discretization_key()))
    def get_ref_mass_mat(out_grp, in_grp):
        if out_grp == in_grp:
            from meshmode.discretization.poly_element import mass_matrix

            return actx.freeze(
                actx.from_numpy(
                    np.asarray(
                        mass_matrix(out_grp),
                        order="C"
                    )
                )
            )

        from modepy import vandermonde
        basis = out_grp.basis_obj()
        vand = vandermonde(basis.functions, out_grp.unit_nodes)
        o_vand = vandermonde(basis.functions, in_grp.unit_nodes)
        vand_inv_t = np.linalg.inv(vand).T

        weights = in_grp.quadrature_rule().weights
        return actx.freeze(
            actx.from_numpy(
                np.asarray(
                    np.einsum("j,ik,jk->ij", weights, vand_inv_t, o_vand),
                    order="C"
                )
            )
        )

    return get_ref_mass_mat(out_element_group, in_element_group)


def _apply_mass_operator(
        dcoll: DiscretizationCollection, dd_out, dd_in, vec):
    if not isinstance(vec, DOFArray):
        return map_array_container(
            partial(_apply_mass_operator, dcoll, dd_out, dd_in), vec
        )

    from grudge.geometry import area_element

    in_discr = dcoll.discr_from_dd(dd_in)
    out_discr = dcoll.discr_from_dd(dd_out)

    actx = vec.array_context
    area_elements = area_element(actx, dcoll, dd=dd_in,
            _use_geoderiv_connection=actx.supports_nonscalar_broadcasting)

    return DOFArray(
        actx,
        data=tuple(
            actx.einsum("ij,ej,ej->ei",
                        reference_mass_matrix(
                            actx,
                            out_element_group=out_grp,
                            in_element_group=in_grp
                        ),
                        ae_i,
                        vec_i,
                        arg_names=("mass_mat", "jac", "vec"),
                        tagged=(FirstAxisIsElementsTag(),))

            for in_grp, out_grp, ae_i, vec_i in zip(
                in_discr.groups,
                out_discr.groups,
                area_elements,
                vec
            )
        )
    )


def mass(dcoll: DiscretizationCollection, *args) -> ArrayOrContainerT:
    r"""Return the action of the DG mass matrix on a vector (or vectors)
    of :class:`~meshmode.dof_array.DOFArray`\ s, *vec*. In the case of
    *vec* being an :class:`~arraycontext.container.ArrayContainer`,
    the mass operator is applied component-wise.

    May be called with ``(vec)`` or ``(dd_in, vec)``.

    Specifically, this function applies the mass matrix elementwise on a
    vector of coefficients :math:`\mathbf{f}` via:
    :math:`\mathbf{M}_{E}\mathbf{f}|_E`, where

    .. math::

        \left(\mathbf{M}_{E}\right)_{ij} = \int_E \phi_i \cdot \phi_j\,\mathrm{d}x,

    where :math:`\phi_i` are local polynomial basis functions on :math:`E`.

    :arg dd_in: a :class:`~grudge.dof_desc.DOFDesc`, or a value convertible to one,
        describing the type of quadrature discretization to be used in the volume.
        Defaults to the base volume discretization if not provided.
    :arg vec: a :class:`~meshmode.dof_array.DOFArray` or an
        :class:`~arraycontext.container.ArrayContainer` of them.
    :returns: a :class:`~meshmode.dof_array.DOFArray` or an
        :class:`~arraycontext.container.ArrayContainer` of them.
    """

    if len(args) == 1:
        vec, = args
        dd_in = dof_desc.DOFDesc("vol", dof_desc.DISCR_TAG_BASE)
    elif len(args) == 2:
        dd_in, vec = args
    else:
        raise TypeError("invalid number of arguments")

    return _apply_mass_operator(dcoll, dof_desc.DD_VOLUME, dd_in, vec)

# }}}


# {{{ Mass inverse operator

def reference_inverse_mass_matrix(actx: ArrayContext, element_group):
    @keyed_memoize_in(
        actx, reference_inverse_mass_matrix,
        lambda grp: grp.discretization_key())
    def get_ref_inv_mass_mat(grp):
        from modepy import inverse_mass_matrix
        basis = grp.basis_obj()

        return actx.freeze(
            actx.from_numpy(
                np.asarray(
                    inverse_mass_matrix(basis.functions, grp.unit_nodes),
                    order="C"
                )
            )
        )

    return get_ref_inv_mass_mat(element_group)


def _apply_inverse_mass_operator(
        dcoll: DiscretizationCollection, dd_out, dd_in, vec):
    if not isinstance(vec, DOFArray):
        return map_array_container(
            partial(_apply_inverse_mass_operator, dcoll, dd_out, dd_in), vec
        )

    from grudge.geometry import area_element

    actx = vec.array_context
    discr = dcoll.discr_from_dd(dd_out)
    inv_area_elements = 1./area_element(actx, dcoll, dd=dd_out,
            _use_geoderiv_connection=actx.supports_nonscalar_broadcasting)
    group_data = []
    for grp, jac_inv, vec_i in zip(discr.groups, inv_area_elements, vec):

        ref_mass_inverse = reference_inverse_mass_matrix(actx,
                                                         element_group=grp)

        # FIXME: Use overintegration in WADG for non-affine groups
        # based on https://arxiv.org/pdf/1608.03836.pdf. That is,
        # use *dd_in* to compute non-affine geometry terms
        # and evaluate ref_Minv * ref_M * (1/jac_det) * ref_Minv.
        # Related issue: https://github.com/inducer/grudge/issues/173
        group_data.append(
            actx.einsum("ei,ij,ej->ei",
                        jac_inv,
                        ref_mass_inverse,
                        vec_i,
                        tagged=(FirstAxisIsElementsTag(),))
        )

    return DOFArray(actx, data=tuple(group_data))


def inverse_mass(dcoll: DiscretizationCollection, *args) -> ArrayOrContainerT:
    r"""Return the action of the DG mass matrix inverse on a vector
    (or vectors) of :class:`~meshmode.dof_array.DOFArray`\ s, *vec*.
    In the case of *vec* being an :class:`~arraycontext.container.ArrayContainer`,
    the inverse mass operator is applied component-wise.

    May be called with ``(vec)`` or ``(dd_in, vec)``.

    For affine elements :math:`E`, the element-wise mass inverse
    is computed directly as the inverse of the (physical) mass matrix:

    .. math::

        \left(\mathbf{M}_{J^e}\right)_{ij} =
            \int_{\widehat{E}} \widehat{\phi}_i\cdot\widehat{\phi}_j J^e
            \mathrm{d}\widehat{x},

    where :math:`\widehat{\phi}_i` are basis functions over the reference
    element :math:`\widehat{E}`, and :math:`J^e` is the (constant) Jacobian
    scaling factor (see :func:`grudge.geometry.area_element`).

    For non-affine :math:`E`, :math:`J^e` is not constant. In this case, a
    weight-adjusted approximation is used instead following [Chan_2016]_:

    .. math::

        \mathbf{M}_{J^e}^{-1} \approx
            \widehat{\mathbf{M}}^{-1}\mathbf{M}_{1/J^e}\widehat{\mathbf{M}}^{-1},

    where :math:`\widehat{\mathbf{M}}` is the reference mass matrix on
    :math:`\widehat{E}`.

    :arg dd_in: a :class:`~grudge.dof_desc.DOFDesc`, or a value convertible to one,
        describing the type of quadrature discretization to be used in the volume.
        Defaults to the base volume discretization if not provided.
    :arg vec: a :class:`~meshmode.dof_array.DOFArray` or an
        :class:`~arraycontext.container.ArrayContainer` of them.
    :returns: a :class:`~meshmode.dof_array.DOFArray` or an
        :class:`~arraycontext.container.ArrayContainer` of them.
    """

    if len(args) == 1:
        vec, = args
        dd_in = dof_desc.DD_VOLUME
    elif len(args) == 2:
        dd_in, vec = args
    else:
        raise TypeError("invalid number of arguments")

    return _apply_inverse_mass_operator(
        dcoll, dof_desc.DD_VOLUME, dd_in, vec
    )

# }}}


# {{{ Face mass operator

def reference_face_mass_matrix(
        actx: ArrayContext, face_element_group, vol_element_group, dtype):
    @keyed_memoize_in(
        actx, reference_face_mass_matrix,
        lambda face_grp, vol_grp: (face_grp.discretization_key(),
                                   vol_grp.discretization_key()))
    def get_ref_face_mass_mat(face_grp, vol_grp):
        nfaces = vol_grp.mesh_el_group.nfaces
        assert face_grp.nelements == nfaces * vol_grp.nelements

        matrix = np.empty(
            (vol_grp.nunit_dofs,
            nfaces,
            face_grp.nunit_dofs),
            dtype=dtype
        )

        import modepy as mp
        from meshmode.discretization import ElementGroupWithBasis
        from meshmode.discretization.poly_element import \
            QuadratureSimplexElementGroup

        n = vol_grp.order
        m = face_grp.order
        vol_basis = vol_grp.basis_obj()
        faces = mp.faces_for_shape(vol_grp.shape)

        for iface, face in enumerate(faces):
            # If the face group is defined on a higher-order
            # quadrature grid, use the underlying quadrature rule
            if isinstance(face_grp, QuadratureSimplexElementGroup):
                face_quadrature = face_grp.quadrature_rule()
                if face_quadrature.exact_to < m:
                    raise ValueError(
                        "The face quadrature rule is only exact for polynomials "
                        f"of total degree {face_quadrature.exact_to}. Please "
                        "ensure a quadrature rule is used that is at least "
                        f"exact for degree {m}."
                    )
            else:
                # NOTE: This handles the general case where
                # volume and surface quadrature rules may have different
                # integration orders
                face_quadrature = mp.quadrature_for_space(
                    mp.space_for_shape(face, 2*max(n, m)),
                    face
                )

            # If the group has a nodal basis and is unisolvent,
            # we use the basis on the face to compute the face mass matrix
            if (isinstance(face_grp, ElementGroupWithBasis)
                    and face_grp.space.space_dim == face_grp.nunit_dofs):

                face_basis = face_grp.basis_obj()

                # Sanity check for face quadrature accuracy. Not integrating
                # degree N + M polynomials here is asking for a bad time.
                if face_quadrature.exact_to < m + n:
                    raise ValueError(
                        "The face quadrature rule is only exact for polynomials "
                        f"of total degree {face_quadrature.exact_to}. Please "
                        "ensure a quadrature rule is used that is at least "
                        f"exact for degree {n+m}."
                    )

                matrix[:, iface, :] = mp.nodal_mass_matrix_for_face(
                    face, face_quadrature,
                    face_basis.functions, vol_basis.functions,
                    vol_grp.unit_nodes,
                    face_grp.unit_nodes,
                )
            else:
                # Otherwise, we use a routine that is purely quadrature-based
                # (no need for explicit face basis functions)
                matrix[:, iface, :] = mp.nodal_quad_mass_matrix_for_face(
                    face,
                    face_quadrature,
                    vol_basis.functions,
                    vol_grp.unit_nodes,
                )

        return actx.freeze(actx.from_numpy(matrix))

    return get_ref_face_mass_mat(face_element_group, vol_element_group)


def _apply_face_mass_operator(dcoll: DiscretizationCollection, dd_in, vec):
    if not isinstance(vec, DOFArray):
        return map_array_container(
            partial(_apply_face_mass_operator, dcoll, dd_in), vec
        )

    from grudge.geometry import area_element

    volm_discr = dcoll.discr_from_dd(dof_desc.DD_VOLUME)
    face_discr = dcoll.discr_from_dd(dd_in)
    dtype = vec.entry_dtype
    actx = vec.array_context

    assert len(face_discr.groups) == len(volm_discr.groups)
    surf_area_elements = area_element(actx, dcoll, dd=dd_in,
            _use_geoderiv_connection=actx.supports_nonscalar_broadcasting)

    return DOFArray(
        actx,
        data=tuple(
            actx.einsum("ifj,fej,fej->ei",
                        reference_face_mass_matrix(
                                actx,
                                face_element_group=afgrp,
                                vol_element_group=vgrp,
                                dtype=dtype),
                        surf_ae_i.reshape(
                                vgrp.mesh_el_group.nfaces,
                                vgrp.nelements,
                                -1),
                        vec_i.reshape(
                                vgrp.mesh_el_group.nfaces,
                                vgrp.nelements,
                                afgrp.nunit_dofs),
                        arg_names=("ref_face_mass_mat", "jac_surf", "vec"),
                        tagged=(FirstAxisIsElementsTag(),))

            for vgrp, afgrp, vec_i, surf_ae_i in zip(
                volm_discr.groups,
                face_discr.groups,
                vec,
                surf_area_elements
            )
        )
    )


def face_mass(dcoll: DiscretizationCollection, *args) -> ArrayOrContainerT:
    r"""Return the action of the DG face mass matrix on a vector (or vectors)
    of :class:`~meshmode.dof_array.DOFArray`\ s, *vec*. In the case of
    *vec* being an arbitrary :class:`~arraycontext.container.ArrayContainer`,
    the face mass operator is applied component-wise.

    May be called with ``(vec)`` or ``(dd_in, vec)``.

    Specifically, this function applies the face mass matrix elementwise on a
    vector of coefficients :math:`\mathbf{f}` as the sum of contributions for
    each face :math:`f \subset \partial E`:

    .. math::

        \sum_{f=1}^{N_{\text{faces}} } \mathbf{M}_{f, E}\mathbf{f}|_f,

    where

    .. math::

        \left(\mathbf{M}_{f, E}\right)_{ij} =
            \int_{f \subset \partial E} \phi_i(s)\psi_j(s)\,\mathrm{d}s,

    where :math:`\phi_i` are (volume) polynomial basis functions on :math:`E`
    evaluated on the face :math:`f`, and :math:`\psi_j` are basis functions for
    a polynomial space defined on :math:`f`.

    :arg dd_in: a :class:`~grudge.dof_desc.DOFDesc`, or a value convertible to one,
        describing the type of quadrature discretization on the skeleton of the
        mesh (``"all_faces"``).
        Defaults to the base discretization if not provided.
    :arg vec: a :class:`~meshmode.dof_array.DOFArray` or an
        :class:`~arraycontext.container.ArrayContainer` of them.
    :returns: a :class:`~meshmode.dof_array.DOFArray` or an
        :class:`~arraycontext.container.ArrayContainer` of them.
    """

    if len(args) == 1:
        vec, = args
        dd_in = dof_desc.DOFDesc("all_faces", dof_desc.DISCR_TAG_BASE)
    elif len(args) == 2:
        dd_in, vec = args
    else:
        raise TypeError("invalid number of arguments")

    return _apply_face_mass_operator(dcoll, dd_in, vec)

# }}}


# vim: foldmethod=marker
