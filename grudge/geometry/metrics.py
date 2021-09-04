"""
.. currentmodule:: grudge.geometry

Coordinate transformations
--------------------------

.. autofunction:: forward_metric_nth_derivative
.. autofunction:: forward_metric_derivative_mat
.. autofunction:: inverse_metric_derivative_mat

.. autofunction:: first_fundamental_form
.. autofunction:: inverse_first_fundamental_form

Geometry terms
--------------

.. autofunction:: inverse_surface_metric_derivative
.. autofunction:: pseudoscalar
.. autofunction:: area_element

Normal vectors
--------------

.. autofunction:: normal

Curvature tensors
-----------------

.. autofunction:: second_fundamental_form
.. autofunction:: shape_operator
.. autofunction:: summed_curvature
"""

__copyright__ = """
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


import numpy as np

from arraycontext import thaw, freeze, ArrayContext
from meshmode.dof_array import DOFArray

from grudge import DiscretizationCollection
import grudge.dof_desc as dof_desc

from grudge.dof_desc import (
    DD_VOLUME, DOFDesc, DISCR_TAG_BASE
)

from pymbolic.geometric_algebra import MultiVector

from pytools.obj_array import make_obj_array
from pytools import memoize_in


from arraycontext import register_multivector_as_array_container
register_multivector_as_array_container()


# {{{ Metric computations

def forward_metric_nth_derivative(
        actx: ArrayContext, dcoll: DiscretizationCollection,
        xyz_axis, ref_axes, dd=None) -> DOFArray:
    r"""Pointwise metric derivatives representing repeated derivatives of the
    physical coordinate enumerated by *xyz_axis*: :math:`x_{\mathrm{xyz\_axis}}`
    with respect to the coordiantes on the reference element :math:`\xi_i`:

    .. math::

        D^\alpha x_{\mathrm{xyz\_axis}} =
        \frac{\partial^{|\alpha|} x_{\mathrm{xyz\_axis}} }{
            \partial \xi_1^{\alpha_1}\cdots \partial \xi_m^{\alpha_m}}

    where :math:`\alpha` is a multi-index described by *ref_axes*.

    :arg xyz_axis: an integer denoting which physical coordinate to
        differentiate.
    :arg ref_axes: a :class:`tuple` of tuples indicating indices of
        coordinate axes of the reference element to the number of derivatives
        which will be taken. For example, the value ``((0, 2), (1, 1))``
        indicates taking the second derivative with respect to the first
        axis and the first derivative with respect to the second
        axis. Each axis must occur only once and the tuple must be sorted
        by the axis index.
    :arg dd: a :class:`~grudge.dof_desc.DOFDesc`, or a value convertible to one.
        Defaults to the base volume discretization.
    :returns: a :class:`~meshmode.dof_array.DOFArray` containing the pointwise
        metric derivative at each nodal coordinate.
    """
    if dd is None:
        dd = DD_VOLUME

    inner_dd = dd.with_discr_tag(DISCR_TAG_BASE)

    if isinstance(ref_axes, int):
        ref_axes = ((ref_axes, 1),)

    if not isinstance(ref_axes, tuple):
        raise ValueError("ref_axes must be a tuple")

    if tuple(sorted(ref_axes)) != ref_axes:
        raise ValueError("ref_axes must be sorted")

    if len(set(ref_axes)) != len(ref_axes):
        raise ValueError("ref_axes must not contain an axis more than once")

    from pytools import flatten
    flat_ref_axes = flatten([rst_axis] * n for rst_axis, n in ref_axes)

    from meshmode.discretization import num_reference_derivative

    vec = num_reference_derivative(
        dcoll.discr_from_dd(inner_dd),
        flat_ref_axes,
        thaw(dcoll.discr_from_dd(inner_dd).nodes(), actx)[xyz_axis]
    )

    if dd.uses_quadrature():
        vec = dcoll.connection_from_dds(inner_dd, dd)(vec)

    return vec


def forward_metric_derivative_vector(
        actx: ArrayContext, dcoll: DiscretizationCollection, rst_axis, dd=None
        ) -> np.ndarray:
    r"""Computes an object array containing the forward metric derivatives
    of each physical coordinate.

    :arg rst_axis: a :class:`tuple` of tuples indicating indices of
        coordinate axes of the reference element to the number of derivatives
        which will be taken.
    :arg dd: a :class:`~grudge.dof_desc.DOFDesc`, or a value convertible to one.
        Defaults to the base volume discretization.
    :returns: an object array of :class:`~meshmode.dof_array.DOFArray`\ s
        containing the pointwise metric derivatives at each nodal coordinate.
    """
    return make_obj_array([
        forward_metric_nth_derivative(actx, dcoll, i, rst_axis, dd=dd)
        for i in range(dcoll.ambient_dim)
        ]
    )


def forward_metric_derivative_mv(
        actx: ArrayContext, dcoll: DiscretizationCollection, rst_axis, dd=None
        ) -> MultiVector:
    r"""Computes a :class:`pymbolic.geometric_algebra.MultiVector` containing
    the forward metric derivatives of each physical coordinate.

    :arg rst_axis: a :class:`tuple` of tuples indicating indices of
        coordinate axes of the reference element to the number of derivatives
        which will be taken.
    :arg dd: a :class:`~grudge.dof_desc.DOFDesc`, or a value convertible to one.
        Defaults to the base volume discretization.
    :returns: a :class:`pymbolic.geometric_algebra.MultiVector` containing
        the forward metric derivatives in each physical coordinate.
    """
    return MultiVector(
        forward_metric_derivative_vector(actx, dcoll, rst_axis, dd=dd)
    )


def forward_metric_derivative_mat(
        actx: ArrayContext, dcoll: DiscretizationCollection, dd=None
        ) -> np.ndarray:
    r"""Computes the forward metric derivative matrix, also commonly
    called the Jacobian matrix, with entries defined as the
    forward metric derivatives:

    .. math::

        J = \left\lbrack
            \frac{\partial x_i}{\partial \xi_j}
            \right\rbrack_{(0, 0) \leq (i, j) \leq (n, m)}

    where :math:`x_1, \dots, x_n` denote the physical coordinates and
    :math:`\xi_1, \dots, \xi_m` denote coordinates on the reference element.
    Note that, in the case of immersed manifolds, `J` is not necessarily
    a square matrix.

    :arg dd: a :class:`~grudge.dof_desc.DOFDesc`, or a value convertible to one.
        Defaults to the base volume discretization.
    :returns: a matrix containing the evaluated forward metric derivatives
        of each physical coordinate, with respect to each reference coordinate.
    """
    ambient_dim = dcoll.ambient_dim

    if dd is None:
        dd = DD_VOLUME

    dim = dcoll.discr_from_dd(dd).dim

    result = np.zeros((ambient_dim, dim), dtype=object)
    for j in range(dim):
        result[:, j] = forward_metric_derivative_vector(actx, dcoll, j, dd=dd)

    return result


def first_fundamental_form(actx: ArrayContext, dcoll: DiscretizationCollection,
        dd=None) -> np.ndarray:
    r"""Computes the first fundamental form using the Jacobian matrix:

    .. math::

        \begin{bmatrix}
            E & F \\ F & G
        \end{bmatrix} :=
        \begin{bmatrix}
            (\partial_u x)^2 & \partial_u x \partial_v x \\
            \partial_u x \partial_v x & (\partial_v x)^2
        \end{bmatrix} =
        J^T \cdot J

    where :math:`u, v` are coordinates on the parameterized surface and
    :math:`x(u, v)` defines a parameterized region. Here, :math:`J` is the
    corresponding Jacobian matrix.

    :arg dd: a :class:`~grudge.dof_desc.DOFDesc`, or a value convertible to one.
        Defaults to the base volume discretization.
    :returns: a matrix containing coefficients of the first fundamental
        form.
    """
    if dd is None:
        dd = DD_VOLUME

    mder = forward_metric_derivative_mat(actx, dcoll, dd=dd)

    return mder.T.dot(mder)


def inverse_metric_derivative_mat(
        actx: ArrayContext, dcoll: DiscretizationCollection, dd=None
        ) -> np.ndarray:
    r"""Computes the inverse metric derivative matrix, which is
    the inverse of the Jacobian (forward metric derivative) matrix.

    :arg dd: a :class:`~grudge.dof_desc.DOFDesc`, or a value convertible to one.
        Defaults to the base volume discretization.
    :returns: a matrix containing the evaluated inverse metric derivatives.
    """
    ambient_dim = dcoll.ambient_dim

    if dd is None:
        dd = DD_VOLUME

    dim = dcoll.discr_from_dd(dd).dim

    result = np.zeros((ambient_dim, dim), dtype=object)
    for i in range(dim):
        for j in range(ambient_dim):
            result[i, j] = inverse_metric_derivative(
                actx, dcoll, i, j, dd=dd
            )

    return result


def inverse_first_fundamental_form(
        actx: ArrayContext, dcoll: DiscretizationCollection, dd=None
        ) -> np.ndarray:
    r"""Computes the inverse of the first fundamental form:

    .. math::

        \begin{bmatrix}
            E & F \\ F & G
        \end{bmatrix}^{-1} =
        \frac{1}{E G - F^2}
        \begin{bmatrix}
            G & -F \\ -F & E
        \end{bmatrix}

    where :math:`E, F, G` are coefficients of the first fundamental form.

    :arg dd: a :class:`~grudge.dof_desc.DOFDesc`, or a value convertible to one.
        Defaults to the base volume discretization.
    :returns: a matrix containing coefficients of the inverse of the
        first fundamental form.
    """
    if dd is None:
        dd = DD_VOLUME

    dim = dcoll.discr_from_dd(dd).dim

    if dcoll.ambient_dim == dim:
        inv_mder = inverse_metric_derivative_mat(actx, dcoll, dd=dd)
        inv_form1 = inv_mder.dot(inv_mder.T)
    else:
        form1 = first_fundamental_form(actx, dcoll, dd=dd)

        if dim == 1:
            inv_form1 = 1.0 / form1
        elif dim == 2:
            (E, F), (_, G) = form1      # noqa: N806
            inv_form1 = 1.0 / (E * G - F * F) * np.stack(
                [make_obj_array([G, -F]),
                 make_obj_array([-F, E])]
            )
        else:
            raise ValueError(f"{dim}D surfaces not supported" % dim)

    return inv_form1


def inverse_metric_derivative(
        actx: ArrayContext, dcoll: DiscretizationCollection, rst_axis, xyz_axis, dd
        ) -> DOFArray:
    r"""Computes the inverse metric derivative of the physical
    coordinate enumerated by *xyz_axis* with respect to the
    reference axis *rst_axis*.

    :arg rst_axis: an integer denoting the reference coordinate axis.
    :arg xyz_axis: an integer denoting the physical coordinate axis.
    :arg dd: a :class:`~grudge.dof_desc.DOFDesc`, or a value convertible to one.
        Defaults to the base volume discretization.
    :returns: a :class:`~meshmode.dof_array.DOFArray` containing the
        inverse metric derivative at each nodal coordinate.
    """

    dim = dcoll.dim
    if dim != dcoll.ambient_dim:
        raise ValueError(
            "Not clear what inverse_metric_derivative means if "
            "the derivative matrix is not square!"
        )

    par_vecs = [forward_metric_derivative_mv(actx, dcoll, rst, dd)
                for rst in range(dim)]

    # Yay Cramer's rule!
    from functools import reduce, partial
    from operator import xor as outerprod_op
    outerprod = partial(reduce, outerprod_op)

    def outprod_with_unit(i, at):
        unit_vec = np.zeros(dim)
        unit_vec[i] = 1

        vecs = par_vecs[:]
        vecs[at] = MultiVector(unit_vec)

        return outerprod(vecs)

    volume_pseudoscalar_inv = outerprod(
        forward_metric_derivative_mv(actx, dcoll, rst_axis, dd)
        for rst_axis in range(dim)
    ).inv()

    result = (outprod_with_unit(xyz_axis, rst_axis)
              * volume_pseudoscalar_inv).as_scalar()

    return result


def inverse_surface_metric_derivative(
        actx: ArrayContext, dcoll: DiscretizationCollection,
        rst_axis, xyz_axis, dd=None):
    r"""Computes the inverse surface metric derivative of the physical
    coordinate enumerated by *xyz_axis* with respect to the
    reference axis *rst_axis*. These geometric terms are used in the
    transformation of physical gradients.

    :arg rst_axis: an integer denoting the reference coordinate axis.
    :arg xyz_axis: an integer denoting the physical coordinate axis.
    :arg dd: a :class:`~grudge.dof_desc.DOFDesc`, or a value convertible to one.
        Defaults to the base volume discretization.
    :returns: a :class:`~meshmode.dof_array.DOFArray` containing the
        inverse metric derivative at each nodal coordinate.
    """
    dim = dcoll.dim
    ambient_dim = dcoll.ambient_dim

    if dd is None:
        dd = DD_VOLUME
    dd = dof_desc.as_dofdesc(dd)

    @memoize_in(dcoll, (inverse_surface_metric_derivative, dd,
                        rst_axis, xyz_axis, actx.supports_nonscalar_broadcasting))
    def _inv_surf_metric_deriv():
        if ambient_dim == dim:
            imd = inverse_metric_derivative(
                actx, dcoll, rst_axis, xyz_axis, dd=dd
            )
        else:
            inv_form1 = inverse_first_fundamental_form(actx, dcoll, dd=dd)
            imd = sum(
                inv_form1[rst_axis, d]*forward_metric_nth_derivative(
                    actx, dcoll, xyz_axis, d, dd=dd
                ) for d in range(dim)
            )

        if actx.supports_nonscalar_broadcasting:
            imd = dcoll._base_to_geoderiv_connection(dd)(imd)

        return freeze(imd, actx)

    return thaw(_inv_surf_metric_deriv(), actx)


def _signed_face_ones(
        actx: ArrayContext, dcoll: DiscretizationCollection, dd
        ) -> DOFArray:

    assert dd.is_trace()

    # NOTE: ignore quadrature_tags on dd, since we only care about
    # the face_id here
    all_faces_conn = dcoll.connection_from_dds(
        DD_VOLUME, DOFDesc(dd.domain_tag)
    )
    signed_face_ones = dcoll.discr_from_dd(dd).zeros(
        actx, dtype=dcoll.real_dtype
    ) + 1
    for igrp, grp in enumerate(all_faces_conn.groups):
        for batch in grp.batches:
            i = actx.thaw(batch.to_element_indices)
            grp_field = signed_face_ones[igrp].reshape(-1)
            grp_field[i] = \
                (2.0 * (batch.to_element_face % 2) - 1.0) * grp_field[i]

    return signed_face_ones


def parametrization_derivative(
        actx: ArrayContext, dcoll: DiscretizationCollection, dd
        ) -> MultiVector:
    r"""Computes the product of forward metric derivatives spanning the
    tangent space with topological dimension *dim*.

    :arg dd: a :class:`~grudge.dof_desc.DOFDesc`, or a value convertible to one.
        Defaults to the base volume discretization.
    :returns: a :class:`pymbolic.geometric_algebra.MultiVector` containing
        the product of metric derivatives.
    """
    if dd is None:
        dd = DD_VOLUME

    dim = dcoll.discr_from_dd(dd).dim
    if dim == 0:
        from pymbolic.geometric_algebra import get_euclidean_space

        return MultiVector(
            _signed_face_ones(actx, dcoll, dd),
            space=get_euclidean_space(dcoll.ambient_dim)
        )

    from pytools import product

    return product(
        forward_metric_derivative_mv(actx, dcoll, rst_axis, dd)
        for rst_axis in range(dim)
    )


def pseudoscalar(actx: ArrayContext, dcoll: DiscretizationCollection,
        dd=None) -> MultiVector:
    r"""Computes the field of pseudoscalars for the domain/discretization
    identified by *dd*.

    :arg dd: a :class:`~grudge.dof_desc.DOFDesc`, or a value convertible to one.
        Defaults to the base volume discretization.
    :returns: A :class:`~pymbolic.geometric_algebra.MultiVector` of
        :class:`~meshmode.dof_array.DOFArray`\ s.
    """
    if dd is None:
        dd = DD_VOLUME

    return parametrization_derivative(actx, dcoll, dd).project_max_grade()


def area_element(
        actx: ArrayContext, dcoll: DiscretizationCollection, dd=None
        ) -> DOFArray:
    r"""Computes the scale factor used to transform integrals from reference
    to global space.

    :arg dd: a :class:`~grudge.dof_desc.DOFDesc`, or a value convertible to one.
        Defaults to the base volume discretization.
    :returns: a :class:`~meshmode.dof_array.DOFArray` containing the transformed
        volumes for each element.
    """
    if dd is None:
        dd = DD_VOLUME

    @memoize_in(dcoll, (area_element, dd, actx.supports_nonscalar_broadcasting))
    def _area_elements():
        result = actx.np.sqrt(
            pseudoscalar(actx, dcoll, dd=dd).norm_squared())

        if actx.supports_nonscalar_broadcasting:
            result = dcoll._base_to_geoderiv_connection(dd)(result)

        return freeze(result, actx)

    return thaw(_area_elements(), actx)

# }}}


# {{{ Surface normal vectors

def rel_mv_normal(
        actx: ArrayContext, dcoll: DiscretizationCollection, dd=None) -> MultiVector:
    r"""Computes surface normals at each nodal location as a
    :class:`~pymbolic.geometric_algebra.MultiVector` relative to the
    pseudoscalar of the discretization described by *dd*.

    :arg dd: a :class:`~grudge.dof_desc.DOFDesc`, or a value convertible to one.
    """

    dd = dof_desc.as_dofdesc(dd)

    # NOTE: Don't be tempted to add a sign here. As it is, it produces
    # exterior normals for positively oriented curves.

    pder = pseudoscalar(actx, dcoll, dd=dd) / area_element(actx, dcoll, dd=dd)

    # Dorst Section 3.7.2
    return pder << pder.I.inv()


def mv_normal(
        actx: ArrayContext, dcoll: DiscretizationCollection, dd,
        ) -> MultiVector:
    """Exterior unit normal as a :class:`~pymbolic.geometric_algebra.MultiVector`.
    This supports both volume discretizations
    (where ambient == topological dimension) and surface discretizations
    (where ambient == topological dimension + 1). In the latter case, extra
    processing ensures that the returned normal is in the local tangent space
    of the element at the point where the normal is being evaluated.

    :arg dd: a :class:`~grudge.dof_desc.DOFDesc` as the surface discretization.
    :returns: a :class:`~pymbolic.geometric_algebra.MultiVector`
        containing the unit normals.
    """
    dd = dof_desc.as_dofdesc(dd)

    @memoize_in(dcoll, (mv_normal, dd, actx.supports_nonscalar_broadcasting))
    def _normal():
        dim = dcoll.discr_from_dd(dd).dim
        ambient_dim = dcoll.ambient_dim

        if dim == ambient_dim:
            raise ValueError("may only request normals on domains whose topological "
                    f"dimension ({dim}) differs from "
                    f"their ambient dimension ({ambient_dim})")

        if dim == ambient_dim - 1:
            result = rel_mv_normal(actx, dcoll, dd=dd)
        else:
            # NOTE: In the case of (d - 2)-dimensional curves, we don't really have
            # enough information on the face to decide what an "exterior face normal"
            # is (e.g the "normal" to a 1D curve in 3D space is actually a
            # "normal plane")
            #
            # The trick done here is that we take the surface normal, move it to the
            # face and then take a cross product with the face tangent to get the
            # correct exterior face normal vector.
            assert dim == ambient_dim - 2

            from grudge.op import project

            volm_normal = MultiVector(
                project(dcoll, dof_desc.DD_VOLUME, dd,
                        rel_mv_normal(
                            actx, dcoll,
                            dd=dof_desc.DD_VOLUME
                        ).as_vector(dtype=object))
            )
            pder = pseudoscalar(actx, dcoll, dd=dd)

            mv = -(volm_normal ^ pder) << volm_normal.I.inv()

            result = mv / actx.np.sqrt(mv.norm_squared())

        if actx.supports_nonscalar_broadcasting:
            result = dcoll._base_to_geoderiv_connection(dd)(result)

        return freeze(result, actx)

    n = _normal()
    return thaw(n, actx)


def normal(actx: ArrayContext, dcoll: DiscretizationCollection, dd):
    """Get the unit normal to the specified surface discretization, *dd*.
    This supports both volume discretizations
    (where ambient == topological dimension) and surface discretizations
    (where ambient == topological dimension + 1). In the latter case, extra
    processing ensures that the returned normal is in the local tangent space
    of the element at the point where the normal is being evaluated.

    :arg dd: a :class:`~grudge.dof_desc.DOFDesc` as the surface discretization.
    :returns: an object array of :class:`~meshmode.dof_array.DOFArray`
        containing the unit normals at each nodal location.
    """
    return mv_normal(actx, dcoll, dd).as_vector(dtype=object)

# }}}


# {{{ Curvature computations

def second_fundamental_form(
        actx: ArrayContext, dcoll: DiscretizationCollection, dd=None
        ) -> np.ndarray:
    r"""Computes the second fundamental form:

    .. math::

        S(x) = \begin{bmatrix}
            \partial_{uu} x\cdot n & \partial_{uv} x\cdot n \\
            \partial_{uv} x\cdot n & \partial_{vv} x\cdot n
        \end{bmatrix}

    where :math:`n` is the surface normal, :math:`x(u, v)` defines a parameterized
    surface, and :math:`u,v` are coordinates on the parameterized surface.

    :arg dd: a :class:`~grudge.dof_desc.DOFDesc`, or a value convertible to one.
    :returns: a rank-2 object array describing second fundamental form.
    """

    if dd is None:
        dd = DD_VOLUME

    dim = dcoll.discr_from_dd(dd).dim
    normal = rel_mv_normal(actx, dcoll, dd=dd).as_vector(dtype=object)

    if dim == 1:
        second_ref_axes = [((0, 2),)]
    elif dim == 2:
        second_ref_axes = [((0, 2),), ((0, 1), (1, 1)), ((1, 2),)]
    else:
        raise ValueError("%dD surfaces not supported" % dim)

    from pytools import flatten

    form2 = np.empty((dim, dim), dtype=object)

    for ref_axes in second_ref_axes:
        i, j = flatten([rst_axis] * n for rst_axis, n in ref_axes)

        ruv = make_obj_array(
            [forward_metric_nth_derivative(actx, dcoll, xyz_axis, ref_axes, dd=dd)
             for xyz_axis in range(dcoll.ambient_dim)]
        )
        form2[i, j] = form2[j, i] = normal.dot(ruv)

    return form2


def shape_operator(actx: ArrayContext, dcoll: DiscretizationCollection,
        dd=None) -> np.ndarray:
    r"""Computes the shape operator (also called the curvature tensor) containing
    second order derivatives:

    .. math::

        C(x) = \begin{bmatrix}
            \partial_{uu} x & \partial_{uv} x \\
            \partial_{uv} x & \partial_{vv} x
        \end{bmatrix}

    where :math:`x(u, v)` defines a parameterized surface, and :math:`u,v` are
    coordinates on the parameterized surface.

    :arg dd: a :class:`~grudge.dof_desc.DOFDesc`, or a value convertible to one.
    :returns: a rank-2 object array describing the shape operator.
    """

    inv_form1 = inverse_first_fundamental_form(actx, dcoll, dd=dd)
    form2 = second_fundamental_form(actx, dcoll, dd=dd)

    return -form2.dot(inv_form1)


def summed_curvature(actx: ArrayContext, dcoll: DiscretizationCollection,
        dd=None) -> DOFArray:
    r"""Computes the sum of the principal curvatures:

    .. math::

        \kappa = \operatorname{Trace}(C(x))

    where :math:`x(u, v)` defines a parameterized surface, :math:`u,v` are
    coordinates on the parameterized surface, and :math:`C(x)` is the shape
    operator.

    :arg dd: a :class:`~grudge.dof_desc.DOFDesc`, or a value convertible to one.
    :returns: a :class:`~meshmode.dof_array.DOFArray` containing the summed
        curvature at each nodal coordinate.
    """

    if dd is None:
        dd = DD_VOLUME

    dim = dcoll.ambient_dim - 1

    if dcoll.ambient_dim == 1:
        return 0.0

    if dcoll.ambient_dim == dim:
        return 0.0

    return np.trace(shape_operator(actx, dcoll, dd=dd))

# }}}


# vim: foldmethod=marker
