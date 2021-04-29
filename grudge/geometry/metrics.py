
import numpy as np

from grudge.dof_desc import (
    DD_VOLUME, DOFDesc
)

from meshmode.dof_array import thaw

from pymbolic.geometric_algebra import MultiVector

from pytools.obj_array import make_obj_array
from pytools import memoize_on_first_arg


def forward_metric_nth_derivative(actx, dcoll, xyz_axis, ref_axes, dd=None):
    r"""
    Pointwise metric derivatives representing repeated derivatives to *vec*

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
    """
    if dd is None:
        dd = DD_VOLUME

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
        dcoll.discr_from_dd(DD_VOLUME),
        flat_ref_axes,
        thaw(actx, dcoll.discr_from_dd(DD_VOLUME).nodes())[xyz_axis]
    )

    if dd.uses_quadrature():
        vec = dcoll.connection_from_dds(DD_VOLUME, dd)(vec)

    return vec


def forward_metric_derivative_vector(actx, dcoll, rst_axis, dd=None):

    return make_obj_array([
        forward_metric_nth_derivative(actx, dcoll, i, rst_axis, dd=dd)
        for i in range(dcoll.ambient_dim)
        ]
    )


def forward_metric_derivative_mv(actx, dcoll, rst_axis, dd=None):

    return MultiVector(
        forward_metric_derivative_vector(actx, dcoll, rst_axis, dd=dd)
    )


def forward_metric_derivative_mat(actx, dcoll, dd=None):

    ambient_dim = dcoll.ambient_dim
    dim = dcoll.dim

    result = np.zeros((ambient_dim, dim), dtype=object)
    for j in range(dim):
        result[:, j] = forward_metric_derivative_vector(actx, dcoll, j, dd=dd)

    return result


def first_fundamental_form(actx, dcoll, dd):

    mder = forward_metric_derivative_mat(actx, dcoll, dd)

    return mder.T.dot(mder)


def inverse_metric_derivative_mat(actx, dcoll, dd=None):

    ambient_dim = dcoll.ambient_dim
    dim = dcoll.dim

    result = np.zeros((ambient_dim, dim), dtype=object)
    for i in range(dim):
        for j in range(ambient_dim):
            result[i, j] = inverse_metric_derivative(
                actx, dcoll, i, j, dd=dd
            )

    return result


@memoize_on_first_arg
def inverse_first_fundamental_form(actx, dcoll, dd):

    if dcoll.ambient_dim == dcoll.dim:
        inv_mder = inverse_metric_derivative_mat(actx, dcoll, dd)
        inv_form1 = inv_mder.dot(inv_mder.T)
    else:
        form1 = first_fundamental_form(actx, dcoll, dd)

        if dim == 1:
            inv_form1 = np.array([[1.0/form1[0, 0]]], dtype=object)
        elif dim == 2:
            (E, F), (_, G) = form1      # noqa: N806
            inv_form1 = 1.0 / (E * G - F * F) * np.array(
                [[G, -F],
                 [-F, E]], dtype=object
            )
        else:
            raise ValueError("%dD surfaces not supported" % dim)

    return inv_form1


@memoize_on_first_arg
def inverse_metric_derivative(actx, dcoll, rst_axis, xyz_axis, dd):

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

    return (outprod_with_unit(xyz_axis, rst_axis)
            * volume_pseudoscalar_inv).as_scalar()


@memoize_on_first_arg
def inverse_surface_metric_derivative(actx, dcoll, rst_axis, xyz_axis, dd=None):

    if dcoll.ambient_dim == dcoll.dim:
        imd = inverse_metric_derivative(
            actx, dcoll, rst_axis, xyz_axis, dd=dd
        )
    else:
        inv_form1 = inverse_first_fundamental_form(actx, dcoll, dd=dd)
        imd = sum(
            inv_form1[rst_axis, d]*forward_metric_nth_derivative(
                actx, dcoll, d, rst_axis, dd=dd
            ) for d in range(dim)
        )

    return imd


@memoize_on_first_arg
def _signed_face_ones(actx, dcoll, dd):

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


def parametrization_derivative(actx, dcoll, dd):

    if dd.is_volume():
        dim = dcoll.dim
    else:
        dim = dcoll.dim - 1

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


def pseudoscalar(actx, dcoll, dd):

    return parametrization_derivative(
        actx, dcoll, dd
    ).project_max_grade()


@memoize_on_first_arg
def area_element(actx, dcoll, dd=None):

    return actx.np.sqrt(
        pseudoscalar(actx, dcoll, dd=dd).norm_squared()
    )
