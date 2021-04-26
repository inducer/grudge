
import numpy as np

from grudge.dof_desc import (
    DD_VOLUME, DOFDesc
)

from meshmode.dof_array import thaw

from pymbolic.geometric_algebra import MultiVector

from pytools.obj_array import make_obj_array
from pytools import memoize_on_first_arg


def forward_metric_nth_derivative(dcoll, ref_axes, vec, dd=None):
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

    vol_discr = dcoll.discr_from_dd(DD_VOLUME)
    vec = num_reference_derivative(vol_discr, flat_ref_axes, vec)

    if dd.uses_quadrature():
        vec = dcoll.connection_from_dds(DD_VOLUME, dd)(vec)

    return vec


def forward_metric_derivative_vector(dcoll, rst_axis, vec, dd=None):

    return make_obj_array([
        forward_metric_nth_derivative(dcoll, rst_axis, vec[i], dd=dd)
        for i in range(dcoll.ambient_dim)
        ]
    )


def forward_metric_derivative_mv(dcoll, rst_axis, vec, dd=None):

    return MultiVector(
        forward_metric_derivative_vector(dcoll, rst_axis, vec, dd=dd)
    )


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


def parametrization_derivative(actx, dcoll, vec, dd):

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
        forward_metric_derivative_mv(dcoll, rst_axis, vec, dd)
        for rst_axis in range(dim)
    )


def pseudoscalar(actx, dcoll, vec, dd):

    return parametrization_derivative(
        actx, dcoll, vec, dd
    ).project_max_grade()


@memoize_on_first_arg
def area_element(actx, dcoll, dd=None):

    if dd is None:
        dd = DD_VOLUME

    nodes = thaw(actx, dcoll.discr_from_dd(DD_VOLUME).nodes())

    return actx.np.sqrt(
        pseudoscalar(actx, dcoll, nodes, dd=dd).norm_squared()
    )
