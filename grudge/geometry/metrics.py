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

from grudge.dof_desc import (
    DD_VOLUME, DOFDesc, DISCR_TAG_BASE
)
from meshmode.dof_array import thaw
from pymbolic.geometric_algebra import MultiVector
from pytools.obj_array import make_obj_array
from pytools import memoize_on_first_arg


# {{{ Metric computations

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
        thaw(actx, dcoll.discr_from_dd(inner_dd).nodes())[xyz_axis]
    )

    if dd.uses_quadrature():
        vec = dcoll.connection_from_dds(inner_dd, dd)(vec)

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


def inverse_first_fundamental_form(actx, dcoll, dd):

    dim = dcoll.dim
    if dcoll.ambient_dim == dim:
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

    result = (outprod_with_unit(xyz_axis, rst_axis)
              * volume_pseudoscalar_inv).as_scalar()

    return result


@memoize_on_first_arg
def inverse_surface_metric_derivative(actx, dcoll, rst_axis, xyz_axis, dd=None):

    dim = dcoll.dim
    if dcoll.ambient_dim == dim:
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


def parametrization_derivative(actx, dcoll, dim, dd):
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


def pseudoscalar(actx, dcoll, dim=None, dd=None):
    if dd is None:
        dd = DD_VOLUME

    if dim is None:
        dim = dcoll.discr_from_dd(dd).dim

    return parametrization_derivative(
        actx, dcoll, dim, dd
    ).project_max_grade()


@memoize_on_first_arg
def area_element(actx, dcoll, dim=None, dd=None):

    return actx.np.sqrt(
        pseudoscalar(actx, dcoll, dim=dim, dd=dd).norm_squared()
    )

# }}}


# {{{ Surface normal vectors

def surface_normal(actx, dcoll, dim=None, dd=None):
    import grudge.dof_desc as dof_desc

    dd = dof_desc.as_dofdesc(dd)
    dim = dim or dcoll.discr_from_dd(dd).dim

    # NOTE: Don't be tempted to add a sign here. As it is, it produces
    # exterior normals for positively oriented curves.

    pder = pseudoscalar(actx, dcoll, dim=dim, dd=dd) \
        / area_element(actx, dcoll, dim=dim, dd=dd)

    # Dorst Section 3.7.2
    return pder << pder.I.inv()


def mv_normal(actx, dcoll, dd):
    """Exterior unit normal as a :class:`~pymbolic.geometric_algebra.MultiVector`."""
    import grudge.dof_desc as dof_desc

    dd = dof_desc.as_dofdesc(dd)
    if not dd.is_trace():
        raise ValueError("may only request normals on boundaries")

    dim = dcoll.discr_from_dd(dd).dim
    ambient_dim = dcoll.ambient_dim

    if dim == ambient_dim - 1:
        return surface_normal(actx, dcoll, dim=dim, dd=dd)

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
    import grudge.dof_desc as dof_desc

    volm_normal = MultiVector(
        project(dcoll, dof_desc.DD_VOLUME, dd,
                surface_normal(
                    actx, dcoll,
                    dim=dim + 1,
                    dd=dof_desc.DD_VOLUME
                ).as_vector(dtype=object))
    )
    pder = pseudoscalar(actx, dcoll, dd=dd)

    mv = -(volm_normal ^ pder) << volm_normal.I.inv()

    return mv / actx.np.sqrt(mv.norm_squared())


def normal(actx, dcoll, dd):
    return mv_normal(actx, dcoll, dd).as_vector(dtype=object)

# }}}


# vim: foldmethod=marker
