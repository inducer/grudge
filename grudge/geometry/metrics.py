
from grudge import sym
from pymbolic.geometric_algebra import MultiVector
from pytools.obj_array import make_obj_array


def forward_metric_nth_derivative(dcoll, xyz_axis, ref_axes, vec, dd=None):
    r"""
    Pointwise metric derivatives representing repeated derivatives to *vec*

    .. math::

        \frac{\partial^n x_{\mathrm{xyz\_axis}} }{\partial r_{\mathrm{ref\_axes}}}

    where *ref_axes* is a multi-index description.
    """
    if dd is None:
        dd = sym.DD_VOLUME

    if isinstance(ref_axes, int):
        ref_axes = ((ref_axes, 1),)

    if not isinstance(ref_axes, tuple):
        raise ValueError("ref_axes must be a tuple")

    if tuple(sorted(ref_axes)) != ref_axes:
        raise ValueError("ref_axes must be sorted")

    if len(dict(ref_axes)) != len(ref_axes):
        raise ValueError("ref_axes must not contain an axis more than once")

    from meshmode.discretization import num_reference_derivative

    vol_discr = dcoll.discr_from_dd(sym.DD_VOLUME)
    vec = num_reference_derivative(vol_discr, ref_axes, vec)

    if dd.uses_quadrature():
        vec = dcoll.connection_from_dds(sym.DD_VOLUME, dd)(vec)

    return vec


def forward_metric_derivative_vector(dcoll, rst_axis, vec, dd=None):
    return make_obj_array([
            forward_metric_nth_derivative(dcoll, i, rst_axis, vec[i], dd=dd)
            for i in range(dcoll.ambient_dim)])


def forward_metric_derivative_mv(dcoll, rst_axis, vec, dd=None):
    return MultiVector(
        forward_metric_derivative_vector(dcoll, rst_axis, vec, dd=dd)
    )
