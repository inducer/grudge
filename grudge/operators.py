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


from pytools import keyed_memoize_in
from pytools.obj_array import obj_array_vectorize
from meshmode.array_context import FirstAxisIsElementsTag
from meshmode.dof_array import DOFArray

import numpy as np
import grudge.dof_desc as dof_desc


# {{{ tags

class HasElementwiseMatvecTag(FirstAxisIsElementsTag):
    pass


class MassOperatorTag(HasElementwiseMatvecTag):
    pass

# }}}


# {{{ Derivative operator

def reference_derivative_matrices(actx, element_group):
    @keyed_memoize_in(
        actx, reference_derivative_matrices,
        lambda grp: grp.discretization_key())
    def get_ref_derivative_mats(grp):
        from meshmode.discretization.poly_element import diff_matrices
        return tuple(
            actx.freeze(actx.from_numpy(dfmat))
            for dfmat in diff_matrices(grp)
        )

    return get_ref_derivative_mats(element_group)


def _compute_local_gradient(dcoll, vec, xyz_axis):
    from grudge.geometry import \
        inverse_surface_metric_derivative

    discr = dcoll.discr_from_dd(dof_desc.DD_VOLUME)
    actx = vec.array_context

    # FIXME: Each individual term comes out as (result,)
    inverse_jac_t = [
        inverse_surface_metric_derivative(
            actx, dcoll, rst_axis, xyz_axis
        )[0] for rst_axis in range(dcoll.dim)
    ]

    data = []
    for grp, vec_i in zip(discr.groups, vec):
        data.append(sum(
            actx.einsum(
                "ij,ej,ej->ei",
                reference_derivative_matrices(actx, grp)[rst_axis],
                vec_i,
                inverse_jac_t[rst_axis],
                arg_names=("ref_diff_mat", "vec", "inv_jac_t"),
                tagged=(HasElementwiseMatvecTag(),)
            ) for rst_axis in range(dcoll.dim))
        )
    return DOFArray(actx, data=tuple(data))


def local_gradient(dcoll, vec):
    from pytools.obj_array import make_obj_array
    return make_obj_array(
        [_compute_local_gradient(dcoll, vec, xyz_axis)
         for xyz_axis in range(dcoll.dim)]
    )

# }}}


# {{{ Mass operator

def reference_mass_matrix(actx, out_element_group, in_element_group):
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


def _apply_mass_operator(dcoll, dd_out, dd_in, vec):
    from grudge.geometry import area_element

    in_discr = dcoll.discr_from_dd(dd_in)
    out_discr = dcoll.discr_from_dd(dd_out)

    actx = vec.array_context
    area_elements = area_element(actx, dcoll, dd=dd_in)
    return DOFArray(
        actx,
        tuple(
            actx.einsum(
                "ij,ej,ej->ei",
                reference_mass_matrix(
                    actx,
                    out_element_group=out_grp,
                    in_element_group=in_grp
                ),
                ae_i,
                vec_i,
                arg_names=("mass_mat", "jac_det", "vec"),
                tagged=(MassOperatorTag(),))

            for in_grp, out_grp, ae_i, vec_i in zip(
                    in_discr.groups, out_discr.groups, area_elements, vec)
        )
    )


def mass_operator(dcoll, *args):
    if len(args) == 1:
        vec, = args
        dd = dof_desc.DOFDesc("vol", dof_desc.QTAG_NONE)
    elif len(args) == 2:
        dd, vec = args
    else:
        raise TypeError("invalid number of arguments")

    if isinstance(vec, np.ndarray):
        return obj_array_vectorize(
            lambda el: mass_operator(dcoll, dd, el), vec
        )

    return _apply_mass_operator(dcoll, dof_desc.DD_VOLUME, dd, vec)

# }}}


# {{{ Mass inverse operator

def reference_inverse_mass_matrix(actx, element_group):
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


def _apply_inverse_mass_operator(dcoll, dd_out, dd_in, vec):
    from grudge.geometry import area_element

    if dd_out != dd_in:
        raise ValueError(
            "Cannot compute inverse of a mass matrix mapping "
            "between different element groups; inverse is not "
            "guaranteed to be well-defined"
        )
    discr = dcoll.discr_from_dd(dd_in)
    use_wadg = not all(grp.is_affine for grp in discr.groups)

    actx = vec.array_context
    inv_area_elements = 1./area_element(actx, dcoll, dd=dd_in)
    if use_wadg:
        # FIXME: Think of how to compose existing functions here...
        # NOTE: Rewritten for readability/debuggability
        grps = discr.groups
        data = []
        for grp, jac_inv, x in zip(grps, inv_area_elements, vec):
            ref_mass = reference_mass_matrix(actx,
                                             out_element_group=grp,
                                             in_element_group=grp)
            ref_mass_inv = reference_inverse_mass_matrix(actx,
                                                         element_group=grp)
            data.append(
                # Based on https://arxiv.org/pdf/1608.03836.pdf
                # true_Minv ~ ref_Minv * ref_M * (1/jac_det) * ref_Minv
                actx.einsum("ik,km,em,mj,ej->ei",
                            ref_mass_inv, ref_mass, jac_inv, ref_mass_inv, x,
                            tagged=(MassOperatorTag(),))
            )
        return DOFArray(actx, data=tuple(data))
    else:
        return DOFArray(
            actx,
            tuple(
                actx.einsum("ij,ej,ej->ei",
                            reference_inverse_mass_matrix(
                                actx,
                                element_group=grp
                            ),
                            iae_i,
                            vec_i,
                            arg_names=("mass_inv_mat", "jac_det_inv", "vec"),
                            tagged=(MassOperatorTag(),))

                for grp, iae_i, vec_i in zip(discr.groups,
                                             inv_area_elements, vec)
            )
        )


def inverse_mass_operator(dcoll, vec):
    if isinstance(vec, np.ndarray):
        return obj_array_vectorize(
            lambda el: inverse_mass_operator(dcoll, el), vec
        )

    return _apply_inverse_mass_operator(
        dcoll, dof_desc.DD_VOLUME, dof_desc.DD_VOLUME, vec
    )

# }}}


# vim: foldmethod=marker
