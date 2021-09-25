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


from arraycontext import (
    ArrayContext,
    map_array_container,
    thaw
)
from functools import partial
from meshmode.transform_metadata import FirstAxisIsElementsTag

from grudge.discretization import DiscretizationCollection

from meshmode.dof_array import DOFArray

from pytools import keyed_memoize_in

import numpy as np


def quadrature_based_mass_matrix(
        actx: ArrayContext, base_element_group, vol_quad_element_group):
    """todo.
    """
    @keyed_memoize_in(
        actx, quadrature_based_mass_matrix,
        lambda base_grp, vol_quad_grp: (base_grp.discretization_key(),
                                        vol_quad_grp.discretization_key()))
    def get_ref_quad_mass_mat(base_grp, vol_quad_grp):
        from grudge.interpolation import volume_quadrature_interpolation_matrix

        vdm_q = actx.to_numpy(
            thaw(
                volume_quadrature_interpolation_matrix(actx, base_grp, vol_quad_grp),
                actx
            )
        )
        weights = np.diag(vol_quad_grp.quadrature_rule().weights)

        return actx.freeze(actx.from_numpy(vdm_q.T @ weights @ vdm_q))

    return get_ref_quad_mass_mat(base_element_group, vol_quad_element_group)


def quadrature_based_inverse_mass_matrix(
        actx: ArrayContext, base_element_group, vol_quad_element_group):
    """todo.
    """
    @keyed_memoize_in(
        actx, quadrature_based_inverse_mass_matrix,
        lambda base_grp, vol_quad_grp: (base_grp.discretization_key(),
                                        vol_quad_grp.discretization_key()))
    def get_ref_quad_inv_mass_mat(base_grp, vol_quad_grp):
        mass_mat = actx.to_numpy(
            thaw(quadrature_based_mass_matrix(actx,
                                              base_grp,
                                              vol_quad_grp), actx)
        )
        return actx.freeze(actx.from_numpy(np.linalg.inv(mass_mat)))

    return get_ref_quad_inv_mass_mat(base_element_group, vol_quad_element_group)


def _apply_inverse_sbp_mass_operator(
        dcoll: DiscretizationCollection, dd_quad, vec):
    if not isinstance(vec, DOFArray):
        return map_array_container(
            partial(_apply_inverse_sbp_mass_operator, dcoll, dd_quad), vec
        )

    from grudge.geometry import area_element

    actx = vec.array_context
    discr = dcoll.discr_from_dd("vol")
    quad_discr = dcoll.discr_from_dd(dd_quad)
    jacobian_dets_inv = 1./area_element(
        actx, dcoll,
        _use_geoderiv_connection=actx.supports_nonscalar_broadcasting
    )
    return DOFArray(
        actx,
        data=tuple(
            actx.einsum("ei,ij,ej->ei",
                        jac_inv_i,
                        quadrature_based_inverse_mass_matrix(
                            actx,
                            base_grp,
                            quad_grp
                        ),
                        vec_i,
                        arg_names=("jac_inv", "mass_inv", "vec"),
                        tagged=(FirstAxisIsElementsTag(),))

            for base_grp, quad_grp, vec_i, jac_inv_i in zip(
                discr.groups,
                quad_discr.groups,
                vec,
                jacobian_dets_inv
            )
        )
    )


def inverse_sbp_mass(dcoll: DiscretizationCollection, dq, vec):
    """todo.
    """
    return _apply_inverse_sbp_mass_operator(dcoll, dq, vec)
