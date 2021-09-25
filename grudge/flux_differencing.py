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
    freeze
)
from arraycontext.container import ArrayOrContainerT
from typing import Callable
from functools import partial
from meshmode.transform_metadata import FirstAxisIsElementsTag

from grudge.discretization import DiscretizationCollection

from meshmode.dof_array import DOFArray

from pytools import memoize_in, keyed_memoize_in

import numpy as np


def skew_symmetric_hybridized_sbp_operators(
        actx: ArrayContext,
        base_element_group,
        vol_quad_element_group,
        face_quad_element_group, dtype):
    """todo.
    """
    @keyed_memoize_in(
        actx, skew_symmetric_hybridized_sbp_operators,
        lambda base_grp, quad_vol_grp, face_quad_grp: (
            base_grp.discretization_key(),
            quad_vol_grp.discretization_key(),
            face_quad_grp.discretization_key()
        )
    )
    def get_skew_symetric_hybridized_diff_mats(
            base_grp, quad_vol_grp, face_quad_grp):
        from meshmode.discretization.poly_element import diff_matrices
        from modepy import faces_for_shape, face_normal
        from grudge.projection import volume_quadrature_l2_projection_matrix
        from grudge.interpolation import (
            volume_quadrature_interpolation_matrix,
            surface_quadrature_interpolation_matrix
        )

        # {{{ Volume operators

        weights = quad_vol_grp.quadrature_rule().weights
        vdm_q = actx.to_numpy(
            volume_quadrature_interpolation_matrix(actx, base_grp, quad_vol_grp)
        )
        mass_mat = weights * vdm_q.T @ vdm_q
        p_mat = actx.to_numpy(
            volume_quadrature_l2_projection_matrix(actx, base_grp, quad_vol_grp)
        )

        # }}}

        # {{{ Surface operators

        faces = faces_for_shape(base_grp.shape)
        nfaces = len(faces)
        # NOTE: assumes same quadrature rule on all faces
        face_weights = np.tile(face_quad_grp.quadrature_rule().weights, nfaces)
        face_normals = [face_normal(face) for face in faces]
        nnods_per_face = face_quad_grp.nunit_dofs
        e = np.ones(shape=(nnods_per_face,))
        nrstj = [
            # nsrtJ = nhat * Jhatf, where nhat is the reference normal
            # and Jhatf is the Jacobian det. of the transformation from
            # the face of the reference element to the reference face.
            np.concatenate([np.sign(nhat[idx])*e for nhat in face_normals])
            for idx in range(base_grp.dim)
        ]
        b_mats = [np.diag(face_weights*nrstj[d]) for d in range(base_grp.dim)]
        vf_mat = actx.to_numpy(
            surface_quadrature_interpolation_matrix(
                actx,
                base_element_group=base_grp,
                face_quad_element_group=face_quad_grp,
                dtype=dtype
            )
        )
        zero_mat = np.zeros((nfaces*nnods_per_face, nfaces*nnods_per_face),
                            dtype=dtype)

        # }}}

        # {{{ Hybridized (volume + surface) operators

        q_mats = [p_mat.T @ mass_mat @ diff_mat @ p_mat
                  for diff_mat in diff_matrices(base_grp)]
        e_mat = vf_mat @ p_mat
        q_skew_hybridized = np.asarray(
            [
                np.block(
                    [[q_mats[d] - q_mats[d].T, e_mat.T @ b_mats[d]],
                    [-b_mats[d] @ e_mat, zero_mat]]
                ) for d in range(base_grp.dim)
            ],
            order="C"
        )

        # }}}

        return actx.freeze(actx.from_numpy(q_skew_hybridized))

    return get_skew_symetric_hybridized_diff_mats(
        base_element_group,
        vol_quad_element_group,
        face_quad_element_group
    )


def _single_axis_hybridized_sbp_derivative_kernel(
        dcoll, dq, df, xyz_axis, flux_matrix):
    if not isinstance(flux_matrix, DOFArray):
        return map_array_container(
            partial(_single_axis_hybridized_sbp_derivative_kernel,
                    dcoll, dq, df, xyz_axis),
            flux_matrix
        )

    from grudge.geometry import \
        area_element, inverse_surface_metric_derivative
    from grudge.interpolation import (
        volume_and_surface_interpolation_matrix,
        volume_and_surface_quadrature_interpolation
    )

    # FIXME: This is kinda meh
    def inverse_jac_matrix():
        @memoize_in(dcoll, (_single_axis_hybridized_sbp_derivative_kernel, dq, df))
        def _inv_surf_metric_deriv():
            mat = actx.np.stack(
                [
                    actx.np.stack(
                        [
                            volume_and_surface_quadrature_interpolation(
                                dcoll, dq, df,
                                area_element(actx, dcoll)
                                * inverse_surface_metric_derivative(
                                    actx, dcoll,
                                    rst_ax, xyz_ax
                                )
                            ) for rst_ax in range(dcoll.dim)
                        ]
                    ) for xyz_ax in range(dcoll.ambient_dim)
                ]
            )
            return freeze(mat, actx)
        return _inv_surf_metric_deriv()

    actx = flux_matrix.array_context
    discr = dcoll.discr_from_dd("vol")
    quad_volm_discr = dcoll.discr_from_dd(dq)
    quad_face_discr = dcoll.discr_from_dd(df)
    return DOFArray(
        actx,
        data=tuple(
            # r for rst axis
            actx.einsum("ik,rej,rij,eij->ek",
                        volume_and_surface_interpolation_matrix(
                            actx,
                            base_element_group=bgrp,
                            vol_quad_element_group=qvgrp,
                            face_quad_element_group=qafgrp,
                            dtype=fmat_i.dtype
                        ),
                        ijm_i[xyz_axis],
                        skew_symmetric_hybridized_sbp_operators(
                            actx,
                            bgrp,
                            qvgrp,
                            qafgrp,
                            fmat_i.dtype
                        ),
                        fmat_i,
                        arg_names=("Vh_mat_t", "inv_jac_t", "ref_Q_mat", "F_mat"),
                        tagged=(FirstAxisIsElementsTag(),))

            for bgrp, qvgrp, qafgrp, fmat_i, ijm_i in zip(
                discr.groups,
                quad_volm_discr.groups,
                quad_face_discr.groups,
                flux_matrix,
                inverse_jac_matrix()
            )
        )
    )


def _apply_skew_symmetric_hybrid_diff_operator(
        dcoll: DiscretizationCollection, dq, df, flux_matrices):
    # flux matrices for each component are a vector of matrices (as DOFArrays)
    # for each spatial dimension
    if not (isinstance(flux_matrices, np.ndarray) and flux_matrices.dtype == "O"):
        return map_array_container(
            partial(_apply_skew_symmetric_hybrid_diff_operator, dcoll, dq, df),
            flux_matrices
        )

    def _sbp_hybrid_diff_helper(diff_func, mats):
        if not isinstance(mats, np.ndarray):
            raise TypeError("argument must be an object array")
        assert mats.dtype == object
        return sum(diff_func(i, mat_i) for i, mat_i in enumerate(mats))

    return _sbp_hybrid_diff_helper(
        lambda i, flux_mat_i: _single_axis_hybridized_sbp_derivative_kernel(
            dcoll, dq, df, i, flux_mat_i),
        flux_matrices
    )


def volume_flux_differencing(
        dcoll: DiscretizationCollection,
        flux: Callable[[ArrayOrContainerT, ArrayOrContainerT], ArrayOrContainerT],
        dq, df, vec: ArrayOrContainerT) -> ArrayOrContainerT:
    """todo.
    """
    from arraycontext import to_numpy, from_numpy

    def _reshape_to_numpy(shape, ary):
        if not isinstance(ary, DOFArray):
            return map_array_container(partial(_reshape_to_numpy, shape), ary)

        actx = ary.array_context
        return DOFArray(
            actx,
            data=tuple(
                to_numpy(subary.reshape(grp.nelements, *shape), actx)
                for grp, subary in zip(
                    dcoll.discr_from_dd("vol").groups,
                    ary
                )
            )
        )

    actx = vec.array_context
    # FIXME: better way to do the reshaping here?
    flux_matrices = from_numpy(flux(_reshape_to_numpy((1, -1), vec),
                                    _reshape_to_numpy((-1, 1), vec)), actx)
    return _apply_skew_symmetric_hybrid_diff_operator(
        dcoll,
        dq, df,
        flux_matrices
    )


# vim: foldmethod=marker
