"""Grudge module for flux-differencing in entropy-stable DG methods

Flux-differencing
-----------------

.. autofunction:: volume_flux_differencing
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


from arraycontext import (
    ArrayContext,
    map_array_container,
    freeze
)
from arraycontext.container import ArrayOrContainerT

from functools import partial

from meshmode.transform_metadata import FirstAxisIsElementsTag
from meshmode.dof_array import DOFArray

from grudge.discretization import DiscretizationCollection
from grudge.dof_desc import DOFDesc

from pytools import memoize_in, keyed_memoize_in

import numpy as np


def _reference_skew_symmetric_hybridized_sbp_operators(
        actx: ArrayContext,
        base_element_group,
        vol_quad_element_group,
        face_quad_element_group, dtype):
    @keyed_memoize_in(
        actx, _reference_skew_symmetric_hybridized_sbp_operators,
        lambda base_grp, quad_vol_grp, face_quad_grp: (
            base_grp.discretization_key(),
            quad_vol_grp.discretization_key(),
            face_quad_grp.discretization_key()))
    def get_reference_skew_symetric_hybridized_diff_mats(
            base_grp, quad_vol_grp, face_quad_grp):
        from meshmode.discretization.poly_element import diff_matrices
        from modepy import faces_for_shape, face_normal
        from grudge.interpolation import (
            volume_quadrature_interpolation_matrix,
            surface_quadrature_interpolation_matrix
        )
        from grudge.op import reference_inverse_mass_matrix

        # {{{ Volume operators

        weights = quad_vol_grp.quadrature_rule().weights
        vdm_q = actx.to_numpy(
            volume_quadrature_interpolation_matrix(actx, base_grp, quad_vol_grp))
        inv_mass_mat = actx.to_numpy(
            reference_inverse_mass_matrix(actx, base_grp))
        p_mat = inv_mass_mat @ (vdm_q.T * weights)

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
                face_quad_element_group=face_quad_grp))
        zero_mat = np.zeros((nfaces*nnods_per_face, nfaces*nnods_per_face),
                            dtype=dtype)

        # }}}

        # {{{ Hybridized (volume + surface) operators

        q_mats = [p_mat.T @ (weights * vdm_q.T @ vdm_q) @ diff_mat @ p_mat
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

    return get_reference_skew_symetric_hybridized_diff_mats(
        base_element_group,
        vol_quad_element_group,
        face_quad_element_group
    )


def _single_axis_hybridized_derivative_kernel(
        dcoll, dd_quad, dd_face_quad, xyz_axis, flux_matrix):
    if not dcoll._has_affine_groups():
        raise NotImplementedError("Not implemented for non-affine elements yet.")

    if not isinstance(flux_matrix, DOFArray):
        return map_array_container(
            partial(_single_axis_hybridized_derivative_kernel,
                    dcoll, dd_quad, dd_face_quad, xyz_axis),
            flux_matrix
        )

    from grudge.geometry import \
        area_element, inverse_surface_metric_derivative
    from grudge.interpolation import (
        volume_and_surface_interpolation_matrix,
        volume_and_surface_quadrature_interpolation
    )

    actx = flux_matrix.array_context

    # FIXME: This is kinda meh
    def inverse_jac_matrix():
        @memoize_in(
            dcoll,
            (_single_axis_hybridized_derivative_kernel, dd_quad, dd_face_quad))
        def _inv_surf_metric_deriv():
            return freeze(
                actx.np.stack(
                    [
                        actx.np.stack(
                            [
                                volume_and_surface_quadrature_interpolation(
                                    dcoll, dd_quad, dd_face_quad,
                                    area_element(actx, dcoll)
                                    * inverse_surface_metric_derivative(
                                        actx, dcoll,
                                        rst_ax, xyz_axis
                                    )
                                ) for rst_ax in range(dcoll.dim)
                            ]
                        ) for xyz_axis in range(dcoll.ambient_dim)
                    ]
                ),
                actx
            )
        return _inv_surf_metric_deriv()

    return DOFArray(
        actx,
        data=tuple(
            # r for rst axis
            actx.einsum("ik,rej,rij,eij->ek",
                        volume_and_surface_interpolation_matrix(
                            actx,
                            base_element_group=bgrp,
                            vol_quad_element_group=qvgrp,
                            face_quad_element_group=qafgrp
                        ),
                        ijm_i[xyz_axis],
                        _reference_skew_symmetric_hybridized_sbp_operators(
                            actx,
                            bgrp,
                            qvgrp,
                            qafgrp,
                            fmat_i.dtype
                        ),
                        fmat_i,
                        arg_names=("Vh_mat_t", "inv_jac_t", "Q_mat", "F_mat"),
                        tagged=(FirstAxisIsElementsTag(),))

            for bgrp, qvgrp, qafgrp, fmat_i, ijm_i in zip(
                dcoll.discr_from_dd("vol").groups,
                dcoll.discr_from_dd(dd_quad).groups,
                dcoll.discr_from_dd(dd_face_quad).groups,
                flux_matrix,
                inverse_jac_matrix()
            )
        )
    )


def volume_flux_differencing(
        dcoll: DiscretizationCollection,
        dd_quad: DOFDesc,
        dd_face_quad: DOFDesc,
        flux_matrices: ArrayOrContainerT) -> ArrayOrContainerT:
    r"""Computes the volume contribution of the DG divergence operator using
    flux-differencing:

    .. math::

       \mathrm{VOL} = \sum_{i=1}^{d}
        \begin{bmatrix}
            \mathbf{V}_q \\ \mathbf{V}_f
        \end{bmatrix}^T
        \left(
            \left( \mathbf{Q}_{i} - \mathbf{Q}^T_{i} \right)
            \circ \mathbf{F}_{i}
        \right)\mathbf{1}

    where :math:`\circ` denotes the
    `Hadamard product <https://en.wikipedia.org/wiki/Hadamard_product_(matrices)>`__,
    :math:`\mathbf{F}_{i}` are matrices whose entries are computed
    as the evaluation of an entropy-conserving two-point flux function
    (e.g. :func:`grudge.models.euler.divergence_flux_chandrashekar`)
    and :math:`\mathbf{Q}_{i} - \mathbf{Q}^T_{i}` are the skew-symmetric
    hybridized differentiation operators defined in (15) of
    `this paper <https://arxiv.org/pdf/1902.01828.pdf>`__.

    :arg flux_matrices: a :class:`~meshmode.dof_array.DOFArray` or an
        :class:`~arraycontext.container.ArrayContainer` of them containing
        evaluations of two-point flux.
    :returns: a :class:`~meshmode.dof_array.DOFArray` or an
        :class:`~arraycontext.container.ArrayContainer` of them.
    """
    from grudge.op import _div_helper

    return _div_helper(
        dcoll,
        lambda _, i, flux_mat_i: _single_axis_hybridized_derivative_kernel(
            dcoll, dd_quad, dd_face_quad, i, flux_mat_i),
        flux_matrices
    )


# vim: foldmethod=marker
