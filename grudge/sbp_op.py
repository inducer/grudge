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
from arraycontext.container import ArrayOrContainerT
from typing import Callable
from functools import partial
from meshmode.transform_metadata import FirstAxisIsElementsTag

from grudge.discretization import DiscretizationCollection

from meshmode.dof_array import DOFArray

from pytools import memoize_in, keyed_memoize_in

import numpy as np


def volume_quadrature_interpolation_matrix(
        actx: ArrayContext, base_element_group, vol_quad_element_group):
    """todo.
    """
    @keyed_memoize_in(
        actx, volume_quadrature_interpolation_matrix,
        lambda base_grp, vol_quad_grp: (base_grp.discretization_key(),
                                        vol_quad_grp.discretization_key()))
    def get_volume_vand(base_grp, vol_quad_grp):
        from modepy import vandermonde

        basis = base_grp.basis_obj()
        vdm_inv = np.linalg.inv(vandermonde(basis.functions,
                                            base_grp.unit_nodes))
        vdm_q = vandermonde(basis.functions, vol_quad_grp.unit_nodes) @ vdm_inv
        return actx.freeze(actx.from_numpy(vdm_q))

    return get_volume_vand(base_element_group, vol_quad_element_group)


def volume_quadrature_interpolation(dcoll: DiscretizationCollection, dq, vec):
    if not isinstance(vec, DOFArray):
        return map_array_container(
            partial(volume_quadrature_interpolation, dcoll, dq), vec
        )

    actx = vec.array_context
    discr = dcoll.discr_from_dd("vol")
    quad_discr = dcoll.discr_from_dd(dq)

    return DOFArray(
        actx,
        data=tuple(
            actx.einsum("ij,ej->ei",
                        volume_quadrature_interpolation_matrix(
                            actx,
                            base_element_group=bgrp,
                            vol_quad_element_group=qgrp
                        ),
                        vec_i,
                        arg_names=("Vq_mat", "vec"),
                        tagged=(FirstAxisIsElementsTag(),))

            for bgrp, qgrp, vec_i in zip(discr.groups, quad_discr.groups, vec)
        )
    )


def surface_quadrature_interpolation_matrix(
        actx: ArrayContext, base_element_group, face_quad_element_group, dtype):
    """todo.
    """
    @keyed_memoize_in(
        actx, surface_quadrature_interpolation_matrix,
        lambda base_grp, face_quad_grp: (base_grp.discretization_key(),
                                         face_quad_grp.discretization_key()))
    def get_surface_vand(base_grp, face_quad_grp):
        nfaces = base_grp.mesh_el_group.nfaces
        assert face_quad_grp.nelements == nfaces * base_grp.nelements

        from modepy import vandermonde, faces_for_shape

        basis = base_grp.basis_obj()
        vdm_inv = np.linalg.inv(vandermonde(basis.functions,
                                            base_grp.unit_nodes))
        faces = faces_for_shape(base_grp.shape)
        # NOTE: Assumes same quadrature rule on each face
        face_quadrature = face_quad_grp.quadrature_rule()

        surface_nodes = faces[0].map_to_volume(face_quadrature.nodes)
        for fidx in range(1, nfaces):
            surface_nodes = np.append(
                surface_nodes,
                faces[fidx].map_to_volume(face_quadrature.nodes),
                axis=1
            )
        vdm_f = vandermonde(basis.functions, surface_nodes) @ vdm_inv
        return actx.freeze(actx.from_numpy(vdm_f))

    return get_surface_vand(base_element_group, face_quad_element_group)


def quadrature_surface_interpolation(dcoll: DiscretizationCollection, df, vec):
    if not isinstance(vec, DOFArray):
        return map_array_container(
            partial(quadrature_surface_interpolation, dcoll, df), vec
        )

    actx = vec.array_context
    dtype = vec.entry_dtype
    discr = dcoll.discr_from_dd("vol")
    quad_face_discr = dcoll.discr_from_dd(df)

    return DOFArray(
        actx,
        data=tuple(
            actx.einsum("ij,ej->ei",
                        surface_quadrature_interpolation_matrix(
                            actx,
                            base_element_group=bgrp,
                            face_quad_element_group=qfgrp,
                            dtype=dtype
                        ),
                        vec_i,
                        arg_names=("Vf_mat", "vec"),
                        tagged=(FirstAxisIsElementsTag(),))

            for bgrp, qfgrp, vec_i in zip(discr.groups,
                                          quad_face_discr.groups, vec)
        )
    )


def volume_and_surface_interpolation_matrix(
        actx: ArrayContext,
        base_element_group,
        vol_quad_element_group,
        face_quad_element_group, dtype):
    """todo.
    """
    @keyed_memoize_in(
        actx, volume_and_surface_interpolation_matrix,
        lambda base_grp, vol_quad_grp, face_quad_grp: (
            base_grp.discretization_key(),
            vol_quad_grp.discretization_key(),
            face_quad_grp.discretization_key()))
    def get_vol_surf_interpolation_matrix(base_grp, vol_quad_grp, face_quad_grp):
        vq_mat = actx.to_numpy(
            thaw(
                volume_quadrature_interpolation_matrix(
                    actx, base_grp, vol_quad_grp
                ),
                actx
            )
        )
        vf_mat = actx.to_numpy(
            thaw(
                surface_quadrature_interpolation_matrix(
                    actx, base_grp, face_quad_grp, dtype
                ),
                actx
            )
        )
        return actx.freeze(actx.from_numpy(np.block([[vq_mat], [vf_mat]])))

    return get_vol_surf_interpolation_matrix(
        base_element_group, vol_quad_element_group, face_quad_element_group
    )


def volume_and_surface_quadrature_interpolation(
        dcoll: DiscretizationCollection, dq, df, vec):
    """todo.
    """
    if not isinstance(vec, DOFArray):
        return map_array_container(
            partial(volume_and_surface_quadrature_interpolation,
                    dcoll, dq, df), vec
        )

    actx = vec.array_context
    dtype = vec.entry_dtype
    discr = dcoll.discr_from_dd("vol")
    quad_volm_discr = dcoll.discr_from_dd(dq)
    quad_face_discr = dcoll.discr_from_dd(df)

    return DOFArray(
        actx,
        data=tuple(
            actx.einsum("ij,ej->ei",
                        volume_and_surface_interpolation_matrix(
                            actx,
                            base_element_group=bgrp,
                            vol_quad_element_group=qvgrp,
                            face_quad_element_group=qfgrp,
                            dtype=dtype
                        ),
                        vec_i,
                        arg_names=("Vh_mat", "vec"),
                        tagged=(FirstAxisIsElementsTag(),))

            for bgrp, qvgrp, qfgrp, vec_i in zip(
                discr.groups,
                quad_volm_discr.groups,
                quad_face_discr.groups, vec)
        )
    )


def volume_and_surface_projection_matrix(
        actx: ArrayContext,
        base_element_group,
        vol_quad_element_group,
        face_quad_element_group, dtype):
    """todo.
    """
    @keyed_memoize_in(
        actx, volume_and_surface_projection_matrix,
        lambda base_grp, vol_quad_grp, face_quad_grp: (
            base_grp.discretization_key(),
            vol_quad_grp.discretization_key(),
            face_quad_grp.discretization_key()))
    def get_vol_surf_l2_projection_matrix(base_grp, vol_quad_grp, face_quad_grp):
        vq_mat = actx.to_numpy(
            thaw(
                volume_quadrature_interpolation_matrix(
                    actx, base_grp, vol_quad_grp
                ),
                actx
            )
        )
        vf_mat = actx.to_numpy(
            thaw(
                surface_quadrature_interpolation_matrix(
                    actx, base_grp, face_quad_grp, dtype
                ),
                actx
            )
        )
        minv = actx.to_numpy(
            thaw(
                quadrature_based_inverse_mass_matrix(actx, base_grp, vol_quad_grp),
                actx
            )
        )
        return actx.freeze(actx.from_numpy(minv @ np.block([[vq_mat], [vf_mat]]).T))

    return get_vol_surf_l2_projection_matrix(
        base_element_group, vol_quad_element_group, face_quad_element_group
    )


def volume_and_surface_quadrature_projection(
        dcoll: DiscretizationCollection, dq, df, vec):
    """todo.
    """
    if not isinstance(vec, DOFArray):
        return map_array_container(
            partial(volume_and_surface_quadrature_projection,
                    dcoll, dq, df), vec
        )

    actx = vec.array_context
    dtype = vec.entry_dtype
    discr = dcoll.discr_from_dd("vol")
    quad_volm_discr = dcoll.discr_from_dd(dq)
    quad_face_discr = dcoll.discr_from_dd(df)

    return DOFArray(
        actx,
        data=tuple(
            actx.einsum("ij,ej->ei",
                        volume_and_surface_projection_matrix(
                            actx,
                            base_element_group=bgrp,
                            vol_quad_element_group=qvgrp,
                            face_quad_element_group=qfgrp,
                            dtype=dtype
                        ),
                        vec_i,
                        arg_names=("Ph_mat", "vec"),
                        tagged=(FirstAxisIsElementsTag(),))

            for bgrp, qvgrp, qfgrp, vec_i in zip(
                discr.groups,
                quad_volm_discr.groups,
                quad_face_discr.groups,
                vec
            )
        )
    )


def quadrature_based_mass_matrix(
        actx: ArrayContext, base_element_group, vol_quad_element_group):
    """todo.
    """
    @keyed_memoize_in(
        actx, quadrature_based_mass_matrix,
        lambda base_grp, vol_quad_grp: (base_grp.discretization_key(),
                                        vol_quad_grp.discretization_key()))
    def get_ref_quad_mass_mat(base_grp, vol_quad_grp):
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


def volume_quadrature_l2_projection_matrix(
        actx: ArrayContext, base_element_group, vol_quad_element_group):
    """todo.
    """
    @keyed_memoize_in(
        actx, volume_quadrature_l2_projection_matrix,
        lambda base_grp, vol_quad_grp: (base_grp.discretization_key(),
                                        vol_quad_grp.discretization_key()))
    def get_ref_l2_proj_mat(base_grp, vol_quad_grp):
        vdm_q = actx.to_numpy(
            thaw(
                volume_quadrature_interpolation_matrix(
                    actx, base_grp, vol_quad_grp
                ),
                actx
            )
        )
        weights = np.diag(vol_quad_grp.quadrature_rule().weights)
        inv_mass_mat = actx.to_numpy(
            thaw(quadrature_based_inverse_mass_matrix(
                actx, base_grp, vol_quad_grp), actx)
        )
        return actx.freeze(actx.from_numpy(inv_mass_mat @ (vdm_q.T @ weights)))

    return get_ref_l2_proj_mat(base_element_group, vol_quad_element_group)


def volume_quadrature_project(dcoll: DiscretizationCollection, dd_q, vec):
    if not isinstance(vec, DOFArray):
        return map_array_container(
            partial(volume_quadrature_project, dcoll, dd_q), vec
        )

    actx = vec.array_context
    discr = dcoll.discr_from_dd("vol")
    quad_discr = dcoll.discr_from_dd(dd_q)

    return DOFArray(
        actx,
        data=tuple(
            actx.einsum("ij,ej->ei",
                        volume_quadrature_l2_projection_matrix(
                            actx,
                            base_element_group=bgrp,
                            vol_quad_element_group=qgrp
                        ),
                        vec_i,
                        arg_names=("Pq_mat", "vec"),
                        tagged=(FirstAxisIsElementsTag(),))

            for bgrp, qgrp, vec_i in zip(
                discr.groups,
                quad_discr.groups,
                vec
            )
        )
    )


def boundary_integration_matrices(
        actx: ArrayContext, base_element_group, face_quad_element_group):
    """todo.
    """
    @keyed_memoize_in(
        actx, boundary_integration_matrices,
        lambda base_grp, face_quad_grp: (base_grp.discretization_key(),
                                         face_quad_grp.discretization_key()))
    def get_ref_boundary_mats(base_grp, face_quad_grp):
        from modepy import faces_for_shape

        dim = base_grp.dim
        faces = faces_for_shape(base_grp.shape)
        nfaces = len(faces)
        # NOTE: assumes same quadrature rule on all faces
        face_quad_rule = face_quad_grp.quadrature_rule()
        face_quad_weights = np.tile(face_quad_rule.weights, nfaces)
        nq_per_face = face_quad_grp.nunit_dofs

        from modepy import face_normal

        face_normals = [face_normal(face) for face in faces]

        e = np.ones(shape=(nq_per_face,))
        nrstj = np.array(
            # nsrtJ = nhat * Jhatf, where nhat is the reference normal
            # and Jhatf is the Jacobian det. of the transformation from
            # the face of the reference element to the reference face.
            [np.concatenate([np.sign(nhat[idx])*e for nhat in face_normals])
             for idx in range(dim)]
        )
        return actx.freeze(
            actx.from_numpy(
                np.asarray(
                    [np.diag(face_quad_weights*nrstj[d])
                     for d in range(base_grp.dim)]
                )
            )
        )

    return get_ref_boundary_mats(base_element_group, face_quad_element_group)


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

        mass_mat = actx.to_numpy(
            thaw(quadrature_based_mass_matrix(actx, base_grp, quad_vol_grp), actx)
        )
        p_mat = actx.to_numpy(
            thaw(
                volume_quadrature_l2_projection_matrix(actx, base_grp, quad_vol_grp),
                actx
            )
        )
        vf_mat = actx.to_numpy(
            thaw(
                surface_quadrature_interpolation_matrix(
                    actx,
                    base_element_group=base_grp,
                    face_quad_element_group=face_quad_grp,
                    dtype=dtype
                ), actx
            )
        )
        b_mats = actx.to_numpy(
            thaw(
                boundary_integration_matrices(
                    actx, base_grp, face_quad_grp
                ),
                actx
            )
        )
        zero_mat = np.zeros(b_mats[-1].shape, dtype=dtype)
        q_mats = np.asarray(
            [p_mat.T @ mass_mat @ diff_mat @ p_mat
             for diff_mat in diff_matrices(base_grp)]
        )
        e_mat = vf_mat @ p_mat
        q_skew_hybridized = np.asarray(
            [np.block([[q_mats[d] - q_mats[d].T, e_mat.T @ b_mats[d]],
                       [-b_mats[d] @ e_mat, zero_mat]])
             for d in range(base_grp.dim)]
        )
        return actx.freeze(actx.from_numpy(q_skew_hybridized))

    return get_skew_symetric_hybridized_diff_mats(
        base_element_group,
        vol_quad_element_group,
        face_quad_element_group
    )


def _apply_skew_symmetric_hybrid_diff_operator(
        dcoll: DiscretizationCollection, vgeo, dq, df, vec):
    if not isinstance(vec, DOFArray):
        return map_array_container(
            partial(_apply_skew_symmetric_hybrid_diff_operator,
                    dcoll, vgeo, dq, df), vec
        )

    actx = vec.array_context
    dtype = vec.entry_dtype
    discr = dcoll.discr_from_dd("vol")
    quad_volm_discr = dcoll.discr_from_dd(dq)
    quad_face_discr = dcoll.discr_from_dd(df)

    return DOFArray(
        actx,
        data=tuple(
            actx.einsum("xrej,rij,eij->ei",
                        geo_i,
                        skew_symmetric_hybridized_sbp_operators(
                            actx,
                            bgrp,
                            qvgrp,
                            qafgrp,
                            dtype
                        ),
                        vec_i,
                        arg_names=("Gmat", "Qskew", "Feval"),
                        tagged=(FirstAxisIsElementsTag(),))

            for bgrp, qvgrp, qafgrp, vec_i, geo_i in zip(
                discr.groups,
                quad_volm_discr.groups,
                quad_face_discr.groups,
                vec,
                vgeo
            )
        )
    )


def volume_flux_differencing(
        dcoll: DiscretizationCollection,
        flux: Callable[[ArrayOrContainerT, ArrayOrContainerT], ArrayOrContainerT],
        dq, df, vec) -> ArrayOrContainerT:
    """todo.
    """
    from arraycontext import to_numpy, from_numpy, freeze
    from grudge.geometry import area_element, inverse_surface_metric_derivative

    def _reshape(shape, ary):
        if not isinstance(ary, DOFArray):
            return map_array_container(partial(_reshape, shape), ary)

        actx = ary.array_context
        return DOFArray(
            actx,
            data=tuple(
                # FIXME: relying on numpy broadcasting for now
                to_numpy(subary.reshape(grp.nelements, *shape), actx)
                for grp, subary in zip(
                    dcoll.discr_from_dd("vol").groups,
                    ary
                )
            )
        )

    # FIXME: assumes we can ask for array container
    # (can go away once we dont need from_numpy)
    actx = vec.array_context

    # FIXME: Hack here to get geometry terms onto the quadrature grid
    # NOTE: Exploit affine-ness
    def inverse_jac_matrix():
        @memoize_in(dcoll, (volume_flux_differencing, dq, df))
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
                                    rst_axis, xyz_axis
                                )
                            ) for rst_axis in range(dcoll.dim)
                        ]
                    ) for xyz_axis in range(dcoll.ambient_dim)
                ]
            )
            return freeze(mat, actx)
        return _inv_surf_metric_deriv()

    return volume_and_surface_quadrature_projection(
        dcoll,
        dq, df,
        _apply_skew_symmetric_hybrid_diff_operator(
            dcoll,
            inverse_jac_matrix(),
            dq, df,
            from_numpy(
                # FIXME: better way to do the reshaping here?
                flux(_reshape((1, -1), vec), _reshape((-1, 1), vec)),
                actx
            )
        )
    )

def _apply_inverse_sbp_mass_operator(
        dcoll: DiscretizationCollection, dd_quad, vec):
    if not isinstance(vec, DOFArray):
        return map_array_container(
            partial(_apply_inverse_sbp_mass_operator, dcoll, dd_quad), vec
        )

    actx = vec.array_context
    discr = dcoll.discr_from_dd("vol")
    quad_discr = dcoll.discr_from_dd(dd_quad)

    return DOFArray(
        actx,
        data=tuple(
            actx.einsum("ij,ej->ei",
                        quadrature_based_inverse_mass_matrix(
                            actx,
                            base_grp,
                            quad_grp
                        ),
                        vec_i,
                        arg_names=("mass_inv", "vec"),
                        tagged=(FirstAxisIsElementsTag(),))

            for base_grp, quad_grp, vec_i in zip(
                discr.groups, quad_discr.groups, vec)
        )
    )


def inverse_sbp_mass(dcoll: DiscretizationCollection, dq, vec):
    """todo.
    """
    return _apply_inverse_sbp_mass_operator(dcoll, dq, vec)
