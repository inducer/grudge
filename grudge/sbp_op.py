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
    make_loopy_program,
    thaw
)
from meshmode.transform_metadata import FirstAxisIsElementsTag

from grudge.discretization import DiscretizationCollection

from meshmode.dof_array import DOFArray

from pytools import keyed_memoize_in
from pytools.obj_array import obj_array_vectorize

import numpy as np


def quadrature_based_mass_matrix(actx: ArrayContext, element_group):
    """todo
    """
    @keyed_memoize_in(
        actx, quadrature_based_mass_matrix,
        lambda grp: grp.discretization_key())
    def get_ref_quad_mass_mat(grp):
        from modepy import vandermonde
        basis_fcts = grp.basis_obj().functions
        quad_rule = grp.quadrature_rule()
        vand = vandermonde(basis_fcts, quad_rule.nodes)
        weights = np.diag(quad_rule.weights)

        return actx.freeze(actx.from_numpy(vand.T @ weights @ vand))

    return get_ref_quad_mass_mat(element_group)


def quadrature_based_inverse_mass_matrix(actx: ArrayContext, element_group):
    """todo
    """
    @keyed_memoize_in(
        actx, quadrature_based_inverse_mass_matrix,
        lambda grp: grp.discretization_key())
    def get_ref_quad_inv_mass_mat(grp):
        mass_mat = thaw(quadrature_based_mass_matrix(grp), actx)
        return actx.freeze(actx.from_numpy(mass_mat.inv))

    return get_ref_quad_inv_mass_mat(element_group)


def quadrature_based_l2_projection_matrix(actx: ArrayContext, element_group):
    """todo
    """
    @keyed_memoize_in(
        actx, quadrature_based_l2_projection_matrix,
        lambda grp: grp.discretization_key())
    def get_ref_l2_proj_mat(grp):
        from modepy import vandermonde

        quad_rule = grp.quadrature_rule()
        vand = vandermonde(grp.basis_obj().functions, quad_rule.nodes)
        weights = np.diag(quad_rule.weights)
        inv_mass_mat = thaw(quadrature_based_inverse_mass_matrix(grp), actx)

        return actx.freeze(actx.from_numpy(inv_mass_mat @ vand.T @ weights))

    return get_ref_l2_proj_mat(element_group)


def quadrature_based_stiffness_matrices(actx: ArrayContext, element_group):
    """todo
    """
    @keyed_memoize_in(
        actx, quadrature_based_stiffness_matrices,
        lambda grp: grp.discretization_key())
    def get_quad_ref_derivative_mats(grp):
        from meshmode.discretization.poly_element import diff_matrices

        mass_mat = thaw(quadrature_based_mass_matrix(grp), actx)
        quad_l2_proj_mat = thaw(quadrature_based_l2_projection_matrix(grp), actx)

        return actx.freeze(
            actx.from_numpy(
                np.asarray(
                    [quad_l2_proj_mat.T @ mass_mat @ dfmat @ quad_l2_proj_mat
                     for dfmat in diff_matrices(grp)]
                )
            )
        )

    return get_quad_ref_derivative_mats(element_group)


def surface_extrapolation_matrix(
    actx: ArrayContext, face_element_group, vol_element_group):
    """todo
    """
    @keyed_memoize_in(
        actx, surface_extrapolation_matrix,
        lambda face_grp, vol_grp: (face_grp.discretization_key(),
                                   vol_grp.discretization_key()))
    def get_surface_extrapolation_mat(face_grp, vol_grp):
        import modepy as mp
        basis = vol_grp.basis_obj()
        surface_quad_nodes = face_grp.unit_nodes
        faces = mp.faces_for_shape(face_grp.shape)

        # Array containing all surface quadrature nodes
        # in volume coordinates
        all_surface_nodes = np.array([
            face.map_to_volume(surface_quad_nodes)
            for face in faces
        ])

        vand_face = mp.vandermonde(basis.functions, all_surface_nodes)
        quad_l2_proj_mat = thaw(quadrature_based_l2_projection_matrix(grp), actx)

        return actx.freeze(actx.from_numpy(vand_face @ quad_l2_proj_mat))

    return get_surface_extrapolation_mat(face_element_group, vol_element_group)


def boundary_matrices(actx: ArrayContext, face_element_group):
    """todo
    """
    @keyed_memoize_in(
        actx, boundary_matrices,
        lambda face_grp: face_grp.discretization_key())
    def get_ref_boundary_mats(face_grp):
        import modepy as mp

        faces = mp.faces_for_shape(face_grp.shape)
        face_normals = [mp.face_normal(face) for face in faces]
        # NOTE: assumes same quadrature rule on all faces
        face_quad_weights = face_grp.weights
        nfaces = len(faces)

        return actx.freeze(
            actx.from_numpy(
                np.asarray(
                    [np.diag(np.concatenate([face_quad_weights*face_normals[i][dim]
                                             for i in range(nfaces)], axis=None))
                     for dim in range(face_grp.dim)]
                )
            )
        )

    return get_ref_boundary_mats(face_element_group)


def hybridized_sbp_operators(
        actx: ArrayContext, face_element_group, vol_element_group):
    """todo
    """
    @keyed_memoize_in(
        actx, hybridized_sbp_operators,
        lambda face_grp, vol_grp: (face_grp.discretization_key(),
                                   vol_grp.discretization_key()))
    def get_hybridized_sbp_mats(face_grp, vol_grp):
        q_mats = thaw(quadrature_based_stiffness_matrices(vol_grp), actx)
        e_mat = thaw(surface_extrapolation_matrix(face_grp, vol_grp), actx)
        b_mats = thaw(boundary_matrices(face_grp), actx)

        return actx.freeze(
            actx.from_numpy(
                np.asarray(
                    [0.5 * np.block(
                        [[q_mats[dim] - q_mats[dim].T, e_mat.T @ b_mats[dim]],
                         [-b_mats[dim] @ e_mat, b_mats[dim]]]
                    ) for dim in range(vol_grp.dim)]
                )
            )
        )

    return get_hybridized_sbp_mats(face_element_group, vol_element_group)


def _apply_hybridized_sbp_operator(dcoll, dd_v, dd_f, vec):
    if isinstance(vec, np.ndarray):
        return obj_array_vectorize(
            lambda vi: _apply_hybridized_sbp_operator(dcoll, dd_v, dd_f, vi), vec
        )

    pass


def weak_hybridized_local_sbp(
    dcoll: DiscretizationCollection, vec, dd_v=None, dd_f=None):
    """todo
    """
    if dd_v is None:
        dd_v = dof_desc.DOFDesc("vol", dof_desc.DISCR_TAG_BASE)

    if dd_f is None:
        dd_f = dof_desc.DOFDesc("all_faces", dof_desc.DISCR_TAG_BASE)

    return _apply_hybridized_sbp_operator(dcoll, dd_v, dd_f, vec)
