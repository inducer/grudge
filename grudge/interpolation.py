"""
.. currentmodule:: grudge.op

Interpolation
-------------

.. autofunction:: interp
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


import numpy as np

from arraycontext import (
    ArrayContext,
    map_array_container
)
from arraycontext import ArrayOrContainerT

from functools import partial

from meshmode.transform_metadata import FirstAxisIsElementsTag

from grudge.discretization import DiscretizationCollection
from grudge.dof_desc import DOFDesc

from meshmode.dof_array import DOFArray

from pytools import keyed_memoize_in


# FIXME: Should revamp interp and make clear distinctions
# between projection and interpolations.
# Related issue: https://github.com/inducer/grudge/issues/38
def interp(dcoll: DiscretizationCollection, src, tgt, vec):
    from warnings import warn
    warn("'interp' currently calls to 'project'",
         UserWarning, stacklevel=2)

    from grudge.projection import project

    return project(dcoll, src, tgt, vec)


# {{{ Interpolation matrices

def volume_quadrature_interpolation_matrix(
        actx: ArrayContext, base_element_group, vol_quad_element_group):
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


def surface_quadrature_interpolation_matrix(
        actx: ArrayContext, base_element_group, face_quad_element_group):
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


def volume_and_surface_interpolation_matrix(
        actx: ArrayContext,
        base_element_group, vol_quad_element_group, face_quad_element_group):
    @keyed_memoize_in(
        actx, volume_and_surface_interpolation_matrix,
        lambda base_grp, vol_quad_grp, face_quad_grp: (
            base_grp.discretization_key(),
            vol_quad_grp.discretization_key(),
            face_quad_grp.discretization_key()))
    def get_vol_surf_interpolation_matrix(base_grp, vol_quad_grp, face_quad_grp):
        vq_mat = actx.to_numpy(
            volume_quadrature_interpolation_matrix(
                actx,
                base_element_group=base_grp,
                vol_quad_element_group=vol_quad_grp))
        vf_mat = actx.to_numpy(
            surface_quadrature_interpolation_matrix(
                actx,
                base_element_group=base_grp,
                face_quad_element_group=face_quad_grp))
        return actx.freeze(actx.from_numpy(np.block([[vq_mat], [vf_mat]])))

    return get_vol_surf_interpolation_matrix(
        base_element_group, vol_quad_element_group, face_quad_element_group
    )

# }}}


def volume_and_surface_quadrature_interpolation(
        dcoll: DiscretizationCollection,
        dd_quad: DOFDesc,
        dd_face_quad: DOFDesc,
        vec: ArrayOrContainerT) -> ArrayOrContainerT:
    """todo.
    """
    if not isinstance(vec, DOFArray):
        return map_array_container(
            partial(volume_and_surface_quadrature_interpolation,
                    dcoll, dd_quad, dd_face_quad), vec
        )

    actx = vec.array_context
    discr = dcoll.discr_from_dd("vol")
    quad_volm_discr = dcoll.discr_from_dd(dd_quad)
    quad_face_discr = dcoll.discr_from_dd(dd_face_quad)

    return DOFArray(
        actx,
        data=tuple(
            actx.einsum("ij,ej->ei",
                        volume_and_surface_interpolation_matrix(
                            actx,
                            base_element_group=bgrp,
                            vol_quad_element_group=qvgrp,
                            face_quad_element_group=qfgrp
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


# vim: foldmethod=marker
