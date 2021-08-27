__copyright__ = "Copyright (C) 2021 University of Illinois Board of Trustees"

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

from grudge import DiscretizationCollection
from grudge.dof_desc import DOFDesc, DISCR_TAG_BASE, DISCR_TAG_QUAD
import grudge.sbp_op as sbp_op

import pytest

from grudge.array_context import PytestPyOpenCLArrayContextFactory
from arraycontext import pytest_generate_tests_for_array_contexts
pytest_generate_tests = pytest_generate_tests_for_array_contexts(
        [PytestPyOpenCLArrayContextFactory])

from arraycontext.container.traversal import thaw

import logging

logger = logging.getLogger(__name__)


@pytest.mark.parametrize("dim", [1, 2, 3])
@pytest.mark.parametrize("order", [1, 2, 3, 4])
def test_reference_element_sbp_operators(actx_factory, dim, order):
    actx = actx_factory()

    from meshmode.mesh.generation import generate_regular_rect_mesh

    nel_1d = 5
    box_ll = -5.0
    box_ur = 5.0
    mesh = generate_regular_rect_mesh(
        a=(box_ll,)*dim,
        b=(box_ur,)*dim,
        nelements_per_axis=(nel_1d,)*dim)

    from meshmode.discretization.poly_element import \
        default_simplex_group_factory, QuadratureSimplexGroupFactory

    dcoll = DiscretizationCollection(
        actx, mesh,
        discr_tag_to_group_factory={
            DISCR_TAG_BASE: default_simplex_group_factory(dim, order),
            DISCR_TAG_QUAD: QuadratureSimplexGroupFactory(2*order)
        }
    )

    dd_q = DOFDesc("vol", DISCR_TAG_QUAD)
    dd_f = DOFDesc("all_faces", DISCR_TAG_QUAD)

    volm_discr = dcoll.discr_from_dd("vol")
    quad_discr = dcoll.discr_from_dd(dd_q)
    quad_face_discr = dcoll.discr_from_dd(dd_f)
    dtype = volm_discr.zeros(actx).entry_dtype

    from meshmode.discretization.poly_element import diff_matrices

    for vgrp, qgrp, qfgrp in zip(volm_discr.groups,
                                 quad_discr.groups,
                                 quad_face_discr.groups):
        nq_vol = qgrp.nunit_dofs
        nq_faces = vgrp.shape.nfaces * qfgrp.nunit_dofs
        nq_total = nq_vol + nq_faces

        mass_mat = actx.to_numpy(
            thaw(
                sbp_op.quadrature_based_mass_matrix(
                    actx,
                    base_element_group=vgrp,
                    vol_quad_element_group=qgrp
                ),
                actx
            )
        )
        p_mat = actx.to_numpy(
            thaw(
                sbp_op.quadrature_based_l2_projection_matrix(
                    actx,
                    base_element_group=vgrp,
                    vol_quad_element_group=qgrp
                ),
                actx
            )
        )
        vdm_q = actx.to_numpy(
            thaw(
                sbp_op.volume_quadrature_interpolation_matrix(
                    actx,
                    base_element_group=vgrp,
                    vol_quad_element_group=qgrp
                ),
                actx
            )
        )

        # Checks Pq @ Vq = Minv @ Vq.T @ W @ Vq = I
        assert np.allclose(p_mat @ vdm_q,
                           np.identity(len(mass_mat)), rtol=1e-15)

        vf_mat = actx.to_numpy(
            thaw(
                sbp_op.surface_quadrature_interpolation_matrix(
                    actx,
                    base_element_group=vgrp,
                    face_quad_element_group=qfgrp,
                    dtype=dtype
                ),
                actx
            )
        )
        b_mats = actx.to_numpy(
            thaw(sbp_op.boundary_matrices(
                actx, vgrp, qfgrp), actx)
        )
        q_mats = np.asarray(
            [p_mat.T @ mass_mat @ diff_mat @ p_mat
                for diff_mat in diff_matrices(vgrp)]
        )
        e_mat = vf_mat @ p_mat

        qrst_skew_hybrid = actx.to_numpy(
            thaw(
                sbp_op.hybridized_sbp_operators(
                    actx, vgrp,
                    qgrp, qfgrp, dtype
                ),
                actx
            )
        )
        ones = np.ones(shape=(nq_total,))
        zeros = np.zeros(shape=(nq_total,))
        for idx in range(dim):
            # Checks the generalized SBP property:
            # Qi + Qi.T = E.T @ Bi @ E
            # c.f. Lemma 1. https://arxiv.org/pdf/1708.01243.pdf
            assert np.allclose(q_mats[idx] + q_mats[idx].T,
                               e_mat.T @ b_mats[idx] @ e_mat, rtol=1e-14)

            # Checks the SBP-like property for the skew hybridized operator
            # Qiskew + Qiskew.T = [0 0; 0 Bi]
            # c.f. Theorem 1 and Lemma 1. https://arxiv.org/pdf/1902.01828.pdf
            residual = qrst_skew_hybrid[idx] + qrst_skew_hybrid[idx].T
            residual[nq_vol:nq_vol+nq_faces, nq_vol:nq_vol+nq_faces] -= b_mats[idx]
            assert np.allclose(residual, np.zeros(residual.shape), rtol=1e-14)

            # Checks quadrature condition for: Qiskew @ ones = zeros
            # Qiskew + Qiskew.T = [0 0; 0 Bi]
            # c.f. Lemma 2. https://arxiv.org/pdf/1902.01828.pdf
            assert np.allclose(np.dot(qrst_skew_hybrid[idx], ones),
                               zeros, rtol=1e-14)


# You can test individual routines by typing
# $ python test_grudge.py 'test_routine()'

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        pytest.main([__file__])
