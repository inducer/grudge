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

import meshmode.mesh.generation as mgen

from pytools.obj_array import make_obj_array

from grudge import op, DiscretizationCollection
from grudge.dof_desc import DOFDesc
import grudge.sbp_op as sbp_op

import pytest

from grudge.array_context import PytestPyOpenCLArrayContextFactory
from arraycontext import pytest_generate_tests_for_array_contexts
pytest_generate_tests = pytest_generate_tests_for_array_contexts(
        [PytestPyOpenCLArrayContextFactory])

from arraycontext.container.traversal import thaw

from meshmode.dof_array import DOFArray, flat_norm
from meshmode.transform_metadata import FirstAxisIsElementsTag

import logging

logger = logging.getLogger(__name__)


@pytest.mark.parametrize("dim", [1, 2, 3])
@pytest.mark.parametrize("order", [2, 3])
def test_skew_hybridized_constant_preserving(actx_factory, dim, order):
    actx = actx_factory()

    from meshmode.mesh.generation import generate_regular_rect_mesh

    nel_1d = 5
    box_ll = -5.0
    box_ur = 5.0
    mesh = generate_regular_rect_mesh(
        a=(box_ll,)*dim,
        b=(box_ur,)*dim,
        nelements_per_axis=(nel_1d,)*dim)

    from grudge import DiscretizationCollection
    from grudge.dof_desc import DISCR_TAG_BASE, DISCR_TAG_QUAD
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
    face_discr = dcoll.discr_from_dd("all_faces")
    quad_discr = dcoll.discr_from_dd(dd_q)
    quad_face_discr = dcoll.discr_from_dd(dd_f)

    dtype = volm_discr.zeros(actx).entry_dtype

    q1 = DOFArray(
        actx,
        data=tuple(
            actx.einsum("dij,ej->ei",
                        sbp_op.hybridized_sbp_operators(
                            actx, fgrp, qfgrp,
                            vgrp, qgrp, dtype
                        ),
                        actx.from_numpy(
                            np.ones(
                                (vgrp.nelements,
                                 (qgrp.nunit_dofs
                                  + (qgrp.mesh_el_group.nfaces \
                                      * qfgrp.nunit_dofs))
                                )
                            )
                        ),
                        arg_names=("Q_mat", "one"),
                        tagged=(FirstAxisIsElementsTag(),))

            for vgrp, fgrp, qgrp, qfgrp in zip(
                volm_discr.groups,
                face_discr.groups,
                quad_discr.groups,
                quad_face_discr.groups)
        )
    )

    # ugly hack to get around cl.Array(float)...
    hopefully_zero = flat_norm(q1)
    assert bool(hopefully_zero < 1e-12)


# You can test individual routines by typing
# $ python test_grudge.py 'test_routine()'

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        pytest.main([__file__])
