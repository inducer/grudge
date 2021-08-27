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

from pytools.obj_array import make_obj_array

from arraycontext.container.traversal import thaw

import logging

logger = logging.getLogger(__name__)


@pytest.mark.parametrize("order", [2, 3, 4])
def test_entropy_projected_variables(actx_factory, order):

    def simple_smooth_function(actx, dcoll, t=0, gamma=1.4, dd=None):
        _beta = 5
        _center = np.zeros(shape=(dim,))
        _velocity = np.zeros(shape=(dim,))

        vortex_loc = _center + t * _velocity

        # Coordinates relative to vortex center
        nodes = thaw(dcoll.nodes(dd), actx)
        x_rel = nodes[0] - vortex_loc[0]
        y_rel = nodes[1] - vortex_loc[1]
 
        r = actx.np.sqrt(x_rel ** 2 + y_rel ** 2)
        expterm = _beta * actx.np.exp(1 - r ** 2)
        u = _velocity[0] - expterm * y_rel / (2 * np.pi)
        v = _velocity[1] + expterm * x_rel / (2 * np.pi)
        velocity = make_obj_array([u, v])
        mass = (1 - (gamma - 1) / (16 * gamma * np.pi ** 2)
                * expterm ** 2) ** (1 / (gamma - 1))
        momentum = mass * velocity
        p = mass ** gamma

        energy = p / (gamma - 1) + mass / 2 * (u ** 2 + v ** 2)

        result = np.empty((2+dim,), dtype=object)
        result[0] = mass
        result[1] = energy
        result[2:dim+2] = momentum

        return result

    actx = actx_factory()

    from meshmode.mesh.generation import generate_regular_rect_mesh

    dim = 2
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
    state = simple_smooth_function(actx, dcoll)
    ddq = DOFDesc("vol", DISCR_TAG_QUAD)

    from grudge.models.euler import \
        conservative_to_entropy_vars, entropy_to_conservative_vars

    entropy_vars = conservative_to_entropy_vars(dcoll, state)
    convserved_vars = entropy_to_conservative_vars(dcoll, entropy_vars)

    from meshmode.dof_array import flat_norm

    assert bool(flat_norm(convserved_vars - state, 2) < 1e-13)

    # # Compute uq = Vq * u
    # uq = sbp_op.quadrature_volume_interpolation(dcoll, ddq, state)
    # # Compute the entropy variables from state data: vq = v(uq)
    # vq = conservative_to_entropy_vars(dcoll, uq)
    # # Project the entropy variables to get modal coefficients
    # # v = Pq * vq
    # v = sbp_op.quadrature_project(dcoll, ddq, vq)
    # # Interpolate coefficients to quadrature grid:
    # # vtildeq = Vq * v
    # vtildeq = sbp_op.quadrature_volume_interpolation(dcoll, ddq, v)
    # # Construct the "auxiliary conservative variables" using
    # # the projected entropy variables:
    # # utildeq = q(vtildeq)
    # utildeq = entropy_to_conservative_vars(dcoll, vtildeq)


# You can test individual routines by typing
# $ python test_grudge.py 'test_routine()'

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        pytest.main([__file__])
