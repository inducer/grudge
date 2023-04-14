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

import pytest

from grudge.array_context import \
    PytestPyOpenCLArrayContextFactory, PytestPytatoPyOpenCLArrayContextFactory
from arraycontext import (
    pytest_generate_tests_for_array_contexts, thaw,
)
pytest_generate_tests = pytest_generate_tests_for_array_contexts(
        [PytestPyOpenCLArrayContextFactory,
         PytestPytatoPyOpenCLArrayContextFactory])

import grudge.op as op

import numpy as np

import logging
logger = logging.getLogger(__name__)


@pytest.mark.parametrize("order", [1, 2, 3])
@pytest.mark.parametrize("esdg", [False, True])
def test_euler_vortex_convergence(actx_factory, order, esdg):
    from meshmode.mesh.generation import generate_regular_rect_mesh

    from grudge import DiscretizationCollection
    from grudge.dof_desc import DISCR_TAG_BASE, DISCR_TAG_QUAD
    from grudge.dt_utils import h_max_from_volume
    from grudge.models.euler import (
        vortex_initial_condition,
        EulerOperator,
        EntropyStableEulerOperator
    )
    from grudge.shortcuts import rk4_step

    from meshmode.discretization.poly_element import \
        (default_simplex_group_factory,
         QuadratureSimplexGroupFactory)

    from pytools.convergence import EOCRecorder

    actx = actx_factory()
    eoc_rec = EOCRecorder()
    quad_tag = DISCR_TAG_QUAD
    if esdg:
        operator_cls = EntropyStableEulerOperator
    else:
        operator_cls = EulerOperator

    if esdg and not actx.supports_nonscalar_broadcasting:
        pytest.xfail(
            "Flux-differencing computations requires an array context "
            "that supports non-scalar broadcasting"
        )

    for resolution in [8, 16, 32]:

        # {{{ discretization

        mesh = generate_regular_rect_mesh(
            a=(0, -5),
            b=(20, 5),
            nelements_per_axis=(2*resolution, resolution),
            periodic=(True, True))

        discr_tag_to_group_factory = {
            DISCR_TAG_BASE: default_simplex_group_factory(base_dim=2, order=order),
            DISCR_TAG_QUAD: QuadratureSimplexGroupFactory(2*order)
        }

        dcoll = DiscretizationCollection(
            actx, mesh,
            discr_tag_to_group_factory=discr_tag_to_group_factory
        )
        h_max = actx.to_numpy(h_max_from_volume(dcoll, dim=dcoll.ambient_dim))
        nodes = actx.thaw(dcoll.nodes())

        # }}}

        euler_operator = operator_cls(
            dcoll,
            flux_type="lf",
            gamma=1.4,
            quadrature_tag=quad_tag
        )

        def rhs(t, q):
            return euler_operator.operator(t, q)

        compiled_rhs = actx.compile(rhs)

        fields = vortex_initial_condition(nodes)

        from grudge.dt_utils import h_min_from_volume

        cfl = 0.125
        cn = 0.5*(order + 1)**2
        dt = cfl * actx.to_numpy(h_min_from_volume(dcoll)) / cn
        final_time = dt * 10

        logger.info("Timestep size: %g", dt)

        # {{{ time stepping

        step = 0
        t = 0.0
        last_q = None
        while t < final_time:
            fields = actx.thaw(actx.freeze(fields))
            fields = rk4_step(fields, t, dt, compiled_rhs)
            t += dt
            logger.info("[%04d] t = %.5f", step, t)
            last_q = fields
            last_t = t
            step += 1

        # }}}

        error_l2 = op.norm(
            dcoll,
            last_q - vortex_initial_condition(nodes, t=last_t),
            2
        )
        error_l2 = actx.to_numpy(error_l2)
        logger.info("h_max %.5e error %.5e", h_max, error_l2)
        eoc_rec.add_data_point(h_max, error_l2)

    logger.info("\n%s", eoc_rec.pretty_print(abscissa_label="h",
                                             error_label="L2 Error"))
    assert eoc_rec.order_estimate() >= order + 0.5


def test_entropy_variable_roundtrip(actx_factory):
    from grudge.models.euler import (
        entropy_to_conservative_vars,
        conservative_to_entropy_vars,
        vortex_initial_condition
    )

    actx = actx_factory()
    gamma = 1.4  # Adiabatic expansion factor for single-gas Euler model

    from meshmode.mesh.generation import generate_regular_rect_mesh

    dim = 2
    res = 5
    mesh = generate_regular_rect_mesh(
        a=(0, -5),
        b=(20, 5),
        nelements_per_axis=(2*res, res),
        periodic=(True, True))

    from meshmode.discretization.poly_element import \
        default_simplex_group_factory
    from grudge import DiscretizationCollection
    from grudge.dof_desc import DISCR_TAG_BASE

    order = 3
    dcoll = DiscretizationCollection(
        actx, mesh,
        discr_tag_to_group_factory={
            DISCR_TAG_BASE: default_simplex_group_factory(dim, order)
        }
    )

    # Fields in conserved variables
    fields = vortex_initial_condition(thaw(dcoll.nodes(), actx))

    # Map back and forth between entropy and conserved vars
    fields_ev = conservative_to_entropy_vars(fields, gamma)
    ev_fields_to_cons = entropy_to_conservative_vars(fields_ev, gamma)
    residual = ev_fields_to_cons - fields

    assert actx.to_numpy(op.norm(dcoll, residual.mass, np.inf)) < 1e-13
    assert actx.to_numpy(op.norm(dcoll, residual.energy, np.inf)) < 1e-13
    assert (
        actx.to_numpy(op.norm(dcoll, residual.momentum[i], np.inf)) < 1e-13
        for i in range(dim)
    )


# You can test individual routines by typing
# $ python test_grudge.py 'test_routine()'

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        pytest.main([__file__])

# vim: fdm=marker
