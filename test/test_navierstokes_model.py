from __future__ import annotations


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

import logging

import numpy as np
import pytest

from arraycontext import (
    ArrayContextFactory,
    pytest_generate_tests_for_array_contexts,
)

from grudge import op
from grudge.array_context import PytestPyOpenCLArrayContextFactory


logger = logging.getLogger(__name__)
pytest_generate_tests = pytest_generate_tests_for_array_contexts(
    [PytestPyOpenCLArrayContextFactory])


@pytest.mark.parametrize("order", [2, 3])
def test_poiseuille_convergence(actx_factory: ArrayContextFactory, order):
    r"""Convergence test for Poiseuille channel flow.

    Tests that the spatial residual of the compressible Navier-Stokes operator
    with BR1 viscous fluxes converges at order :math:`p + 1` when evaluated at
    the exact Poiseuille steady-state solution.

    The exact steady solution (x-momentum) is

    .. math::

        u_x(y) = \frac{G}{2\mu} y(H - y), \quad
        \rho = \rho_0 = 1, \quad p(x) = p_0 - G x.

    For this solution the mass and x-momentum equations are satisfied exactly,
    so their residuals should converge to zero as :math:`O(h^{p+1})`.
    """
    from meshmode.mesh.generation import generate_box_mesh
    from pytools.convergence import EOCRecorder

    from grudge.discretization import make_discretization_collection
    from grudge.dof_desc import DISCR_TAG_BASE
    from grudge.dt_utils import h_max_from_volume
    from grudge.models.navierstokes import (
        CNSOperator,
        InflowBC,
        NoSlipBC,
        OutflowBC,
        poiseuille_flow,
    )

    actx = actx_factory()

    # Flow parameters (low Mach number: p0 >> rho0 * u_max^2 / 2)
    gamma = 1.4
    gas_const = 1.0
    mu = 1.0      # high viscosity → small u_max → low Mach
    Pr = 0.72
    G = 1.0
    H = 1.0
    p0 = 100.0    # reference pressure (M ~ 0.01)
    rho0 = 1.0
    T_wall = p0 / (rho0 * gas_const)

    def exact_state(x_vec, t=0):
        return poiseuille_flow(
            x_vec,
            mu=mu,
            gamma=gamma,
            gas_const=gas_const,
            pressure_drop=G,
            channel_height=H,
            rho0=rho0,
            p0=p0,
        )

    eoc_mom = EOCRecorder()
    eoc_mass = EOCRecorder()

    for resolution in [4, 8, 16]:
        from meshmode.discretization.poly_element import (
            default_simplex_group_factory,
        )

        n = resolution + 1
        mesh = generate_box_mesh(
            axis_coords=[np.linspace(0, 1, n), np.linspace(0, H, n)],
            boundary_tag_to_face={
                "left": ["-x"],
                "right": ["+x"],
                "walls": ["-y", "+y"],
            },
        )

        dcoll = make_discretization_collection(
            actx,
            mesh,
            discr_tag_to_group_factory={
                DISCR_TAG_BASE: default_simplex_group_factory(
                    base_dim=2, order=order
                )
            },
        )

        h_max = actx.to_numpy(h_max_from_volume(dcoll, dim=dcoll.ambient_dim))
        nodes = actx.thaw(dcoll.nodes())

        bdry_conditions = {
            "left": InflowBC(exact_state),
            "right": OutflowBC(),
            "walls": NoSlipBC(T_wall=T_wall),
        }

        cns_op = CNSOperator(
            dcoll,
            bdry_conditions,
            mu=mu,
            Pr=Pr,
            gamma=gamma,
            gas_const=gas_const,
            flux_type="lf",
        )

        fields = exact_state(nodes)
        rhs = cns_op.operator(actx, 0.0, fields)

        # The mass equation is ∂ρ/∂t = 0 for Poiseuille flow:
        # ∇·(ρu) = 0 since ρ=const and u independent of x.
        err_mass = actx.to_numpy(op.norm(dcoll, rhs.mass, 2))

        # The x-momentum equation is ∂(ρu_x)/∂t = 0 for exact Poiseuille:
        # -∂p/∂x + μ ∂²u_x/∂y² = G - G = 0.
        err_mom = actx.to_numpy(op.norm(dcoll, rhs.momentum[0], 2))

        logger.info(
            "order=%d  resolution=%d  h=%.3e  "
            "|rhs_mass|=%.3e  |rhs_mom_x|=%.3e",
            order, resolution, h_max, err_mass, err_mom,
        )

        eoc_mass.add_data_point(h_max, max(err_mass, 1e-16))
        eoc_mom.add_data_point(h_max, max(err_mom, 1e-16))

    logger.info("Mass convergence:\n%s", eoc_mass.pretty_print(
        abscissa_label="h", error_label="|rhs_mass|"))
    logger.info("Momentum-x convergence:\n%s", eoc_mom.pretty_print(
        abscissa_label="h", error_label="|rhs_mom_x|"))

    # Expect spatial convergence at (at least) order p + 0.5
    assert eoc_mass.order_estimate() >= order + 0.5, (
        f"Mass residual EOC {eoc_mass.order_estimate():.2f} < {order + 0.5}")
    assert eoc_mom.order_estimate() >= order + 0.5, (
        f"Momentum-x residual EOC {eoc_mom.order_estimate():.2f} < {order + 0.5}")


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        pytest.main([__file__])



@pytest.mark.parametrize("order", [2, 3])
def test_poiseuille_convergence(actx_factory: ArrayContextFactory, order):
    r"""Convergence test for Poiseuille channel flow.

    Tests that the spatial error of the compressible Navier-Stokes operator
    with BR1 viscous fluxes converges at order :math:`p + 1` for a
    low-Mach-number Poiseuille channel flow.

    The exact steady solution is

    .. math::

        u_x(y) = \frac{G}{2\mu} y(H - y), \quad
        u_y = 0, \quad
        \rho = \rho_0, \quad
        p(x) = p_0 - G x,

    with isothermal walls at :math:`T_0 = p_0/(\rho_0 R)`.  For low Mach
    number (:math:`p_0 \gg \rho_0 u_{\max}^2 / 2`) this is an approximate
    steady solution of the compressible NS equations and the dominant error
    is the :math:`O(h^{p+1})` spatial discretization error.
    """
    from meshmode.mesh.generation import generate_box_mesh
    from pytools.convergence import EOCRecorder

    from grudge.discretization import make_discretization_collection
    from grudge.dof_desc import DISCR_TAG_BASE
    from grudge.dt_utils import h_max_from_volume, h_min_from_volume
    from grudge.models.navierstokes import (
        CNSOperator,
        InflowBC,
        NoSlipBC,
        OutflowBC,
        poiseuille_flow,
    )
    from grudge.shortcuts import rk4_step

    actx = actx_factory()

    # Flow parameters (low Mach number: p0 >> rho0 * u_max^2 / 2)
    gamma = 1.4
    gas_const = 1.0
    mu = 1.0          # high viscosity → low u_max
    Pr = 0.72
    G = 1.0           # pressure-gradient parameter
    H = 1.0           # channel height
    p0 = 100.0        # reference pressure  (M ~ u_max / c_sound << 1)
    rho0 = 1.0
    # u_max = G*H^2/(8*mu) = 0.125,  c_sound = sqrt(gamma*p0) ≈ 11.8  → M ≈ 0.01

    T_wall = p0 / (rho0 * gas_const)  # wall temperature from ideal gas law

    def exact_state(x_vec, t=0):
        return poiseuille_flow(
            x_vec,
            mu=mu,
            gamma=gamma,
            gas_const=gas_const,
            pressure_drop=G,
            channel_height=H,
            rho0=rho0,
            p0=p0,
        )

    eoc_rec = EOCRecorder()

    for resolution in [4, 8, 16]:
        from meshmode.discretization.poly_element import (
            default_simplex_group_factory,
        )

        # Channel mesh: [0,1] x [0,H] with labeled boundaries
        n = resolution + 1  # number of *points* per axis
        mesh = generate_box_mesh(
            axis_coords=[np.linspace(0, 1, n), np.linspace(0, H, n)],
            boundary_tag_to_face={
                "left": ["-x"],
                "right": ["+x"],
                "walls": ["-y", "+y"],
            },
        )

        dcoll = make_discretization_collection(
            actx,
            mesh,
            discr_tag_to_group_factory={
                DISCR_TAG_BASE: default_simplex_group_factory(
                    base_dim=2, order=order
                )
            },
        )

        h_max = actx.to_numpy(h_max_from_volume(dcoll, dim=dcoll.ambient_dim))
        nodes = actx.thaw(dcoll.nodes())

        bdry_conditions = {
            "left": InflowBC(exact_state),
            "right": OutflowBC(),
            "walls": NoSlipBC(T_wall=T_wall),
        }

        cns_op = CNSOperator(
            dcoll,
            bdry_conditions,
            mu=mu,
            Pr=Pr,
            gamma=gamma,
            gas_const=gas_const,
            flux_type="lf",
        )

        def rhs(t, q, op=cns_op):
            return op.operator(actx, t, q)

        compiled_rhs = actx.compile(rhs)

        fields = exact_state(nodes)

        # Use a CFL-limited timestep
        cfl = 0.1
        cn = 0.5 * (order + 1) ** 2
        dt = cfl * actx.to_numpy(h_min_from_volume(dcoll)) / cn
        final_time = dt * 5

        logger.info("resolution=%d, h_max=%.3e, dt=%.3e", resolution, h_max, dt)

        t = 0.0
        step = 0
        while t < final_time:
            fields = actx.thaw(actx.freeze(fields))
            fields = rk4_step(fields, t, dt, compiled_rhs)
            t += dt
            step += 1

        # Compare against the exact solution
        error_l2 = actx.to_numpy(
            op.norm(dcoll, fields - exact_state(nodes, t=t), 2)
        )
        logger.info("h_max=%.5e  L2-error=%.5e", h_max, error_l2)
        eoc_rec.add_data_point(h_max, error_l2)

    logger.info("\n%s", eoc_rec.pretty_print(
        abscissa_label="h", error_label="L2 Error"))

    # Expect spatial convergence at (at least) order p + 0.5
    assert eoc_rec.order_estimate() >= order + 0.5


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        pytest.main([__file__])
