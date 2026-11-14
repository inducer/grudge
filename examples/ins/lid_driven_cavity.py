__copyright__ = "Copyright (C) 2026 Kaushik Kulkarni"

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


from collections.abc import Callable
from math import factorial
from typing import Literal, TypeVar, cast

import numpy as np

import pyopencl as cl
import pyopencl.tools as cl_tools
from arraycontext import ArrayContext, BcastUntilActxArray, outer
from meshmode.discretization.connection import FACE_RESTR_ALL
from meshmode.discretization.poly_element import (
    PolynomialWarpAndBlend2DRestrictingGroupFactory,
    QuadratureSimplexGroupFactory,
)
from meshmode.dof_array import DOFArray
from meshmode.mesh.generation import generate_regular_rect_mesh
from pytools import obj_array

import grudge.geometry as geo
from grudge import op
from grudge.array_context import PyOpenCLArrayContext
from grudge.discretization import (
    DiscretizationCollection,
    make_discretization_collection,
)
from grudge.dof_desc import (
    DD_VOLUME_ALL,
    DISCR_TAG_BASE,
    DISCR_TAG_QUAD,
)
from grudge.geometry import metrics
from grudge.shortcuts import lsrk4_step
from grudge.trace_pair import (
    TracePair,
    local_interior_trace_pair,
    tracepair_with_discr_tag,
)


T = TypeVar("T", obj_array.ObjectArray1D[DOFArray], DOFArray)


def grad_term(
    dcoll: DiscretizationCollection,
    actx: ArrayContext,
    p: DOFArray,
) -> obj_array.ObjectArray1D[DOFArray]:
    r"""
    Returns :math:`G(p)` from Eq. (11), Section 2.1 of [Karakus2018]_.

    Use central traces on interior faces and :math:`p^*=p^-` on the
    velocity-Dirichlet boundaries, as prescribed in Section 2.1, p.5.

    :arg p: A scalar :class:`~meshmode.dof_array.DOFArray` on the base
        volume discretization, representing pressure field.
    """
    dd_vol = DD_VOLUME_ALL
    dd_allfaces = dd_vol.trace(FACE_RESTR_ALL)
    tpair = local_interior_trace_pair(dcoll, p)
    normal = geo.normal(actx, dcoll, tpair.dd)
    # Eq. (11): n*(p_star-p_minus)
    flux_correction = op.project(
        dcoll, tpair.dd, dd_allfaces, normal * (tpair.diff / 2)
    )

    # On both lid and walls, n*(p_star-p_minus) = 0.
    return cast("obj_array.ObjectArray1D[DOFArray]", op.local_grad(dcoll, p)) + cast(
        "obj_array.ObjectArray1D[DOFArray]",
        op.inverse_mass(dcoll, op.face_mass(dcoll, dd_allfaces, flux_correction)),
    )


def divergence_term(
    dcoll: DiscretizationCollection,
    actx: ArrayContext,
    u: obj_array.ObjectArray1D[DOFArray],
) -> DOFArray:
    r"""
    Returns :math:`D(u)` from Eq. (12), Section 2.1 of [Karakus2018]_.

    Use central traces on interior faces and :math:`u^*=g_D` on the
    cavity boundaries, as prescribed in Section 2.1, p.5.

    :arg u: A ``dcoll.dim`` dimensional :mod:`numpy` object array containing
        velocity :class:`~meshmode.dof_array.DOFArray` components on the base
        volume discretization. The mesh must have ``"lid"`` and ``"walls"``
        tags.

    The mesh must tag the top (+y) as ``"lid"`` and other faces as ``"walls"``.
    Both fields use exterior velocity ``(LID_VELOCITY, 0, ...)`` on the lid
    and zero on the stationary walls.
    """
    dd_vol = DD_VOLUME_ALL
    dd_allfaces = dd_vol.trace(FACE_RESTR_ALL)
    tpair = local_interior_trace_pair(dcoll, u)
    normal = geo.normal(actx, dcoll, tpair.dd)
    jump = tpair.diff
    # Eq. (12): (u_star-u_minus).n.
    correction = sum(jump[1:] * normal[1:], start=jump[0] * normal[0]) / 2
    flux_correction = op.project(dcoll, tpair.dd, dd_allfaces, correction)

    # Enforce Dirichlet BC on lid: (g_D-u_minus).n = -u_minus.n.
    dd_lid = dd_vol.trace("lid")
    u_lid = op.project(dcoll, dd_vol, dd_lid, u)
    normal_lid = geo.normal(actx, dcoll, dd_lid)
    lid_correction = -sum(u_lid[1:] * normal_lid[1:], start=u_lid[0] * normal_lid[0])
    flux_correction = flux_correction + op.project(
        dcoll, dd_lid, dd_allfaces, lid_correction
    )

    # Enforce Dirichlet BC on walls: g_D = 0.
    dd_walls = dd_vol.trace("walls")
    u_walls = op.project(dcoll, dd_vol, dd_walls, u)
    normal_walls = geo.normal(actx, dcoll, dd_walls)
    walls_correction = -sum(
        u_walls[1:] * normal_walls[1:], start=u_walls[0] * normal_walls[0]
    )
    flux_correction = flux_correction + op.project(
        dcoll, dd_walls, dd_allfaces, walls_correction
    )

    return cast("DOFArray", op.local_div(dcoll, u)) + cast(
        "DOFArray",
        op.inverse_mass(dcoll, op.face_mass(dcoll, dd_allfaces, flux_correction)),
    )


def laplacian_sipdg_velocity_term(
    dcoll: DiscretizationCollection,
    actx: ArrayContext,
    u: obj_array.ObjectArray1D[DOFArray],
    *,
    lid_velocity: np.floating,
) -> obj_array.ObjectArray1D[DOFArray]:
    r"""
    Returns the mass-weighted SIPDG Laplacian (i.e. :math:`ML(u)`) in Eq.
    (17b).

    :arg u: A ``dcoll.dim`` dimensional :mod:`numpy` object array containing
        one DOFArray per velocity component on the base volume grid.
    :arg lid_velocity: Prescribed x-velocity on the lid. All other components
        and all stationary walls have zero data. Use zero inside CG so the
        action stays linear; evaluate at zero input with nonzero lid data
        separately to obtain the boundary load.

    The mesh must tag the top (+y) as ``"lid"`` and other faces as ``"walls"``.
    Both fields use exterior velocity ``(LID_VELOCITY, 0, ...)`` on the lid
    and zero on the stationary walls.
    """
    dd_vol = DD_VOLUME_ALL
    dd_allfaces = dd_vol.trace(FACE_RESTR_ALL)
    order = max(group.order for group in dcoll.discr_from_dd(dd_vol).groups)
    # penalty factor in SIPDG (see Eq. (14) of Sec 2.1)
    penalty_factor = (order + 1) * (order + dcoll.dim) / dcoll.dim
    reference_volume = 2**dcoll.dim / factorial(dcoll.dim)
    reference_face_area = 2 ** (dcoll.dim - 1) / factorial(dcoll.dim - 1)
    element_volume = metrics.area_element(actx, dcoll, dd_vol) * reference_volume
    volume_tpair = local_interior_trace_pair(dcoll, element_volume)
    u_tpair = local_interior_trace_pair(dcoll, u)
    # Rows are velocity components; columns are spatial derivative directions.
    grad_u = cast("obj_array.ObjectArray2D[DOFArray]", op.local_grad(dcoll, u))
    grad_tpair = local_interior_trace_pair(dcoll, grad_u)
    dd_int = u_tpair.dd
    normal = geo.normal(actx, dcoll, dd_int)
    face_area = metrics.area_element(actx, dcoll, dd_int) * reference_face_area

    # See Eq. (14) of Sec 2.1
    tau = (
        penalty_factor
        * face_area
        * actx.np.maximum(1 / volume_tpair.interior, 1 / volume_tpair.exterior)
    )
    avg_grad = grad_tpair.avg
    normal_derivative = avg_grad @ normal
    # See Eq. (13) of Sec 2.1
    face_flux = op.project(
        dcoll, dd_int, dd_allfaces, normal_derivative + tau * u_tpair.diff
    )
    symmetry_flux = op.project(
        dcoll,
        dd_int,
        dd_allfaces,
        cast("obj_array.ObjectArray2D[DOFArray]", outer(u_tpair.diff / 2, normal)),
    )

    # Enforce Dirichlet BC on lid: u_star = g_D, as prescribed in Section 2.1, p.5.
    dd_lid = dd_vol.trace("lid")
    u_lid = op.project(dcoll, dd_vol, dd_lid, u)
    grad_lid = op.project(dcoll, dd_vol, dd_lid, grad_u)
    normal_lid = geo.normal(actx, dcoll, dd_lid)
    lid_area = metrics.area_element(actx, dcoll, dd_lid) * reference_face_area
    tau_lid = (
        2
        * penalty_factor
        * lid_area
        / op.project(dcoll, dd_vol, dd_lid, element_volume)
    )
    u_star_lid = obj_array.new_1d([
        actx.np.zeros_like(u_lid[0]) + lid_velocity,
        *(actx.np.zeros_like(component) for component in u_lid[1:]),
    ])
    lid_flux_correction = u_star_lid - u_lid
    lid_normal_derivative = grad_lid @ normal_lid
    face_flux = face_flux + op.project(
        dcoll,
        dd_lid,
        dd_allfaces,
        lid_normal_derivative + tau_lid * lid_flux_correction,
    )
    symmetry_flux = symmetry_flux + op.project(
        dcoll,
        dd_lid,
        dd_allfaces,
        cast(
            "obj_array.ObjectArray2D[DOFArray]",
            outer(lid_flux_correction, normal_lid),
        ),
    )

    # Enforce Dirichlet BC on walls: u_star = g_D = 0.
    dd_walls = dd_vol.trace("walls")
    u_walls = op.project(dcoll, dd_vol, dd_walls, u)
    grad_walls = op.project(dcoll, dd_vol, dd_walls, grad_u)
    normal_walls = geo.normal(actx, dcoll, dd_walls)
    wall_area = metrics.area_element(actx, dcoll, dd_walls) * reference_face_area
    tau_walls = (
        2
        * penalty_factor
        * wall_area
        / op.project(dcoll, dd_vol, dd_walls, element_volume)
    )
    u_star_walls = actx.np.zeros_like(u_walls)
    walls_flux_correction = u_star_walls - u_walls
    walls_normal_derivative = grad_walls @ normal_walls
    face_flux = face_flux + op.project(
        dcoll,
        dd_walls,
        dd_allfaces,
        walls_normal_derivative + tau_walls * walls_flux_correction,
    )
    symmetry_flux = symmetry_flux + op.project(
        dcoll,
        dd_walls,
        dd_allfaces,
        cast(
            "obj_array.ObjectArray2D[DOFArray]",
            outer(walls_flux_correction, normal_walls),
        ),
    )

    # Eq. (13)'s test-gradient term: D^T M (M^-1 M_face symmetry_flux).
    # weak_local_div is the transpose derivative action, without a minus sign.
    symmetry_lift = cast(
        "obj_array.ObjectArray2D[DOFArray]",
        op.inverse_mass(dcoll, op.face_mass(dcoll, dd_allfaces, symmetry_flux)),
    )
    mass_laplacian = -cast(
        "obj_array.ObjectArray1D[DOFArray]",
        op.weak_local_div(dcoll, grad_u + symmetry_lift),
    ) + cast(
        "obj_array.ObjectArray1D[DOFArray]",
        op.face_mass(dcoll, dd_allfaces, face_flux),
    )
    return mass_laplacian


def laplacian_sipdg_pressure_term(
    dcoll: DiscretizationCollection,
    actx: ArrayContext,
    p: DOFArray,
) -> DOFArray:
    r"""
    Returns the mass-weighted SIPDG Laplacian, i.e. :math:`ML(p)`, for Eq.
    (17c).

    :arg p: A scalar pressure-increment DOFArray on the base volume grid.
    """
    dd_vol = DD_VOLUME_ALL
    dd_allfaces = dd_vol.trace(FACE_RESTR_ALL)
    order = max(group.order for group in dcoll.discr_from_dd(dd_vol).groups)
    penalty_factor = (order + 1) * (order + dcoll.dim) / dcoll.dim
    reference_volume = 2**dcoll.dim / factorial(dcoll.dim)
    reference_face_area = 2 ** (dcoll.dim - 1) / factorial(dcoll.dim - 1)
    element_volume = metrics.area_element(actx, dcoll, dd_vol) * reference_volume
    volume_tpair = local_interior_trace_pair(dcoll, element_volume)
    p_tpair = local_interior_trace_pair(dcoll, p)
    grad_p = cast("obj_array.ObjectArray1D[DOFArray]", op.local_grad(dcoll, p))
    grad_tpair = local_interior_trace_pair(dcoll, grad_p)
    dd_int = p_tpair.dd
    normal = geo.normal(actx, dcoll, dd_int)
    face_area = metrics.area_element(actx, dcoll, dd_int) * reference_face_area
    tau = (
        penalty_factor
        * face_area
        * actx.np.maximum(1 / volume_tpair.interior, 1 / volume_tpair.exterior)
    )
    avg_grad = grad_tpair.avg
    normal_derivative = sum(avg_grad[1:] * normal[1:], start=avg_grad[0] * normal[0])
    # Eq. (13), with jump = exterior-interior.
    face_flux = op.project(
        dcoll, dd_int, dd_allfaces, normal_derivative + tau * p_tpair.diff
    )
    symmetry_flux = op.project(dcoll, dd_int, dd_allfaces, normal * (p_tpair.diff / 2))

    # Homogeneous Neumann: both boundary face terms are zero for our case (i.e.
    # lid driven cavity).

    # Eq. (13)'s test-gradient term: D^T M (M^-1 M_face symmetry_flux).
    # weak_local_div is the transpose derivative action, without a minus sign.
    symmetry_lift = cast(
        "obj_array.ObjectArray1D[DOFArray]",
        op.inverse_mass(dcoll, op.face_mass(dcoll, dd_allfaces, symmetry_flux)),
    )
    mass_laplacian = -cast(
        "DOFArray", op.weak_local_div(dcoll, grad_p + symmetry_lift)
    ) + cast("DOFArray", op.face_mass(dcoll, dd_allfaces, face_flux))
    return mass_laplacian


def nonlinear_convective_term(
    dcoll: DiscretizationCollection,
    actx: ArrayContext,
    ubar: obj_array.ObjectArray1D[DOFArray],
    utilde: obj_array.ObjectArray1D[DOFArray],
    lid_velocity: np.floating,
) -> obj_array.ObjectArray1D[DOFArray]:
    r"""
    Returns :math:`\tilde N(\bar u, \tilde u)` in Eq. (21). We use the local
    Lax--Friedrichs flux following Eq. (21) in Section 2.3 of [Karakus2018]_.

    :arg ubar: :math:`\bar{\mathbf{U}}` in Eq. (19) of  [Karakus2018]_ corresponding
        to the interpolated velocity in Eulerian frame in a Lagrangian
        subcycling subproblem.
    :arg ubar: :math:`\tilde{\mathbf{U}}_i` in Eq. (19) of  [Karakus2018]_ corresponding
        to the solution of a Lagrangian subcycling subproblem.

    The mesh must tag the top (+y) as ``"lid"`` and other faces as ``"walls"``.
    Both fields use exterior velocity ``(LID_VELOCITY, 0, ...)`` on the lid
    and zero on the stationary walls.
    """
    dd_vol = DD_VOLUME_ALL
    dd_quad = dd_vol.with_discr_tag(DISCR_TAG_QUAD)
    dd_allfaces_quad = dd_quad.trace(FACE_RESTR_ALL)

    # Sec (5.3) says "Use a sufficiently high-order quadrature rule to prevent
    # aliasing error" -> use overintegration here.
    ubar_quad = op.project(dcoll, dd_vol, dd_quad, ubar)
    utilde_quad = op.project(dcoll, dd_vol, dd_quad, utilde)

    # Eq. (21): grudge contracts the last tensor axis with the derivative,
    # so rows index transported components and columns index flux directions.
    convective_flux = cast(
        "obj_array.ObjectArray2D[DOFArray]", outer(utilde_quad, ubar_quad)
    )

    def normal_flux(
        ubar_tpair: TracePair[obj_array.ObjectArray1D[DOFArray]],
        utilde_tpair: TracePair[obj_array.ObjectArray1D[DOFArray]],
    ) -> obj_array.ObjectArray1D[DOFArray]:
        normal = geo.normal(actx, dcoll, ubar_tpair.dd)
        ubar_minus_dot_n = sum(
            ubar_tpair.interior[1:] * normal[1:],
            start=ubar_tpair.interior[0] * normal[0],
        )
        ubar_plus_dot_n = sum(
            ubar_tpair.exterior[1:] * normal[1:],
            start=ubar_tpair.exterior[0] * normal[0],
        )
        # lf_lambda corresponds to Lax Friedrich flux's \tilde{\Lambda} which is
        # defined in Sec 2.3 (p. 8)
        lf_lambda = actx.np.maximum(
            actx.np.abs(ubar_minus_dot_n), actx.np.abs(ubar_plus_dot_n)
        )
        # numerical_flux corresponds to $\tilde{\mathbf{F}}^*$ in Sec 2.3 (p 8)
        numerical_flux = (
            utilde_tpair.interior * ubar_minus_dot_n
            + utilde_tpair.exterior * ubar_plus_dot_n
        ) / 2 - lf_lambda * utilde_tpair.diff / 2
        return op.project(dcoll, ubar_tpair.dd, dd_allfaces_quad, numerical_flux)

    ubar_tpair = local_interior_trace_pair(dcoll, ubar)
    utilde_tpair = local_interior_trace_pair(dcoll, utilde)
    face_flux = normal_flux(
        tracepair_with_discr_tag(dcoll, DISCR_TAG_QUAD, ubar_tpair),
        tracepair_with_discr_tag(dcoll, DISCR_TAG_QUAD, utilde_tpair),
    )

    # Enforce Dirichlet BC on lid.
    dd_lid = dd_quad.trace("lid")
    ubar_lid = op.project(dcoll, dd_vol, dd_lid, ubar)
    utilde_lid = op.project(dcoll, dd_vol, dd_lid, utilde)
    g_lid = actx.np.zeros_like(utilde_lid)
    assert isinstance(g_lid, np.ndarray)
    g_lid[0] = g_lid[0] + lid_velocity
    face_flux = face_flux + normal_flux(
        TracePair(dd_lid, interior=ubar_lid, exterior=g_lid),
        TracePair(dd_lid, interior=utilde_lid, exterior=g_lid),
    )

    # Enforce Dirichlet BC on walls.
    dd_walls = dd_quad.trace("walls")
    ubar_walls = op.project(dcoll, dd_vol, dd_walls, ubar)
    utilde_walls = op.project(dcoll, dd_vol, dd_walls, utilde)
    g_walls = actx.np.zeros_like(utilde_walls)
    face_flux = face_flux + normal_flux(
        TracePair(dd_walls, interior=ubar_walls, exterior=g_walls),
        TracePair(dd_walls, interior=utilde_walls, exterior=g_walls),
    )

    return op.inverse_mass(
        dcoll,
        -cast("DOFArray", op.weak_local_div(dcoll, dd_quad, convective_flux))
        + cast("DOFArray", op.face_mass(dcoll, dd_allfaces_quad, face_flux)),
    )  # pyright: ignore[reportReturnType]


def advection_rhs(
    dcoll: DiscretizationCollection,
    actx: ArrayContext,
    n_s: int,
    dt: np.floating,
    t: np.floating,
    u_0: obj_array.ObjectArray1D[DOFArray],
    u_1: obj_array.ObjectArray1D[DOFArray] | None,
    *,
    lid_velocity: np.floating,
) -> obj_array.ObjectArray1D[DOFArray]:
    r"""
    Computes :math:`\hat{\mathbf{U}}` in the advection step, Eq. (17a), of
    [Karakus2018]_ for time *t*. We use the Lagrangian Subcyling Method which is
    described in Section 2.3 of the same paper.

    :arg n_s: Number of advection subcycles per macro timestep. [Karakus2018]_
        refer to this as :math:`N_s`.
    :arg dt: Macro timestep. [Karakus2018]_ refer to this as :math:`\Delta t`.
    :arg u_0: Velocity field in the Eulerian frame at the current time,
        :math:`\mathbf{U}(\mathbf{x}, t)`.
    :arg u_1: Previous Eulerian velocity, :math:`\mathbf{U}(\mathbf{x},t-\Delta t)`.
        Pass *None* for BDF1 startup when no previous velocity is available.
    :arg lid_velocity: Prescribed x-velocity on the lid.

    .. note::

        This routine uses BDF2 for the outer time discretization and a LSERK4
        method for each advection substep. Extensions to general-order BDF
        methods and alternative advection time integrators are left for future
        work.
    """
    # {{{ certain variables are comptime

    assert isinstance(n_s, int)

    # }}}

    dt_s = dt / n_s

    # {{{ Solve the subproblem corresponding to $\tilde{\mathbf{U}}_0$

    def u_tilde_0_rhs(
        t_stage: np.floating,
        y: obj_array.ObjectArray1D[DOFArray],
    ) -> obj_array.ObjectArray1D[DOFArray]:
        # u_bar corresponds to $\bar{\mathbf{U}}$ defined in Eq.(19) of Sec 2.3
        u_bar = u_0 if u_1 is None else u_0 + ((t_stage - t) / dt) * (u_0 - u_1)
        return -nonlinear_convective_term(
            dcoll,
            actx,
            u_bar,
            y,
            lid_velocity,
        )

    u_tilde_0 = u_0
    for isubcycle in range(n_s):
        u_tilde_0 = lsrk4_step(
            u_tilde_0,
            t + isubcycle * dt_s,
            dt_s,
            u_tilde_0_rhs,
        )

    # }}}

    # {{{ Solve the subproblem corresponding to $\tilde{\mathbf{U}}_1$

    if u_1 is None:
        return u_tilde_0

    def u_tilde_1_rhs(
        t_stage: np.floating,
        y: obj_array.ObjectArray1D[DOFArray],
    ) -> obj_array.ObjectArray1D[DOFArray]:
        # u_bar corresponds to $\bar{\mathbf{U}}$ defined in Eq.(19) of Sec 2.3
        u_bar = u_0 + ((t_stage - t) / dt) * (u_0 - u_1)
        return -nonlinear_convective_term(
            dcoll,
            actx,
            u_bar,
            y,
            lid_velocity,
        )

    u_tilde_1 = u_1
    for isubcycle in range(2 * n_s):
        u_tilde_1 = lsrk4_step(
            u_tilde_1,
            t - dt + isubcycle * dt_s,
            dt_s,
            u_tilde_1_rhs,
        )

    # }}}

    # Eq. (18): both transported histories are now evaluated at t+dt.
    return 2 * u_tilde_0 - 0.5 * u_tilde_1


def screened_poisson_residual(
    dcoll: DiscretizationCollection,
    actx: ArrayContext,
    dt: np.floating,
    reynolds: np.floating,
    is_first_timestep: bool,
    u_hat_hat: obj_array.ObjectArray1D[DOFArray],
    u_hat: obj_array.ObjectArray1D[DOFArray],
    p: DOFArray,
    *,
    lid_velocity: np.floating,
) -> obj_array.ObjectArray1D[DOFArray]:
    r"""
    Returns the mass-weighted residual of Eq. (17b) of [Karakus2018]_,
    as:

    .. math::

        r = ML\left(\hat{\hat{\mathbf{U}}};g_D\right)
            - \lambda M\hat{\hat{\mathbf{U}}}
            + M\left(\frac{\hat{\mathbf{U}}}{\nu\Delta t}
            - \frac{1}{\nu}\mathbf{G}P^n\right),
        \qquad \lambda=\frac{\gamma}{\nu\Delta t}.

    We use :math:`\gamma=1` for BDF1 startup and :math:`\gamma=3/2` for BDF2,
    and :math:`\nu=|\mathrm{lid\_velocity}|/\mathrm{Re}` for the unit cavity.

    :arg dt: Macro timestep, referred by :math:`\Delta t` in the paper.
    :arg reynolds: Reynolds number based on the lid speed and unit
        cavity length.
    :arg is_first_timestep: Is this the first timestep?
    :arg u_hat_hat: Guess velocity-vector iterate in the linear solve.
    :arg u_hat: Advection result :math:`\hat{\mathbf{U}}` from Eq. (18),
        replacing Eq. (17a) under subcycling.
    :arg p: Pressure field :math:`P^n` on the base volume grid.
    :arg lid_velocity: Prescribed nonzero x-velocity on the lid; other
        components and stationary walls have zero prescribed velocity.
    """

    # Ensure is_first_timestep is known at comptime.
    assert isinstance(is_first_timestep, bool)
    gamma = 1.0 if is_first_timestep else 1.5
    inv_nu = reynolds / abs(lid_velocity)

    return (
        laplacian_sipdg_velocity_term(dcoll, actx, u_hat_hat, lid_velocity=lid_velocity)
        - (
            (gamma * inv_nu / dt)
            * cast("obj_array.ObjectArray1D[DOFArray]", op.mass(dcoll, u_hat_hat))
        )
        + cast(
            "obj_array.ObjectArray1D[DOFArray]",
            op.mass(
                dcoll,
                (inv_nu / dt) * u_hat - inv_nu * grad_term(dcoll, actx, p),
            ),
        )
    )


def screened_poisson_action(
    dcoll: DiscretizationCollection,
    actx: ArrayContext,
    dt: np.floating,
    reynolds: np.floating,
    is_first_timestep: bool,
    u: obj_array.ObjectArray1D[DOFArray],
    *,
    lid_velocity: np.floating,
) -> obj_array.ObjectArray1D[DOFArray]:
    r"""
    Similar contract as :func:`screened_poisson_residual`, but evaluates the
    action of the of the linear operator instead of the resdual. Returns:

    .. math::

        Au = -M(u;0) + \frac{\gamma}{\nu\Delta t}Mu,
        \qquad \nu=|\mathrm{lid\_velocity}|/\mathrm{Re}.
    """
    # ensure is_first_timestep is known at comptime.
    assert isinstance(is_first_timestep, bool)
    gamma = 1.0 if is_first_timestep else 1.5
    inv_nu = reynolds / abs(lid_velocity)

    return -laplacian_sipdg_velocity_term(
        dcoll, actx, u, lid_velocity=0 * lid_velocity
    ) + cast(
        "obj_array.ObjectArray1D[DOFArray]",
        (gamma * inv_nu / dt)
        * cast("obj_array.ObjectArray1D[DOFArray]", op.mass(dcoll, u)),
    )


def pressure_poisson_residual(
    dcoll: DiscretizationCollection,
    actx: ArrayContext,
    dt: np.floating,
    is_first_timestep: bool,
    delta_p: DOFArray,
    u_hat_hat: obj_array.ObjectArray1D[DOFArray],
) -> DOFArray:
    r"""Return the mass-weighted residual of Eq. (17c) of [Karakus2018]_, as

    .. math::

        r = ML(\delta P) - \frac{\gamma}{\Delta t}
            M D\left(\hat{\hat{\mathbf{U}}}\right).

    Here :math:`\gamma=1` for BDF1 startup and :math:`\gamma=3/2` for BDF2.
    Use the first-order pressure increment :math:`\delta P=P^{n+1}-P^n`,
    as in Section 4. Homogeneous Neumann conditions leave a constant
    nullspace; the linear solver must handle it and RHS compatibility.

    :arg dt: Macro timestep :math:`\Delta t`.
    :arg is_first_timestep: Is this the first timestep?
    :arg delta_p: Scalar pressure-increment iterate on the base volume grid.
    :arg u_hat_hat: Velocity obtained from Eq. (17b).
    """
    # ensure is_first_timestep is known at comptime.
    assert isinstance(is_first_timestep, bool)
    gamma = 1.0 if is_first_timestep else 1.5

    return laplacian_sipdg_pressure_term(dcoll, actx, delta_p) - cast(
        "DOFArray",
        (gamma / dt) * op.mass(dcoll, divergence_term(dcoll, actx, u_hat_hat)),
    )


def pressure_poisson_action(
    dcoll: DiscretizationCollection,
    actx: ArrayContext,
    delta_p: DOFArray,
) -> DOFArray:
    r"""See :func:`pressure_poisson_residual`. Evaluate

    .. math::

        A\delta P = -ML(\delta P).
    """
    return -laplacian_sipdg_pressure_term(dcoll, actx, delta_p)


def preconditioned_cg(
    actx: ArrayContext,
    x0: T,
    f_residual: Callable[[T], T],
    f_action: Callable[[T], T],
    f_preconditioner: Callable[[T], T],
    tolerance: np.floating,
    maxiter: int,
) -> T:
    r"""
    Returns the solution to linear equation :math:`A x = b` with the
    preconditioner :math:`M` as computed using the preconditioned Conjugate
    Gradient Method.

    :arg x0: Initial guess vector.
    :arg f_residual: For a vector x, returns the residual :math:`b - A x`.
    :arg f_action: For a vector x, returns the action :math:`A x`.
    :arg f_preconditioner: For a residual vector r, returns the solution, z, in
        the equation :math:`M z = r`.
    :arg tolerance: Stop the iterative solver once the 2-norm of the residual
        is at most ``max(tolerance*norm(r0), tolerance)``, where ``r0`` is the
        residual of *x0* (libparanumal's ``ABS/REL-INITRESID`` criterion).
    :arg maxiter: Maximum iteration count.
    """
    # See <https://en.wikipedia.org/wiki/Conjugate_gradient_method#The_preconditioned_conjugate_gradient_method>.

    x_prev = x0
    r_prev = f_residual(x_prev)
    r0_norm = actx.to_numpy(actx.np.linalg.norm(r_prev))
    stop_norm = max(tolerance * r0_norm, tolerance)
    if r0_norm <= stop_norm:
        return x_prev
    z_prev = f_preconditioner(r_prev)
    p_prev = z_prev
    k = 0
    while k < maxiter:
        actioned_p_prev = f_action(p_prev)
        rho_prev = actx.np.vdot(r_prev, z_prev)
        alpha = rho_prev / actx.np.vdot(p_prev, actioned_p_prev)
        x = x_prev + cast("T", BcastUntilActxArray(actx, alpha) * p_prev)
        r = r_prev - cast("T", BcastUntilActxArray(actx, alpha) * actioned_p_prev)
        if actx.to_numpy(actx.np.linalg.norm(r)) <= stop_norm:
            return x
        z = f_preconditioner(r)
        beta = actx.np.vdot(r, z) / rho_prev
        p = z + cast("T", BcastUntilActxArray(actx, beta) * p_prev)
        x_prev, r_prev, z_prev, p_prev = x, r, z, p
        k += 1

    raise RuntimeError(f"PCG did not converge after {maxiter} iterations.")


def project_pressure_nullspace(actx: ArrayContext, v: DOFArray) -> DOFArray:
    r"""Return :math:`v'`, where

    .. math::

        v' = v - \frac{\mathbf{1}^{T}v}{\mathbf{1}^{T}\mathbf{1}}\mathbf{1}.

    Here :math:`\mathbf{1}` is the all-ones coefficient vector.
    """
    pressure_ones = actx.np.zeros_like(v) + 1
    pressure_ones_dot = actx.np.vdot(pressure_ones, pressure_ones)
    mean = actx.np.vdot(pressure_ones, v) / pressure_ones_dot
    return cast(
        "DOFArray",
        v - BcastUntilActxArray(actx, mean) * pressure_ones,
    )


def main(
    actx: ArrayContext,
    dim: int,
    reynolds: np.floating,
    nel_1d: int,
    order: int,
    n_s: int,
    dt: np.floating,
    t_final: np.floating,
    lid_velocity: np.floating,
    *,
    plot_to: Literal["pdf", "xwindow"] | None = None,
) -> None:
    r"""
    Implement a lid driven cavity solver using the scheme of [Karakus2018]_.

    :arg dim: Spatial dimension.
    :arg reynolds: Reynolds number to use.
    :arg nel_1d: Number of elements along one axis to be passed to
        :func:`generate_regular_rect_mesh`.
    :arg order: Use :math:`P^{\mathrm{disc}}_{\mathrm{order}}` as test and trial
        function spaces.
    :arg n_s: Number of advection subcycles per macro timestep.
    :arg plot_to: Plot nodal velocity magnitude in 2D: ``"pdf"`` saves
        ``lid-driven-cavity-final.pdf``; ``"xwindow"`` displays a window.
        Only the final field is plotted. ``None`` disables plotting.

    .. [Karakus2018] Ali Karakus, Noel Chalmers, Kasia Swirydowicz,
       and Timothy Warburton. *A GPU Accelerated Discontinuous Galerkin
       Incompressible Flow Solver* (2018).
       `arXiv:1801.00246 <https://arxiv.org/abs/1801.00246>`_.
    """
    mesh = generate_regular_rect_mesh(
        a=(0.0,) * dim,
        b=(1.0,) * dim,
        nelements_per_axis=(nel_1d,) * dim,
        boundary_tag_to_face={
            "lid": ["+y"],
            "walls": [
                sign + axis
                for axis in "xyz"[:dim]
                for sign in ("-", "+")
                if sign + axis != "+y"
            ],
        },
    )

    dcoll = make_discretization_collection(
        actx,
        mesh,
        discr_tag_to_group_factory={
            DISCR_TAG_BASE: PolynomialWarpAndBlend2DRestrictingGroupFactory(order),
            DISCR_TAG_QUAD: QuadratureSimplexGroupFactory(3 * order),
        },
    )

    u_n = obj_array.new_1d([dcoll.zeros(actx) for _ in range(dim)])
    u_n_minus_1: obj_array.ObjectArray1D[DOFArray] | None = None
    p = dcoll.zeros(actx)
    delta_p = dcoll.zeros(actx)

    t = 0.0

    while t < t_final:
        # Eq. (17a)
        u_hat = advection_rhs(
            dcoll, actx, n_s, dt, t, u_n, u_n_minus_1, lid_velocity=lid_velocity
        )
        # Eq. (17b)
        # {{{ use preconditioned CG to perform the linear solve

        is_first_timestep = u_n_minus_1 is None
        gamma = 1.0 if is_first_timestep else 1.5
        inv_screening = abs(lid_velocity) * dt / (gamma * reynolds)

        u_hat_hat = preconditioned_cg(
            actx,
            x0=u_n,
            f_residual=lambda u, first=is_first_timestep, u_hat=u_hat, p=p: (
                screened_poisson_residual(
                    dcoll,
                    actx,
                    dt,
                    reynolds,
                    first,
                    u,
                    u_hat,
                    p,
                    lid_velocity=lid_velocity,
                )
            ),
            f_action=lambda u, is_first_timestep=is_first_timestep: (
                screened_poisson_action(
                    dcoll,
                    actx,
                    dt,
                    reynolds,
                    is_first_timestep,
                    u,
                    lid_velocity=lid_velocity,
                )
            ),
            f_preconditioner=lambda r, inv_screening=inv_screening: (
                inv_screening
                * cast("obj_array.ObjectArray1D[DOFArray]", op.inverse_mass(dcoll, r))
            ),
            tolerance=np.float64(1.0e-8),
            maxiter=2000,
        )

        # }}}

        # {{{ Eq. (17c): Solve Pressure Poisson Equation.

        # Note [Karakus2018_] use pMG-AMG scheme to solve the Pressure Poisson
        # Equation, we use a plain Conjugate Gradient scheme to solve it. Using
        # pMG-AMG scheme is left as a TODO.
        delta_p = preconditioned_cg(
            actx,
            x0=delta_p,
            f_residual=lambda delta_p, u_hat_hat=u_hat_hat, first=is_first_timestep: (
                project_pressure_nullspace(
                    actx,
                    pressure_poisson_residual(
                        dcoll,
                        actx,
                        dt,
                        first,
                        project_pressure_nullspace(actx, delta_p),
                        u_hat_hat,
                    ),
                )
            ),
            f_action=lambda delta_p: project_pressure_nullspace(
                actx,
                pressure_poisson_action(
                    dcoll, actx, project_pressure_nullspace(actx, delta_p)
                ),
            ),
            f_preconditioner=lambda r: r,
            tolerance=np.float64(1.0e-8),
            maxiter=2000,
        )
        delta_p = project_pressure_nullspace(actx, delta_p)

        # }}}

        # {{{ Eq. (17d): Evolve u and p.

        u_n_minus_1, u_n = (
            u_n,
            u_hat_hat - (dt / gamma) * grad_term(dcoll, actx, delta_p),
        )
        p = p + delta_p

        # }}}

        t += dt
        print(f"At {t=}.")

    if plot_to is not None:
        if dim != 2:
            raise NotImplementedError(
                "Velocity magnitude plotting is only supported in 2D"
            )
        from matplotlib import rc_context
        from matplotlib.figure import Figure
        from matplotlib.tri import Triangulation

        from arraycontext import flatten

        with rc_context({"text.usetex": True, "font.family": "serif"}):
            if plot_to == "xwindow":
                import matplotlib.pyplot as plt

                fig, ax = plt.subplots()
            else:
                fig = Figure()
                ax = fig.subplots()

            nodes = actx.thaw(dcoll.nodes())
            x, y = [actx.to_numpy(flatten(node, actx)) for node in nodes]
            uabs = actx.to_numpy(flatten(actx.np.sqrt(sum(u_n**2)), actx))

            # {{{ DG to CG discretization for plotting

            points, inverse, counts = np.unique(
                np.round(np.column_stack((x, y)), decimals=12),
                axis=0,
                return_inverse=True,
                return_counts=True,
            )
            uabs = np.bincount(inverse, weights=uabs) / counts

            # }}}

            triangulation = Triangulation(points[:, 0], points[:, 1])
            fig.set_facecolor("white")
            ax.set_facecolor("white")
            levels = np.linspace(0, max(float(np.max(uabs)), 1e-15), 11)
            contours = ax.tricontour(
                triangulation, uabs, levels=levels, colors="black", linewidths=0.8
            )
            ax.set(xlabel="$x$", ylabel="$y$", aspect="equal")
            ax.clabel(contours, inline=True, fmt="%.3g", fontsize=8, colors="black")
            ax.set_title(rf"Velocity magnitude $|\mathbf{{u}}|$ at $t = {t:.6g}$")
            if plot_to == "pdf":
                fig.savefig("lid-driven-cavity-final.pdf")
            else:
                assert plot_to == "xwindow"
                plt.show()


if __name__ == "__main__":
    ctx = cl.create_some_context()
    cq = cl.CommandQueue(ctx)
    alloc = cl_tools.MemoryPool(cl_tools.ImmediateAllocator(cq))
    actx = PyOpenCLArrayContext(cq, alloc)
    main(
        actx,
        dim=2,
        reynolds=np.float64(100.0),
        nel_1d=32,
        order=4,
        n_s=4,
        dt=np.float64(0.01),
        # t_final=0.1 does not reach steady state. we use smaller t_final for CI
        # purposes. For steady state use t_final=80 to compare with Ghia et al.
        t_final=np.float64(0.1),
        lid_velocity=np.float64(1.0),
        plot_to=None,
    )
