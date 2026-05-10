# pyright: reportUnknownArgumentType=false

"""Grudge operators modeling compressible, viscous flows (Navier-Stokes).

Uses the Bassi-Rebay 1 (BR1) scheme :cite:`Bassi1997` for the viscous fluxes.

Model definitions
-----------------

.. autoclass:: CNSOperator

Predefined initial conditions
------------------------------

.. autofunction:: poiseuille_flow

Helper routines and array containers
-------------------------------------

.. autofunction:: compute_temperature
.. autofunction:: compute_viscous_gradient
.. autofunction:: compute_ns_viscous_flux
.. autofunction:: ns_viscous_numerical_flux

Boundary conditions
-------------------

.. autoclass:: ViscousBCObject
.. autoclass:: NoSlipBC
.. autoclass:: InflowBC
.. autoclass:: OutflowBC
"""

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

from abc import ABCMeta, abstractmethod
from typing import TYPE_CHECKING

import numpy as np

from arraycontext import ArrayContext
from meshmode.dof_array import DOFArray
from pytools import obj_array

import grudge.geometry as geo
from grudge import op
from grudge.dof_desc import (
    DISCR_TAG_BASE,
    FACE_RESTR_ALL,
    VTAG_ALL,
    BoundaryDomainTag,
    DOFDesc,
    as_dofdesc,
)


def _as_bdry_dd(btag, discr_tag=DISCR_TAG_BASE) -> DOFDesc:
    """Convert *btag* (a BTAG_* class or a string name) to a :class:`DOFDesc`.

    :func:`~grudge.dof_desc.as_dofdesc` only accepts BTAG_* classes, not
    plain string boundary names.  This helper handles both.
    """
    return DOFDesc(BoundaryDomainTag(btag, VTAG_ALL), discr_tag)
from grudge.models import HyperbolicOperator
from grudge.models.euler import (
    ConservedEulerField,
    compute_wavespeed,
    conservative_to_primitive_vars,
    euler_numerical_flux,
    euler_volume_flux,
)
from grudge.trace_pair import TracePair


if TYPE_CHECKING:
    from grudge.discretization import DiscretizationCollection


# {{{ Temperature

def compute_temperature(
        cv_state: ConservedEulerField, *,
        gamma: float = 1.4,
        gas_const: float = 1.0) -> DOFArray:
    """Compute the temperature from conserved variables.

    Uses the ideal gas relation :math:`p = \\rho R T`, so
    :math:`T = p / (\\rho R)`.

    :arg cv_state: A :class:`~grudge.models.euler.ConservedEulerField` with
        the conserved variables.
    :arg gamma: The isentropic expansion factor (default 1.4).
    :arg gas_const: The specific gas constant :math:`R` (default 1.0).
    :returns: A :class:`~meshmode.dof_array.DOFArray` with the temperature.
    """
    rho, u, p = conservative_to_primitive_vars(cv_state, gamma=gamma)
    return p / (rho * gas_const)

# }}}


# {{{ BR1 gradient

def compute_viscous_gradient(
        actx: ArrayContext,
        dcoll: DiscretizationCollection,
        scalar_field: DOFArray,
        bdry_pairs: list,
        dd_allfaces: DOFDesc) -> np.ndarray:
    r"""Compute the BR1 gradient of a scalar field.

    Implements the Bassi-Rebay 1 (BR1) auxiliary-variable gradient:

    .. math::

        \boldsymbol{\sigma} = M^{-1}\!\left[-S^T q
        + \text{face\_mass}\!\left(\hat{q}\, \mathbf{n}\right)\right],

    where :math:`\hat{q} = \{q\}` (the average) at interior faces and
    :math:`\hat{q} = q_{\text{bc}}` at boundary faces.

    :arg scalar_field: A :class:`~meshmode.dof_array.DOFArray` on the
        volume discretization.
    :arg bdry_pairs: A list of ``(dd_bc, q_bc)`` pairs where *q_bc* is
        the boundary value of *scalar_field* on the boundary *dd_bc*.
    :arg dd_allfaces: A :class:`~grudge.dof_desc.DOFDesc` for all faces.
    :returns: A :class:`numpy.ndarray` of shape ``(dim,)`` containing
        :class:`~meshmode.dof_array.DOFArray`\ s for each gradient component.
    """
    def _facial_flux(tpair):
        normal = geo.normal(actx, dcoll, tpair.dd)
        dd_af = tpair.dd.with_domain_tag(
            BoundaryDomainTag(FACE_RESTR_ALL, VTAG_ALL))
        return op.project(dcoll, tpair.dd, dd_af, tpair.avg * normal)

    # Sum over interior trace pairs (using average {q})
    all_face_terms = sum(
        _facial_flux(tpair)
        for tpair in op.interior_trace_pairs(dcoll, scalar_field)
    )

    # Add boundary contributions (using prescribed BC values)
    for dd_bc, q_bc in bdry_pairs:
        normal = geo.normal(actx, dcoll, dd_bc)
        dd_af = dd_bc.with_domain_tag(BoundaryDomainTag(FACE_RESTR_ALL, VTAG_ALL))
        all_face_terms = all_face_terms + op.project(
            dcoll, dd_bc, dd_af, q_bc * normal)

    # σ = -M^{-1}(S^T q) + M^{-1} face_mass(q̂ ⊗ n)
    #   = -weak_local_grad(q) + M^{-1} face_mass(q̂ ⊗ n)
    return (
        -op.weak_local_grad(dcoll, scalar_field)
        + op.inverse_mass(
            dcoll,
            op.face_mass(dcoll, dd_allfaces, all_face_terms))
    )

# }}}


# {{{ Viscous flux

def compute_ns_viscous_flux(
        cv_state: ConservedEulerField,
        grad_u: np.ndarray,
        grad_T: np.ndarray,
        *,
        gamma: float = 1.4,
        mu: float,
        kappa: float) -> ConservedEulerField:
    r"""Compute the viscous flux tensor :math:`\boldsymbol{\sigma}_{\rm vis}`.

    The viscous flux is a :class:`~grudge.models.euler.ConservedEulerField`
    containing:

    .. math::

        \boldsymbol{\sigma}_{\rm vis}^{\rm mom}_{ij} &= \tau_{ij}, \\
        \boldsymbol{\sigma}_{\rm vis}^{\rm energy}_{j} &=
            \sum_i \tau_{ij}\, u_i - q_j,

    where :math:`\tau_{ij} = \mu(\partial_j u_i + \partial_i u_j
    - \tfrac{2}{3}\delta_{ij}\nabla\cdot\mathbf{u})` is the viscous stress
    tensor and :math:`q_j = -\kappa\,\partial_j T` is the heat flux.

    :arg cv_state: The conserved state.
    :arg grad_u: A ``(dim, dim)`` :class:`numpy.ndarray` where
        ``grad_u[i][j] = ∂u_i/∂x_j``.
    :arg grad_T: A ``(dim,)`` :class:`numpy.ndarray` where
        ``grad_T[j] = ∂T/∂x_j``.
    :arg mu: Dynamic viscosity.
    :arg kappa: Thermal conductivity.
    :returns: A :class:`~grudge.models.euler.ConservedEulerField` with the
        viscous flux components.
    """
    rho, u, p = conservative_to_primitive_vars(cv_state, gamma=gamma)
    dim = len(u)

    # Velocity divergence: ∇·u = Σ_k ∂u_k/∂x_k
    div_u = sum(grad_u[k][k] for k in range(dim))

    # Viscous stress tensor: τ_{ij} = μ(∂u_i/∂x_j + ∂u_j/∂x_i − 2/3 δ_{ij} ∇·u)
    tau = np.empty((dim, dim), dtype=object)
    for i in range(dim):
        for j in range(dim):
            tau[i, j] = mu * (grad_u[i][j] + grad_u[j][i])
            if i == j:
                tau[i, j] -= (2.0 / 3.0) * mu * div_u

    # Viscous energy flux: σ_e_j = Σ_i τ_{ij} u_i − q_j
    # where q_j = −κ ∂T/∂x_j, so −q_j = κ ∂T/∂x_j
    sigma_e = np.empty((dim,), dtype=object)
    for j in range(dim):
        sigma_e[j] = (
            sum(tau[i, j] * u[i] for i in range(dim))
            + kappa * grad_T[j]
        )

    # Zero mass flux (no viscous mass transport)
    zero_mass = obj_array.new_1d([0 * u[k] for k in range(dim)])

    return ConservedEulerField(
        mass=zero_mass,
        energy=sigma_e,
        momentum=tau,
    )


def ns_viscous_numerical_flux(
        actx: ArrayContext,
        dcoll: DiscretizationCollection,
        tpair: TracePair) -> ConservedEulerField:
    """Compute the BR1 numerical viscous flux for a trace pair of *sigma_vis*.

    Uses the average :math:`\\{\\boldsymbol{\\sigma}_{\\rm vis}\\}` at each
    interface as the numerical flux (the BR1 choice).

    :arg tpair: A :class:`~grudge.trace_pair.TracePair` holding the viscous
        flux :math:`\\boldsymbol{\\sigma}_{\\rm vis}` on each side of a face.
    :returns: The projected face flux
        :math:`\\{\\boldsymbol{\\sigma}_{\\rm vis}\\}\\cdot\\hat{n}`.
    """
    dd_allfaces = tpair.dd.with_domain_tag(
        BoundaryDomainTag(FACE_RESTR_ALL, VTAG_ALL))
    normal = geo.normal(actx, dcoll, tpair.dd)
    return op.project(dcoll, tpair.dd, dd_allfaces, tpair.avg @ normal)

# }}}


# {{{ Boundary conditions

class ViscousBCObject(metaclass=ABCMeta):
    """Abstract base class for viscous boundary conditions.

    Subclasses must implement :meth:`inviscid_boundary_tpair` for the
    inviscid (Euler) flux and :meth:`grad_bc_data` for the BR1 gradient.
    """

    @abstractmethod
    def inviscid_boundary_tpair(
            self,
            actx: ArrayContext,
            dcoll: DiscretizationCollection,
            dd_bc: DOFDesc,
            state: ConservedEulerField, t: float = 0) -> TracePair:
        """Return the :class:`~grudge.trace_pair.TracePair` used for
        the inviscid numerical flux at boundary *dd_bc*."""

    @abstractmethod
    def grad_bc_data(
            self,
            actx: ArrayContext,
            dcoll: DiscretizationCollection,
            dd_bc: DOFDesc,
            state: ConservedEulerField,
            *,
            gamma: float = 1.4,
            gas_const: float = 1.0) -> tuple[np.ndarray, DOFArray]:
        """Return ``(u_bc, T_bc)`` used for the BR1 gradient at *dd_bc*.

        :returns: A tuple ``(u_bc, T_bc)`` where *u_bc* is a ``(dim,)``
            object array of velocity components and *T_bc* is a
            :class:`~meshmode.dof_array.DOFArray` for temperature.
        """

    def viscous_boundary_flux(
            self,
            actx: ArrayContext,
            dcoll: DiscretizationCollection,
            dd_bc: DOFDesc,
            sigma_vis: ConservedEulerField) -> ConservedEulerField:
        """Return the viscous flux contribution at a boundary face.

        The default implementation uses the one-sided interior value of
        *sigma_vis*, consistent with the BR1 scheme.
        """
        dd_vol = as_dofdesc("vol", DISCR_TAG_BASE)
        dd_af = dd_bc.with_domain_tag(
            BoundaryDomainTag(FACE_RESTR_ALL, VTAG_ALL))
        normal = geo.normal(actx, dcoll, dd_bc)
        sigma_int = op.project(dcoll, dd_vol, dd_bc, sigma_vis)
        return op.project(dcoll, dd_bc, dd_af, sigma_int @ normal)


class NoSlipBC(ViscousBCObject):
    """No-slip (solid wall) boundary condition.

    Enforces zero velocity at the wall.  Temperature can be either
    prescribed (*T_wall* is given) for an isothermal wall, or reflected
    (adiabatic, zero normal gradient) when *T_wall* is ``None``.

    :arg T_wall: Wall temperature.  If ``None``, use the interior
        temperature (adiabatic approximation).
    """

    def __init__(self, T_wall=None):
        self.T_wall = T_wall

    def inviscid_boundary_tpair(
            self,
            actx: ArrayContext,
            dcoll: DiscretizationCollection,
            dd_bc: DOFDesc,
            state: ConservedEulerField, t: float = 0) -> TracePair:
        dd_vol = as_dofdesc("vol", DISCR_TAG_BASE)
        nhat = geo.normal(actx, dcoll, dd_bc)
        interior = op.project(dcoll, dd_vol, dd_bc, state)
        # Reflect momentum (no-slip: u → −u at wall,
        # effectively removing normal component and reversing tangential)
        return TracePair(
            dd_bc,
            interior=interior,
            exterior=ConservedEulerField(
                mass=interior.mass,
                energy=interior.energy,
                momentum=(
                    interior.momentum
                    - 2.0 * nhat * np.dot(interior.momentum, nhat)
                ),
            ),
        )

    def grad_bc_data(
            self,
            actx: ArrayContext,
            dcoll: DiscretizationCollection,
            dd_bc: DOFDesc,
            state: ConservedEulerField,
            *,
            gamma: float = 1.4,
            gas_const: float = 1.0) -> tuple:
        dd_vol = as_dofdesc("vol", DISCR_TAG_BASE)
        dim = dcoll.dim
        # Zero velocity at wall
        zeros = op.project(dcoll, dd_vol, dd_bc, state.mass) * 0
        u_bc = obj_array.new_1d([zeros for _ in range(dim)])
        # Temperature: prescribed wall value or interior
        if self.T_wall is not None:
            T_bc = zeros + self.T_wall
        else:
            T_int = compute_temperature(
                op.project(dcoll, dd_vol, dd_bc, state),
                gamma=gamma, gas_const=gas_const)
            T_bc = T_int
        return u_bc, T_bc


class InflowBC(ViscousBCObject):
    """Inflow boundary condition with prescribed state.

    :arg prescribed_state: A callable ``f(x_vec, t=0)`` returning a
        :class:`~grudge.models.euler.ConservedEulerField` with the
        prescribed inflow state.
    """

    def __init__(self, prescribed_state):
        self.prescribed_state = prescribed_state

    def inviscid_boundary_tpair(
            self,
            actx: ArrayContext,
            dcoll: DiscretizationCollection,
            dd_bc: DOFDesc,
            state: ConservedEulerField, t: float = 0) -> TracePair:
        dd_vol = as_dofdesc("vol", DISCR_TAG_BASE)
        return TracePair(
            dd_bc,
            interior=op.project(dcoll, dd_vol, dd_bc, state),
            exterior=self.prescribed_state(
                actx.thaw(dcoll.nodes(dd_bc)), t=t),
        )

    def grad_bc_data(
            self,
            actx: ArrayContext,
            dcoll: DiscretizationCollection,
            dd_bc: DOFDesc,
            state: ConservedEulerField,
            *,
            gamma: float = 1.4,
            gas_const: float = 1.0) -> tuple:
        x_vec = actx.thaw(dcoll.nodes(dd_bc))
        ext_state = self.prescribed_state(x_vec, t=0)
        _, u_bc, _ = conservative_to_primitive_vars(ext_state, gamma=gamma)
        T_bc = compute_temperature(ext_state, gamma=gamma, gas_const=gas_const)
        return u_bc, T_bc


class OutflowBC(ViscousBCObject):
    """Outflow (extrapolation) boundary condition.

    Copies the interior state to the exterior so that no information
    is reflected back into the domain.  This is appropriate for
    subsonic outflow when the flow is nearly fully developed.
    """

    def inviscid_boundary_tpair(
            self,
            actx: ArrayContext,
            dcoll: DiscretizationCollection,
            dd_bc: DOFDesc,
            state: ConservedEulerField, t: float = 0) -> TracePair:
        dd_vol = as_dofdesc("vol", DISCR_TAG_BASE)
        interior = op.project(dcoll, dd_vol, dd_bc, state)
        return TracePair(dd_bc, interior=interior, exterior=interior)

    def grad_bc_data(
            self,
            actx: ArrayContext,
            dcoll: DiscretizationCollection,
            dd_bc: DOFDesc,
            state: ConservedEulerField,
            *,
            gamma: float = 1.4,
            gas_const: float = 1.0) -> tuple:
        dd_vol = as_dofdesc("vol", DISCR_TAG_BASE)
        interior = op.project(dcoll, dd_vol, dd_bc, state)
        _, u_bc, _ = conservative_to_primitive_vars(interior, gamma=gamma)
        T_bc = compute_temperature(interior, gamma=gamma, gas_const=gas_const)
        return u_bc, T_bc

# }}}


# {{{ Poiseuille flow

def poiseuille_flow(x_vec, *, mu, gamma=1.4, gas_const=1.0,
                    pressure_drop=1.0, channel_height=1.0,
                    rho0=1.0, p0=10.0):
    r"""Initial/boundary condition for 2-D Poiseuille channel flow.

    Returns the exact steady-state solution for viscous flow between two
    parallel plates at :math:`y = 0` and :math:`y = H`, driven by a
    constant pressure gradient :math:`-G = dp/dx`:

    .. math::

        u_x(y) &= \frac{G}{2\mu}\,y\,(H - y), \\
        u_y    &= 0, \\
        \rho   &= \rho_0, \\
        p(x)   &= p_0 - G\,x, \\
        T      &= \frac{p}{\rho_0 R}.

    :arg x_vec: Coordinate arrays ``(x, y)``.
    :arg mu: Dynamic viscosity.
    :arg gamma: Isentropic expansion factor (default 1.4).
    :arg gas_const: Specific gas constant :math:`R` (default 1.0).
    :arg pressure_drop: Pressure gradient parameter :math:`G` (default 1.0).
    :arg channel_height: Channel height :math:`H` (default 1.0).
    :arg rho0: Reference density (default 1.0).
    :arg p0: Reference pressure at :math:`x = 0` (default 10.0).
    :returns: A :class:`~grudge.models.euler.ConservedEulerField`.
    """
    x, y = x_vec
    G = pressure_drop
    H = channel_height

    rho = rho0 + 0 * x  # constant density field
    u_x = G / (2 * mu) * y * (H - y)
    u_y = 0 * x
    p = p0 - G * x

    rho_u = obj_array.new_1d([rho * u_x, rho * u_y])
    rho_e = p / (gamma - 1) + 0.5 * (rho * u_x**2 + rho * u_y**2)

    return ConservedEulerField(mass=rho, energy=rho_e, momentum=rho_u)

# }}}


# {{{ CNS operator

class CNSOperator(HyperbolicOperator):
    r"""Compressible Navier-Stokes operator using the BR1 viscous scheme.

    Discretizes

    .. math::

        \partial_t \mathbf{Q}
        + \nabla\cdot(\mathbf{F}_{\rm inv} - \boldsymbol{\sigma}_{\rm vis}) = 0,

    where :math:`\mathbf{F}_{\rm inv}` is the inviscid Euler flux (see
    :func:`~grudge.models.euler.euler_volume_flux`) and
    :math:`\boldsymbol{\sigma}_{\rm vis}` is the viscous flux (see
    :func:`compute_ns_viscous_flux`).  Gradients for the viscous flux are
    computed with the BR1 scheme (:func:`compute_viscous_gradient`).

    :arg dcoll: A :class:`~grudge.discretization.DiscretizationCollection`.
    :arg bdry_conditions: A :class:`dict` mapping boundary tags to
        :class:`ViscousBCObject` instances.
    :arg mu: Dynamic viscosity.
    :arg Pr: Prandtl number (used to derive :math:`\kappa = \mu c_p / Pr`).
    :arg gamma: Isentropic expansion factor (default 1.4).
    :arg gas_const: Specific gas constant :math:`R` (default 1.0).
    :arg flux_type: Inviscid numerical flux type; ``"lf"`` for
        Lax-Friedrichs or ``"central"`` for central fluxes.
    :arg quadrature_tag: Optional quadrature tag for overintegration of
        the inviscid terms.
    """

    def __init__(
            self,
            dcoll: DiscretizationCollection,
            bdry_conditions: dict | None = None,
            *,
            mu: float,
            Pr: float = 0.72,
            gamma: float = 1.4,
            gas_const: float = 1.0,
            flux_type: str = "lf",
            quadrature_tag=None):
        self.dcoll = dcoll
        self.bdry_conditions = bdry_conditions or {}
        self.mu = mu
        # κ = μ c_p / Pr,  c_p = γ R / (γ−1)
        self.kappa = mu * (gamma * gas_const / (gamma - 1)) / Pr
        self.gamma = gamma
        self.gas_const = gas_const
        self.flux_type = flux_type
        self.lf_stabilization = (flux_type == "lf")
        self.qtag = quadrature_tag

    def max_characteristic_velocity(self, actx: ArrayContext, **kwargs):
        state = kwargs["state"]
        return compute_wavespeed(actx, state, gamma=self.gamma)

    def operator(self, actx: ArrayContext, t, q):
        dcoll = self.dcoll
        gamma = self.gamma
        gas_const = self.gas_const
        mu = self.mu
        kappa = self.kappa
        qtag = self.qtag
        dq = as_dofdesc("vol", qtag)
        df = as_dofdesc("all_faces", qtag)
        dd_vol = as_dofdesc("vol", DISCR_TAG_BASE)
        dd_allfaces = as_dofdesc("all_faces", DISCR_TAG_BASE)

        def interp_to_quad(u):
            return op.project(dcoll, "vol", dq, u)

        # ----------------------------------------------------------------
        # Part 1 – Inviscid (Euler) contribution
        # RHS_inv = weak_local_div(F_inv) − M^{−1} face_mass(F̂_inv · n)
        # ----------------------------------------------------------------
        euler_vol_fluxes = op.weak_local_div(
            dcoll, dq,
            interp_to_quad(euler_volume_flux(dcoll, q, gamma=gamma))
        )

        euler_face_fluxes = sum(
            euler_numerical_flux(
                actx, dcoll,
                op.tracepair_with_discr_tag(dcoll, qtag, tpair),
                gamma=gamma,
                lf_stabilization=self.lf_stabilization,
            )
            for tpair in op.interior_trace_pairs(dcoll, q)
        )

        if self.bdry_conditions:
            euler_bc_fluxes = sum(
                euler_numerical_flux(
                    actx, dcoll,
                    self.bdry_conditions[btag].inviscid_boundary_tpair(
                        actx, dcoll,
                        _as_bdry_dd(btag, qtag or DISCR_TAG_BASE),
                        q, t=t,
                    ),
                    gamma=gamma,
                    lf_stabilization=self.lf_stabilization,
                )
                for btag in self.bdry_conditions
            )
            euler_face_fluxes = euler_face_fluxes + euler_bc_fluxes

        # ----------------------------------------------------------------
        # Part 2 – BR1 gradients of primitive variables
        # ----------------------------------------------------------------
        rho, u, p = conservative_to_primitive_vars(q, gamma=gamma)
        T = compute_temperature(q, gamma=gamma, gas_const=gas_const)
        dim = dcoll.dim

        # Collect boundary values (u_bc, T_bc) from each BC object
        u_bdry_vals: dict[int, list] = {i: [] for i in range(dim)}
        T_bdry_vals: list = []

        for btag, bc in self.bdry_conditions.items():
            dd_bc = _as_bdry_dd(btag)
            u_bc, T_bc = bc.grad_bc_data(
                actx, dcoll, dd_bc, q,
                gamma=gamma, gas_const=gas_const)
            for i in range(dim):
                u_bdry_vals[i].append((dd_bc, u_bc[i]))
            T_bdry_vals.append((dd_bc, T_bc))

        # grad_u[i][j] = ∂u_i/∂x_j
        grad_u = np.empty((dim, dim), dtype=object)
        for i in range(dim):
            grad_u_i = compute_viscous_gradient(
                actx, dcoll, u[i],
                u_bdry_vals[i], dd_allfaces)
            for j in range(dim):
                grad_u[i, j] = grad_u_i[j]

        # grad_T[j] = ∂T/∂x_j
        grad_T = compute_viscous_gradient(
            actx, dcoll, T, T_bdry_vals, dd_allfaces)

        # ----------------------------------------------------------------
        # Part 3 – Viscous flux and its DG contribution
        # NS: ∂Q/∂t + ∇·(F_inv − σ_vis) = 0
        # So: ∂Q/∂t = (Euler RHS) + (−∇·σ_vis)
        #           = euler_rhs
        #             − weak_local_div(σ_vis)
        #             + M^{−1} face_mass(σ̂_vis · n)
        # ----------------------------------------------------------------
        sigma_vis = compute_ns_viscous_flux(
            q, grad_u, grad_T,
            gamma=gamma, mu=mu, kappa=kappa)

        # Volume viscous term (note the minus sign in the final assembly)
        vis_vol_fluxes = op.weak_local_div(dcoll, sigma_vis)

        # Interior viscous face fluxes (BR1: average σ_vis)
        vis_face_fluxes = sum(
            ns_viscous_numerical_flux(actx, dcoll, tpair)
            for tpair in op.interior_trace_pairs(dcoll, sigma_vis)
        )

        # Boundary viscous face fluxes
        if self.bdry_conditions:
            vis_bc_fluxes = sum(
                self.bdry_conditions[btag].viscous_boundary_flux(
                    actx, dcoll,
                    _as_bdry_dd(btag),
                    sigma_vis,
                )
                for btag in self.bdry_conditions
            )
            vis_face_fluxes = vis_face_fluxes + vis_bc_fluxes

        # ----------------------------------------------------------------
        # Combine and return
        # RHS = M^{−1}[ (euler_vol − vis_vol)
        #               − face_mass(euler_faces − vis_faces) ]
        # ----------------------------------------------------------------
        return op.inverse_mass(
            dcoll,
            (euler_vol_fluxes - vis_vol_fluxes)
            - op.face_mass(
                dcoll, df,
                euler_face_fluxes - vis_face_fluxes,  # type: ignore[operator]
            ),
        )

# }}}


# vim: foldmethod=marker
