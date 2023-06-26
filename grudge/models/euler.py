"""Grudge operators modeling compressible, inviscid flows (Euler)

Model definitions
-----------------

.. autoclass:: EulerOperator
.. autoclass:: EntropyStableEulerOperator

Predefined initial conditions
-----------------------------

.. autofunction:: vortex_initial_condition

Helper routines and array containers
------------------------------------

.. autoclass:: ConservedEulerField

.. autofunction:: conservative_to_primitive_vars
.. autofunction:: compute_wavespeed

.. autofunction:: euler_volume_flux
.. autofunction:: euler_numerical_flux

.. autofunction:: divergence_flux_chandrashekar
.. autofunction:: entropy_stable_numerical_flux_chandrashekar
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

from abc import ABCMeta, abstractmethod

from dataclasses import dataclass

from arraycontext import (
    dataclass_array_container,
    with_container_arithmetic,
    map_array_container, thaw,
    outer
)

from meshmode.dof_array import DOFArray

from grudge.discretization import DiscretizationCollection
from grudge.dof_desc import DOFDesc, DISCR_TAG_BASE, as_dofdesc
from grudge.models import HyperbolicOperator
from grudge.trace_pair import TracePair

from pytools.obj_array import make_obj_array

import grudge.op as op


# {{{ Array containers for the Euler model

@with_container_arithmetic(bcast_obj_array=False,
                           bcast_container_types=(DOFArray, np.ndarray),
                           matmul=True,
                           rel_comparison=True)
@dataclass_array_container
@dataclass(frozen=True)
class ConservedEulerField:
    mass: DOFArray
    energy: DOFArray
    momentum: np.ndarray

    @property
    def array_context(self):
        return self.mass.array_context

    @property
    def dim(self):
        return len(self.momentum)

# }}}


# {{{ Predefined initial conditions for the Euler model

def vortex_initial_condition(
        x_vec, t=0, center=5, mach_number=0.5, epsilon=1, gamma=1.4):
    """Initial condition adapted from Section 2 (equation 2) of:

    K. Mattsson, M. Svärd, M. Carpenter, and J. Nordström (2006).
    High-order accurate computations for unsteady aerodynamics.
    `DOI <https://doi.org/10.1016/j.compfluid.2006.02.004>`__.
    """
    x, y = x_vec
    actx = x.array_context

    fxyt = 1 - (((x - center) - t)**2 + y**2)
    expterm = actx.np.exp(fxyt/2)
    c = (epsilon**2 * (gamma - 1) * mach_number**2)/(8*np.pi**2)

    u = 1 - (epsilon*y/(2*np.pi)) * expterm
    v = ((epsilon*(x - center) - t)/(2*np.pi)) * expterm

    velocity = make_obj_array([u, v])
    rho = (1 - c * actx.np.exp(fxyt)) ** (1 / (gamma - 1))
    p = (rho ** gamma)/(gamma * mach_number**2)

    rhou = rho * velocity
    rhoe = p * (1/(gamma - 1)) + 0.5 * sum(rhou * velocity)

    return ConservedEulerField(mass=rho, energy=rhoe, momentum=rhou)

# }}}


# {{{ Variable transformation and helper routines

def conservative_to_primitive_vars(cv_state: ConservedEulerField, gamma: float):
    """Converts from conserved variables (density, momentum, total energy)
    into primitive variables (density, velocity, pressure).

    :arg cv_state: A :class:`ConservedEulerField` containing the conserved
        variables.
    :arg gamma: The isentropic expansion factor.
    :returns: A :class:`Tuple` containing the primitive variables:
        (density, velocity, pressure).
    """
    rho = cv_state.mass
    rho_e = cv_state.energy
    rho_u = cv_state.momentum
    u = rho_u / rho
    p = (gamma - 1) * (rho_e - 0.5 * sum(rho_u * u))

    return (rho, u, p)


def conservative_to_entropy_vars(cv_state: ConservedEulerField, gamma: float):
    """Converts from conserved variables (density, momentum, total energy)
    into entropy variables.

    :arg cv_state: A :class:`ConservedEulerField` containing the conserved
        variables.
    :arg gamma: The isentropic expansion factor for a single-species gas
        (default set to 1.4).
    :returns: A :class:`ConservedEulerField` containing the entropy variables.
    """
    actx = cv_state.array_context
    rho, u, p = conservative_to_primitive_vars(cv_state, gamma)

    u_square = sum(v ** 2 for v in u)
    s = actx.np.log(p) - gamma*actx.np.log(rho)
    rho_p = rho / p

    return ConservedEulerField(
        mass=((gamma - s)/(gamma - 1)) - 0.5 * rho_p * u_square,
        energy=-rho_p,
        momentum=rho_p * u
    )


def entropy_to_conservative_vars(ev_state: ConservedEulerField, gamma: float):
    """Converts from entropy variables into conserved variables
    (density, momentum, total energy).

    :arg ev_state: A :class:`ConservedEulerField` containing the entropy
        variables.
    :arg gamma: The isentropic expansion factor.
    :returns: A :class:`ConservedEulerField` containing the conserved variables.
    """
    actx = ev_state.array_context
    # See Hughes, Franca, Mallet (1986) A new finite element
    # formulation for CFD: (DOI: 10.1016/0045-7825(86)90127-1)
    inv_gamma_minus_one = 1/(gamma - 1)

    # Convert to entropy `-rho * s` used by Hughes, France, Mallet (1986)
    ev_state = ev_state * (gamma - 1)
    v1 = ev_state.mass
    v2t4 = ev_state.momentum
    v5 = ev_state.energy

    v_square = sum(v**2 for v in v2t4)
    s = gamma - v1 + v_square/(2*v5)
    rho_iota = (
        ((gamma - 1) / (-v5)**gamma)**(inv_gamma_minus_one)
    ) * actx.np.exp(-s * inv_gamma_minus_one)

    return ConservedEulerField(
        mass=-rho_iota * v5,
        energy=rho_iota * (1 - v_square/(2*v5)),
        momentum=rho_iota * v2t4
    )


def compute_wavespeed(cv_state: ConservedEulerField, gamma: float):
    """Computes the total translational wavespeed.

    :arg cv_state: A :class:`ConservedEulerField` containing the conserved
        variables.
    :arg gamma: The isentropic expansion factor.
    :returns: A :class:`~meshmode.dof_array.DOFArray` containing local wavespeeds.
    """
    actx = cv_state.array_context
    rho, u, p = conservative_to_primitive_vars(cv_state, gamma)

    return actx.np.sqrt(np.dot(u, u)) + actx.np.sqrt(gamma * (p / rho))

# }}}


# {{{ Boundary condition types

class InviscidBCObject(metaclass=ABCMeta):

    def __init__(self, *, prescribed_state=None) -> None:
        self.prescribed_state = prescribed_state

    @abstractmethod
    def boundary_tpair(
            self,
            dcoll: DiscretizationCollection,
            dd_bc: DOFDesc,
            restricted_state: ConservedEulerField, t=0):
        pass


class PrescribedBC(InviscidBCObject):

    def boundary_tpair(
            self,
            dcoll: DiscretizationCollection,
            dd_bc: DOFDesc,
            restricted_state: ConservedEulerField, t=0):
        actx = restricted_state.array_context

        return TracePair(
            dd_bc,
            interior=restricted_state,
            exterior=self.prescribed_state(thaw(dcoll.nodes(dd_bc), actx), t=t)
        )


class InviscidWallBC(InviscidBCObject):

    def boundary_tpair(
            self,
            dcoll: DiscretizationCollection,
            dd_bc: DOFDesc,
            restricted_state: ConservedEulerField, t=0):
        actx = restricted_state.array_context
        nhat = thaw(dcoll.normal(dd_bc), actx)
        interior = restricted_state

        return TracePair(
            dd_bc,
            interior=interior,
            exterior=ConservedEulerField(
                mass=interior.mass,
                energy=interior.energy,
                momentum=(
                    interior.momentum - 2.0 * nhat * np.dot(interior.momentum, nhat)
                )
            )
        )

# }}}


# {{{ Euler operator

def euler_volume_flux(
        dcoll: DiscretizationCollection,
        cv_state: ConservedEulerField, gamma: float):
    """Computes the (non-linear) volume flux for the
    Euler operator.

    :arg cv_state: A :class:`ConservedEulerField` containing the conserved
        variables.
    :arg gamma: The isentropic expansion factor.
    :returns: A :class:`ConservedEulerField` containing the volume fluxes.
    """
    rho, u, p = conservative_to_primitive_vars(cv_state, gamma)

    return ConservedEulerField(
        mass=cv_state.momentum,
        energy=u * (cv_state.energy + p),
        momentum=rho * outer(u, u) + np.eye(dcoll.dim) * p
    )


def euler_numerical_flux(
        dcoll: DiscretizationCollection, tpair: TracePair,
        gamma: float, dissipation=False):
    """Computes the interface numerical flux for the Euler operator.

    :arg tpair: A :class:`grudge.trace_pair.TracePair` containing the conserved
        variables on the interior and exterior sides of element facets.
    :arg gamma: The isentropic expansion factor.
    :arg dissipation: A boolean denoting whether to apply Lax-Friedrichs
        dissipation.
    :returns: A :class:`ConservedEulerField` containing the interface fluxes.
    """
    q_ll = tpair.int
    q_rr = tpair.ext
    actx = q_ll.array_context

    flux_tpair = TracePair(
        tpair.dd,
        interior=euler_volume_flux(dcoll, q_ll, gamma),
        exterior=euler_volume_flux(dcoll, q_rr, gamma)
    )
    num_flux = flux_tpair.avg
    normal = thaw(dcoll.normal(tpair.dd), actx)

    if dissipation:
        # Compute jump penalization parameter
        lam = actx.np.maximum(compute_wavespeed(q_ll, gamma),
                              compute_wavespeed(q_rr, gamma))
        num_flux -= lam*outer(tpair.diff, normal)/2

    return num_flux @ normal


class EulerOperator(HyperbolicOperator):
    r"""This operator discretizes the Euler equations:

    .. math::

        \partial_t \mathbf{Q} + \nabla\cdot\mathbf{F} = 0,

    where :math:`\mathbf{Q}` is the state vector containing density, momentum, and
    total energy, and :math:`\mathbf{F}` is the vector of inviscid fluxes
    (see :func:`euler_volume_flux`)
    """

    def __init__(self, dcoll: DiscretizationCollection,
                 bdry_conditions=None,
                 flux_type="lf",
                 gamma=1.4,
                 quadrature_tag=None):
        self.dcoll = dcoll
        self.bdry_conditions = bdry_conditions
        self.flux_type = flux_type
        self.gamma = gamma
        self.lf_stabilization = flux_type == "lf"
        self.qtag = quadrature_tag

    def max_characteristic_velocity(self, actx, **kwargs):
        state = kwargs["state"]
        return compute_wavespeed(state, self.gamma)

    def operator(self, t, q):
        dcoll = self.dcoll
        gamma = self.gamma
        qtag = self.qtag

        dissipation = self.lf_stabilization

        dd_base = as_dofdesc("vol", DISCR_TAG_BASE)
        dd_vol_quad = as_dofdesc("vol", qtag)
        dd_face_quad = as_dofdesc("all_faces", qtag)

        def interp_to_quad_surf(tpair):
            dd = tpair.dd
            dd_quad = dd.with_discr_tag(qtag)
            return TracePair(
                dd_quad,
                interior=op.project(dcoll, dd, dd_quad, tpair.int),
                exterior=op.project(dcoll, dd, dd_quad, tpair.ext)
            )

        interior_trace_pairs = [
            interp_to_quad_surf(tpair)
            for tpair in op.interior_trace_pairs(dcoll, q)
        ]

        # Compute volume derivatives
        volume_derivs = op.weak_local_div(
            dcoll, dd_vol_quad,
            euler_volume_flux(
                dcoll, op.project(dcoll, dd_base, dd_vol_quad, q), gamma)
        )

        # Compute interior interface fluxes
        interface_fluxes = (
            sum(
                op.project(dcoll, qtpair.dd, dd_face_quad,
                           euler_numerical_flux(dcoll, qtpair, gamma,
                                                dissipation=dissipation))
                for qtpair in interior_trace_pairs
            )
        )

        # Compute boundary fluxes
        if self.bdry_conditions is not None:
            for btag in self.bdry_conditions:
                boundary_condition = self.bdry_conditions[btag]
                dd_bc = as_dofdesc(btag).with_discr_tag(qtag)
                bc_flux = op.project(
                    dcoll,
                    dd_bc,
                    dd_face_quad,
                    euler_numerical_flux(
                        dcoll,
                        boundary_condition.boundary_tpair(
                            dcoll=dcoll,
                            dd_bc=dd_bc,
                            restricted_state=op.project(dcoll, dd_base, dd_bc, q),
                            t=t
                        ),
                        gamma,
                        dissipation=dissipation
                    )
                )
                interface_fluxes = interface_fluxes + bc_flux

        return op.inverse_mass(
            dcoll,
            volume_derivs - op.face_mass(dcoll, dd_face_quad, interface_fluxes)
        )

# }}}


# {{{ Entropy stable Euler operator

def divergence_flux_chandrashekar(
        dcoll: DiscretizationCollection,
        q_left: ConservedEulerField,
        q_right: ConservedEulerField, gamma: float):
    """Two-point volume flux based on the entropy conserving
    and kinetic energy preserving two-point flux in:

    Chandrashekar (2013) Kinetic Energy Preserving and Entropy Stable Finite
    Volume Schemes for Compressible Euler and Navier-Stokes Equations:
    `DOI <https://doi.org/10.4208/cicp.170712.010313a>`__.

    :args q_left: A :class:`ConservedEulerField` containing the "left" state.
    :args q_right: A :class:`ConservedEulerField` containing the "right" state.
    :arg gamma: The isentropic expansion factor.
    """
    dim = dcoll.dim
    actx = q_left.array_context

    def ln_mean(x: DOFArray, y: DOFArray, epsilon=1e-4):
        f2 = (x * (x - 2 * y) + y * y) / (x * (x + 2 * y) + y * y)
        return actx.np.where(
            actx.np.less(f2, epsilon),
            (x + y) / (2 + f2*2/3 + f2*f2*2/5 + f2*f2*f2*2/7),
            (y - x) / actx.np.log(y / x)
        )

    rho_left, u_left, p_left = conservative_to_primitive_vars(q_left, gamma)
    rho_right, u_right, p_right = conservative_to_primitive_vars(q_right, gamma)

    beta_left = 0.5 * rho_left / p_left
    beta_right = 0.5 * rho_right / p_right
    specific_kin_left = 0.5 * sum(v**2 for v in u_left)
    specific_kin_right = 0.5 * sum(v**2 for v in u_right)

    rho_avg = 0.5 * (rho_left + rho_right)
    rho_mean = ln_mean(rho_left,  rho_right)
    beta_mean = ln_mean(beta_left, beta_right)
    beta_avg = 0.5 * (beta_left + beta_right)
    u_avg = 0.5 * (u_left + u_right)
    p_mean = 0.5 * rho_avg / beta_avg

    velocity_square_avg = specific_kin_left + specific_kin_right

    mass_flux = rho_mean * u_avg
    momentum_flux = outer(mass_flux, u_avg) + np.eye(dim) * p_mean
    energy_flux = (
        mass_flux * 0.5 * (1/(gamma - 1)/beta_mean - velocity_square_avg)
        + np.dot(momentum_flux, u_avg)
    )

    return ConservedEulerField(mass=mass_flux,
                               energy=energy_flux,
                               momentum=momentum_flux)


def entropy_stable_numerical_flux_chandrashekar(
        dcoll: DiscretizationCollection, tpair: TracePair,
        gamma: float, dissipation=False):
    """Entropy stable numerical flux based on the entropy conserving
    and kinetic energy preserving two-point flux in:

    Chandrashekar (2013) Kinetic Energy Preserving and Entropy Stable Finite
    Volume Schemes for Compressible Euler and Navier-Stokes Equations
    `DOI <https://doi.org/10.4208/cicp.170712.010313a>`__.

    :arg tpair: A :class:`grudge.trace_pair.TracePair` containing the conserved
        variables on the interior and exterior sides of element facets.
    :arg gamma: The isentropic expansion factor.
    :arg dissipation: A boolean denoting whether to apply Lax-Friedrichs
        dissipation.
    :returns: A :class:`ConservedEulerField` containing the interface fluxes.
    """
    q_int = tpair.int
    q_ext = tpair.ext
    actx = q_int.array_context

    num_flux = divergence_flux_chandrashekar(
        dcoll, q_left=q_int, q_right=q_ext, gamma=gamma)
    normal = thaw(dcoll.normal(tpair.dd), actx)

    if dissipation:
        # Compute jump penalization parameter
        lam = actx.np.maximum(compute_wavespeed(q_int, gamma),
                              compute_wavespeed(q_ext, gamma))
        num_flux -= lam*outer(tpair.diff, normal)/2

    return num_flux @ normal


class EntropyStableEulerOperator(EulerOperator):
    """Discretizes the Euler equations using an entropy-stable
    discontinuous Galerkin discretization as outlined in (15)
    of `this paper <https://arxiv.org/pdf/1902.01828.pdf>`__.
    """

    def operator(self, t, q):
        from grudge.projection import volume_quadrature_project
        from grudge.interpolation import \
            volume_and_surface_quadrature_interpolation

        dcoll = self.dcoll
        gamma = self.gamma
        qtag = self.qtag
        dissipation = self.lf_stabilization

        dd_base = DOFDesc("vol", DISCR_TAG_BASE)
        dd_vol_quad = DOFDesc("vol", qtag)
        dd_face_quad = DOFDesc("all_faces", qtag)

        # Convert to projected entropy variables: v_q = P_q v(u_q)
        proj_entropy_vars = \
            volume_quadrature_project(
                dcoll, dd_vol_quad,
                conservative_to_entropy_vars(
                    # Interpolate state to vol quad grid: u_q = V_q u
                    op.project(dcoll, dd_base, dd_vol_quad, q), gamma))

        def modified_conserved_vars_tpair(tpair):
            dd = tpair.dd
            dd_quad = dd.with_discr_tag(qtag)
            # Interpolate entropy variables to the surface quadrature grid
            ev_tpair = op.project(dcoll, dd, dd_quad, tpair)
            return TracePair(
                dd_quad,
                # Convert interior and exterior states to conserved variables
                interior=entropy_to_conservative_vars(ev_tpair.int, gamma),
                exterior=entropy_to_conservative_vars(ev_tpair.ext, gamma)
            )

        # Compute interior trace pairs containing the modified conserved
        # variables (in terms of projected entropy variables)
        interior_trace_pairs = [
            modified_conserved_vars_tpair(tpair)
            for tpair in op.interior_trace_pairs(dcoll, proj_entropy_vars)
        ]

        from functools import partial
        from grudge.flux_differencing import volume_flux_differencing

        def _reshape(shape, ary):
            if not isinstance(ary, DOFArray):
                return map_array_container(partial(_reshape, shape), ary)

            return DOFArray(ary.array_context, data=tuple(
                subary.reshape(grp.nelements, *shape)
                # Just need group for determining the number of elements
                for grp, subary in zip(dcoll.discr_from_dd(dd_base).groups, ary)))

        # Compute the (modified) conserved state in terms of the projected
        # entropy variables on both the volume and surface nodes
        qtilde_vol_and_surf = \
            entropy_to_conservative_vars(
                # Interpolate projected entropy variables to
                # volume + surface quadrature grids
                volume_and_surface_quadrature_interpolation(
                    dcoll, dd_vol_quad, dd_face_quad, proj_entropy_vars), gamma)

        # FIXME: These matrices are actually symmetric. Could make use
        # of that to avoid redundant computation.
        flux_matrices = divergence_flux_chandrashekar(
            dcoll,
            _reshape((1, -1), qtilde_vol_and_surf),
            _reshape((-1, 1), qtilde_vol_and_surf),
            gamma
        )

        # Compute volume derivatives using flux differencing
        volume_derivs = -volume_flux_differencing(
            dcoll, dd_vol_quad, dd_face_quad, flux_matrices)

        # Computing interface numerical fluxes
        interface_fluxes = (
            sum(
                op.project(dcoll, qtpair.dd, dd_face_quad,
                           entropy_stable_numerical_flux_chandrashekar(
                               dcoll, qtpair, gamma, dissipation=dissipation))
                for qtpair in interior_trace_pairs
            )
        )

        # Compute boundary fluxes
        if self.bdry_conditions is not None:
            for btag in self.bdry_conditions:
                boundary_condition = self.bdry_conditions[btag]
                dd_bc = as_dofdesc(btag).with_discr_tag(qtag)
                bc_flux = op.project(
                    dcoll,
                    dd_bc,
                    dd_face_quad,
                    entropy_stable_numerical_flux_chandrashekar(
                        dcoll,
                        boundary_condition.boundary_tpair(
                            dcoll=dcoll,
                            dd_bc=dd_bc,
                            # Pass modified conserved state to be used as
                            # the "interior" state for computing the boundary
                            # trace pair
                            restricted_state=entropy_to_conservative_vars(
                                op.project(
                                    dcoll, dd_base, dd_bc, proj_entropy_vars),
                                gamma
                            ),
                            t=t
                        ),
                        gamma,
                        dissipation=dissipation
                    )
                )
                interface_fluxes = interface_fluxes + bc_flux

        return op.inverse_mass(
            dcoll,
            volume_derivs - op.face_mass(dcoll, dd_face_quad, interface_fluxes)
        )

# }}}

# vim: foldmethod=marker
