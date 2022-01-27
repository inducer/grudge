"""Grudge operators modeling compressible, inviscid flows (Euler)

Model definitions
-----------------

.. autoclass:: EulerOperator

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
    thaw,
    dataclass_array_container,
    with_container_arithmetic,
    map_array_container
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

def conservative_to_primitive_vars(cv_state: ConservedEulerField, gamma=1.4):
    """Converts from conserved variables (density, momentum, total energy)
    into primitive variables (density, velocity, pressure).

    :arg cv_state: A :class:`ConservedEulerField` containing the conserved
        variables.
    :arg gamma: The isentropic expansion factor for a single-species gas
        (default set to 1.4).
    :returns: A :class:`Tuple` containing the primitive variables:
        (density, velocity, pressure).
    """
    rho = cv_state.mass
    rho_e = cv_state.energy
    rho_u = cv_state.momentum
    u = rho_u / rho
    p = (gamma - 1) * (rho_e - 0.5 * sum(rho_u * u))

    return (rho, u, p)


def conservative_to_entropy_vars(cv_state, gamma=1.4):
    """Converts from conserved variables (density, momentum, total energy)
    into entropy variables.

    :arg cv_state: A :class:`ConservedEulerField` containing the conserved
        variables.
    :arg gamma: The isentropic expansion factor for a single-species gas
        (default set to 1.4).
    :returns: A :class:`ConservedEulerField` containing the entropy variables.
    """
    actx = cv_state.array_context
    rho, u, p = conservative_to_primitive_vars(cv_state, gamma=gamma)

    u_square = sum(v ** 2 for v in u)
    s = actx.np.log(p) - gamma*actx.np.log(rho)
    rho_p = rho / p

    return ConservedEulerField(
        mass=((gamma - s)/(gamma - 1)) - 0.5 * rho_p * u_square,
        energy=-rho_p,
        momentum=rho_p * u
    )


def entropy_to_conservative_vars(ev_state, gamma=1.4):
    """Converts from entropy variables into conserved variables
    (density, momentum, total energy).

    :arg ev_state: A :class:`ConservedEulerField` containing the entropy
        variables.
    :arg gamma: The isentropic expansion factor for a single-species gas
        (default set to 1.4).
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


def compute_wavespeed(cv_state: ConservedEulerField, gamma=1.4):
    """Computes the total translational wavespeed.

    :arg cv_state: A :class:`ConservedEulerField` containing the conserved
        variables.
    :arg gamma: The isentropic expansion factor for a single-species gas
        (default set to 1.4).
    :returns: A :class:`~meshmode.dof_array.DOFArray` containing local wavespeeds.
    """
    actx = cv_state.array_context
    rho, u, p = conservative_to_primitive_vars(cv_state, gamma=gamma)

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
        dcoll: DiscretizationCollection, cv_state: ConservedEulerField, gamma=1.4):
    """Computes the (non-linear) volume flux for the
    Euler operator.

    :arg cv_state: A :class:`ConservedEulerField` containing the conserved
        variables.
    :arg gamma: The isentropic expansion factor for a single-species gas
        (default set to 1.4).
    :returns: A :class:`ConservedEulerField` containing the volume fluxes.
    """
    from arraycontext import outer

    rho, u, p = conservative_to_primitive_vars(cv_state, gamma=gamma)

    return ConservedEulerField(
        mass=cv_state.momentum,
        energy=u * (cv_state.energy + p),
        momentum=rho * outer(u, u) + np.eye(dcoll.dim) * p
    )


def euler_numerical_flux(
        dcoll: DiscretizationCollection, tpair: TracePair,
        gamma=1.4, lf_stabilization=False):
    """Computes the interface numerical flux for the Euler operator.

    :arg tpair: A :class:`grudge.trace_pair.TracePair` containing the conserved
        variables on the interior and exterior sides of element facets.
    :arg gamma: The isentropic expansion factor for a single-species gas
        (default set to 1.4).
    :arg lf_stabilization: A boolean denoting whether to apply Lax-Friedrichs
        dissipation.
    :returns: A :class:`ConservedEulerField` containing the interface fluxes.
    """
    dd_intfaces = tpair.dd
    dd_allfaces = dd_intfaces.with_dtag("all_faces")
    q_ll = tpair.int
    q_rr = tpair.ext
    actx = q_ll.array_context

    flux_tpair = TracePair(
        tpair.dd,
        interior=euler_volume_flux(dcoll, q_ll, gamma=gamma),
        exterior=euler_volume_flux(dcoll, q_rr, gamma=gamma)
    )
    num_flux = flux_tpair.avg
    normal = thaw(dcoll.normal(dd_intfaces), actx)

    if lf_stabilization:
        from arraycontext import outer

        # Compute jump penalization parameter
        lam = actx.np.maximum(compute_wavespeed(q_ll, gamma=gamma),
                              compute_wavespeed(q_rr, gamma=gamma))
        num_flux -= lam*outer(tpair.diff, normal)/2

    return op.project(dcoll, dd_intfaces, dd_allfaces, num_flux @ normal)


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
        return compute_wavespeed(state, gamma=self.gamma)

    def operator(self, t, q):
        dcoll = self.dcoll
        gamma = self.gamma
        qtag = self.qtag

        dd_base = DOFDesc("vol", DISCR_TAG_BASE)
        dq = DOFDesc("vol", qtag)
        df = DOFDesc("all_faces", qtag)
        df_int = DOFDesc("int_faces", qtag)

        def interp_to_quad_surf(u):
            return TracePair(
                df_int,
                interior=op.project(dcoll, "int_faces", df_int, u.int),
                exterior=op.project(dcoll, "int_faces", df_int, u.ext)
            )

        # Compute volume fluxes
        volume_fluxes = op.weak_local_div(
            dcoll, dq,
            euler_volume_flux(
                dcoll, op.project(dcoll, dd_base, dq, q), gamma=gamma)
        )

        # Compute interior interface fluxes
        interface_fluxes = (
            sum(
                euler_numerical_flux(
                    dcoll,
                    interp_to_quad_surf(tpair),
                    gamma=gamma,
                    lf_stabilization=self.lf_stabilization
                ) for tpair in op.interior_trace_pairs(dcoll, q)
            )
        )

        # Compute boundary fluxes
        if self.bdry_conditions is not None:
            for btag in self.bdry_conditions:
                boundary_condition = self.bdry_conditions[btag]
                dd_bc = as_dofdesc(btag).with_discr_tag(qtag)
                bc_flux = euler_numerical_flux(
                    dcoll,
                    boundary_condition.boundary_tpair(
                        dcoll=dcoll,
                        dd_bc=dd_bc,
                        restricted_state=op.project(dcoll, dd_base, dd_bc, q),
                        t=t
                    ),
                    gamma=gamma,
                    lf_stabilization=self.lf_stabilization
                )
                interface_fluxes = interface_fluxes + bc_flux

        return op.inverse_mass(
            dcoll,
            volume_fluxes - op.face_mass(dcoll, df, interface_fluxes)
        )

# }}}


# {{{ Entropy stable Euler operator

def flux_chandrashekar(dcoll: DiscretizationCollection, q_ll, q_rr, gamma=1.4):
    """Two-point volume flux based on the entropy conserving
    and kinetic energy preserving two-point flux in:

    - Chandrashekar (2013) Kinetic Energy Preserving and Entropy Stable Finite
    Volume Schemes for Compressible Euler and Navier-Stokes Equations
    [DOI](https://doi.org/10.4208/cicp.170712.010313a)

    :args q_ll: an array container for the "left" state.
    :args q_rr: an array container for the "right" state.
    :arg gamma: The isentropic expansion factor for a single-species gas
        (default set to 1.4).
    """
    dim = dcoll.dim
    actx = q_ll.array_context

    def ln_mean(x: DOFArray, y: DOFArray, epsilon=1e-4):
        f2 = (x * (x - 2 * y) + y * y) / (x * (x + 2 * y) + y * y)
        return actx.np.where(
            actx.np.less(f2, epsilon),
            (x + y) / (2 + f2*2/3 + f2*f2*2/5 + f2*f2*f2*2/7),
            (y - x) / actx.np.log(y / x)
        )

    rho_ll, u_ll, p_ll = conservative_to_primitive_vars(q_ll, gamma=gamma)
    rho_rr, u_rr, p_rr = conservative_to_primitive_vars(q_rr, gamma=gamma)

    beta_ll = 0.5 * rho_ll / p_ll
    beta_rr = 0.5 * rho_rr / p_rr
    specific_kin_ll = 0.5 * sum(v**2 for v in u_ll)
    specific_kin_rr = 0.5 * sum(v**2 for v in u_rr)

    rho_avg = 0.5 * (rho_ll + rho_rr)
    rho_mean = ln_mean(rho_ll,  rho_rr)
    beta_mean = ln_mean(beta_ll, beta_rr)
    beta_avg = 0.5 * (beta_ll + beta_rr)
    u_avg = 0.5 * (u_ll + u_rr)
    p_mean = 0.5 * rho_avg / beta_avg

    velocity_square_avg = specific_kin_ll + specific_kin_rr

    mass_flux = rho_mean * u_avg
    momentum_flux = np.outer(mass_flux, u_avg) + np.eye(dim) * p_mean
    energy_flux = (
        mass_flux * 0.5 * (1/(gamma - 1)/beta_mean - velocity_square_avg)
        + np.dot(momentum_flux, u_avg)
    )

    return ConservedEulerField(mass=mass_flux,
                               energy=energy_flux,
                               momentum=momentum_flux)


def entropy_stable_numerical_flux_chandrashekar(
        dcoll: DiscretizationCollection, tpair: TracePair,
        gamma=1.4, lf_stabilization=False):
    """Entropy stable numerical flux based on the entropy conserving
    and kinetic energy preserving two-point flux in:

    - Chandrashekar (2013) Kinetic Energy Preserving and Entropy Stable Finite
    Volume Schemes for Compressible Euler and Navier-Stokes Equations
    [DOI](https://doi.org/10.4208/cicp.170712.010313a)
    """
    dd_intfaces = tpair.dd
    dd_allfaces = dd_intfaces.with_dtag("all_faces")
    q_ll = tpair.int
    q_rr = tpair.ext
    actx = q_ll.array_context

    num_flux = flux_chandrashekar(dcoll, q_ll, q_rr, gamma=gamma)
    normal = thaw(dcoll.normal(dd_intfaces), actx)

    if lf_stabilization:
        from arraycontext import outer

        rho_ll, u_ll, p_ll = conservative_to_primitive_vars(q_ll, gamma=gamma)
        rho_rr, u_rr, p_rr = conservative_to_primitive_vars(q_rr, gamma=gamma)

        def compute_wavespeed(rho, u, p):
            return (
                actx.np.sqrt(np.dot(u, u)) + actx.np.sqrt(gamma * (p / rho))
            )

        # Compute jump penalization parameter
        lam = actx.np.maximum(compute_wavespeed(rho_ll, u_ll, p_ll),
                              compute_wavespeed(rho_rr, u_rr, p_rr))
        num_flux -= lam*outer(tpair.diff, normal)/2

    return op.project(dcoll, dd_intfaces, dd_allfaces, num_flux @ normal)


class EntropyStableEulerOperator(EulerOperator):

    def operator(self, t, q):
        from grudge.projection import volume_quadrature_project
        from grudge.interpolation import \
            volume_and_surface_quadrature_interpolation

        gamma = self.gamma
        qtag = self.qtag

        dd_base = DOFDesc("vol", DISCR_TAG_BASE)
        dq = DOFDesc("vol", qtag)
        df = DOFDesc("all_faces", qtag)

        dcoll = self.dcoll

        # Interpolate cv_state to vol quad grid: u_q = V_q u
        q_quad = op.project(dcoll, dd_base, dq, q)

        # Convert to projected entropy variables: v_q = V_h P_q v(u_q)
        entropy_vars = conservative_to_entropy_vars(q_quad, gamma=gamma)
        proj_entropy_vars = volume_quadrature_project(dcoll, dq, entropy_vars)

        # Compute conserved state in terms of the (interpolated)
        # projected entropy variables on the quad grid
        qtilde_allquad = \
            entropy_to_conservative_vars(
                # Interpolate projected entropy variables to
                # volume + surface quadrature grids
                volume_and_surface_quadrature_interpolation(
                    dcoll, dq, df, proj_entropy_vars
                ),
                gamma=gamma)

        # Compute volume derivatives using flux differencing
        from functools import partial
        from grudge.flux_differencing import volume_flux_differencing

        def _reshape(shape, ary):
            if not isinstance(ary, DOFArray):
                return map_array_container(partial(_reshape, shape), ary)

            return DOFArray(ary.array_context, data=tuple(
                subary.reshape(grp.nelements, *shape)
                # Just need group for determining the number of elements
                for grp, subary in zip(dcoll.discr_from_dd(dd_base).groups, ary)))

        flux_matrices = flux_chandrashekar(dcoll,
                                           _reshape((1, -1), qtilde_allquad),
                                           _reshape((-1, 1), qtilde_allquad),
                                           gamma=gamma)

        flux_diff = volume_flux_differencing(dcoll, dq, df, flux_matrices)

        def entropy_tpair(tpair):
            dd = tpair.dd
            dd_quad = dd.with_discr_tag(qtag)
            # Interpolate entropy variables to the surface quadrature grid
            ev_tpair = op.project(dcoll, dd, dd_quad, tpair)
            return TracePair(
                dd_quad,
                # Convert interior and exterior states to conserved variables
                interior=entropy_to_conservative_vars(ev_tpair.int, gamma=gamma),
                exterior=entropy_to_conservative_vars(ev_tpair.ext, gamma=gamma)
            )

        # Computing interface numerical fluxes
        interface_fluxes = (
            sum(
                entropy_stable_numerical_flux_chandrashekar(
                    dcoll,
                    entropy_tpair(tpair),
                    gamma=gamma,
                    lf_stabilization=self.lf_stabilization
                    # NOTE: Trace pairs consist of the projected entropy variables
                    # which will be converted to conserved variables in
                    # *entropy_tpair*
                ) for tpair in op.interior_trace_pairs(dcoll, proj_entropy_vars)
            )
        )

        # Compute boundary fluxes
        if self.bdry_conditions is not None:
            for btag in self.bdry_conditions:
                boundary_condition = self.bdry_conditions[btag]
                dd_bc = as_dofdesc(btag).with_discr_tag(qtag)
                bc_flux = entropy_stable_numerical_flux_chandrashekar(
                    dcoll,
                    boundary_condition.boundary_tpair(
                        dcoll=dcoll,
                        dd_bc=dd_bc,
                        # Pass modified conserved state to be used as
                        # the "interior" state for computing the boundary
                        # trace pair
                        restricted_state=entropy_to_conservative_vars(
                            op.project(dcoll, dd_base, dd_bc, proj_entropy_vars),
                            gamma=gamma
                        ),
                        t=t
                    ),
                    gamma=gamma,
                    lf_stabilization=self.lf_stabilization
                )
                interface_fluxes = interface_fluxes + bc_flux

        return op.inverse_mass(
            dcoll,
            -flux_diff - op.face_mass(dcoll, df, interface_fluxes)
        )

# }}}

# vim: foldmethod=marker
