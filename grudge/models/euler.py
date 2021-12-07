"""Grudge operators modeling compressible, inviscid flows (Euler)"""

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
    with_container_arithmetic
)

from meshmode.dof_array import DOFArray

from grudge.discretization import DiscretizationCollection
from grudge.dof_desc import DOFDesc, DISCR_TAG_BASE, as_dofdesc
from grudge.models import HyperbolicOperator
from grudge.trace_pair import TracePair

import grudge.op as op


# {{{ Array container for the Euler model

@with_container_arithmetic(bcast_obj_array=False,
                           bcast_container_types=(DOFArray, np.ndarray),
                           matmul=True,
                           rel_comparison=True)
@dataclass_array_container
@dataclass(frozen=True)
class EulerContainer:
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


# {{{ Variable transformation and helper routines

def conservative_to_primitive_vars(cv_state, gamma=1.4):
    """Converts from conserved variables (density, momentum, total energy)
    into primitive variables (density, velocity, pressure).

    :arg cv_state: A :class:`EulerContainer` containing the conserved
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

    return rho, u, p


def primitive_to_conservative_vars(prim_vars, gamma=1.4):
    """Converts from primitive variables (density, velocity, pressure)
    into conserved variables (density, momentum, total energy).

    :arg prim_vars: A :class:`Tuple` containing the primitive variables:
        (density, velocity, pressure).
    :arg gamma: The isentropic expansion factor for a single-species gas
        (default set to 1.4).
    :returns: A :class:`EulerContainer` containing the conserved
        variables.
    """
    rho, u, p = prim_vars
    inv_gamma_minus_one = 1/(gamma - 1)
    rhou = rho * u
    rhoe = p * inv_gamma_minus_one + 0.5 * sum(rhou * u)

    return EulerContainer(mass=rho, energy=rhoe, momentum=rhou)


def compute_wavespeed(cv_state, gamma=1.4):
    """Computes the total translational wavespeed.

    :arg cv_state: A :class:`EulerContainer` containing the conserved
        variables.
    :arg gamma: The isentropic expansion factor for a single-species gas
        (default set to 1.4).
    :returns: A :class:`DOFArray` containing local wavespeeds.
    """
    actx = cv_state.array_context
    rho, u, p = conservative_to_primitive_vars(cv_state, gamma=gamma)

    return (
        actx.np.sqrt(np.dot(u, u)) + actx.np.sqrt(gamma * (p / rho))
    )

# }}}


# {{{ Boundary condition types

class InviscidBCObject(metaclass=ABCMeta):

    def __init__(self, *, prescribed_state=None) -> None:
        self.prescribed_state = prescribed_state

    @abstractmethod
    def boundary_tpair(self, dcoll, dd_bc, state, t=0):
        pass


class PrescribedBC(InviscidBCObject):

    def boundary_tpair(self, dcoll, dd_bc, state, t=0):
        actx = state.array_context
        dd_base = as_dofdesc("vol").with_discr_tag(DISCR_TAG_BASE)

        return TracePair(
            dd_bc,
            interior=op.project(dcoll, dd_base, dd_bc, state),
            exterior=self.prescribed_state(thaw(dcoll.nodes(dd_bc), actx), t=t)
        )


class InviscidWallBC(InviscidBCObject):

    def boundary_tpair(self, dcoll, dd_bc, state, t=0):
        actx = state.array_context
        dd_base = as_dofdesc("vol").with_discr_tag(DISCR_TAG_BASE)
        nhat = thaw(dcoll.normal(dd_bc), actx)
        interior = op.project(dcoll, dd_base, dd_bc, state)

        return TracePair(
            dd_bc,
            interior=interior,
            exterior=EulerContainer(
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
        dcoll: DiscretizationCollection, cv_state, gamma=1.4):
    """Computes the (non-linear) volume flux for the
    Euler operator.

    :arg cv_state: A :class:`EulerContainer` containing the conserved
        variables.
    :arg gamma: The isentropic expansion factor for a single-species gas
        (default set to 1.4).
    :returns: A :class:`EulerContainer` containing the volume fluxes.
    """
    from arraycontext import outer

    rho, u, p = conservative_to_primitive_vars(cv_state, gamma=gamma)

    return EulerContainer(
        mass=cv_state.momentum,
        energy=u * (cv_state.energy + p),
        momentum=rho * outer(u, u) + np.eye(dcoll.dim) * p
    )


def euler_numerical_flux(
        dcoll: DiscretizationCollection, tpair: TracePair,
        gamma=1.4, lf_stabilization=False):
    """Computes the interface numerical flux for the Euler operator.

    :arg tpair: A :class:`TracePair` containing the conserved
        variables on the interior and exterior sides of element facets.
    :arg gamma: The isentropic expansion factor for a single-species gas
        (default set to 1.4).
    :arg lf_stabilization: A boolean denoting whether to apply Lax-Friedrichs
        dissipation.
    :returns: A :class:`EulerContainer` containing the interface fluxes.
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
        dq = DOFDesc("vol", qtag)
        df = DOFDesc("all_faces", qtag)
        df_int = DOFDesc("int_faces", qtag)

        def interp_to_quad(u):
            return op.project(dcoll, "vol", dq, u)

        def interp_to_quad_surf(u):
            return TracePair(
                df_int,
                interior=op.project(dcoll, "int_faces", df_int, u.int),
                exterior=op.project(dcoll, "int_faces", df_int, u.ext)
            )

        # Compute volume fluxes
        volume_fluxes = op.weak_local_div(
            dcoll, dq,
            interp_to_quad(euler_volume_flux(dcoll, q, gamma=gamma))
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
            bc_fluxes = sum(
                euler_numerical_flux(
                    dcoll,
                    self.bdry_conditions[btag].boundary_tpair(
                        dcoll,
                        as_dofdesc(btag).with_discr_tag(qtag),
                        q,
                        t=t
                    ),
                    gamma=gamma,
                    lf_stabilization=self.lf_stabilization
                ) for btag in self.bdry_conditions
            )
            interface_fluxes = interface_fluxes + bc_fluxes

        return op.inverse_mass(
            dcoll,
            volume_fluxes - op.face_mass(dcoll, df, interface_fluxes)
        )

# }}}


# {{{ Limiter

def limiter_zhang_shu(
        dcoll: DiscretizationCollection, quad_tag, state):
    """Implements the positivity-preserving limiter of
    - Zhang, Shu (2011)
        Maximum-principle-satisfying and positivity-preserving high-order schemes
        for conservation laws: survey and new developments
        [DOI](https://doi.org/10.1098/rspa.2011.0153)

    This limiter is currently only applied to the ``mass'' component of the state.

    :quad_tag: A quadrature tag denoting the volume quadrature discretization.
    :state: A :class:`EulerContainer` containing the state with components
        ``mass'', ``energy'', and ``momentum.''
    :returns: A :class:`EulerContainer` containing the limited state.
    """
    actx = state.array_context

    # Interpolate state to quadrature grid and
    # compute nodal and elementwise max/mins
    dd_base = as_dofdesc("vol")
    dd_quad = dd_base.with_discr_tag(quad_tag)
    mass = op.project(dcoll, "vol", dd_quad, state.mass)
    _mmax = op.nodal_max(dcoll, dd_quad, mass)
    _mmin = op.nodal_min(dcoll, dd_quad, mass)
    _mmax_i = op.elementwise_max(dcoll, mass)
    _mmin_i = op.elementwise_min(dcoll, mass)

    # Compute cell averages of the state
    from grudge.geometry import area_element

    inv_area_elements = 1./area_element(
        actx, dcoll,
        _use_geoderiv_connection=actx.supports_nonscalar_broadcasting)
    mass_cell_avgs = \
        inv_area_elements * op.elementwise_integral(dcoll, state.mass)

    # Compute minmod factor
    theta = actx.np.minimum(
        1,
        actx.np.minimum(
            abs((_mmax - mass_cell_avgs)/(_mmax_i - mass_cell_avgs)),
            abs((_mmin - mass_cell_avgs)/(_mmin_i - mass_cell_avgs))
        )
    )

    return EulerContainer(
        # Limit only mass
        mass=theta*(state.mass - mass_cell_avgs) + mass_cell_avgs,
        energy=state.energy,
        momentum=state.momentum
    )

# }}}


# vim: foldmethod=marker
