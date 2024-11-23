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

from abc import ABCMeta, abstractmethod
from dataclasses import dataclass

import numpy as np

from arraycontext import (
    ArrayContext,
    dataclass_array_container,
    with_container_arithmetic,
)
from meshmode.dof_array import DOFArray
from pytools.obj_array import make_obj_array

import grudge.geometry as geo
import grudge.op as op
from grudge.discretization import DiscretizationCollection
from grudge.dof_desc import DISCR_TAG_BASE, DOFDesc, as_dofdesc
from grudge.models import HyperbolicOperator
from grudge.trace_pair import TracePair


# {{{ Array containers for the Euler model

@with_container_arithmetic(bcast_obj_array=False,
                           bcast_container_types=(DOFArray, np.ndarray),
                           matmul=True,
                           rel_comparison=True,
                           )
@dataclass_array_container
@dataclass(frozen=True)
class ConservedEulerField:
    mass: DOFArray
    energy: DOFArray
    momentum: np.ndarray

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

    return rho, u, p


def compute_wavespeed(actx: ArrayContext, cv_state: ConservedEulerField, gamma=1.4):
    """Computes the total translational wavespeed.

    :arg cv_state: A :class:`ConservedEulerField` containing the conserved
        variables.
    :arg gamma: The isentropic expansion factor for a single-species gas
        (default set to 1.4).
    :returns: A :class:`~meshmode.dof_array.DOFArray` containing local wavespeeds.
    """
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
            actx: ArrayContext,
            dcoll: DiscretizationCollection,
            dd_bc: DOFDesc,
            state: ConservedEulerField, t=0):
        pass


class PrescribedBC(InviscidBCObject):

    def boundary_tpair(
            self,
            actx: ArrayContext,
            dcoll: DiscretizationCollection,
            dd_bc: DOFDesc,
            state: ConservedEulerField, t=0):
        actx = state.array_context
        dd_base = as_dofdesc("vol", DISCR_TAG_BASE)

        return TracePair(
            dd_bc,
            interior=op.project(dcoll, dd_base, dd_bc, state),
            exterior=self.prescribed_state(actx.thaw(dcoll.nodes(dd_bc)), t=t)
        )


class InviscidWallBC(InviscidBCObject):

    def boundary_tpair(
            self,
            actx: ArrayContext,
            dcoll: DiscretizationCollection,
            dd_bc: DOFDesc,
            state: ConservedEulerField, t=0):
        dd_base = as_dofdesc("vol", DISCR_TAG_BASE)
        nhat = geo.normal(actx, dcoll, dd_bc)
        interior = op.project(dcoll, dd_base, dd_bc, state)

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
        momentum=rho * outer(u, u) + np.eye(dcoll.dim, dtype=object) * p
    )


def euler_numerical_flux(
        actx: ArrayContext,
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
    from grudge.dof_desc import FACE_RESTR_ALL, VTAG_ALL, BoundaryDomainTag

    dd_intfaces = tpair.dd
    dd_allfaces = dd_intfaces.with_domain_tag(
        BoundaryDomainTag(FACE_RESTR_ALL, VTAG_ALL)
        )
    q_ll = tpair.int
    q_rr = tpair.ext

    flux_tpair = TracePair(
        tpair.dd,
        interior=euler_volume_flux(dcoll, q_ll, gamma=gamma),
        exterior=euler_volume_flux(dcoll, q_rr, gamma=gamma)
    )
    num_flux = flux_tpair.avg
    normal = geo.normal(actx, dcoll, dd_intfaces)

    if lf_stabilization:
        from arraycontext import outer

        # Compute jump penalization parameter
        lam = actx.np.maximum(compute_wavespeed(actx, q_ll, gamma=gamma),
                              compute_wavespeed(actx, q_rr, gamma=gamma))
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

    def max_characteristic_velocity(self, actx: ArrayContext, **kwargs):
        state = kwargs["state"]
        return compute_wavespeed(actx, state, gamma=self.gamma)

    def operator(self, actx: ArrayContext, t, q):
        dcoll = self.dcoll
        gamma = self.gamma
        qtag = self.qtag
        dq = as_dofdesc("vol", qtag)
        df = as_dofdesc("all_faces", qtag)

        def interp_to_quad(u):
            return op.project(dcoll, "vol", dq, u)

        # Compute volume fluxes
        volume_fluxes = op.weak_local_div(
            dcoll, dq,
            interp_to_quad(euler_volume_flux(dcoll, q, gamma=gamma))
        )

        # Compute interior interface fluxes
        interface_fluxes = (
            sum(
                euler_numerical_flux(
                    actx, dcoll,
                    op.tracepair_with_discr_tag(dcoll, qtag, tpair),
                    gamma=gamma,
                    lf_stabilization=self.lf_stabilization
                ) for tpair in op.interior_trace_pairs(dcoll, q)
            )
        )

        # Compute boundary fluxes
        if self.bdry_conditions is not None:
            bc_fluxes = sum(
                euler_numerical_flux(
                    actx, dcoll,
                    self.bdry_conditions[btag].boundary_tpair(
                        actx, dcoll,
                        as_dofdesc(btag, qtag),
                        q,
                        t=t
                    ),
                    gamma=gamma,
                    lf_stabilization=self.lf_stabilization
                ) for btag in self.bdry_conditions
            )
            interface_fluxes = interface_fluxes + bc_fluxes

        return op.inverse_mass(
            dcoll, dq,
            volume_fluxes - op.face_mass(dcoll, df, interface_fluxes)
        )

# }}}


# vim: foldmethod=marker
