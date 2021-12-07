"""Euler operators"""

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

from dataclasses import dataclass
from arraycontext import (
    thaw,
    dataclass_array_container,
    with_container_arithmetic
)
from functools import partial

from meshmode.dof_array import DOFArray
from grudge.dof_desc import DOFDesc, DISCR_TAG_BASE, as_dofdesc

from grudge.models import HyperbolicOperator
from grudge.trace_pair import TracePair

import grudge.op as op


@with_container_arithmetic(bcast_obj_array=False,
                           bcast_container_types=(DOFArray, np.ndarray),
                           matmul=True,
                           rel_comparison=True)
@dataclass_array_container
@dataclass(frozen=True)
class EulerState:
    mass: DOFArray
    energy: DOFArray
    momentum: np.ndarray

    @property
    def array_context(self):
        return self.mass.array_context

    @property
    def dim(self):
        return len(self.momentum)

    def join(self):
        """Call :func:`join_conserved` on *self*."""
        return join_conserved(
            dim=self.dim,
            mass=self.mass,
            energy=self.energy,
            momentum=self.momentum)


def join_conserved(dim, mass, energy, momentum):

    def _aux_shape(ary, leading_shape):
        """:arg leading_shape: a tuple with which ``ary.shape``
        is expected to begin.
        """
        if (isinstance(ary, np.ndarray) and ary.dtype == object
                and not isinstance(ary, DOFArray)):
            naxes = len(leading_shape)
            if ary.shape[:naxes] != leading_shape:
                raise ValueError(
                    "array shape does not start with expected leading dimensions"
                )
            return ary.shape[naxes:]
        else:
            if leading_shape != ():
                raise ValueError(
                    "array shape does not start with expected leading dimensions"
                )
            return ()

    aux_shapes = [
        _aux_shape(mass, ()),
        _aux_shape(energy, ()),
        _aux_shape(momentum, (dim,))]

    from pytools import single_valued
    aux_shape = single_valued(aux_shapes)

    result = np.empty((2+dim,) + aux_shape, dtype=object)
    result[0] = mass
    result[1] = energy
    result[2:dim+2] = momentum

    return result


def make_eulerstate(dim, q):
    return EulerState(mass=q[0], energy=q[1], momentum=q[2:2+dim])


def euler_flux(dcoll, cv_state, gamma=1.4):
    """todo.
    """
    dim = dcoll.dim

    rho = cv_state.mass
    rho_e = cv_state.energy
    rho_u = cv_state.momentum
    u = rho_u / rho
    u_square = sum(v ** 2 for v in u)
    p = (gamma - 1) * (rho_e - 0.5 * rho * u_square)

    return EulerState(
        mass=rho_u,
        energy=u * (rho_e + p),
        momentum=np.outer(rho_u, u) + np.eye(dim) * p
    )


def euler_numerical_flux(dcoll, tpair, gamma=1.4, lf_stabilization=False):
    """todo.
    """
    dd_intfaces = tpair.dd
    dd_allfaces = dd_intfaces.with_dtag("all_faces")
    q_ll = tpair.int
    q_rr = tpair.ext
    actx = q_ll.array_context

    flux_tpair = TracePair(tpair.dd,
                           interior=euler_flux(dcoll, q_ll, gamma=gamma),
                           exterior=euler_flux(dcoll, q_rr, gamma=gamma))
    num_flux = flux_tpair.avg
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


class EulerOperator(HyperbolicOperator):

    def __init__(self, dcoll, bdry_conditions=None,
                 flux_type="lf", gamma=1.4, gas_const=287.1,
                 quadrature_tag=None):
        self.dcoll = dcoll
        self.bdry_conditions = bdry_conditions
        self.flux_type = flux_type
        self.gamma = gamma
        self.gas_const = gas_const
        self.lf_stabilization = flux_type == "lf"
        self.qtag = quadrature_tag

    def operator(self, t, q):
        dcoll = self.dcoll
        gamma = self.gamma
        qtag = self.qtag
        dq = DOFDesc("vol", qtag)
        df = DOFDesc("all_faces", qtag)
        df_int = DOFDesc("int_faces", qtag)

        def to_quad_vol(u):
            return op.project(dcoll, "vol", dq, u)

        def to_quad_tpair(u):
            return TracePair(
                df_int,
                interior=op.project(dcoll, "int_faces", df_int, u.int),
                exterior=op.project(dcoll, "int_faces", df_int, u.ext)
            )

        euler_flux_faces = (
            sum(
                euler_numerical_flux(
                    dcoll,
                    to_quad_tpair(tpair),
                    gamma=gamma,
                    lf_stabilization=self.lf_stabilization
                ) for tpair in op.interior_trace_pairs(dcoll, q)
            )
        )

        if self.bdry_conditions is not None:
            bc_fluxes = sum(
                euler_numerical_flux(
                    dcoll,
                    bc.boundary_tpair(dcoll, q, t=t),
                    gamma=gamma,
                    lf_stabilization=self.lf_stabilization
                ) for bc in self.bdry_conditions
            )
            euler_flux_faces = euler_flux_faces + bc_fluxes

        return op.inverse_mass(
            dcoll,
            op.weak_local_div(
                dcoll, dq, to_quad_vol(euler_flux(dcoll, q, gamma=gamma))
            ) - op.face_mass(dcoll, df, euler_flux_faces),
            dd_quad=dq
        )

    def max_characteristic_velocity(self, actx, **kwargs):
        state = kwargs["state"]
        gamma = self.gamma
        rho, u, p = conservative_to_primitive_vars(state, gamma=gamma)

        return actx.np.sqrt(np.dot(u, u)) + actx.np.sqrt(gamma * (p / rho))

    def state_to_mathematical_entropy(self, state):
        actx = state.array_context
        gamma = self.gamma
        rho, _, p = conservative_to_primitive_vars(state, gamma=gamma)
        s = actx.np.log(p) - gamma*actx.np.log(rho)

        return -rho * s / (gamma - 1)


# {{{ Entropy stable operator

def conservative_to_primitive_vars(cv_state, gamma=1.4):
    """todo.
    """
    rho = cv_state.mass
    rho_e = cv_state.energy
    rho_u = cv_state.momentum
    u = rho_u / rho
    p = (gamma - 1) * (rho_e - 0.5 * sum(rho_u * u))

    return rho, u, p


def conservative_to_entropy_vars(actx, cv_state, gamma=1.4):
    """todo.
    """
    rho, u, p = conservative_to_primitive_vars(cv_state, gamma=gamma)

    u_square = sum(v ** 2 for v in u)
    s = actx.np.log(p) - gamma*actx.np.log(rho)
    rho_p = rho / p

    return EulerState(mass=((gamma - s)/(gamma - 1)) - 0.5 * rho_p * u_square,
                      energy=-rho_p,
                      momentum=rho_p * u)


def entropy_to_conservative_vars(actx, ev_state, gamma=1.4):
    """todo.
    """
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

    return EulerState(mass=-rho_iota * v5,
                      energy=rho_iota * (1 - v_square/(2*v5)),
                      momentum=rho_iota * v2t4)


def flux_chandrashekar(dcoll, gamma, q_ll, q_rr):
    """Entropy conserving two-point flux by Chandrashekar (2013)
    Kinetic Energy Preserving and Entropy Stable Finite Volume Schemes
    for Compressible Euler and Navier-Stokes Equations
    [DOI: 10.4208/cicp.170712.010313a](https://doi.org/10.4208/cicp.170712.010313a)

    :args q_ll: an array container for the "left" state
    :args q_rr: an array container for the "right" state
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

    return EulerState(mass=mass_flux,
                      energy=energy_flux,
                      momentum=momentum_flux)


def entropy_stable_numerical_flux_chandrashekar(
        dcoll, tpair, gamma=1.4, lf_stabilization=False):
    """Entropy stable numerical flux based on the entropy conserving flux in
    Chandrashekar (2013) Kinetic Energy Preserving and Entropy Stable Finite
    Volume Schemes for Compressible Euler and Navier-Stokes Equations
    [DOI: 10.4208/cicp.170712.010313a](https://doi.org/10.4208/cicp.170712.010313a)
    """
    dd_intfaces = tpair.dd
    dd_allfaces = dd_intfaces.with_dtag("all_faces")
    q_ll = tpair.int
    q_rr = tpair.ext
    actx = q_ll.array_context

    num_flux = flux_chandrashekar(dcoll, gamma, q_ll, q_rr)
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
        dq = DOFDesc("vol", qtag)
        df = DOFDesc("all_faces", qtag)

        dcoll = self.dcoll
        actx = q.array_context

        # Interpolate cv_state to vol quad grid: u_q = V_q u
        q_quad = op.project(dcoll, "vol", dq, q)

        # Convert to projected entropy variables: v_q = V_h P_q v(u_q)
        entropy_vars = conservative_to_entropy_vars(actx, q_quad, gamma=gamma)
        proj_entropy_vars = volume_quadrature_project(dcoll, dq, entropy_vars)

        # Compute conserved state in terms of the (interpolated)
        # projected entropy variables on the quad grid
        qtilde_allquad = \
            entropy_to_conservative_vars(
                actx,
                # Interpolate projected entropy variables to
                # volume + surface quadrature grids
                volume_and_surface_quadrature_interpolation(
                    dcoll, dq, df, proj_entropy_vars
                ),
                gamma=gamma)

        # Compute volume derivatives using flux differencing
        from grudge.flux_differencing import volume_flux_differencing

        flux_diff = volume_flux_differencing(
            dcoll,
            partial(flux_chandrashekar, dcoll, gamma),
            dq, df,
            qtilde_allquad
        )

        def entropy_tpair(tpair):
            dd_intfaces = tpair.dd
            dd_intfaces_quad = dd_intfaces.with_discr_tag(qtag)
            # Interpolate entropy variables to the surface quadrature grid
            vtilde_tpair = \
                op.project(dcoll, dd_intfaces, dd_intfaces_quad, tpair)
            return TracePair(
                dd_intfaces_quad,
                # Convert interior and exterior states to conserved variables
                interior=entropy_to_conservative_vars(
                    actx, vtilde_tpair.int, gamma=gamma
                ),
                exterior=entropy_to_conservative_vars(
                    actx, vtilde_tpair.ext, gamma=gamma
                )
            )

        # Computing interface numerical fluxes
        num_fluxes_bdry = (
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

        # Compute boundary numerical fluxes
        if self.bdry_conditions is not None:
            bc_fluxes = sum(
                entropy_stable_numerical_flux_chandrashekar(
                    dcoll,
                    bc.boundary_tpair(dcoll, q, t=t),
                    gamma=gamma,
                    lf_stabilization=self.lf_stabilization
                ) for bc in self.bdry_conditions
            )
            num_fluxes_bdry = num_fluxes_bdry + bc_fluxes

        return op.inverse_mass(
            dcoll,
            -flux_diff - op.face_mass(dcoll, df, num_fluxes_bdry),
            dd_quad=dq
        )

# }}}


# {{{ Boundary conditions

class BCObject:
    def __init__(self, dd, *, prescribed_state=None) -> None:
        self.dd = dd
        self.prescribed_state = prescribed_state

    def boundary_tpair(self, dcoll, state, t=0):
        raise NotImplementedError("Boundary pair method not implemented.")


class PrescribedBC(BCObject):

    def boundary_tpair(self, dcoll, state, t=0):
        actx = state.array_context
        dd_bcq = self.dd
        dd_base = as_dofdesc("vol").with_discr_tag(DISCR_TAG_BASE)

        return TracePair(
            dd_bcq,
            interior=op.project(dcoll, dd_base, dd_bcq, state),
            exterior=self.prescribed_state(thaw(dcoll.nodes(dd_bcq), actx), t=t)
        )


class AdiabaticSlipBC(BCObject):

    def boundary_tpair(self, dcoll, state, t=0):
        actx = state.array_context
        dd_bcq = self.dd
        dd_base = as_dofdesc("vol").with_discr_tag(DISCR_TAG_BASE)
        nhat = thaw(dcoll.normal(dd_bcq), actx)
        interior = op.project(dcoll, dd_base, dd_bcq, state)

        return TracePair(
            dd_bcq,
            interior=interior,
            exterior=EulerState(
                mass=interior.mass,
                energy=interior.energy,
                momentum=(
                    interior.momentum - 2.0 * nhat * np.dot(interior.momentum, nhat)
                )
            )
        )

# }}}


# {{{ Limiter

def positivity_preserving_limiter(dcoll, quad_tag, state):
    actx = state.array_context

    # Interpolate state to quadrature grid
    dd_quad = as_dofdesc("vol").with_discr_tag(quad_tag)
    density = op.project(dcoll, "vol", dd_quad, state.mass)

    # Compute nodal and elementwise max/mins
    _mmax = op.nodal_max(dcoll, dd_quad, density)
    _mmin = op.nodal_min(dcoll, dd_quad, density)
    _mmax_i = op.elementwise_max(dcoll, density)
    _mmin_i = op.elementwise_min(dcoll, density)

    # Compute cell averages of the state
    from grudge.geometry import area_element

    inv_area_elements = 1./area_element(
        actx, dcoll,
        _use_geoderiv_connection=actx.supports_nonscalar_broadcasting)
    state_cell_avgs = \
        inv_area_elements * op.elementwise_integral(dcoll, state.mass)

    # Compute minmod factor
    theta = actx.np.minimum(
        1,
        actx.np.minimum(
            abs((_mmax - state_cell_avgs)/(_mmax_i - state_cell_avgs)),
            abs((_mmin - state_cell_avgs)/(_mmin_i - state_cell_avgs))
        )
    )

    return EulerState(
        # Limit only mass
        mass=theta*(state.mass - state_cell_avgs) + state_cell_avgs,
        energy=state.energy,
        momentum=state.momentum
    )

# }}}


# vim: foldmethod=marker
