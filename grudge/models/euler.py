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
from grudge.dof_desc import DOFDesc, DISCR_TAG_BASE, DISCR_TAG_QUAD, as_dofdesc

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


def euler_numerical_flux(
        dcoll, tpair, gamma=1.4, lf_stabilization=False):
    """todo.
    """
    dd_intfaces = tpair.dd
    dd_allfaces = dd_intfaces.with_dtag("all_faces")
    q_int = tpair.int
    q_ext = tpair.ext
    actx = q_int.array_context

    normal = thaw(dcoll.normal(dd_intfaces), actx)
    num_flux = 0.5 * (euler_flux(dcoll, q_int, gamma=gamma)
                      + euler_flux(dcoll, q_ext, gamma=gamma)) @ normal

    if lf_stabilization:
        rho_int = q_int.mass
        rhoe_int = q_int.energy
        rhou_int = q_int.momentum

        rho_ext = q_ext.mass
        rhoe_ext = q_ext.energy
        rhou_ext = q_ext.momentum

        v_int = rhou_int / rho_int
        v_ext = rhou_ext / rho_ext

        p_int = (gamma - 1) * (rhoe_int - 0.5 * sum(rhou_int * v_int))
        p_ext = (gamma - 1) * (rhoe_ext - 0.5 * sum(rhou_ext * v_ext))

        # compute jump penalization parameter
        lam = actx.np.maximum(
            actx.np.sqrt(gamma * (p_int / rho_int))
            + actx.np.sqrt(np.dot(v_int, v_int)),
            actx.np.sqrt(gamma * (p_ext / rho_ext))
            + actx.np.sqrt(np.dot(v_ext, v_ext))
        )
        num_flux -= 0.5 * lam * (q_ext - q_int)

    return op.project(dcoll, dd_intfaces, dd_allfaces, num_flux)


class EulerOperator(HyperbolicOperator):

    def __init__(self, dcoll, bdry_conditions=None,
                 flux_type="lf", gamma=1.4, gas_const=287.1):
        self.dcoll = dcoll
        self.bdry_conditions = bdry_conditions
        self.flux_type = flux_type
        self.gamma = gamma
        self.gas_const = gas_const
        self.lf_stabilization = flux_type == "lf"

    def operator(self, t, q):
        dcoll = self.dcoll
        gamma = self.gamma
        dim = q.dim

        dq = DOFDesc("vol", DISCR_TAG_QUAD)
        df = DOFDesc("all_faces", DISCR_TAG_QUAD)
        df_int = DOFDesc("int_faces", DISCR_TAG_QUAD)

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

        vol_div = make_eulerstate(
            dim=dim,
            q=op.weak_local_div(
                dcoll, dq, to_quad_vol(euler_flux(dcoll, q, gamma=gamma)).join()
            )
        )

        return op.inverse_mass(
            dcoll, dq,
            vol_div - op.face_mass(dcoll, df, euler_flux_faces)
        )

    def max_characteristic_velocity(self, actx, **kwargs):
        q = kwargs["state"]
        rho = q.mass
        rhoe = q.energy
        rhov = q.momentum
        v = rhov / rho
        gamma = self.gamma
        p = (gamma - 1) * (rhoe - 0.5 * sum(rhov * v))

        return actx.np.sqrt(np.dot(v, v)) + actx.np.sqrt(gamma * (p / rho))


# {{{ Entropy stable operator

def conservative_to_entropy_vars(actx, cv_state, gamma=1.4):
    """todo.
    """
    rho = cv_state.mass
    rho_e = cv_state.energy
    rho_u = cv_state.momentum
    u = rho_u / rho
    u_square = sum(v ** 2 for v in u)
    p = (gamma - 1) * (rho_e - 0.5 * rho * u_square)
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


def entropy_projection(
        actx, dcoll, dd_q, dd_f, cv_state, gamma=1.4):
    """todo.
    """
    from grudge.projection import volume_quadrature_project
    from grudge.interpolation import \
        volume_and_surface_quadrature_interpolation

    # Interpolate cv_state to vol quad grid: u_q = V_q u
    cv_state_q = op.project(dcoll, "vol", dd_q, cv_state)
    # Convert to entropy variables: v_q = v(u_q)
    ev_state_q = \
        conservative_to_entropy_vars(actx, cv_state_q, gamma=gamma)
    # Project entropy variables and interpolate the result to the
    # volume and surface quadrature nodes:
    # vtilde = [vtilde_q; vtilde_f] = [V_q; V_f] .* P_q * v_q
    # NOTE: Potential optimization: fuse [V_q; V_f] .* P_q
    ev_state = volume_quadrature_project(dcoll, dd_q, ev_state_q)
    aux_ev_state_q = volume_and_surface_quadrature_interpolation(
        dcoll, dd_q, dd_f, ev_state
    )
    # Convert from project entropy to conservative variables:
    # utilde = [utilde_q; utilde_f] = u(vtilde) = u([vtilde_q; vtilde_f])
    aux_cv_state_q = \
        entropy_to_conservative_vars(actx, aux_ev_state_q, gamma=gamma)
    return aux_cv_state_q, ev_state


def flux_chandrashekar(dcoll, gamma, qi, qj):
    """Entropy conserving two-point flux by Chandrashekar (2013)
    Kinetic Energy Preserving and Entropy Stable Finite Volume Schemes
    for Compressible Euler and Navier-Stokes Equations
    [DOI: 10.4208/cicp.170712.010313a](https://doi.org/10.4208/cicp.170712.010313a)

    :args qi: an array container for the "left" state
    :args qj: an array container for the "right" state
    """
    dim = dcoll.dim
    actx = qi.array_context

    def log_mean(x: DOFArray, y: DOFArray, epsilon=1e-4):
        zeta = x / y
        f = (zeta - 1) / (zeta + 1)
        u = f*f
        ff = actx.np.where(actx.np.less(u, epsilon),
                           1 + u/3 + u*u/5 + u*u*u/7,
                           actx.np.log(zeta)/2/f)
        return (x + y) / (2*ff)

    rho_i = qi.mass
    rhoe_i = qi.energy
    rhou_i = qi.momentum

    rho_j = qj.mass
    rhoe_j = qj.energy
    rhou_j = qj.momentum

    v_i = rhou_i / rho_i
    v_j = rhou_j / rho_j

    p_i = (gamma - 1) * (rhoe_i - 0.5 * sum(rhou_i * v_i))
    p_j = (gamma - 1) * (rhoe_j - 0.5 * sum(rhou_j * v_j))

    beta_i = 0.5 * rho_i / p_i
    beta_j = 0.5 * rho_j / p_j
    specific_kin_i = 0.5 * sum(v**2 for v in v_i)
    specific_kin_j = 0.5 * sum(v**2 for v in v_j)
    v_avg = 0.5 * (v_i + v_j)
    velocity_square_avg = (
        2 * sum(vi_avg**2 for vi_avg in v_avg)
        - (specific_kin_i + specific_kin_j)
    )

    rho_avg = 0.5 * (rho_i + rho_j)
    beta_avg = 0.5 * (beta_i + beta_j)
    p_avg = 0.5 * rho_avg / beta_avg
    rho_mean = log_mean(rho_i, rho_j)
    beta_mean = log_mean(beta_i, beta_j)
    e_avg = (
        (rho_mean / (2 * beta_mean * (gamma - 1)))
        + 0.5 * velocity_square_avg
    )
    rho_mean_v_avg = rho_mean * v_avg

    return EulerState(
        mass=rho_mean_v_avg,
        energy=v_avg * (e_avg + p_avg),
        momentum=np.outer(rho_mean_v_avg, v_avg) + np.eye(dim) * p_avg
    )


def entropy_stable_numerical_flux_chandrashekar(
        dcoll, tpair, gamma=1.4, lf_stabilization=False):
    """Entropy stable numerical flux based on the entropy conserving flux in
    Chandrashekar (2013) Kinetic Energy Preserving and Entropy Stable Finite
    Volume Schemes for Compressible Euler and Navier-Stokes Equations
    [DOI: 10.4208/cicp.170712.010313a](https://doi.org/10.4208/cicp.170712.010313a)
    """
    dd_intfaces = tpair.dd
    dd_intfaces_base = dd_intfaces.with_discr_tag(DISCR_TAG_BASE)
    dd_allfaces = dd_intfaces.with_dtag("all_faces")
    q_int = tpair.int
    q_ext = tpair.ext
    actx = q_int.array_context

    flux = flux_chandrashekar(dcoll, gamma, q_int, q_ext)
    # FIXME: Because of the affineness of the geometry, this normal technically
    # does not need to be interpolated to the quadrature grid.
    normal = thaw(dcoll.normal(dd_intfaces_base), actx)
    num_flux = flux @ normal

    if lf_stabilization:
        # compute jump penalization parameter
        # FIXME: Move into *flux_chandrashekar*
        rho_int = q_int.mass
        rhoe_int = q_int.energy
        rhou_int = q_int.momentum
        rho_ext = q_ext.mass
        rhoe_ext = q_ext.energy
        rhou_ext = q_ext.momentum
        v_int = rhou_int / rho_int
        v_ext = rhou_ext / rho_ext
        p_int = (gamma - 1) * (rhoe_int - 0.5 * sum(rhou_int * v_int))
        p_ext = (gamma - 1) * (rhoe_ext - 0.5 * sum(rhou_ext * v_ext))

        lam = actx.np.maximum(
            actx.np.sqrt(gamma * (p_int / rho_int))
            + actx.np.sqrt(np.dot(v_int, v_int)),
            actx.np.sqrt(gamma * (p_ext / rho_ext))
            + actx.np.sqrt(np.dot(v_ext, v_ext))
        )
        num_flux -= 0.5 * lam * (q_ext - q_int)

    return op.project(dcoll, dd_intfaces, dd_allfaces, num_flux)


class EntropyStableEulerOperator(EulerOperator):

    def operator(self, t, q):
        gamma = self.gamma
        dq = DOFDesc("vol", DISCR_TAG_QUAD)
        df = DOFDesc("all_faces", DISCR_TAG_QUAD)

        dcoll = self.dcoll
        actx = q.array_context

        # Get the projected entropy variables and state
        qtilde_allquad, proj_entropy_vars = entropy_projection(
            actx, dcoll, dq, df, q, gamma=gamma)

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
            dd_intfaces_quad = dd_intfaces.with_discr_tag(DISCR_TAG_QUAD)
            vtilde_tpair = op.project(
                dcoll, dd_intfaces, dd_intfaces_quad, tpair)
            return TracePair(
                dd_intfaces_quad,
                # int and ext states are terms of the entropy-projected
                # conservative vars
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
            dcoll, dq,
            -flux_diff - op.face_mass(dcoll, df, num_fluxes_bdry)
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


# vim: foldmethod=marker
