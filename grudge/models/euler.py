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

from grudge.models import HyperbolicOperator
from grudge.trace_pair import TracePair

import grudge.op as op


def euler_flux(dcoll, cv_state, gamma=1.4):
    """todo.
    """
    dim = dcoll.dim

    rho = cv_state[0]
    rho_e = cv_state[1]
    rho_u = cv_state[2:dim+2]
    u = rho_u / rho
    u_square = sum(v ** 2 for v in u)
    p = (gamma - 1) * (rho_e - 0.5 * rho * u_square)

    flux = np.empty((2+dim, dim), dtype=object)
    flux[0, :] = rho_u
    flux[1, :] = u * (rho_e + p)
    flux[2:dim+2, :] = np.outer(rho_u, u) + np.eye(dim) * p

    return flux


def euler_numerical_flux(
        dcoll, tpair, gamma=1.4, lf_stabilization=False):
    """todo.
    """
    dim = dcoll.dim
    dd_intfaces = tpair.dd
    dd_allfaces = dd_intfaces.with_dtag("all_faces")
    q_int = tpair.int
    q_ext = tpair.ext
    actx = q_int[0].array_context

    normal = thaw(dcoll.normal(dd_intfaces), actx)
    num_flux = 0.5 * (euler_flux(dcoll, q_int, gamma=gamma)
                      + euler_flux(dcoll, q_ext, gamma=gamma)) @ normal

    if lf_stabilization:
        rho_int = q_int[0]
        rhoe_int = q_int[1]
        rhou_int = q_int[2:dim+2]

        rho_ext = q_ext[0]
        rhoe_ext = q_ext[1]
        rhou_ext = q_ext[2:dim+2]

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


def euler_boundary_numerical_flux_prescribed(
        dcoll, cv_state, cv_prescribe, dd_bc,
        qtag=None, gamma=1.4, lf_stabilization=False):
    """todo.
    """
    dd_bcq = dd_bc.with_qtag(qtag)
    cv_state_btag = op.project(dcoll, "vol", dd_bc, cv_state)
    cv_bcq = op.project(dcoll, dd_bc, dd_bcq, cv_state_btag)

    bdry_tpair = TracePair(
        dd_bcq,
        interior=cv_bcq,
        exterior=op.project(dcoll, dd_bc, dd_bcq, cv_prescribe)
    )
    return euler_numerical_flux(
        dcoll, bdry_tpair, gamma=gamma, lf_stabilization=lf_stabilization)


class EulerOperator(HyperbolicOperator):

    def __init__(self, dcoll, bdry_fcts=None, initial_condition=None,
                 flux_type="lf", gamma=1.4, gas_const=287.1):
        self.dcoll = dcoll
        self.bdry_fcts = bdry_fcts
        self.initial_condition = initial_condition
        self.flux_type = flux_type
        self.gamma = gamma
        self.gas_const = gas_const
        self.lf_stabilization = flux_type == "lf"

    def operator(self, t, q):
        from grudge.dof_desc import DOFDesc, DISCR_TAG_QUAD, as_dofdesc

        dcoll = self.dcoll
        actx = q[0].array_context
        gamma = self.gamma

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
            + sum(
                euler_boundary_numerical_flux_prescribed(
                    dcoll,
                    q,
                    self.bdry_fcts[btag](thaw(self.dcoll.nodes(btag), actx), t),
                    dd_bc=as_dofdesc(btag),
                    qtag=DISCR_TAG_QUAD,
                    gamma=gamma,
                    lf_stabilization=self.lf_stabilization,
                ) for btag in self.bdry_fcts
            )
        )
        return op.inverse_mass(
            dcoll,
            op.weak_local_div(
                dcoll, dq, to_quad_vol(euler_flux(dcoll, q, gamma=gamma))
            )
            - op.face_mass(dcoll, df, euler_flux_faces)
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
        return _join_fields(
            dim=self.dim,
            mass=self.mass,
            energy=self.energy,
            momentum=self.momentum
        )


def _join_fields(dim, mass, energy, momentum):
    def _aux_shape(ary, leading_shape):
        from meshmode.dof_array import DOFArray
        if (isinstance(ary, np.ndarray) and ary.dtype == object
                and not isinstance(ary, DOFArray)):
            naxes = len(leading_shape)
            if ary.shape[:naxes] != leading_shape:
                raise ValueError("array shape does not start with expected leading "
                        "dimensions")
            return ary.shape[naxes:]
        else:
            if leading_shape != ():
                raise ValueError("array shape does not start with expected leading "
                        "dimensions")
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


def conservative_to_entropy_vars(actx, dcoll, cv_state, gamma=1.4):
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


def entropy_to_conservative_vars(actx, dcoll, ev_state, gamma=1.4):
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
    from grudge.sbp_op import (volume_quadrature_project,
                               volume_quadrature_interpolation,
                               volume_and_surface_quadrature_interpolation)

    # Interpolate cv_state to vol quad grid: u_q = V_q u
    cv_state_q = volume_quadrature_interpolation(dcoll, dd_q, cv_state)
    # Convert to entropy variables: v_q = v(u_q)
    ev_state_q = conservative_to_entropy_vars(
        actx, dcoll, cv_state_q, gamma=gamma)
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
    aux_cv_state_q = entropy_to_conservative_vars(
        actx, dcoll, aux_ev_state_q, gamma=gamma)
    return aux_cv_state_q, ev_state


def flux_chandrashekar(dcoll, gamma, use_numpy, qi, qj):
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
        # FIXME: else branch doesn't work right
        # when DOFArray contains numpy arrays
        if use_numpy:
            grp_data = []
            for xi, yi in zip(x, y):
                zeta = xi / yi
                f = (zeta - 1) / (zeta + 1)
                u = f*f
                # NOTE: RuntimeWarning: invalid value encountered in true_divide
                # will occur when using numpy due to eager evaluation of
                # np.log(zeta)/2/f (including at points where np.log is ill-defined).
                # However, the resulting nans are tossed out because of the
                # np.where conditional
                ff = np.where(u < epsilon,
                              1 + u/3 + u*u/5 + u*u*u/7,
                              np.log(zeta)/2/f)
                grp_data.append((xi + yi) / (2*ff))
            return DOFArray(actx, data=tuple(grp_data))
        else:
            zeta = x / y
            f = (zeta - 1) / (zeta + 1)
            u = f*f
            ff = actx.np.where(u < epsilon,
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
    dd_allfaces = dd_intfaces.with_dtag("all_faces")
    q_int = tpair.int
    q_ext = tpair.ext
    actx = q_int.array_context

    flux = flux_chandrashekar(dcoll, gamma, False, q_int, q_ext)
    normal = thaw(dcoll.normal(dd_intfaces), actx)
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


def entropy_stable_boundary_numerical_flux_prescribed(
        dcoll, proj_ev_state, cv_prescribe, dd_bcq,
        gamma=1.4, t=0.0, lf_stabilization=False):
    """todo.
    """
    actx = proj_ev_state.array_context
    x_bcq = thaw(dcoll.nodes(dd_bcq), actx)
    ev_bcq = op.project(dcoll, "vol", dd_bcq, proj_ev_state)
    bdry_tpair = TracePair(
        dd_bcq,
        # interior state in terms of the entropy-projected conservative vars
        interior=entropy_to_conservative_vars(
            actx, dcoll, ev_bcq, gamma=gamma
        ),
        exterior=cv_prescribe(x_bcq, t=t)
    )
    return entropy_stable_numerical_flux_chandrashekar(
        dcoll, bdry_tpair, gamma=gamma, lf_stabilization=lf_stabilization)


class EntropyStableEulerOperator(EulerOperator):

    def operator(self, t, q):
        from grudge.dof_desc import DOFDesc, DISCR_TAG_QUAD, as_dofdesc
        from grudge.sbp_op import volume_flux_differencing, inverse_sbp_mass

        gamma = self.gamma
        dq = DOFDesc("vol", DISCR_TAG_QUAD)
        df = DOFDesc("all_faces", DISCR_TAG_QUAD)

        dcoll = self.dcoll
        actx = q.array_context

        # Get the projected entropy variables and state
        qtilde_allquad, proj_entropy_vars = entropy_projection(
            actx, dcoll, dq, df, q, gamma=gamma)

        # Compute volume derivatives using flux differencing
        flux_diff = volume_flux_differencing(
            dcoll,
            partial(flux_chandrashekar, dcoll, gamma, True),
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
                    actx, dcoll, vtilde_tpair.int, gamma=gamma
                ),
                exterior=entropy_to_conservative_vars(
                    actx, dcoll, vtilde_tpair.ext, gamma=gamma
                )
            )

        # Computing interface and boundary numerical fluxes
        num_fluxes_bdry = (
            sum(
                entropy_stable_numerical_flux_chandrashekar(
                    dcoll,
                    entropy_tpair(tpair),
                    gamma=gamma,
                    lf_stabilization=self.lf_stabilization
                ) for tpair in op.interior_trace_pairs(dcoll, proj_entropy_vars)
            )
            + sum(
                # Boundary conditions (prescribed)
                entropy_stable_boundary_numerical_flux_prescribed(
                    dcoll,
                    proj_entropy_vars,
                    cv_prescribe=self.bdry_fcts[btag],
                    dd_bcq=as_dofdesc(btag).with_discr_tag(DISCR_TAG_QUAD),
                    t=t,
                    lf_stabilization=self.lf_stabilization
                ) for btag in self.bdry_fcts
            )
        )

        return inverse_sbp_mass(
            dcoll, dq,
            -flux_diff - op.face_mass(dcoll, df, num_fluxes_bdry)
        )

# }}}
