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

from arraycontext import thaw, make_loopy_program

from meshmode.dof_array import DOFArray

from grudge.models import HyperbolicOperator
from grudge.trace_pair import TracePair

from pytools.obj_array import make_obj_array, obj_array_vectorize

import grudge.op as op
import loopy as lp


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
    dim = dcoll.dim
    dd_bcq = dd_bc.with_qtag(qtag)
    actx = cv_state[0].array_context

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
        rho = q[0]
        rhoe = q[1]
        rhov = q[2:self.dcoll.dim+2]
        v = rhov / rho
        gamma = self.gamma
        p = (gamma - 1) * (rhoe - 0.5 * sum(rhov * v))

        return actx.np.sqrt(np.dot(v, v)) + actx.np.sqrt(gamma * (p / rho))


# {{{ Entropy stable operator


def conservative_to_entropy_vars(actx, dcoll, cv_state, gamma=1.4):
    """todo.
    """
    dim = dcoll.dim
    rho = cv_state[0]
    rho_e = cv_state[1]
    rho_u = cv_state[2:dim+2]
    u = rho_u / rho
    u_square = sum(v ** 2 for v in u)
    p = (gamma - 1) * (rho_e - 0.5 * rho * u_square)
    s = actx.np.log(p) - gamma*actx.np.log(rho)
    rho_p = rho / p

    entropy_vars = np.empty((2+dim,), dtype=object)
    entropy_vars[0] = ((gamma - s)/(gamma - 1)) - 0.5 * rho_p * u_square
    entropy_vars[1] = -rho_p
    entropy_vars[2:dim+2] = rho_p * u

    return entropy_vars


def entropy_to_conservative_vars(actx, dcoll, ev_state, gamma=1.4):
    """todo.
    """
    # See Hughes, Franca, Mallet (1986) A new finite element
    # formulation for CFD: (DOI: 10.1016/0045-7825(86)90127-1)
    dim = dcoll.dim
    inv_gamma_minus_one = 1/(gamma - 1)

    # Convert to entropy `-rho * s` used by Hughes, France, Mallet (1986)
    ev_state = ev_state * (gamma - 1)
    v1 = ev_state[0]
    v2t4 = ev_state[2:dim+2]
    v5 = ev_state[1]

    v_square = sum(v**2 for v in v2t4)
    s = gamma - v1 + v_square/(2*v5)
    rho_iota = (
        ((gamma - 1) / (-v5)**gamma)**(inv_gamma_minus_one)
    ) * actx.np.exp(-s * inv_gamma_minus_one)

    conserved_vars = np.empty((2+dim,), dtype=object)
    conserved_vars[0] = -rho_iota * v5
    conserved_vars[1] = rho_iota * (1 - v_square/(2*v5))
    conserved_vars[2:dim+2] = rho_iota * v2t4

    return conserved_vars


def entropy_projection(
        actx, dcoll, dd_q, dd_f, cv_state, gamma=1.4, initial_condition=None):
    """todo.
    """
    from grudge.sbp_op import (volume_quadrature_project,
                               volume_quadrature_interpolation,
                               volume_and_surface_quadrature_interpolation)

    # Interpolate cv_state to vol quad grid: u_q = V_q u
    if initial_condition:
        cv_state_q = initial_condition(thaw(dcoll.nodes(dd_q), actx))
    else:
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


def flux_chandrashekar(q_ll, q_rr, orientation, gamma=1.4):
    """Entropy conserving two-point flux by Chandrashekar (2013)
    Kinetic Energy Preserving and Entropy Stable Finite Volume Schemes
    for Compressible Euler and Navier-Stokes Equations
    [DOI: 10.4208/cicp.170712.010313a](https://doi.org/10.4208/cicp.170712.010313a)

    :args q_ll: a tuple containing the "left" state
    :args q_rr: a tuple containing the "right" state
    :args orientation: an integer denoting the dimension axis;
        e.g. 0 for x-direction, 1 for y-direction, 2 for z-direction.
    """

    rho_ll, rhoe_ll, rhou_ll = q_ll
    rho_rr, rhoe_rr, rhou_rr = q_rr

    v_ll = rhou_ll / rho_ll
    v_rr = rhou_rr / rho_rr

    dim = len(v_ll)

    p_ll = (gamma - 1) * (rhoe_ll - 0.5 * sum(rhou_ll * v_ll))
    p_rr = (gamma - 1) * (rhoe_rr - 0.5 * sum(rhou_rr * v_rr))

    beta_ll = 0.5 * rho_ll / p_ll
    beta_rr = 0.5 * rho_rr / p_rr

    specific_kin_ll = 0.5 * sum(v**2 for v in v_ll)
    specific_kin_rr = 0.5 * sum(v**2 for v in v_rr)
    v_avg = 0.5 * (v_ll + v_rr)
    velocity_square_avg = (
        2 * sum(vi_avg**2 for vi_avg in v_avg)
        - (specific_kin_ll + specific_kin_rr)
    )

    def log_mean(x, y, epsilon=1e-2):
        """Computes the logarithmic mean using a numerically stable
        stable approach outlined in Appendix B of Ismail, Roe (2009).
        Affordable, entropy-consistent Euler flux functions II: Entropy
        production at shocks.
        [DOI: 10.1016/j.jcp.2009.04.021](https://doi.org/10.1016/j.jcp.2009.04.021)
        """
        zeta = x / y
        f = (zeta - 1) / (zeta + 1)
        u = f*f
        if u < epsilon:
            ff = 1 + u / 3 + u*u / 5 + u*u*u / 7
        else:
            ff = np.log(zeta)/2/f
        return (x + y) / (2*ff)

    # Compute the necessary mean values
    rho_avg = 0.5 * (rho_ll + rho_rr)
    rho_mean  = log_mean(rho_ll,  rho_rr)

    beta_mean = log_mean(beta_ll, beta_rr)
    beta_avg = 0.5 * (beta_ll + beta_rr)
    p_avg = 0.5 * rho_avg / beta_avg
    e_avg = (
        (rho_mean / (2 * beta_mean * (gamma - 1)))
        + 0.5 * velocity_square_avg
    )
    rho_mean_v_avg = rho_mean * v_avg
    fxyz_momentum = np.outer(rho_mean_v_avg, v_avg) + np.eye(dim) * p_avg

    f_mass = rho_mean_v_avg[orientation]
    f_momentum = fxyz_momentum[:, orientation]
    f_energy = v_avg[orientation] * (e_avg + p_avg)
    result =  np.array([f_mass, f_energy, f_momentum], dtype=object)
    # NOTE: NaN appears to be coming *in* to the function as q_ll, q_rr...
    # if np.isnan(sum(result)):
    #     # raise ValueError("NaN occurred in two-point flux calculation.")
    #     # print("NaN occurred in two-point flux calculation.")
    #     import ipdb; ipdb.set_trace()
    return result


def volume_flux_differencing(actx, dcoll, dq, df, state, gamma=1.4):
    """Computes the flux differencing operation: ∑_j 2Q[i, j] * f_S(q_i, q_j),
    where Q is a hybridized SBP derivative matrix and f_S is
    an entropy-conservative two-point flux.

    See `flux_chandrashekar` for a concrete implementation of such a two-point
    flux routine.
    """
    from grudge.sbp_op import \
        hybridized_sbp_operators, volume_and_surface_quadrature_interpolation
    from grudge.geometry import area_element, inverse_metric_derivative_mat

    mesh = dcoll.mesh
    dim = dcoll.dim
    dtype = state[0].entry_dtype

    volm_discr = dcoll.discr_from_dd("vol")
    face_discr = dcoll.discr_from_dd("all_faces")
    volm_quad_discr = dcoll.discr_from_dd(dq)
    face_quad_discr = dcoll.discr_from_dd(df)

    # Geometry terms for building the physical derivative operators
    # (interpolated to volume+surface quadrature nodes)
    jacobian_dets = volume_and_surface_quadrature_interpolation(
        dcoll, dq, df, area_element(actx, dcoll)
    )
    drstdxyz = volume_and_surface_quadrature_interpolation(
        dcoll, dq, df, inverse_metric_derivative_mat(actx, dcoll)
    )
    rstxyzj = jacobian_dets * drstdxyz

    rho = state[0]
    rhoe = state[1]
    rhou = state[2:dim+2]

    # Group array lists for the computed result
    QF_mass_data = []
    QF_energy_data = []
    QF_momentum_data = []

    # Group loop
    for gidx, mgrp in enumerate(mesh.groups):
        vgrp = volm_discr.groups[gidx]
        vqgrp = volm_quad_discr.groups[gidx]
        fqgrp = face_quad_discr.groups[gidx]

        Nelements = mgrp.nelements
        Nq_vol = vqgrp.nunit_dofs
        Nq_faces = vqgrp.shape.nfaces * fqgrp.nunit_dofs
        Nq_total = Nq_vol + Nq_faces

        # Get geometric terms for the group
        vgeo_gidx = [[actx.to_numpy(rstxyzj[ridx][cidx][gidx])
                      for ridx in range(dim)]
                     for cidx in range(dim)]

        # Get state values for the group
        rho_gidx = rho[gidx]
        rhoe_gidx = rhoe[gidx]
        rhou_gidx = [ru[gidx] for ru in rhou]

        # Form skew-symmetric SBP hybridized derivative operators
        qmats = actx.to_numpy(
            thaw(hybridized_sbp_operators(actx,
                                          vgrp,
                                          vqgrp, fqgrp,
                                          dtype), actx)
        )
        Qrst_skew = [0.5 * (qmats[d] - qmats[d].T) for d in range(dim)]

        # Group arrays for the Hadamard row-sum
        dQF_rho = np.zeros(shape=(Nelements, Nq_total), dtype=dtype)
        dQF_rhoe = np.zeros(shape=(Nelements, Nq_total), dtype=dtype)
        dQF_rhou = np.zeros(shape=(dim, Nelements, Nq_total), dtype=dtype)

        # Element loop
        for eidx in range(Nelements):
            vgeo = [[vgeo_gidx[ridx][cidx][eidx]
                     for ridx in range(dim)]
                    for cidx in range(dim)]

            # Build physical SBP operators on the element
            Qxyz_skew = [
                2*sum(vgeo[j][d] * Qrst_skew[j] for j in range(dim))
                for d in range(dim)
            ]

            # Element-local state data
            local_rho = actx.to_numpy(rho_gidx[eidx])
            local_rhoe = actx.to_numpy(rhoe_gidx[eidx])
            local_rhou = np.array(
                [actx.to_numpy(rhou_gidx[d][eidx]) for d in range(dim)]
            )

            # Local arrays for the Hadamard row-sum
            dq_rho = np.zeros(shape=(Nq_total,))
            dq_rhoe = np.zeros(shape=(Nq_total,))
            dq_rhou = np.zeros(shape=(dim, Nq_total))

            # Compute flux differencing in each cell and apply the
            # hybridized SBP operator
            for d in range(dim):
                Qskew_d = Qxyz_skew[d]
                # Loop over all (vol + surface) quadrature nodes and compute
                # the Hadamard row-sum
                for i in range(Nq_total):
                    q_i = (local_rho[i],
                           local_rhoe[i],
                           local_rhou[:, i])

                    dq_rho_i = dq_rho[i]
                    dq_rhoe_i = dq_rhoe[i]
                    dq_rhou_i = dq_rhou[:, i]

                    # Loop over all (vol + surface) quadrature nodes
                    for j in range(Nq_total):
                        # Computes only the upper-triangular part of the
                        # hadamard sum (Q .* F). We avoid computing the
                        # lower-triangular part using the fact that Q is
                        # skew-symmetric and F is symmetric.
                        # Also skips subblock of Q which we know is
                        # zero by construction.
                        if j > i and not (i > Nq_vol and j > Nq_vol):
                            # Apply derivative to each component
                            # (mass, energy, momentum)
                            QF_rho_ij, QF_rhoe_ij, QF_rhou_ij = \
                                Qskew_d[i, j] * flux_chandrashekar(
                                    q_i,
                                    (local_rho[j], local_rhoe[j], local_rhou[:, j]),
                                    d,
                                    gamma=gamma
                                )

                            # Accumulate upper triangular part
                            dq_rho_i = dq_rho_i + QF_rho_ij
                            dq_rhoe_i = dq_rhoe_i + QF_rhoe_ij
                            dq_rhou_i = dq_rhou_i + QF_rhou_ij

                            # Accumulate lower triangular part
                            dq_rho[j] = dq_rho[j] - QF_rho_ij
                            dq_rhoe[j] = dq_rhoe[j] - QF_rhoe_ij
                            dq_rhou[:, j] = dq_rhou[:, j] - QF_rhou_ij
                        # end if
                    # end j
                    dq_rho[i] = dq_rho_i
                    dq_rhoe[i] = dq_rhoe_i
                    dq_rhou[:, i] = dq_rhou_i
                # end i
            # end d
            dQF_rho[eidx, :] = dq_rho
            dQF_rhoe[eidx, :] = dq_rhoe
            dQF_rhou[:, eidx, :] = dq_rhou
        # end e

        # Append group data
        QF_mass_data.append(actx.from_numpy(dQF_rho))
        QF_energy_data.append(actx.from_numpy(dQF_rhoe))
        QF_momentum_data.append([actx.from_numpy(dQF_rhou[d])
                                 for d in range(dim)])
    # end g

    # Convert back to mirgecom-like data structure
    QF_mass_dof_ary = DOFArray(actx, data=tuple(QF_mass_data))
    QF_energy_dof_ary = DOFArray(actx, data=tuple(QF_energy_data))
    QF_momentum_dof_ary = make_obj_array(
        [DOFArray(actx,
                  data=tuple(mom_data[d]
                             for mom_data in QF_momentum_data))
                  for d in range(dim)]
    )

    result = np.empty((2+dim,), dtype=object)
    result[0] = QF_mass_dof_ary
    result[1] = QF_energy_dof_ary
    result[2:dim+2] = QF_momentum_dof_ary

    from grudge.sbp_op import volume_and_surface_quadrature_projection

    # Apply Ph = Minv * [Vq @ Wq; Vf @ Wf].T to the result
    return volume_and_surface_quadrature_projection(dcoll, dq, df, result)


def entropy_stable_numerical_flux_chandrashekar(
        dcoll, tpair, gamma=1.4, lf_stabilization=False):
    """Entropy stable numerical flux based on the entropy conserving flux in
    Chandrashekar (2013) Kinetic Energy Preserving and Entropy Stable Finite
    Volume Schemes for Compressible Euler and Navier-Stokes Equations
    [DOI: 10.4208/cicp.170712.010313a](https://doi.org/10.4208/cicp.170712.010313a)
    """
    # NOTE: this version of flux_chandrashekar is written to take full arrays
    # of interior/exterior values (not point-wise like `flux_chandrashekar`).
    dim = dcoll.dim
    dd_intfaces = tpair.dd
    dd_allfaces = dd_intfaces.with_dtag("all_faces")
    q_int = tpair.int
    q_ext = tpair.ext
    actx = q_int[0].array_context

    def log_mean(x, y, epsilon=1e-2):
        zeta = x / y
        f = (zeta - 1) / (zeta + 1)
        u = f*f
        return actx.np.where(
            # if f_squared < epsilon
            u < epsilon,
            (x + y) / (2*(1 + u / 3 + u*u / 5 + u*u*u / 7)),
            # else
            (x + y) / (2*actx.np.log(zeta)/2/f)
        )

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

    beta_int = 0.5 * rho_int / p_int
    beta_ext = 0.5 * rho_ext / p_ext
    specific_kin_int = 0.5 * sum(v**2 for v in v_int)
    specific_kin_ext = 0.5 * sum(v**2 for v in v_ext)
    v_avg = 0.5 * (v_int + v_ext)
    velocity_square_avg = (
        2 * sum(vi_avg**2 for vi_avg in v_avg)
        - (specific_kin_int + specific_kin_ext)
    )

    rho_avg = 0.5 * (rho_int + rho_ext)
    beta_avg = 0.5 * (beta_int + beta_ext)
    p_avg = 0.5 * rho_avg / beta_avg
    rho_mean = log_mean(rho_int, rho_ext)
    beta_mean = log_mean(beta_int, beta_ext)
    e_avg = (
        (rho_mean / (2 * beta_mean * (gamma - 1)))
        + 0.5 * velocity_square_avg
    )
    rho_mean_v_avg = rho_mean * v_avg

    flux = np.empty((2+dim, dim), dtype=object)
    flux[0, :] = rho_mean_v_avg
    flux[1, :] = v_avg * (e_avg + p_avg)
    flux[2:dim+2, :] = np.outer(rho_mean_v_avg, v_avg) + np.eye(dim) * p_avg

    normal = thaw(dcoll.normal(dd_intfaces), actx)
    num_flux = flux @ normal

    if lf_stabilization:
        # compute jump penalization parameter
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
    actx = proj_ev_state[0].array_context
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

        gamma = self.gamma
        dq = DOFDesc("vol", DISCR_TAG_QUAD)
        df = DOFDesc("all_faces", DISCR_TAG_QUAD)

        dcoll = self.dcoll
        actx = q[0].array_context

        print("Computing auxiliary conservative variables...")
        if t == 0:
            qtilde_allquad, proj_entropy_vars = entropy_projection(
                actx, dcoll, dq, df, q, gamma=gamma,
                initial_condition=self.initial_condition
            )
        else:
            qtilde_allquad, proj_entropy_vars = entropy_projection(
                actx, dcoll, dq, df, q, gamma=gamma)
        print("Finished auxiliary conservative variables.")

        print("Performing volume flux differencing...")
        # NOTE: Performs flux differencing in the volume of cells, requires
        # accessing nodes in both the volume and faces. The operation has the
        # form:
        # ∑_d ∑_j Vh.T @ 2Q_d[i, j] * F_d(q_i, q_j)
        # where Q_d is the skew-hybridized SBP operator for the d-th dimension
        # axis, and F_d is an entropy conservative two-point flux. The routine
        # computes the Hadamard (element-wise) multiplication + row sum:
        # dQF1 = Vh.T @ (∑_d (2Q_d * F_d)1)
        dQF1 = volume_flux_differencing(
            actx, dcoll, dq, df, qtilde_allquad, gamma)
        print("Finished volume flux differencing.")

        print("Computing interface numerical fluxes...")
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
        print("Finished computing interface numerical fluxes.")

        print("Applying lifting operators...")
        from grudge.sbp_op import inverse_sbp_mass

        # Compute: sum_i={x,y,z} (Minv @ V_f.T @ Wf @ Jf_i) * f_i
        # Lift operator: Minv @ V_f.T @ Wf
        lifted_fluxes = inverse_sbp_mass(
            dcoll, dq, op.face_mass(dcoll, df, num_fluxes_bdry))
        print("Finished applying lifting operators.")

        from grudge.geometry import area_element
        # Apply inverse cell jacobians
        inv_jacobians = 1./area_element(actx, dcoll)

        return -inv_jacobians*(dQF1 + lifted_fluxes)

# }}}
