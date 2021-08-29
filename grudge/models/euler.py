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

from arraycontext import (
    thaw,
    with_container_arithmetic,
    dataclass_array_container,
)

from collections import namedtuple

from dataclasses import dataclass

from meshmode.dof_array import DOFArray

from grudge.models import HyperbolicOperator
from grudge.trace_pair import TracePair

from pytools.obj_array import make_obj_array, obj_array_vectorize

import grudge.op as op
import grudge.dof_desc as dof_desc


# {{{ Array container utilities

@with_container_arithmetic(bcast_obj_array=False,
                           bcast_container_types=(DOFArray, np.ndarray),
                           matmul=True,
                           rel_comparison=True)
@dataclass_array_container
@dataclass(frozen=True)
class ArrayContainer:
    mass: DOFArray
    energy: DOFArray
    momentum: np.ndarray  # [object array of DOFArrays]

    @property
    def array_context(self):
        return self.mass.array_context

    @property
    def dim(self):
        return len(self.momentum)

    @property
    def velocity(self):
        return self.momentum / self.mass

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

# }}}


class EulerOperator(HyperbolicOperator):

    def __init__(self, dcoll, bdry_fcts=None,
                 flux_type="lf", gamma=1.4, gas_const=287.1):

        self.dcoll = dcoll
        self.bdry_fcts = bdry_fcts
        self.flux_type = flux_type
        self.gamma = gamma
        self.gas_const = gas_const

        dd_modal = dof_desc.DD_VOLUME_MODAL
        dd_volume = dof_desc.DD_VOLUME

        self.map_to_modal = dcoll.connection_from_dds(dd_volume, dd_modal)
        self.map_to_nodal = dcoll.connection_from_dds(dd_modal, dd_volume)

    def operator(self, t, q):
        dcoll = self.dcoll

        # Convert to array container
        q = ArrayContainer(mass=q[0],
                           energy=q[1],
                           momentum=q[2:2+dcoll.dim])

        actx = q.array_context
        nodes = thaw(self.dcoll.nodes(), actx)

        euler_flux_vol = self.euler_flux(q)
        euler_flux_bnd = (
            sum(self.numerical_flux(tpair)
                for tpair in op.interior_trace_pairs(dcoll, q))
            + sum(
                self.boundary_numerical_flux(q, self.bdry_fcts[btag](nodes, t), btag)
                for btag in self.bdry_fcts
            )
        )
        return op.inverse_mass(
            dcoll,
            op.weak_local_div(dcoll, euler_flux_vol.join())
            - op.face_mass(dcoll, euler_flux_bnd.join())
        )

    def euler_flux(self, q):
        p = self.pressure(q)
        mom = q[2:self.dcoll.dim+2]

        return ArrayContainer(
            mass=mom,
            energy=mom * (q[1] + p) / q[0],
            momentum=np.outer(mom, mom) / q[0] + np.eye(self.dcoll.dim)*p
        ).join()

    def numerical_flux(self, q_tpair):
        """Return the numerical flux across a face given the solution on
        both sides *q_tpair*.
        """
        actx = q_tpair.int[0].array_context

        lam = actx.np.maximum(
            self.max_characteristic_velocity(actx, state=q_tpair.int),
            self.max_characteristic_velocity(actx, state=q_tpair.ext)
        )

        normal = thaw(self.dcoll.normal(q_tpair.dd), actx)

        flux_tpair = TracePair(
            q_tpair.dd,
            interior=self.euler_flux(q_tpair.int),
            exterior=self.euler_flux(q_tpair.ext)
        )

        flux_weak = flux_tpair.avg @ normal - lam/2.0*(q_tpair.int - q_tpair.ext)

        return op.project(self.dcoll, q_tpair.dd, "all_faces", flux_weak)

    def boundary_numerical_flux(self, q, q_prescribe, btag):
        """Return the numerical flux across a face given the solution on
        both sides *q_tpair*, with an external state given by a prescribed
        state *q_prescribe* at the boundaries denoted by *btag*.
        """
        actx = q[0].array_context

        bdry_tpair = TracePair(
            btag,
            interior=op.project(self.dcoll, "vol", btag, q),
            exterior=op.project(self.dcoll, "vol", btag, q_prescribe)
        )

        normal = thaw(self.dcoll.normal(bdry_tpair.dd), actx)

        bdry_flux_tpair = TracePair(
            bdry_tpair.dd,
            interior=self.euler_flux(bdry_tpair.int),
            exterior=self.euler_flux(bdry_tpair.ext)
        )

        # lam = actx.np.maximum(
        #     self.max_characteristic_velocity(actx, state=bdry_tpair.int),
        #     self.max_characteristic_velocity(actx, state=bdry_tpair.ext)
        # )

        # flux_weak = 0.5*(bdry_flux_tpair.int - bdry_flux_tpair.ext) @ normal \
        #     - lam/2.0*(bdry_tpair.int - bdry_tpair.ext)
        flux_weak = bdry_flux_tpair.ext @ normal

        return op.project(self.dcoll, bdry_tpair.dd, "all_faces", flux_weak)

    def kinetic_energy(self, q):
        mom = q[2:self.dcoll.dim+2]
        return (0.5 * np.dot(mom, mom) / q[0])

    def internal_energy(self, q):
        return (q[1] - self.kinetic_energy(q))

    def pressure(self, q):
        return self.internal_energy(q) * (self.gamma - 1.0)

    def sound_speed(self, q):
        actx = q[0].array_context
        return actx.np.sqrt(self.gamma / q[0] * self.pressure(q))

    def max_characteristic_velocity(self, actx, **kwargs):
        q = kwargs["state"]
        rho = q[0]
        rhov = q[2:self.dcoll.dim+2]
        v = rhov / rho
        return actx.np.sqrt(np.dot(v, v)) + self.sound_speed(q)


# {{{ Entropy stable operator

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

    momentum_flux = np.outer(rho_u, rho_u) / rho + np.eye(dim) * p

    def _euler_flux(idx):
        flux = np.empty((2+dim,), dtype=object)
        flux[0] = rho_u[idx]
        flux[1] = u[idx] * (rho_e + p)
        flux[2:dim+2] = momentum_flux[idx, :]
        return flux

    return [_euler_flux(idx) for idx in range(dim)]


def conservative_to_entropy_vars(actx, dcoll, cv_state, gamma=1.4):
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


def entropy_projection(actx, dcoll, dd_q, dd_f, cv_state, gamma=1.4):
    """todo.
    """
    from grudge.sbp_op import (quadrature_project,
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
    ev_state = quadrature_project(dcoll, dd_q, ev_state_q)
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

    p_ll = (gamma - 1) * (rhoe_ll - 0.5 * sum(rhou_ll * v_ll))
    p_rr = (gamma - 1) * (rhoe_rr - 0.5 * sum(rhou_rr * v_rr))

    # print(
    #     f"rho_ll = {rho_ll} "  + "\n"
    #     f"rho_rr = {rho_rr} "  + "\n"
    #     f"p_ll = {p_ll} "  + "\n"
    #     f"p_rr = {p_rr} "  + "\n"
    #     f"v_ll = {v_ll} "  + "\n"
    #     f"v_rr = {v_rr} " + "\n"
    # )

    beta_ll = 0.5 * rho_ll / p_ll
    beta_rr = 0.5 * rho_rr / p_rr

    specific_kin_ll = 0.5 * sum(v**2 for v in v_ll)
    specific_kin_rr = 0.5 * sum(v**2 for v in v_rr)

    def log_mean(x, y, epsilon=1e-4):
        """Computes the logarithmic mean using a numerically stable
        stable approach outlined in Appendix B of Ismail, Roe (2009).
        Affordable, entropy-consistent Euler flux functions II: Entropy
        production at shocks.
        [DOI: 10.1016/j.jcp.2009.04.021](https://doi.org/10.1016/j.jcp.2009.04.021)
        """
        f_squared =  (x * (x - 2 * y) + y * y) / (x * (x + 2 * y) + y * y)
        if f_squared < epsilon:
            f = 1 + 1/3 * f_squared + 1/5 * (f_squared**2) + 1/7 * (f_squared**3)
            return (x + y) / (2*f)
        else:
            return (x - y) / np.log(x/y)

    # Compute the necessary mean values
    rho_avg = 0.5 * (rho_ll + rho_rr)
    rho_mean  = log_mean(rho_ll,  rho_rr)

    beta_mean = log_mean(beta_ll, beta_rr)
    beta_avg = 0.5 * (beta_ll + beta_rr)

    v_avg = 0.5 * (v_ll + v_rr)
    p_mean = 0.5 * rho_avg / beta_avg
    velocity_square_avg = specific_kin_ll + specific_kin_rr

    fS_mass = rho_mean * v_avg[orientation]
    fS_momentum = fS_mass * v_avg
    fS_momentum[orientation] += p_mean
    fS_energy = fS_mass * (
        0.5 * (1/(gamma - 1 ) / beta_mean - velocity_square_avg)
        + np.dot(fS_momentum, v_avg)
    )
    return fS_mass, fS_energy, fS_momentum


def volume_flux_differencing(actx, dcoll, dq, df, state, gamma=1.4):
    """Computes the flux differencing operation: ∑_j 2Q[i, j] * f_S(q_i, q_j),
    where Q is a hybridized SBP derivative matrix and f_S is
    an entropy-conservative two-point flux.

    See `flux_chandrashekar` for a concrete implementation of such a two-point
    flux routine.
    """
    from grudge.sbp_op import hybridized_sbp_operators
    from grudge.geometry import area_element, inverse_metric_derivative_mat

    mesh = dcoll.mesh
    dim = dcoll.dim
    dtype = state[0].entry_dtype

    volm_discr = dcoll.discr_from_dd("vol")
    face_discr = dcoll.discr_from_dd("all_faces")
    volm_quad_discr = dcoll.discr_from_dd(dq)
    face_quad_discr = dcoll.discr_from_dd(df)

    # Geometry terms for building the physical derivative operators
    jacobian_dets = area_element(actx, dcoll)
    geo_factors = inverse_metric_derivative_mat(actx, dcoll)

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
        fgrp = face_discr.groups[gidx]
        vqgrp = volm_quad_discr.groups[gidx]
        fqgrp = face_quad_discr.groups[gidx]

        Nelements = mgrp.nelements
        Nq_vol = vqgrp.nunit_dofs
        Nq_faces = vqgrp.shape.nfaces * fqgrp.nunit_dofs
        Nq_total = Nq_vol + Nq_faces

        # Get state values for the group
        rho_gidx = rho[gidx]
        rhoe_gidx = rhoe[gidx]
        rhou_gidx = [ru[gidx] for ru in rhou]

        # Get SBP hybridized derivative operators on the ref element
        qmats = actx.to_numpy(
            thaw(hybridized_sbp_operators(actx,
                                          vgrp,
                                          vqgrp, fqgrp,
                                          dtype), actx)
        )

        # Get geometry factors for the group *gidx*
        jac_k = actx.to_numpy(jacobian_dets[gidx])
        gmat = np.array(
            [[actx.to_numpy(geo_factors[row, col][gidx])
              for col in range(dim)]
             for row in range(dim)]
        )

        # Group arrays for the Hadamard row-sum
        dQF_rho = np.empty(shape=(Nelements, Nq_total), dtype=dtype)
        dQF_rhoe = np.empty(shape=(Nelements, Nq_total), dtype=dtype)
        dQF_rhou = np.empty(shape=(dim, Nelements, Nq_total), dtype=dtype)

        # Element loop
        for eidx in range(Nelements):
            # NOTE: Assumes affine (jacobian det is constant in each cell)
            j_k = jac_k[eidx][0]
            # Build physical SBP operators
            Qrst = [sum(j_k*gmat[d, j][eidx][0]*qmats[j, :, :]
                        for j in range(dim)) for d in range(dim)]

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
                Qskew_d = Qrst[d] - Qrst[d].T
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
                            # (mass, energy, momentum);
                            # flux_chandrashekar returns a tuple of the form
                            # (fS_mass, fS_energy, fS_momentum)
                            fS_mass, fS_energy, fS_momentum = \
                                flux_chandrashekar(
                                    q_i,
                                    (local_rho[j], local_rhoe[j], local_rhou[:, j]),
                                    d,
                                    gamma=gamma)

                            # Compute upper triangular entry of 2Q .* F
                            QF_rho_ij = Qskew_d[i, j] * fS_mass
                            QF_rhoe_ij = Qskew_d[i, j] * fS_energy
                            QF_rhou_ij = Qskew_d[i, j] * fS_momentum

                            # Accumulate upper triangular part
                            dq_rho_i += QF_rho_ij
                            dq_rhoe_i += QF_rhoe_ij
                            dq_rhou_i += QF_rhou_ij

                            # Accumulate lower triangular part
                            dq_rho[j] -= QF_rho_ij
                            dq_rhoe[j] -= QF_rhoe_ij
                            dq_rhou[:, j] -= QF_rhou_ij
                        # end if
                    # end j
                    dq_rho[i] += dq_rho_i
                    dq_rhoe[i] += dq_rhoe_i
                    dq_rhou[:, i] += dq_rhou_i
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

    from grudge.sbp_op import volume_and_surface_quadrature_adjoint

    # Apply Vh.T = [Vq; Vf].T to the result
    return volume_and_surface_quadrature_adjoint(dcoll, dq, df, result)


def trace_pair_flux_chandrashekar(
        dcoll, tpair, gamma=1.4, reshape=False):
    """Entropy conserving numerical flux by Chandrashekar (2013)
    Kinetic Energy Preserving and Entropy Stable Finite Volume Schemes
    for Compressible Euler and Navier-Stokes Equations
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

    def log_mean(x, y, epsilon=1e-4):
        f_squared =  (x * (x - 2 * y) + y * y) / (x * (x + 2 * y) + y * y)
        return actx.np.where(
            # if f_squared < epsilon
            f_squared < epsilon,
            (x + y) / (2*(1 + 1/3 * f_squared
                          + 1/5 * (f_squared**2)
                          + 1/7 * (f_squared**3))),
            # else
            (x - y) / actx.np.log(x/y)
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

    rho_avg = 0.5 * (rho_int + rho_ext)
    beta_avg = 0.5 * (beta_int + beta_ext)
    v_avg = 0.5 * (v_int + v_ext)
    p_mean = 0.5 * rho_avg / beta_avg
    velocity_square_avg = specific_kin_int + specific_kin_ext

    rho_mean = log_mean(rho_int, rho_ext)
    beta_mean = log_mean(beta_int, beta_ext)

    def reshape_face_quad_dofs(dcoll, vec):
        if isinstance(vec, np.ndarray):
            return obj_array_vectorize(
                lambda vi: reshape_face_quad_dofs(dcoll, vi), vec)

        discr = dcoll.discr_from_dd("vol")
        face_discr = dcoll.discr_from_dd(dd_allfaces)

        return DOFArray(
            actx,
            data=tuple(
                vec_i.reshape(
                    vgrp.nelements,
                    vgrp.mesh_el_group.nfaces * fgrp.nunit_dofs
                )
                for vgrp, fgrp, vec_i in zip(
                    discr.groups, face_discr.groups, vec)
            )
        )

    num_fluxes = []
    for idx in range(dim):
        num_flux = np.empty((2+dim,), dtype=object)
        num_flux[0] = rho_mean * v_avg[idx]
        num_flux[2:dim+2] = num_flux[0] * v_avg
        num_flux[2:dim+2][idx] += p_mean
        num_flux[1] = num_flux[0] * (
            0.5 * (1/(gamma - 1 ) / beta_mean - velocity_square_avg)
            + sum(num_flux[2:dim+2] * v_avg)
        )
        num_flux = op.project(dcoll, dd_intfaces, dd_allfaces, num_flux)

        # FIXME: interior trace pairs still have original dof ordering;
        # reshaping to (nelem, n_total_face_dofs)
        if reshape:
            num_flux = reshape_face_quad_dofs(dcoll, num_flux)

        num_fluxes.append(num_flux)

    return num_fluxes


class EntropyStableEulerOperator(EulerOperator):

    def operator(self, t, q):
        from grudge.dof_desc import DOFDesc, DISCR_TAG_QUAD

        gamma = self.gamma
        dq = dof_desc.DOFDesc("vol", DISCR_TAG_QUAD)
        df = dof_desc.DOFDesc("all_faces", DISCR_TAG_QUAD)
        df_int = dof_desc.DOFDesc("int_faces", DISCR_TAG_QUAD)

        dcoll = self.dcoll
        actx = q[0].array_context

        print("Computing auxiliary conservative variables...")
        qtilde_allquad, entropy_vars = entropy_projection(
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

        from grudge.sbp_op import local_interior_trace_pair

        print("Computing interface numerical fluxes...")
        def entropy_tpair(tpair):
            dd_intfaces = tpair.dd
            dd_intfaces_quad = dd_intfaces.with_discr_tag(DISCR_TAG_QUAD)
            vint = tpair.int
            vext = tpair.ext
            vtilde_tpair = op.project(
                dcoll, dd_intfaces, dd_intfaces_quad, tpair)
            return TracePair(
                dd_intfaces_quad,
                interior=entropy_to_conservative_vars(
                    actx, dcoll, vtilde_tpair.int, gamma=gamma
                ),
                exterior=entropy_to_conservative_vars(
                    actx, dcoll, vtilde_tpair.ext, gamma=gamma
                )
            )

        num_fluxes = [
            trace_pair_flux_chandrashekar(dcoll, entropy_tpair(tpair),
                                          gamma=gamma, reshape=True)
            for tpair in op.interior_trace_pairs(dcoll, entropy_vars)
        ]
        # TODO: BCs
        # nodes = thaw(dcoll.nodes(), actx)
        # f_bdry = (
        #     # TODO: Scrutinize BCs
        #     - sum(
        #         self.boundary_numerical_flux(
        #             qtilde, self.bdry_fcts[btag](nodes, t), btag)
        #         for btag in self.bdry_fcts
        #     )
        # )
        print("Finished computing interface numerical fluxes.")

        print("Applying inverse mass and lifting operators...")
        # NOTE: Put everything together by applying lifting on surface terms
        # and the inverse mass matrix
        # du = M.inv * (-∑_d [V_q; V_f].T (2Q_d * F_d)1
        #               -V_f.T B_d (f*_d - f(q_f)))
        from grudge.sbp_op import inverse_sbp_mass, sbp_lift_operator

        # Compute: sum_i={x,y,z} (V_f.T @ B_i @ J_i) * f_iS(q+, q)
        lifted_fluxes = sum(
            sum(
                sbp_lift_operator(dcoll, idx, df, local_fluxes[idx])
                for idx in range(dcoll.dim)
            ) for local_fluxes in num_fluxes
        )
        dqhat = inverse_sbp_mass(dcoll, dq, -dQF1 - lifted_fluxes)
        print("Finished applying mass and lifting operators.")

        #import ipdb; ipdb.set_trace()
        1/0

        return dqhat

# }}}
