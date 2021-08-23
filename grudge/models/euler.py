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

from pytools.obj_array import make_obj_array

import grudge.op as op


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
        mom = q.momentum

        return ArrayContainer(
            mass=mom,
            energy=mom * (q.energy + p) / q.mass,
            momentum=np.outer(mom, mom) / q.mass + np.eye(q.dim)*p
        )

    def numerical_flux(self, q_tpair):
        """Return the numerical flux across a face given the solution on
        both sides *q_tpair*.
        """
        actx = q_tpair.int.array_context

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
        actx = q.array_context

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
        mom = q.momentum
        return (0.5 * np.dot(mom, mom) / q.mass)

    def internal_energy(self, q):
        return (q.energy - self.kinetic_energy(q))

    def pressure(self, q):
        return self.internal_energy(q) * (self.gamma - 1.0)

    def temperature(self, q):
        return (
            (((self.gamma - 1.0) / self.gas_const)
             * self.internal_energy(q) / q.mass)
        )

    def sound_speed(self, q):
        actx = q.array_context
        return actx.np.sqrt(self.gamma / q.mass * self.pressure(q))

    def max_characteristic_velocity(self, actx, **kwargs):
        q = kwargs["state"]
        v = q.velocity
        return actx.np.sqrt(np.dot(v, v)) + self.sound_speed(q)


# {{{ Entropy stable operator

def full_quadrature_state(actx, dcoll, state):
    """Return a concatentated state array of both volume and surface
    degrees of freedom.
    """
    state_faces = op.project(dcoll, "vol", "all_faces", state)
    mesh = dcoll.mesh
    dim = dcoll.dim
    volm_discr = dcoll.discr_from_dd("vol")
    face_discr = dcoll.discr_from_dd("all_faces")

    # Group loop
    mass_data = []
    energy_data = []
    momentum_data = []
    for gidx, _ in enumerate(mesh.groups):
        vgrp = volm_discr.groups[gidx]
        fgrp = face_discr.groups[gidx]

        # Volume and face data for the group *gidx*
        # NOTE: Need to cast to numpy array because
        # actx.np.concatenate(..., axis=1) doesn't work out of the box
        # due to memory layout incompatibilities?
        mass_data.append(actx.from_numpy(
            np.concatenate(
                [actx.to_numpy(state[0][gidx]),
                 actx.to_numpy(state_faces[0][gidx].reshape(
                     vgrp.nelements,
                     vgrp.mesh_el_group.nfaces*fgrp.nunit_dofs))],
                axis=1
            )
        ))
        energy_data.append(actx.from_numpy(
            np.concatenate(
                [actx.to_numpy(state[1][gidx]),
                 actx.to_numpy(state_faces[1][gidx].reshape(
                     vgrp.nelements,
                     vgrp.mesh_el_group.nfaces*fgrp.nunit_dofs))],
                axis=1
            )
        ))
        momentum_data.append([
            actx.from_numpy(
                np.concatenate(
                    [actx.to_numpy(state[2:dim+2][d][gidx]),
                     actx.to_numpy(state_faces[2:dim+2][d][gidx].reshape(
                         vgrp.nelements,
                         vgrp.mesh_el_group.nfaces*fgrp.nunit_dofs))],
                    axis=1
                )
            ) for d in range(dim)
        ])

    mass_dof_ary = DOFArray(actx, data=tuple(mass_data))
    energy_dof_ary = DOFArray(actx, data=tuple(energy_data))
    momentum_dof_ary = make_obj_array(
        [DOFArray(actx,
                  data=tuple(mom_data[d]
                             for mom_data in momentum_data))
                  for d in range(dim)]
    )

    result = np.empty((2+dim,), dtype=object)
    result[0] = mass_dof_ary
    result[1] = energy_dof_ary
    result[2:dim+2] = momentum_dof_ary

    return result


def log_mean(x, y, epsilon=1e-4):
    """Computes the logarithmic mean using a numerically stable
    stable approach outlined in Appendix B of
    Ismail, Roe (2009). Affordable, entropy-consistent Euler flux functions II:
    Entropy production at shocks.
    [DOI: 10.1016/j.jcp.2009.04.021](https://doi.org/10.1016/j.jcp.2009.04.021)
    """
    f_squared =  (x * (x - 2 * y) + y * y) / (x * (x + 2 * y) + y * y)
    if f_squared < epsilon:
        f = 1 + 1/3 * f_squared + 1/5 * (f_squared**2) + 1/7 * (f_squared**3)
        return (x + y) / (2*f)
    else:
        return (x - y) / np.log(x/y)


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

    v_ll = np.array([rhou/rho_ll for rhou in rhou_ll])
    v_rr = np.array([rhou/rho_rr for rhou in rhou_rr])

    p_ll = (gamma - 1) * (
        rhoe_ll - 0.5 * sum(rhov * v for rhov, v in zip(rhou_ll, v_ll))
    )
    p_rr = (gamma - 1) * (
        rhoe_rr - 0.5 * sum(rhov * v for rhov, v in zip(rhou_rr, v_rr))
    )

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


def flux_differencing_kernel(actx, dcoll, quad_state, gamma=1.4):
    mesh = dcoll.mesh
    dim = dcoll.dim

    volm_discr = dcoll.discr_from_dd("vol")
    face_discr = dcoll.discr_from_dd("all_faces")

    mass = quad_state[0]
    energy = quad_state[1]
    momentum = quad_state[2:dim+2]

    # Group loop
    for gidx, mgrp in enumerate(mesh.groups):
        vgrp = volm_discr.groups[gidx]
        fgrp = face_discr.groups[gidx]
        Nq_total = vgrp.nunit_dofs + vgrp.shape.nfaces * fgrp.nunit_dofs

        # Convert group arrays into numpy arrays
        mass_ary = actx.to_numpy(mass[gidx])
        energy_ary = actx.to_numpy(energy[gidx])
        momentum_arys = [actx.to_numpy(momentum[d][gidx]) for d in range(dim)]

        # Element loop
        for eidx in range(mgrp.nelements):
            for d in range(dim):
                for i in range(Nq_total):
                    q_i = (
                        mass_ary[eidx][i],
                        energy_ary[eidx][i],
                        [mom[eidx][i] for mom in momentum_arys]
                    )
                    for j in range(Nq_total):
                        q_j = (
                            mass_ary[eidx][j],
                            energy_ary[eidx][j],
                            [mom[eidx][j] for mom in momentum_arys]
                        )
                        flux_ij = flux_chandrashekar(q_i, q_j, d, gamma=gamma)
                        print(flux_ij)


class EntropyStableEulerOperator(EulerOperator):

    def physical_entropy(self, rho, pressure):
        actx = rho.array_context
        return actx.np.log(pressure) - self.gamma*actx.np.log(rho)

    def conservative_to_entropy_vars(self, cv):
        gamma = self.gamma
        inv_gamma_minus_one = 1/(gamma - 1)

        rho = cv.mass
        rho_e = cv.energy
        velocity = cv.velocity

        v_square = sum(v ** 2 for v in velocity)
        p = self.pressure(cv)
        s = self.physical_entropy(rho, p)
        rho_p = rho / p

        v1 = (gamma - s) * inv_gamma_minus_one - 0.5 * rho_p * v_square
        v2 = -rho_p
        v3 = rho_p * velocity

        return ArrayContainer(mass=v1, energy=v2, momentum=v3)

    def entropy_to_conservative_vars(self, ev):
        actx = ev.array_context
        gamma = self.gamma
        inv_gamma_minus_one = 1/(gamma - 1)

        ev = ev * (gamma - 1)
        v1 = ev.mass
        v2 = ev.momentum
        v3 = ev.energy

        v_square = sum(v**2 for v in v2)
        s = gamma - v1 + v_square/(2*v3)
        rho_iota = (
            ((gamma -1) / (-v3**gamma)**inv_gamma_minus_one)
            * actx.np.exp(-s * inv_gamma_minus_one)
        )

        rho = -rho_iota * v3
        rho_u = rho_iota * v2
        rho_e = rho_iota * (1 - v_square/(2*v3))

        return ArrayContainer(mass=rho, energy=rho_u, momentum=rho_e)

    def operator(self, t, q):
        dcoll = self.dcoll
        actx = q[0].array_context

        quad_state = full_quadrature_state(actx, dcoll, q)
        flux_differencing_kernel(actx, dcoll, quad_state, self.gamma)

        1/0

        # from grudge.sbp_op import weak_hybridized_local_sbp
        # weak_hybridized_local_sbp(dcoll, q)

# }}}
