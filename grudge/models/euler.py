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

from dataclasses import dataclass

from meshmode.dof_array import DOFArray

from grudge.models import HyperbolicOperator
from grudge.trace_pair import TracePair

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


def convert_to_array_container(dim, ary):
    return ArrayContainer(mass=ary[0],
                          energy=ary[1],
                          momentum=ary[2:2+dim])

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


def flux_chandrashekar(cv_ll, cv_rr):
    """Entropy conserving two-point flux by Chandrashekar (2013)
    Kinetic Energy Preserving and Entropy Stable Finite Volume Schemes
    for Compressible Euler and Navier-Stokes Equations
    [DOI: 10.4208/cicp.170712.010313a](https://doi.org/10.4208/cicp.170712.010313a)
    """
    rho_ll, v_ll, p_ll = cv_ll
    rho_rr, v_rr, p_rr = cv_rr

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

    identity = np.eye(len(v_avg))

    fS_mass = rho_mean * vel_avg
    fS_momentum = fS_mass * vel_avg + identity * p_mean
    fS_energy = fS_mass * (
        0.5 * (1/(gamma - 1 ) / beta_mean - velocity_square_avg)
        + np.dot(fS_momentum, vel_avg)
    )
    return fS_mass, fS_energy, fS_momentum


#def hadamard_sum(D, two_point_volume_flux, state, dim)


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

    def compute_volume_integrals(self, q_cv):
        # TODO Hand-write this damn calculation in Python instead
        # of trying to go through loopy


    def operator(self, t, q):
        dcoll = self.dcoll

        # Convert to array container
        q = ArrayContainer(mass=q[0],
                           energy=q[1],
                           momentum=q[2:2+dcoll.dim])

        actx = q.array_context
        nodes = thaw(self.dcoll.nodes(), actx)

        # Modified conservative variables using the entropy variables
        # Step 1: interpolate conserved vars to quadrature grid (if any)
        # q -> V_q p

        # Step 2: Convert conserved variables to entropy variables
        v = self.conservative_to_entropy_vars(q)

        # Step 3: Project the entropy variables
        # FIXME: Assumes quad/interpolation collocated; so Vq = I; Vf = I.
        # v = Minv * [Vq; Vf]^T v = Minv * v
        # v_projected = convert_to_array_container(
        #     dcoll.dim, op.inverse_mass(dcoll, v.join())
        # )

        # Step 4: Convert to conserved vars from projected entropy vars
        # and interpolate to all nodes (volume + face)
        q_projected = self.entropy_to_conservative_vars(v)
        q_projected_faces = op.project(dcoll, "vol", "all_faces", q_projected)

        import ipdb; ipdb.set_trace()

        # Evaluate two-point entropy conservative flux at all combinations
        # of elemental nodes
        fSvv = self.flux_chandrashekar(q_projected, q_projected)
        fSvf = self.flux_chandrashekar(q_projected, q_projected_faces)
        fSfv = self.flux_chandrashekar(q_projected_faces, q_projected)

        import ipdb; ipdb.set_trace()

        fSff = (
            sum(self.flux_chandrashekar(qtpair.int, qtpair.ext)
                for qtpair in op.interior_trace_pairs(dcoll, q_projected))
            + sum(
                self.self.flux_chandrashekar(q_projected, self.bdry_fcts[btag](nodes, t), btag)
                for btag in self.bdry_fcts
            )
        )

# }}}
