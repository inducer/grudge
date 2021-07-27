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
class EulerState:
    density: DOFArray
    total_energy: DOFArray
    momentum: np.ndarray  # [object array of DOFArrays]

    @property
    def array_context(self):
        return self.density.array_context

    @property
    def dim(self):
        return len(self.momentum)

    @property
    def velocity(self):
        return self.momentum / self.density

    def join(self):
        return _join_fields(
            dim=self.dim,
            density=self.density,
            total_energy=self.total_energy,
            momentum=self.momentum
        )


def _join_fields(dim, density, total_energy, momentum):

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
        _aux_shape(density, ()),
        _aux_shape(total_energy, ()),
        _aux_shape(momentum, (dim,))]

    from pytools import single_valued
    aux_shape = single_valued(aux_shapes)

    result = np.empty((2+dim,) + aux_shape, dtype=object)
    result[0] = density
    result[1] = total_energy
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
        q = EulerState(density=q[0],
                       total_energy=q[1],
                       momentum=q[2:2+dcoll.dim])

        euler_flux_vol = self.euler_flux(q)
        euler_flux_bnd = (
            sum(self.numerical_flux(tpair)
                for tpair in op.interior_trace_pairs(dcoll, q))
            + sum(self.boundary_flux(q, btag) for btag in self.bdry_fcts)
        )
        return op.inverse_mass(
            dcoll,
            op.weak_local_div(dcoll, euler_flux_vol.join())
            - op.face_mass(dcoll, euler_flux_bnd.join())
        )

    def euler_flux(self, q):
        p = self.pressure(q)
        mom = q.momentum

        return EulerState(
            density=mom,
            total_energy=mom * (q.total_energy + p) / q.density,
            momentum=np.outer(mom, mom) / q.density + np.eye(q.dim)*p
        )

    def numerical_flux(self, q_tpair):
        """Return the numerical flux across a face given the solution on
        both sides *q_tpair*.
        """
        actx = q_tpair.int.array_context

        def _compute_wavespeed(q):
            v = q.velocity
            return actx.np.sqrt(np.dot(v, v)) + self.sound_speed(q)

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

        flux_weak = flux_tpair.avg @ normal - lam*q_tpair.diff / 2

        return op.project(self.dcoll, q_tpair.dd, "all_faces", flux_weak)

    def boundary_flux(self, q, btag):
        actx = q.array_context
        nhat = thaw(self.dcoll.normal(btag), actx)
        q_bc = op.project(self.dcoll, "vol", btag, q)
        # TODO: This just implements a solid wall BC for now
        # (boundary flux is equal to the interior pressure contribution
        # since v.n = 0 --- we do this by evaluating the euler flux)
        bdry_flux = self.euler_flux(q_bc) @ nhat
        return op.project(self.dcoll, btag, "all_faces", bdry_flux)

    def kinetic_energy(self, q):
        mom = q.momentum
        return (0.5 * np.dot(mom, mom) / q.density)

    def internal_energy(self, q):
        return (q.total_energy - self.kinetic_energy(q))

    def pressure(self, q):
        return self.internal_energy(q) * (self.gamma - 1.0)

    def temperature(self, q):
        return (
            (((self.gamma - 1.0) / self.gas_const)
             * self.internal_energy(q) / q.density)
        )

    def sound_speed(self, q):
        actx = q.array_context
        return actx.np.sqrt(self.gamma / q.density * self.pressure(q))

    def max_characteristic_velocity(self, actx, **kwargs):
        q = kwargs["state"]
        v = q.velocity
        return actx.np.sqrt(np.dot(v, v)) + self.sound_speed(q)
