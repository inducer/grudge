# -*- coding: utf8 -*-
"""Operators modeling advective phenomena."""

from __future__ import division, absolute_import

__copyright__ = "Copyright (C) 2009 Andreas Kloeckner"

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
import numpy.linalg as la

from grudge.models import HyperbolicOperator
from grudge.models.second_order import CentralSecondDerivative
from grudge import sym


# {{{ constant-coefficient advection

class AdvectionOperatorBase(HyperbolicOperator):
    flux_types = [
            "central",
            "upwind",
            "lf"
            ]

    def __init__(self, v, inflow_u, flux_type="central"):
        self.ambient_dim = len(v)
        self.v = v
        self.inflow_u = inflow_u
        self.flux_type = flux_type

    def weak_flux(self, u):
        normal = sym.normal(u. dd, self.ambient_dim)

        v_dot_normal = sym.cse(self.v.dot(normal), "v_dot_normal")
        norm_v = sym.sqrt((self.v**2).sum())

        if self.flux_type == "central":
            return u.avg*v_dot_normal
        elif self.flux_type == "lf":
            return u.avg*v_dot_normal + 0.5*norm_v*(u.int - u.ext)
        elif self.flux_type == "upwind":
            return (
                    v_dot_normal * sym.If(
                        sym.Comparison(v_dot_normal, ">", 0),
                        u.int,  # outflow
                        u.ext,  # inflow
                        ))
        else:
            raise ValueError("invalid flux type")

    def max_eigenvalue(self, t=None, fields=None, discr=None):
        return la.norm(self.v)


class StrongAdvectionOperator(AdvectionOperatorBase):
    def flux(self, u):
        normal = sym.normal(u. dd, self.ambient_dim)
        v_dot_normal = sym.cse(self.v.dot(normal), "v_dot_normal")

        return u.int * v_dot_normal - self.weak_flux(u)

    def sym_operator(self):
        u = sym.var("u")

        def flux(pair):
            return sym.interp(pair.dd, "all_faces")(
                    self.flux(pair))

        return (
                -self.v.dot(sym.nabla(self.ambient_dim)*u)
                + sym.InverseMassOperator()(
                    sym.FaceMassOperator()(
                        flux(sym.int_tpair(u))
                        + flux(sym.bv_tpair(sym.BTAG_ALL, u, self.inflow_u)))))


class WeakAdvectionOperator(AdvectionOperatorBase):
    def flux(self):
        return self.weak_flux()

    def sym_operator(self):
        from grudge.symbolic import (
                Field,
                BoundaryPair,
                get_flux_operator,
                make_stiffness_t,
                InverseMassOperator,
                RestrictToBoundary)

        u = Field("u")

        # boundary conditions -------------------------------------------------
        bc_in = Field("bc_in")
        bc_out = RestrictToBoundary(self.outflow_tag)*u

        stiff_t = make_stiffness_t(self.dimensions)
        m_inv = InverseMassOperator()

        flux_op = get_flux_operator(self.flux())

        return m_inv(np.dot(self.v, stiff_t*u) - (
                    flux_op(u)
                    + flux_op(BoundaryPair(u, bc_in, self.inflow_tag))
                    + flux_op(BoundaryPair(u, bc_out, self.outflow_tag))
                    ))

# }}}


# {{{ variable-coefficient advection

class VariableCoefficientAdvectionOperator(HyperbolicOperator):
    """A class for space- and time-dependent DG advection operators.

    :param advec_v: Adheres to the :class:`grudge.data.ITimeDependentGivenFunction`
      interfacer and is an n-dimensional vector representing the velocity.
    :param bc_u_f: Adheres to the :class:`grudge.data.ITimeDependentGivenFunction`
      interface and is a scalar representing the boundary condition at all
      boundary faces.

    Optionally allows diffusion.
    """

    flux_types = [
            "central",
            "upwind",
            "lf"
            ]

    def __init__(self,
            dimensions,
            advec_v,
            bc_u_f="None",
            flux_type="central",
            diffusion_coeff=None,
            diffusion_scheme=CentralSecondDerivative()):
        self.dimensions = dimensions
        self.advec_v = advec_v
        self.bc_u_f = bc_u_f
        self.flux_type = flux_type
        self.diffusion_coeff = diffusion_coeff
        self.diffusion_scheme = diffusion_scheme

    # {{{ flux ----------------------------------------------------------------
    def flux(self):
        from grudge.flux import (
                make_normal,
                FluxVectorPlaceholder,
                flux_max)
        from pymbolic.primitives import IfPositive

        d = self.dimensions

        w = FluxVectorPlaceholder((1+d)+1)
        u = w[0]
        v = w[1:d+1]
        c = w[1+d]

        normal = make_normal(self.dimensions)

        if self.flux_type == "central":
            return (u.int*np.dot(v.int, normal)
                    + u.ext*np.dot(v.ext, normal)) * 0.5
        elif self.flux_type == "lf":
            n_vint = np.dot(normal, v.int)
            n_vext = np.dot(normal, v.ext)
            return 0.5 * (n_vint * u.int + n_vext * u.ext) \
                   - 0.5 * (u.ext - u.int) \
                   * flux_max(c.int, c.ext)

        elif self.flux_type == "upwind":
            return (
                    IfPositive(
                        np.dot(normal, v.avg),
                        np.dot(normal, v.int) * u.int,  # outflow
                        np.dot(normal, v.ext) * u.ext,  # inflow
                        ))
        else:
            raise ValueError("invalid flux type")
    # }}}

    def bind_characteristic_velocity(self, discr):
        from grudge.symbolic.operators import (
                ElementwiseMaxOperator)
        from grudge.symbolic import make_sym_vector
        velocity_vec = make_sym_vector("v", self.dimensions)
        velocity = ElementwiseMaxOperator()(
                np.dot(velocity_vec, velocity_vec)**0.5)

        compiled = discr.compile(velocity)

        def do(t, u):
            return compiled(v=self.advec_v.volume_interpolant(t, discr))

        return do

    def sym_operator(self, with_sensor=False):
        # {{{ operator preliminaries ------------------------------------------
        from grudge.symbolic import (Field, BoundaryPair, get_flux_operator,
                make_stiffness_t, InverseMassOperator, make_sym_vector,
                ElementwiseMaxOperator, RestrictToBoundary)

        from grudge.symbolic.primitives import make_common_subexpression as cse

        from grudge.symbolic.operators import (
                QuadratureGridUpsampler,
                QuadratureInteriorFacesGridUpsampler)

        to_quad = QuadratureGridUpsampler("quad")
        to_if_quad = QuadratureInteriorFacesGridUpsampler("quad")

        from grudge.tools import join_fields, \
                                ptwise_dot

        u = Field("u")
        v = make_sym_vector("v", self.dimensions)
        c = ElementwiseMaxOperator()(ptwise_dot(1, 1, v, v))

        quad_u = cse(to_quad(u))
        quad_v = cse(to_quad(v))

        w = join_fields(u, v, c)
        quad_face_w = to_if_quad(w)
        # }}}

        # {{{ boundary conditions ---------------------------------------------

        from grudge.mesh import BTAG_ALL
        bc_c = to_quad(RestrictToBoundary(BTAG_ALL)(c))
        bc_u = to_quad(Field("bc_u"))
        bc_v = to_quad(RestrictToBoundary(BTAG_ALL)(v))

        if self.bc_u_f is "None":
            bc_w = join_fields(0, bc_v, bc_c)
        else:
            bc_w = join_fields(bc_u, bc_v, bc_c)

        minv_st = make_stiffness_t(self.dimensions)
        m_inv = InverseMassOperator()

        flux_op = get_flux_operator(self.flux())
        # }}}

        # {{{ diffusion -------------------------------------------------------
        if with_sensor or (
                self.diffusion_coeff is not None and self.diffusion_coeff != 0):
            if self.diffusion_coeff is None:
                diffusion_coeff = 0
            else:
                diffusion_coeff = self.diffusion_coeff

            if with_sensor:
                diffusion_coeff += Field("sensor")

            from grudge.second_order import SecondDerivativeTarget

            # strong_form here allows IPDG to reuse the value of grad u.
            grad_tgt = SecondDerivativeTarget(
                    self.dimensions, strong_form=True,
                    operand=u)

            self.diffusion_scheme.grad(grad_tgt, bc_getter=None,
                    dirichlet_tags=[], neumann_tags=[])

            div_tgt = SecondDerivativeTarget(
                    self.dimensions, strong_form=False,
                    operand=diffusion_coeff*grad_tgt.minv_all)

            self.diffusion_scheme.div(div_tgt,
                    bc_getter=None,
                    dirichlet_tags=[], neumann_tags=[])

            diffusion_part = div_tgt.minv_all
        else:
            diffusion_part = 0

        # }}}

        to_quad = QuadratureGridUpsampler("quad")
        quad_u = cse(to_quad(u))
        quad_v = cse(to_quad(v))

        return m_inv(np.dot(minv_st, cse(quad_v*quad_u))
                - (flux_op(quad_face_w)
                    + flux_op(BoundaryPair(quad_face_w, bc_w, BTAG_ALL)))) \
                            + diffusion_part

    def bind(self, discr, sensor=None):
        compiled_sym_operator = discr.compile(
                self.sym_operator(with_sensor=sensor is not None))

        from grudge.mesh import check_bc_coverage, BTAG_ALL
        check_bc_coverage(discr.mesh, [BTAG_ALL])

        def rhs(t, u):
            kwargs = {}
            if sensor is not None:
                kwargs["sensor"] = sensor(t, u)

            if self.bc_u_f is not "None":
                kwargs["bc_u"] = \
                        self.bc_u_f.boundary_interpolant(t, discr, tag=BTAG_ALL)

            return compiled_sym_operator(
                    u=u,
                    v=self.advec_v.volume_interpolant(t, discr),
                    **kwargs)

        return rhs

    def max_eigenvalue(self, t, fields=None, discr=None):
        # Gives the max eigenvalue of a vector of eigenvalues.
        # As the velocities of each node is stored in the velocity-vector-field
        # a pointwise dot product of this vector has to be taken to get the
        # magnitude of the velocity at each node. From this vector the maximum
        # values limits the timestep.

        from grudge.tools import ptwise_dot
        v = self.advec_v.volume_interpolant(t, discr)
        return discr.nodewise_max(ptwise_dot(1, 1, v, v)**0.5)

# }}}


# vim: foldmethod=marker
