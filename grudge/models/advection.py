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
                - self.v.dot(sym.nabla(self.ambient_dim)*u)
                + sym.InverseMassOperator()(
                    sym.FaceMassOperator()(
                        flux(sym.int_tpair(u))
                        + flux(sym.bv_tpair(sym.BTAG_ALL, u, self.inflow_u))

                        # FIXME: Add back support for inflow/outflow tags
                        #+ flux(sym.bv_tpair(self.inflow_tag, u, bc_in))
                        #+ flux(sym.bv_tpair(self.outflow_tag, u, bc_out))
                        )))


class WeakAdvectionOperator(AdvectionOperatorBase):
    def flux(self, u):
        return self.weak_flux(u)

    def sym_operator(self):
        u = sym.var("u")

        # boundary conditions -------------------------------------------------
        bc_in = self.inflow_u
        # bc_out = sym.interp("vol", self.outflow_tag)(u)

        def flux(pair):
            return sym.interp(pair.dd, "all_faces")(
                    self.flux(pair))

        return sym.InverseMassOperator()(
                np.dot(
                    self.v, sym.stiffness_t(self.ambient_dim)*u)
                - sym.FaceMassOperator()(
                    flux(sym.int_tpair(u))
                    + flux(sym.bv_tpair(sym.BTAG_ALL, u, bc_in))

                    # FIXME: Add back support for inflow/outflow tags
                    #+ flux(sym.bv_tpair(self.inflow_tag, u, bc_in))
                    #+ flux(sym.bv_tpair(self.outflow_tag, u, bc_out))
                    ))

# }}}


# {{{ variable-coefficient advection

class VariableCoefficientAdvectionOperator(HyperbolicOperator):
    def __init__(self, dim, v, inflow_u, flux_type="central"):
        self.ambient_dim = dim 
        self.v = v
        self.inflow_u = inflow_u
        self.flux_type = flux_type
    def flux(self, u): 
        normal = sym.normal(u. dd, self.ambient_dim)
        
        surf_v = sym.interp("vol", u.dd)(self.v)
        
        
        v_dot_normal = sym.cse(np.dot(surf_v,normal), "v_dot_normal")
        norm_v = sym.sqrt(np.sum(self.v**2))
        
        if self.flux_type == "central":
            return u.avg*v_dot_normal
            # versus??
            #return v_dot_normal
            #return (u.int*v_dot_normal
                    #+ u.ext*v_dot_normal) * 0.5

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

    def sym_operator(self):
        u = sym.var("u")

       # boundary conditions -------------------------------------------------
        bc_in = self.inflow_u

        def flux(pair):
            return sym.interp(pair.dd, "all_faces")(
                    self.flux(pair))


        return sym.InverseMassOperator()(
                np.dot(
                    #self.v, sym.stiffness_t(self.ambient_dim)*u)
                    sym.stiffness_t(self.ambient_dim), sym.cse(self.v*u))
                 - sym.FaceMassOperator()(
                    flux(sym.int_tpair(u))
  		 + flux(sym.bv_tpair(sym.BTAG_ALL, u, bc_in))

                    # FIXME: Add back support for inflow/outflow tags
                    #+ flux(sym.bv_tpair(self.inflow_tag, u, bc_in))
                    #+ flux(sym.bv_tpair(self.outflow_tag, u, bc_out))
                   ))



# }}}


# vim: foldmethod=marker
