"""Operators modeling advective phenomena."""

__copyright__ = "Copyright (C) 2009-2017 Andreas Kloeckner, Bogdan Enache"

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
from grudge import sym


# {{{ fluxes

def advection_weak_flux(flux_type, u, velocity):
    normal = sym.normal(u.dd, len(velocity))
    v_dot_n = sym.cse(velocity.dot(normal), "v_dot_normal")

    flux_type = flux_type.lower()
    if flux_type == "central":
        return u.avg * v_dot_n
    elif flux_type == "lf":
        norm_v = sym.sqrt((velocity**2).sum())
        return u.avg * v_dot_n + 0.5 * norm_v * (u.int - u.ext)
    elif flux_type == "upwind":
        u_upwind = sym.If(
                sym.Comparison(v_dot_n, ">", 0),
                u.int,      # outflow
                u.ext       # inflow
                )
        return u_upwind * v_dot_n
    else:
        raise ValueError(f"flux '{flux_type}' is not implemented")

# }}}


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

        if flux_type not in self.flux_types:
            raise ValueError(f"unknown flux type: '{flux_type}'")

    def weak_flux(self, u):
        return advection_weak_flux(self.flux_type, u, self.v)

    def max_eigenvalue(self, t=None, fields=None, discr=None):
        return la.norm(self.v)


class StrongAdvectionOperator(AdvectionOperatorBase):
    def flux(self, u):
        normal = sym.normal(u.dd, self.ambient_dim)
        v_dot_normal = sym.cse(self.v.dot(normal), "v_dot_normal")

        return u.int * v_dot_normal - self.weak_flux(u)

    def sym_operator(self):
        u = sym.var("u")

        def flux(pair):
            return sym.project(pair.dd, "all_faces")(
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

        def flux(pair):
            return sym.project(pair.dd, "all_faces")(
                    self.flux(pair))

        bc_in = self.inflow_u
        # bc_out = sym.project(sym.DD_VOLUME, self.outflow_tag)(u)

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

class VariableCoefficientAdvectionOperator(AdvectionOperatorBase):
    def __init__(self, v, inflow_u, flux_type="central", quad_tag="product"):
        super().__init__(
                v, inflow_u, flux_type=flux_type)

        self.quad_tag = quad_tag

    def flux(self, u):
        surf_v = sym.project(sym.DD_VOLUME, u.dd)(self.v)
        return advection_weak_flux(self.flux_type, u, surf_v)

    def sym_operator(self):
        u = sym.var("u")

        def flux(pair):
            return sym.project(pair.dd, face_dd)(self.flux(pair))

        face_dd = sym.DOFDesc(sym.FACE_RESTR_ALL, self.quad_tag)
        boundary_dd = sym.DOFDesc(sym.BTAG_ALL, self.quad_tag)
        quad_dd = sym.DOFDesc(sym.DTAG_VOLUME_ALL, self.quad_tag)

        to_quad = sym.project(sym.DD_VOLUME, quad_dd)
        stiff_t_op = sym.stiffness_t(self.ambient_dim,
                dd_in=quad_dd, dd_out=sym.DD_VOLUME)

        quad_v = to_quad(self.v)
        quad_u = to_quad(u)

        return sym.InverseMassOperator()(
                sum(stiff_t_op[n](quad_u * quad_v[n])
                    for n in range(self.ambient_dim))
                - sym.FaceMassOperator(face_dd, sym.DD_VOLUME)(
                    flux(sym.int_tpair(u, self.quad_tag))
                    + flux(sym.bv_tpair(boundary_dd, u, self.inflow_u))

                    # FIXME: Add back support for inflow/outflow tags
                    #+ flux(sym.bv_tpair(self.inflow_tag, u, bc_in))
                    #+ flux(sym.bv_tpair(self.outflow_tag, u, bc_out))
                ))
# }}}


# {{{ closed surface advection

def v_dot_n_tpair(velocity, dd=None):
    if dd is None:
        dd = sym.DOFDesc(sym.FACE_RESTR_INTERIOR)

    ambient_dim = len(velocity)
    normal = sym.normal(dd.with_qtag(None), ambient_dim, dim=ambient_dim - 2)

    return sym.int_tpair(velocity.dot(normal),
            qtag=dd.quadrature_tag,
            from_dd=dd.with_qtag(None))


def surface_advection_weak_flux(flux_type, u, velocity):
    v_dot_n = v_dot_n_tpair(velocity, dd=u.dd)
    # NOTE: the normals in v_dot_n point to the exterior of their respective
    # elements, so this is actually just an average
    v_dot_n = sym.cse(0.5 * (v_dot_n.int - v_dot_n.ext), "v_dot_normal")

    flux_type = flux_type.lower()
    if flux_type == "central":
        return u.avg * v_dot_n
    elif flux_type == "lf":
        return u.avg * v_dot_n + 0.5 * sym.fabs(v_dot_n) * (u.int - u.ext)
    else:
        raise ValueError(f"flux '{flux_type}' is not implemented")


class SurfaceAdvectionOperator(AdvectionOperatorBase):
    def __init__(self, v, flux_type="central", quad_tag=None):
        super().__init__(
                v, inflow_u=None, flux_type=flux_type)
        self.quad_tag = quad_tag

    def flux(self, u):
        surf_v = sym.project(sym.DD_VOLUME, u.dd.with_qtag(None))(self.v)
        return surface_advection_weak_flux(self.flux_type, u, surf_v)

    def sym_operator(self):
        u = sym.var("u")

        def flux(pair):
            return sym.project(pair.dd, face_dd)(self.flux(pair))

        face_dd = sym.DOFDesc(sym.FACE_RESTR_ALL, self.quad_tag)
        quad_dd = sym.DOFDesc(sym.DTAG_VOLUME_ALL, self.quad_tag)

        to_quad = sym.project(sym.DD_VOLUME, quad_dd)
        stiff_t_op = sym.stiffness_t(self.ambient_dim,
                dd_in=quad_dd, dd_out=sym.DD_VOLUME)

        quad_v = to_quad(self.v)
        quad_u = to_quad(u)

        return sym.InverseMassOperator()(
                sum(stiff_t_op[n](quad_u * quad_v[n])
                    for n in range(self.ambient_dim))
                - sym.FaceMassOperator(face_dd, sym.DD_VOLUME)(
                    flux(sym.int_tpair(u, self.quad_tag))
                    )
                )

# }}}

# vim: foldmethod=marker
