"""Operators modeling advective phenomena."""

__copyright__ = """
Copyright (C) 2009-2017 Andreas Kloeckner, Bogdan Enache
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
import grudge.op as op
import types

from grudge.models import HyperbolicOperator
from grudge.symbolic.primitives import TracePair

from meshmode.dof_array import thaw

from grudge import sym


# {{{ fluxes

def advection_weak_flux(dcoll, flux_type, u_tpair, velocity):
    r"""Compute the numerical flux for the advection operator
    $(v \cdot \nabla)u$.
    """
    actx = u_tpair.int.array_context
    dd = u_tpair.dd
    normal = thaw(actx, op.normal(dcoll, dd))
    v_dot_n = np.dot(velocity, normal)

    flux_type = flux_type.lower()
    if flux_type == "central":
        return u_tpair.avg * v_dot_n
    elif flux_type == "lf":
        norm_v = np.sqrt(sum(velocity**2))
        return u_tpair.avg * v_dot_n + 0.5 * norm_v * (u_tpair.int - u_tpair.ext)
    elif flux_type == "upwind":
        u_upwind = actx.np.where(v_dot_n > 0, u_tpair.int, u_tpair.ext)
        return u_upwind * v_dot_n
    else:
        raise ValueError(f"flux '{flux_type}' is not implemented")

# }}}


# {{{ constant-coefficient advection

class AdvectionOperatorBase(HyperbolicOperator):
    flux_types = ["central", "upwind", "lf"]

    def __init__(self, dcoll, v, inflow_u, flux_type="central"):
        if flux_type not in self.flux_types:
            raise ValueError(f"unknown flux type: '{flux_type}'")

        if not isinstance(inflow_u, types.LambdaType):
            raise ValueError(
                "inflow_u must be a lambda function of time `t`"
            )

        self.dcoll = dcoll
        self.v = v
        self.inflow_u = inflow_u
        self.flux_type = flux_type

    def weak_flux(self, u_tpair):
        return advection_weak_flux(self.dcoll, self.flux_type, u_tpair, self.v)


class StrongAdvectionOperator(AdvectionOperatorBase):
    def flux(self, u_tpair):
        actx = u_tpair.int.array_context
        dd = u_tpair.dd
        normal = thaw(actx, op.normal(self.dcoll, dd))
        v_dot_normal = np.dot(self.v, normal)

        return u_tpair.int * v_dot_normal - self.weak_flux(u_tpair)

    def operator(self, t, u):
        from meshmode.mesh import BTAG_ALL

        dcoll = self.dcoll
        inflow_u = self.inflow_u(t)

        def flux(tpair):
            return op.project(dcoll, tpair.dd, "all_faces", self.flux(tpair))

        return (
            -self.v.dot(op.local_grad(dcoll, u))
            + op.inverse_mass(
                dcoll,
                op.face_mass(
                    dcoll,
                    flux(op.interior_trace_pair(dcoll, u))
                    + flux(TracePair(BTAG_ALL,
                                     interior=op.project(
                                         dcoll, "vol", BTAG_ALL, u
                                     ),
                                     exterior=inflow_u))

                    # FIXME: Add support for inflow/outflow tags
                    # + flux(TracePair(self.inflow_tag,
                    #                  interior=op.project(
                    #                      dcoll, "vol", self.inflow_tag, u
                    #                  ),
                    #                  exterior=bc_in))
                    # + flux(TracePair(self.outflow_tag,
                    #                  interior=op.project(
                    #                      dcoll, "vol", self.outflow_tag, u
                    #                  ),
                    #                  exterior=bc_out))
                )
            )
        )


class WeakAdvectionOperator(AdvectionOperatorBase):
    def flux(self, u_tpair):
        return self.weak_flux(u_tpair)

    def operator(self, t, u):
        from meshmode.mesh import BTAG_ALL

        dcoll = self.dcoll
        inflow_u = self.inflow_u(t)

        def flux(tpair):
            return op.project(dcoll, tpair.dd, "all_faces", self.flux(tpair))

        return (
            op.inverse_mass(
                dcoll,
                np.dot(self.v, op.weak_local_grad(dcoll, u))
                - op.face_mass(
                    dcoll,
                    flux(op.interior_trace_pair(dcoll, u))
                    + flux(TracePair(BTAG_ALL,
                                     interior=op.project(
                                         dcoll, "vol", BTAG_ALL, u
                                     ),
                                     exterior=inflow_u))

                    # FIXME: Add support for inflow/outflow tags
                    # + flux(TracePair(self.inflow_tag,
                    #                  interior=op.project(
                    #                      dcoll, "vol", self.inflow_tag, u
                    #                  ),
                    #                  exterior=bc_in))
                    # + flux(TracePair(self.outflow_tag,
                    #                  interior=op.project(
                    #                      dcoll, "vol", self.outflow_tag, u
                    #                  ),
                    #                  exterior=bc_out))
                )
            )
        )

# }}}


# {{{ variable-coefficient advection

class VariableCoefficientAdvectionOperator(AdvectionOperatorBase):
    def __init__(self, dcoll, v, inflow_u, flux_type="central", quad_tag=None):
        super().__init__(dcoll, v, inflow_u, flux_type=flux_type)

        if quad_tag is None:
            from grudge.dof_desc import DISCR_TAG_BASE
            quad_tag = DISCR_TAG_BASE

        self.quad_tag = quad_tag

    def flux(self, u_tpair):
        from grudge.dof_desc import DD_VOLUME

        surf_v = op.project(self.dcoll, DD_VOLUME, u_tpair.dd, self.v)
        return advection_weak_flux(self.dcoll, self.flux_type, u_tpair, surf_v)

    def operator(self, t, u):
        from grudge.dof_desc import DOFDesc, DD_VOLUME, DTAG_VOLUME_ALL
        from meshmode.mesh import BTAG_ALL
        from meshmode.discretization.connection import \
            FACE_RESTR_ALL, FACE_RESTR_INTERIOR

        face_dd = DOFDesc(FACE_RESTR_ALL, self.quad_tag)
        boundary_dd = DOFDesc(BTAG_ALL, self.quad_tag)
        quad_dd = DOFDesc(DTAG_VOLUME_ALL, self.quad_tag)

        dcoll = self.dcoll
        inflow_u = self.inflow_u(t)

        def flux(tpair):
            return op.project(dcoll, tpair.dd, face_dd, self.flux(tpair))

        def to_quad(arg):
            return op.project(dcoll, DD_VOLUME, quad_dd, arg)

        def to_quad_int_tpair(arg, quad_tag):
            trace_dd = DOFDesc(FACE_RESTR_INTERIOR, quad_tag)
            i = op.project(dcoll, DD_VOLUME,
                           trace_dd.with_discr_tag(None), arg)
            e = dcoll.opposite_face_connection()(i)
            if trace_dd.uses_quadrature():
                i = op.project(dcoll, trace_dd.with_discr_tag(None),
                               trace_dd, i)
                e = op.project(dcoll, trace_dd.with_discr_tag(None),
                               trace_dd, e)
            return TracePair(trace_dd, interior=i, exterior=e)

        quad_v = to_quad(self.v)
        quad_u = to_quad(u)

        return (
            op.inverse_mass(
                dcoll,
                sum(op.weak_local_d_dx(dcoll, quad_dd, d, quad_u * quad_v[d])
                    for d in range(dcoll.dim))
                - op.face_mass(
                    dcoll,
                    face_dd,
                    flux(to_quad_int_tpair(u, self.quad_tag))
                    + flux(TracePair(boundary_dd,
                                     interior=op.project(
                                         dcoll, DD_VOLUME, boundary_dd, u
                                     ),
                                     exterior=inflow_u))

                    # FIXME: Add support for inflow/outflow tags
                    # + flux(TracePair(self.inflow_tag,
                    #                  interior=op.project(
                    #                      dcoll, DD_VOLUME, self.inflow_tag, u
                    #                  ),
                    #                  exterior=bc_in))
                    # + flux(TracePair(self.outflow_tag,
                    #                  interior=op.project(
                    #                      dcoll, DD_VOLUME, self.outflow_tag, u
                    #                  ),
                    #                  exterior=bc_out))
                )
            )
        )

# }}}


# {{{ closed surface advection

def v_dot_n_tpair(velocity, dd=None):
    from grudge.dof_desc import DOFDesc
    from meshmode.discretization.connection import FACE_RESTR_INTERIOR

    if dd is None:
        dd = DOFDesc(FACE_RESTR_INTERIOR)

    ambient_dim = len(velocity)
    normal = sym.normal(dd.with_discr_tag(None),
                        ambient_dim, dim=ambient_dim - 2)

    return sym.int_tpair(velocity.dot(normal),
            qtag=dd.discretization_tag,
            from_dd=dd.with_discr_tag(None))


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
        from grudge.dof_desc import DD_VOLUME

        surf_v = sym.project(DD_VOLUME, u.dd.with_discr_tag(None))(self.v)
        return surface_advection_weak_flux(self.flux_type, u, surf_v)

    def sym_operator(self):
        from grudge.dof_desc import DOFDesc, DD_VOLUME, DTAG_VOLUME_ALL
        from meshmode.discretization.connection import FACE_RESTR_ALL

        u = sym.var("u")

        def flux(pair):
            return sym.project(pair.dd, face_dd)(self.flux(pair))

        face_dd = DOFDesc(FACE_RESTR_ALL, self.quad_tag)
        quad_dd = DOFDesc(DTAG_VOLUME_ALL, self.quad_tag)

        to_quad = sym.project(DD_VOLUME, quad_dd)
        stiff_t_op = sym.stiffness_t(self.ambient_dim,
                dd_in=quad_dd, dd_out=DD_VOLUME)

        quad_v = to_quad(self.v)
        quad_u = to_quad(u)

        return sym.InverseMassOperator()(
                sum(stiff_t_op[n](quad_u * quad_v[n])
                    for n in range(self.ambient_dim))
                - sym.FaceMassOperator(face_dd, DD_VOLUME)(
                    flux(sym.int_tpair(u, self.quad_tag))
                    )
                )

# }}}

# vim: foldmethod=marker
