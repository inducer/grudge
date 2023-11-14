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
import grudge.geometry as geo

from grudge.models import HyperbolicOperator


# {{{ fluxes

def advection_weak_flux(dcoll, flux_type, u_tpair, velocity):
    r"""Compute the numerical flux for the advection operator
    $(v \cdot \nabla)u$.
    """
    actx = u_tpair.int.array_context
    dd = u_tpair.dd
    normal = geo.normal(actx, dcoll, dd)
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

    def __init__(self, dcoll, v, inflow_u=None, flux_type="central"):
        if flux_type not in self.flux_types:
            raise ValueError(f"unknown flux type: '{flux_type}'")

        if inflow_u is not None:
            if not isinstance(inflow_u, types.LambdaType):
                raise ValueError(
                    "A specified inflow_u must be a lambda function of time `t`"
                )

        self.dcoll = dcoll
        self.v = v
        self.inflow_u = inflow_u
        self.flux_type = flux_type

    def weak_flux(self, u_tpair):
        return advection_weak_flux(self.dcoll, self.flux_type, u_tpair, self.v)

    def max_characteristic_velocity(self, actx, **kwargs):
        return sum(v_i**2 for v_i in self.v)**0.5


class StrongAdvectionOperator(AdvectionOperatorBase):
    def flux(self, u_tpair):
        actx = u_tpair.int.array_context
        dd = u_tpair.dd
        normal = geo.normal(actx, self.dcoll, dd)
        v_dot_normal = np.dot(self.v, normal)

        return u_tpair.int * v_dot_normal - self.weak_flux(u_tpair)

    def operator(self, t, u):
        from meshmode.mesh import BTAG_ALL

        dcoll = self.dcoll

        def flux(tpair):
            return op.project(dcoll, tpair.dd, "all_faces", self.flux(tpair))

        if self.inflow_u is not None:
            inflow_flux = flux(op.bv_trace_pair(dcoll,
                                                BTAG_ALL,
                                                interior=u,
                                                exterior=self.inflow_u(t)))
        else:
            inflow_flux = 0

        return (
            -self.v.dot(op.local_grad(dcoll, u))
            + op.inverse_mass(
                dcoll,
                op.face_mass(
                    dcoll,
                    sum(flux(tpair) for tpair in op.interior_trace_pairs(dcoll, u))
                    + inflow_flux

                    # FIXME: Add support for inflow/outflow tags
                    # + flux(op.bv_trace_pair(dcoll,
                    #                         self.inflow_tag,
                    #                         interior=u,
                    #                         exterior=bc_in))
                    # + flux(op.bv_trace_pair(dcoll,
                    #                         self.outflow_tag,
                    #                         interior=u,
                    #                         exterior=bc_out))
                )
            )
        )


class WeakAdvectionOperator(AdvectionOperatorBase):
    def flux(self, u_tpair):
        return self.weak_flux(u_tpair)

    def operator(self, t, u):
        from meshmode.mesh import BTAG_ALL

        dcoll = self.dcoll

        def flux(tpair):
            return op.project(dcoll, tpair.dd, "all_faces", self.flux(tpair))

        if self.inflow_u is not None:
            inflow_flux = flux(op.bv_trace_pair(dcoll,
                                                BTAG_ALL,
                                                interior=u,
                                                exterior=self.inflow_u(t)))
        else:
            inflow_flux = 0

        return (
            op.inverse_mass(
                dcoll,
                np.dot(self.v, op.weak_local_grad(dcoll, u))
                - op.face_mass(
                    dcoll,
                    sum(flux(tpair) for tpair in op.interior_trace_pairs(dcoll, u))
                    + inflow_flux

                    # FIXME: Add support for inflow/outflow tags
                    # + flux(op.bv_trace_pair(dcoll,
                    #                         self.inflow_tag,
                    #                         interior=u,
                    #                         exterior=bc_in))
                    # + flux(op.bv_trace_pair(dcoll,
                    #                         self.outflow_tag,
                    #                         interior=u,
                    #                         exterior=bc_out))
                )
            )
        )

# }}}


def to_quad_int_tpairs(dcoll, u, quad_tag):
    from grudge.dof_desc import DISCR_TAG_QUAD
    from grudge.trace_pair import TracePair

    if issubclass(quad_tag, DISCR_TAG_QUAD):
        return [
            TracePair(
                tpair.dd.with_discr_tag(quad_tag),
                interior=op.project(
                    dcoll, tpair.dd,
                    tpair.dd.with_discr_tag(quad_tag), tpair.int
                ),
                exterior=op.project(
                    dcoll, tpair.dd,
                    tpair.dd.with_discr_tag(quad_tag), tpair.ext
                )
            ) for tpair in op.interior_trace_pairs(dcoll, u)
        ]
    else:
        return op.interior_trace_pairs(dcoll, u)


# {{{ variable-coefficient advection

class VariableCoefficientAdvectionOperator(AdvectionOperatorBase):
    def __init__(self, dcoll, v, inflow_u, flux_type="central", quad_tag=None):
        super().__init__(dcoll, v, inflow_u, flux_type=flux_type)

        if quad_tag is None:
            from grudge.dof_desc import DISCR_TAG_BASE
            quad_tag = DISCR_TAG_BASE

        self.quad_tag = quad_tag

    def flux(self, u_tpair):
        from grudge.dof_desc import DD_VOLUME_ALL

        surf_v = op.project(self.dcoll, DD_VOLUME_ALL, u_tpair.dd, self.v)
        return advection_weak_flux(self.dcoll, self.flux_type, u_tpair, surf_v)

    def operator(self, t, u):
        from grudge.dof_desc import DOFDesc, DD_VOLUME_ALL, DTAG_VOLUME_ALL
        from meshmode.mesh import BTAG_ALL
        from meshmode.discretization.connection import FACE_RESTR_ALL

        face_dd = DOFDesc(FACE_RESTR_ALL, self.quad_tag)
        boundary_dd = DOFDesc(BTAG_ALL, self.quad_tag)
        quad_dd = DOFDesc(DTAG_VOLUME_ALL, self.quad_tag)

        dcoll = self.dcoll

        def flux(tpair):
            return op.project(dcoll, tpair.dd, face_dd, self.flux(tpair))

        def to_quad(arg):
            return op.project(dcoll, DD_VOLUME_ALL, quad_dd, arg)

        if self.inflow_u is not None:
            inflow_flux = flux(op.bv_trace_pair(dcoll,
                                                boundary_dd,
                                                interior=u,
                                                exterior=self.inflow_u(t)))
        else:
            inflow_flux = 0

        quad_v = to_quad(self.v)
        quad_u = to_quad(u)

        return (
            op.inverse_mass(
                dcoll,
                sum(op.weak_local_d_dx(dcoll, quad_dd, d, quad_u * quad_v[d])
                    for d in range(dcoll.ambient_dim))
                - op.face_mass(
                    dcoll,
                    face_dd,
                    sum(flux(quad_tpair)
                        for quad_tpair in to_quad_int_tpairs(dcoll, u,
                                                             self.quad_tag))
                    + inflow_flux

                    # FIXME: Add support for inflow/outflow tags
                    # + flux(op.bv_trace_pair(dcoll,
                    #                         self.inflow_tag,
                    #                         interior=u,
                    #                         exterior=bc_in))
                    # + flux(op.bv_trace_pair(dcoll,
                    #                         self.outflow_tag,
                    #                         interior=u,
                    #                         exterior=bc_out))
                )
            )
        )

# }}}


# {{{ closed surface advection

def v_dot_n_tpair(actx, dcoll, velocity, trace_dd):
    from grudge.dof_desc import BoundaryDomainTag
    from grudge.trace_pair import TracePair
    from meshmode.discretization.connection import FACE_RESTR_INTERIOR

    normal = geo.normal(actx, dcoll, trace_dd.with_discr_tag(None))
    v_dot_n = velocity.dot(normal)
    i = op.project(dcoll, trace_dd.with_discr_tag(None), trace_dd, v_dot_n)

    assert isinstance(trace_dd.domain_tag, BoundaryDomainTag)
    if trace_dd.domain_tag.tag is FACE_RESTR_INTERIOR:
        e = dcoll.opposite_face_connection(trace_dd.domain_tag)(i)
    else:
        raise ValueError("Unrecognized domain tag: %s" % trace_dd.domain_tag)

    return TracePair(trace_dd, interior=i, exterior=e)


def surface_advection_weak_flux(dcoll, flux_type, u_tpair, velocity):
    actx = u_tpair.int.array_context
    v_dot_n = v_dot_n_tpair(actx, dcoll, velocity, trace_dd=u_tpair.dd)
    # NOTE: the normals in v_dot_n point to the exterior of their respective
    # elements, so this is actually just an average
    v_dot_n = 0.5 * (v_dot_n.int - v_dot_n.ext)

    flux_type = flux_type.lower()
    if flux_type == "central":
        return u_tpair.avg * v_dot_n
    elif flux_type == "lf":
        return (u_tpair.avg * v_dot_n
                + 0.5 * actx.np.fabs(v_dot_n) * (u_tpair.int - u_tpair.ext))
    else:
        raise ValueError(f"flux '{flux_type}' is not implemented")


class SurfaceAdvectionOperator(AdvectionOperatorBase):
    def __init__(self, dcoll, v, flux_type="central", quad_tag=None):
        super().__init__(dcoll, v, inflow_u=None, flux_type=flux_type)

        if quad_tag is None:
            from grudge.dof_desc import DISCR_TAG_BASE
            quad_tag = DISCR_TAG_BASE

        self.quad_tag = quad_tag

    def flux(self, u_tpair):
        from grudge.dof_desc import DD_VOLUME_ALL

        surf_v = op.project(self.dcoll, DD_VOLUME_ALL,
                            u_tpair.dd.with_discr_tag(None), self.v)
        return surface_advection_weak_flux(self.dcoll,
                                           self.flux_type,
                                           u_tpair,
                                           surf_v)

    def operator(self, t, u):
        from grudge.dof_desc import DOFDesc, DD_VOLUME_ALL, DTAG_VOLUME_ALL
        from meshmode.discretization.connection import FACE_RESTR_ALL

        face_dd = DOFDesc(FACE_RESTR_ALL, self.quad_tag)
        quad_dd = DOFDesc(DTAG_VOLUME_ALL, self.quad_tag)

        dcoll = self.dcoll

        def flux(tpair):
            return op.project(dcoll, tpair.dd, face_dd, self.flux(tpair))

        def to_quad(arg):
            return op.project(dcoll, DD_VOLUME_ALL, quad_dd, arg)

        quad_v = to_quad(self.v)
        quad_u = to_quad(u)

        return (
            op.inverse_mass(
                dcoll,
                sum(op.weak_local_d_dx(dcoll, quad_dd, d, quad_u * quad_v[d])
                    for d in range(dcoll.ambient_dim))
                - op.face_mass(
                    dcoll,
                    face_dd,
                    sum(flux(quad_tpair)
                        for quad_tpair in to_quad_int_tpairs(dcoll, u,
                                                             self.quad_tag))
                )
            )
        )

# }}}


# vim: foldmethod=marker
