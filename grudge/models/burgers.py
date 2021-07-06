"""Operator modeling Burgers' equation"""

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

import grudge.op as op
import grudge.dof_desc as dof_desc

from arraycontext import thaw, make_loopy_program

from grudge.models import HyperbolicOperator

from pytools import memoize_in


# {{{ Inviscid operator

def burgers_numerical_flux(dcoll, num_flux_type, u_tpair, max_wavespeed):
    dd = u_tpair.dd
    ut_avg = u_tpair.avg
    actx = ut_avg.array_context
    normal = thaw(dcoll.normal(dd), actx)
    max_wavespeed = op.project(dcoll, "vol", dd, max_wavespeed)

    num_flux_type = num_flux_type.lower()

    central = sum([ut_avg * normal[d] for d in range(dcoll.dim)])
    if num_flux_type == "central":
        return central
    elif num_flux_type == "lf":
        return central - 0.5 * max_wavespeed * u_tpair.diff
    else:
        raise NotImplementedError(f"flux '{num_flux_type}' is not implemented")


class InviscidBurgers(HyperbolicOperator):
    flux_types = ["central", "lf"]

    def __init__(self, dcoll, flux_type="central"):
        if flux_type not in self.flux_types:
            raise NotImplementedError(f"unknown flux type: '{flux_type}'")

        self.dcoll = dcoll
        self.flux_type = flux_type

    def max_characteristic_velocity(self, actx, **kwargs):
        fields = kwargs["fields"]
        return op.elementwise_max(self.dcoll, abs(fields))

    def operator(self, t, u):
        dcoll = self.dcoll
        actx = u.array_context
        max_wavespeed = self.max_characteristic_velocity(actx, fields=u)

        def flux(u):
            return 0.5 * (u**2)

        def numflux(tpair):
            return op.project(dcoll, tpair.dd, "all_faces",
                              burgers_numerical_flux(dcoll, self.flux_type,
                                                     tpair, max_wavespeed))

        return (
            op.inverse_mass(
                dcoll,
                sum(op.weak_local_d_dx(dcoll, d, flux(u))
                    for d in range(dcoll.dim))
                - op.face_mass(
                    dcoll,
                    sum(numflux(tpair)
                        for tpair in op.interior_trace_pairs(dcoll, flux(u)))
                )
            )
        )

# }}}


# {{{ Entropy conservative inviscid operator

def burgers_flux_differencing_mat(actx, state_i, state_j):
    @memoize_in(actx, (burgers_flux_differencing_mat, "flux_diff_knl"))
    def energy_conserving_flux_prg():
        return make_loopy_program(
            [
                "{[iel]: 0 <= iel < nelements}",
                "{[idof]: 0 <= idof < left_nodes}",
                "{[jdof]: 0 <= jdof < right_nodes}"
            ],
            """
            result[iel, idof, jdof] = 1/6 * (
                vec_left[iel, idof] * vec_left[iel, idof]
                + vec_left[iel, idof] * vec_right[iel, jdof]
                + vec_right[iel, jdof] * vec_right[iel, jdof]
            )
            """,
            name="flux_diff_mat"
        )

    return DOFArray(
        actx,
        data=tuple(
            actx.call_loopy(energy_conserving_flux_prg(),
                            vec_left=vec_i,
                            vec_right=vec_j)["result"]

            for vec_i, vec_j in zip(state_i, state_j)
        )
    )


class EntropyConservativeInviscidBurgers(InviscidBurgers):

    def operator(self, t, u):
        dcoll = self.dcoll
        actx = u.array_context

        # dd_q = dof_desc.DOFDesc("vol", dof_desc.DISCR_TAG_BASE)
        dd_f = dof_desc.DOFDesc("all_faces", dof_desc.DISCR_TAG_BASE)
        # u_q = op.project(dcoll, "vol", dd_q, u)
        # u_f = op.project(dcoll, "vol", dd_f, u)

        # F_qq = burgers_flux_differencing_mat(actx, u, u)
        # F_qf = burgers_flux_differencing_mat(actx, u, u_f)
        # F_fq = burgers_flux_differencing_mat(actx, u_f, u)
        # F_ff = burgers_flux_differencing_mat(actx, u_f, u_f)

        normal = thaw(dcoll.normal(dd_f), actx)

        def energy_conserving_fluxes(tpair):
            return op.project(
                dcoll, tpair.dd, dd_f,
                1/6 * (tpair.ext * tpair.int + tpair.ext ** 2)
            )

        return -(
            1/3 * sum(op.local_d_dx(dcoll, d, u ** 2)
                      + u * op.local_d_dx(dcoll, d, u)
                      for d in range(dcoll.dim))
            + op.inverse_mass(
                dcoll,
                op.face_mass(
                    dcoll,
                    sum((sum(energy_conserving_fluxes(tpair)
                             for tpair in op.interior_trace_pairs(dcoll, u))
                         - 1/3 * op.project(dcoll, "vol", dd_f, u ** 2)) * normal[d]
                        for d in range(dcoll.dim)
                    )
                )
            )
        )

# }}}


# vim: foldmethod=marker
