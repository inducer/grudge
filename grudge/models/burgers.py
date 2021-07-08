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

    def entropy_function(self, u):
        """Returns the entropy function.

        :arg u: the state.
        """
        return 0.5 * (u**2)

    def flux(self, u):
        """Returns the flux for the Burgers operator in conservative form.

        :arg u: the state.
        """
        return 0.5 * (u**2)

    def numerical_fluxes(self, u):
        """Returns the numerical fluxes on the interior and boundary faces.

        :arg u: the state.
        """
        actx = u.array_context
        dcoll = self.dcoll
        num_flux_type = self.flux_type.lower()

        def _numerical_flux(utpair):
            dd = utpair.dd
            f_avg = 0.5 * (self.flux(utpair.int) + self.flux(utpair.ext))
            normal = thaw(dcoll.normal(dd), actx)

            central = sum([f_avg * normal[d] for d in range(dcoll.dim)])

            if num_flux_type == "central":
                return op.project(dcoll, dd, "all_faces", central)
            elif num_flux_type == "lf":
                max_wavespeed = op.project(
                    dcoll, "vol", dd,
                    self.max_characteristic_velocity(actx, fields=u)
                )
                lf_flux = central - 0.5 * max_wavespeed * utpair.diff
                return op.project(dcoll, dd, "all_faces", lf_flux)
            else:
                raise NotImplementedError(
                    f"flux '{num_flux_type}' is not implemented"
                )

        # FIXME: Generalzie BC interface
        # dir_bc = op.project(dcoll, "vol", self.bc_tag, self.bc)
        # bdry_fluxes = _numerical_flux(op.bv_trace_pair(dcoll,
        #                                                self.bc_tag,
        #                                                u, dir_bc))
        bdry_fluxes = 0

        return (
            sum(_numerical_flux(tpair) for tpair in op.interior_trace_pairs(dcoll, u))
            + bdry_fluxes
        )

    def operator(self, t, u):
        dcoll = self.dcoll
        dim = dcoll.dim

        return (
            op.inverse_mass(
                dcoll,
                sum(op.weak_local_d_dx(dcoll, d, self.flux(u)) for d in range(dim))
                - op.face_mass(dcoll, self.numerical_fluxes(u))
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
