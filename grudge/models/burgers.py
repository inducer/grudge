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

from arraycontext import thaw

from grudge.models import HyperbolicOperator


# {{{ Inviscid operator

def burgers_numerical_flux(actx, dcoll, num_flux_type, u_tpair, max_wavespeed):
    dd = u_tpair.dd
    normal = thaw(dcoll.normal(dd), actx)
    max_wavespeed = op.project(dcoll, "vol", dd, max_wavespeed)

    num_flux_type = num_flux_type.lower()
    central = (0.5*u_tpair**2).avg @ normal
    if num_flux_type == "central":
        return central
    elif num_flux_type == "lf":
        return central - 0.5 * max_wavespeed * u_tpair.diff
    else:
        raise NotImplementedError(f"flux '{num_flux_type}' is not implemented")


class InviscidBurgers(HyperbolicOperator):
    flux_types = ["central", "lf"]

    def __init__(self, actx, dcoll, flux_type="central"):
        if flux_type not in self.flux_types:
            raise NotImplementedError(f"unknown flux type: '{flux_type}'")

        self.actx = actx
        self.dcoll = dcoll
        self.flux_type = flux_type

    def max_characteristic_velocity(self, actx, **kwargs):
        fields = kwargs["fields"]
        return op.elementwise_max(self.dcoll, abs(fields))

    def operator(self, t, u):
        dcoll = self.dcoll
        actx = self.actx
        max_wavespeed = self.max_characteristic_velocity(actx, fields=u)

        def flux(u):
            return 0.5 * (u**2)

        def numflux(tpair):
            return op.project(dcoll, tpair.dd, "all_faces",
                              burgers_numerical_flux(actx, dcoll,
                                                     self.flux_type,
                                                     tpair, max_wavespeed))

        return (
            op.inverse_mass(
                dcoll,
                sum(op.weak_local_d_dx(dcoll, d, flux(u[d]))
                    for d in range(dcoll.dim))
                - op.face_mass(
                    dcoll,
                    sum(numflux(tpair)
                        for tpair in op.interior_trace_pairs(dcoll, u))
                )
            )
        )

# }}}


# vim: foldmethod=marker
