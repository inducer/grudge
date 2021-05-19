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

import numpy as np
import grudge.op as op

from grudge.models import HyperbolicOperator

from meshmode.dof_array import thaw


# {{{ Numerical flux

def burgers_numerical_flux(dcoll, flux_type, u_tpair, max_wavespeed):
    actx = u_tpair.int.array_context
    dd = u_tpair.dd
    normal = thaw(actx, op.normal(dcoll, dd))
    max_wavespeed = op.project(dcoll, "vol", dd, max_wavespeed)

    flux_type = flux_type.lower()
    central = u_tpair.avg * normal
    if flux_type == "central":
        return central
    elif flux_type == "lf":
        return central + 0.5 * max_wave_speed * (u_tpair.int - u_tpair.ext)
    else:
        raise NotImplementedError(f"flux '{flux_type}' is not implemented")

# }}}


# {{{ Inviscid Operator

class InviscidBurgers(HyperbolicOperator):
    flux_types = ["central", "lf"]

    def __init__(self, dcoll, flux_type="central"):
        if flux_type not in self.flux_types:
            raise NotImplementedError(f"unknown flux type: '{flux_type}'")

        self.dcoll = dcoll
        self.flux_type = flux_type

    def max_wavespeed(self, u):
        actx = self.dcoll._setup_actx
        return actx.np.sqrt(op.elementwise_max(self.dcoll, fields**2))

    def max_eigenvalue(self, t=None, fields=None, discr=None):
        return op.nodal_maximum(fields)

    def operator(self, t, u):
        dcoll = self.dcoll
        actx = u.array_context

        # max_wavespeed = self.max_eigenvalue(fields=u)
        max_wavespeed = self.max_wavespeed(u)

        def flux(u):
            return 0.5 * (u**2)

        def numflux(tpair):
            return op.project(dcoll, tpair.dd, "all_faces",
                              burgers_numerical_flux(dcoll, self.flux_type,
                                                     tpair, max_wavespeed))

        return (
            op.inverse_mass(
                dcoll,
                op.weak_local_grad(dcoll, flux(u))
                - op.face_mass(
                    dcoll,
                    sum(numflux(tpair)
                        for tpair in op.interior_trace_pairs(dcoll, u))
                )
            )
        )

# }}}