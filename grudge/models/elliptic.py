"""Interior penalty DG discretization of elliptic operators."""
from __future__ import annotations


__copyright__ = """
Copyright (C) 2026 University of Illinois Board of Trustees
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

from collections.abc import Callable
from typing import Any

import numpy as np

from meshmode.mesh import BTAG_ALL, BTAG_NONE

import grudge.geometry as geo
from grudge import op
from grudge.dof_desc import FACE_RESTR_ALL, as_dofdesc
from grudge.dt_utils import characteristic_lengthscales
from grudge.models import Operator


class InteriorPenaltyEllipticOperator(Operator):
    r"""Discretize :math:`-\nabla\cdot(\kappa \nabla u)` with SIPG.

    The implementation follows the symmetric interior penalty bilinear form

    .. math::

        a(u, v)
        = \int_\Omega \nabla u \cdot \kappa \nabla v \,dx
          - \sum_{e \in \mathcal{E}}
            \int_e [u]\cdot\{\kappa \nabla v\}
            + [v]\cdot\{\kappa \nabla u\}
            - \frac{\alpha_e}{h_e}[v]\cdot[u] \,ds.
    """

    def __init__(
            self,
            dcoll,
            *,
            kappa: Any = 1.0,
            penalty_factor: float = 10.0,
            dirichlet_tag=BTAG_ALL,
            dirichlet_bc: Any = 0.0,
            neumann_tag=BTAG_NONE,
            neumann_bc: Any = 0.0,
            comm_tag=None):
        self.dcoll = dcoll
        self.kappa = kappa
        self.penalty_factor = penalty_factor
        self.dirichlet_tag = dirichlet_tag
        self.dirichlet_bc = dirichlet_bc
        self.neumann_tag = neumann_tag
        self.neumann_bc = neumann_bc
        self.comm_tag = comm_tag

    @staticmethod
    def _is_active_boundary_tag(tag: Any) -> bool:
        return tag is not None and tag != BTAG_NONE

    def _get_boundary_data(self, actx, dd, bc):
        if isinstance(bc, Callable):
            return bc(actx.thaw(self.dcoll.nodes(dd=dd)))

        if np.isscalar(bc):
            return self.dcoll.discr_from_dd(dd).zeros(actx) + bc

        return bc

    def operator(self, u):
        dcoll = self.dcoll
        actx = u.array_context
        assert actx is not None

        all_faces_dd = as_dofdesc(FACE_RESTR_ALL)
        h = characteristic_lengthscales(actx, dcoll)
        h = h * (dcoll.discr_from_dd(as_dofdesc("vol")).zeros(actx) + 1.0)

        k_grad_u = self.kappa * op.local_grad(dcoll, u)

        # {{{ divergence operator with SIPG penalty flux
        q_flux = 0

        u_tpairs = op.interior_trace_pairs(dcoll, u, comm_tag=self.comm_tag)
        h_tpairs = op.interior_trace_pairs(dcoll, h, comm_tag=self.comm_tag)
        q_tpairs = op.interior_trace_pairs(dcoll, k_grad_u, comm_tag=self.comm_tag)

        for u_tpair, h_tpair, q_tpair in zip(
                u_tpairs, h_tpairs, q_tpairs, strict=True):
            normal = geo.normal(actx, dcoll, u_tpair.dd)
            penalty = self.penalty_factor / h_tpair.avg
            q_flux = q_flux + op.project(
                dcoll,
                u_tpair.dd,
                all_faces_dd,
                np.dot(q_tpair.avg, normal) - penalty * (u_tpair.int - u_tpair.ext))

        if self._is_active_boundary_tag(self.dirichlet_tag):
            dir_dd = as_dofdesc(self.dirichlet_tag)
            dir_normal = geo.normal(actx, dcoll, dir_dd)
            dir_h = op.project(dcoll, "vol", dir_dd, h)
            dir_u = op.project(dcoll, "vol", dir_dd, u)
            dir_q = op.project(dcoll, "vol", dir_dd, k_grad_u)
            dir_bc = self._get_boundary_data(actx, dir_dd, self.dirichlet_bc)
            dir_penalty = self.penalty_factor / dir_h

            q_flux = q_flux + op.project(
                dcoll, dir_dd, all_faces_dd,
                np.dot(dir_q, dir_normal) - dir_penalty * (dir_u - dir_bc))

        if self._is_active_boundary_tag(self.neumann_tag):
            neu_dd = as_dofdesc(self.neumann_tag)
            neu_bc = self._get_boundary_data(actx, neu_dd, self.neumann_bc)
            q_flux = q_flux + op.project(dcoll, neu_dd, all_faces_dd, neu_bc)

        return op.inverse_mass(
            dcoll,
            op.weak_local_div(dcoll, k_grad_u) - op.face_mass(dcoll, q_flux))
        # }}}
