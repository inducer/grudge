from __future__ import annotations

import pytest

import meshmode.mesh.generation as mgen
from arraycontext import NumpyArrayContext
from meshmode.mesh import BTAG_ALL, BTAG_NONE
from pytools.convergence import EOCRecorder

import grudge.geometry as geo
from grudge import op
from grudge.discretization import make_discretization_collection
from grudge.dt_utils import h_max_from_volume
from grudge.models.elliptic import InteriorPenaltyEllipticOperator


@pytest.mark.parametrize("bc_kind", ["dirichlet", "neumann"])
def test_sipg_elliptic_h_convergence_quadratic(bc_kind: str) -> None:
    actx = NumpyArrayContext()
    eoc = EOCRecorder()

    for n in [2, 4, 8]:
        mesh = mgen.generate_regular_rect_mesh(
            a=(-1.0, -1.0), b=(1.0, 1.0), nelements_per_axis=(n, n))
        dcoll = make_discretization_collection(actx, mesh, order=2)

        x = actx.thaw(dcoll.nodes())
        u_exact = x[0]**2 + x[0]*x[1] + x[1]**2 + 1.0
        rhs = 0.0 * x[0] - 4.0

        if bc_kind == "dirichlet":
            dir_bc = op.project(dcoll, "vol", BTAG_ALL, u_exact)

            elliptic_op = InteriorPenaltyEllipticOperator(
                dcoll,
                penalty_factor=10.0,
                dirichlet_tag=BTAG_ALL,
                dirichlet_bc=dir_bc,
                neumann_tag=BTAG_NONE,
            )
        elif bc_kind == "neumann":
            bdry_dd = BTAG_ALL
            bdry_normal = geo.normal(actx, dcoll, bdry_dd)
            grad_u = op.local_grad(dcoll, u_exact)
            grad_u_bdry = op.project(dcoll, "vol", bdry_dd, grad_u)
            neu_bc = grad_u_bdry[0]*bdry_normal[0] + grad_u_bdry[1]*bdry_normal[1]

            elliptic_op = InteriorPenaltyEllipticOperator(
                dcoll,
                penalty_factor=10.0,
                dirichlet_tag=BTAG_NONE,
                neumann_tag=BTAG_ALL,
                neumann_bc=neu_bc,
            )
        else:
            raise ValueError(f"invalid boundary condition type: '{bc_kind}'")

        resid = elliptic_op.operator(u_exact) - rhs
        err = actx.to_numpy(op.norm(dcoll, resid, 2))
        h_max = actx.to_numpy(h_max_from_volume(dcoll))
        eoc.add_data_point(h_max, err)

    assert eoc.max_error() < 5.0e-9 or eoc.order_estimate() > 1.0
