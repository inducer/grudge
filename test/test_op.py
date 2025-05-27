__copyright__ = "Copyright (C) 2021 University of Illinois Board of Trustees"

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


import logging

import numpy as np
import pytest

import meshmode.mesh.generation as mgen
from arraycontext import pytest_generate_tests_for_array_contexts
from meshmode.discretization.poly_element import (
    InterpolatoryEdgeClusteredGroupFactory,
    QuadratureGroupFactory,
)
from meshmode.mesh import BTAG_ALL, TensorProductElementGroup
from pytools.obj_array import make_obj_array

from grudge import geometry, op
from grudge.array_context import PytestPyOpenCLArrayContextFactory
from grudge.discretization import make_discretization_collection
from grudge.dof_desc import (
    DISCR_TAG_BASE,
    DISCR_TAG_QUAD,
    DTAG_VOLUME_ALL,
    FACE_RESTR_ALL,
    VTAG_ALL,
    BoundaryDomainTag,
    as_dofdesc,
)
from grudge.trace_pair import bv_trace_pair


logger = logging.getLogger(__name__)
pytest_generate_tests = pytest_generate_tests_for_array_contexts(
        [PytestPyOpenCLArrayContextFactory])


# {{{ gradient

@pytest.mark.parametrize("form", ["strong", "weak", "weak-overint"])
@pytest.mark.parametrize("dim", [1, 2, 3])
@pytest.mark.parametrize("order", [2, 3])
@pytest.mark.parametrize("warp_mesh", [False, True])
@pytest.mark.parametrize("tpe", [False, True])
@pytest.mark.parametrize(("vectorize", "nested"), [
    (False, False),
    (True, False),
    (True, True)
    ])
def test_gradient(actx_factory, form, dim, order, tpe, vectorize, nested,
                  warp_mesh, visualize=False):
    actx = actx_factory()
    group_cls = None
    if tpe:
        if dim == 1:
            pytest.skip("Skipping 1D test for tensor product elements.")
        group_cls = TensorProductElementGroup

    def vectorize_if_requested(vec):
        if vectorize:
            return make_obj_array([(i+1)*vec for i in range(dim)])
        else:
            return vec

    from pytools.convergence import EOCRecorder
    eoc_rec = EOCRecorder()

    for n in [8, 12, 16] if warp_mesh else [4, 6, 8]:
        if warp_mesh:
            if dim == 1:
                pytest.skip("warped mesh in 1D not implemented")
            mesh = mgen.generate_warped_rect_mesh(
                dim=dim, order=order, nelements_side=n,
                group_cls=group_cls)
        else:
            mesh = mgen.generate_regular_rect_mesh(
                a=(-1,)*dim, b=(1,)*dim,
                nelements_per_axis=(n,)*dim,
                group_cls=group_cls)

        dcoll = make_discretization_collection(
                   actx, mesh,
                   discr_tag_to_group_factory={
                       DISCR_TAG_BASE:
                       InterpolatoryEdgeClusteredGroupFactory(order),
                       DISCR_TAG_QUAD: QuadratureGroupFactory(3 * order)
                   })

        if "overint" in form:
            quad_discr_tag = DISCR_TAG_QUAD
        else:
            quad_discr_tag = DISCR_TAG_BASE

        allfaces_dd_quad = as_dofdesc(FACE_RESTR_ALL, quad_discr_tag)
        vol_dd_base = as_dofdesc(DTAG_VOLUME_ALL)
        vol_dd_quad = vol_dd_base.with_discr_tag(quad_discr_tag)
        bdry_dd_base = as_dofdesc(BTAG_ALL)
        bdry_dd_quad = bdry_dd_base.with_discr_tag(quad_discr_tag)

        x_base = actx.thaw(dcoll.nodes(dd=vol_dd_base))
        bdry_x_base = actx.thaw(dcoll.nodes(bdry_dd_base))

        def f(x):
            result = 1
            for i in range(dim-1):
                result = result * actx.np.sin(np.pi*x[i])
            result = result * actx.np.cos(np.pi/2*x[dim-1])
            return result

        def grad_f(x):
            result = make_obj_array([1 for _ in range(dim)])
            for i in range(dim-1):
                for j in range(i):
                    result[i] = result[i] * actx.np.sin(np.pi*x[j])
                result[i] = result[i] * np.pi*actx.np.cos(np.pi*x[i])
                for j in range(i+1, dim-1):
                    result[i] = result[i] * actx.np.sin(np.pi*x[j])
                result[i] = result[i] * actx.np.cos(np.pi/2*x[dim-1])
            for j in range(dim-1):
                result[dim-1] = result[dim-1] * actx.np.sin(np.pi*x[j])
            result[dim-1] = result[dim-1] * (-np.pi/2*actx.np.sin(np.pi/2*x[dim-1]))
            return result

        def get_flux(u_tpair, dcoll=dcoll):
            dd = u_tpair.dd
            dd_allfaces = dd.with_domain_tag(
                BoundaryDomainTag(FACE_RESTR_ALL, VTAG_ALL)
                )
            normal = geometry.normal(actx, dcoll, dd)
            u_avg = u_tpair.avg
            if vectorize:
                if nested:
                    flux = make_obj_array([u_avg_i * normal for u_avg_i in u_avg])
                else:
                    flux = np.outer(u_avg, normal)
            else:
                flux = u_avg * normal
            return op.project(dcoll, dd, dd_allfaces, flux)

        u_base = vectorize_if_requested(f(x_base))
        bdry_u_base = vectorize_if_requested(f(bdry_x_base))

        if form == "strong":
            grad_u = (
                op.local_grad(dcoll, u_base, nested=nested)
                # No flux terms because u doesn't have inter-el jumps
                )
        elif form.startswith("weak"):
            assert form in ["weak", "weak-overint"]

            # Uses _apply_inverse_mass_operator_quad (if overint)
            grad_u = op.inverse_mass(
                dcoll, vol_dd_quad,
                -op.weak_local_grad(
                    dcoll, vol_dd_quad,
                    op.project(dcoll, vol_dd_base, vol_dd_quad, u_base),
                    nested=nested)
                +
                op.face_mass(dcoll, allfaces_dd_quad,
                             sum(get_flux(
                                 op.project_tracepair(dcoll, allfaces_dd_quad,
                                                      utpair))
                                 for utpair in op.interior_trace_pairs(
                                         dcoll, u_base, volume_dd=vol_dd_base))
                             + get_flux(
                                 op.project_tracepair(
                                     dcoll, bdry_dd_quad,
                                     bv_trace_pair(dcoll, bdry_dd_base, u_base,
                                                   bdry_u_base))))
            )
        else:
            raise ValueError("Invalid form argument.")

        if vectorize:
            expected_grad_u = make_obj_array(
                [(i+1)*grad_f(x_base) for i in range(dim)])
            if not nested:
                expected_grad_u = np.stack(expected_grad_u, axis=0)
        else:
            expected_grad_u = grad_f(x_base)

        if visualize:
            # the code below does not handle the vectorized case
            assert not vectorize

            from grudge.shortcuts import make_visualizer
            vis = make_visualizer(dcoll, vis_order=order if dim == 3 else dim+3)

            filename = (f"test_gradient_{form}_{dim}_{order}"
                f"{'_vec' if vectorize else ''}{'_nested' if nested else ''}.vtu")
            vis.write_vtk_file(filename, [
                ("u", u_base),
                ("grad_u", grad_u),
                ("expected_grad_u", expected_grad_u),
                ], overwrite=True)

        rel_linf_err = actx.to_numpy(
            op.norm(dcoll, grad_u - expected_grad_u, np.inf)
            / op.norm(dcoll, expected_grad_u, np.inf))
        eoc_rec.add_data_point(1./n, rel_linf_err)

    print("L^inf error:")
    print(eoc_rec)
    assert (eoc_rec.order_estimate() >= order - 0.5
                or eoc_rec.max_error() < 1e-11)

# }}}


# {{{ divergence

@pytest.mark.parametrize("form", ["strong", "weak"])
@pytest.mark.parametrize("dim", [1, 2, 3])
@pytest.mark.parametrize("order", [2, 3])
@pytest.mark.parametrize(("vectorize", "nested"), [
    (False, False),
    (True, False),
    (True, True)
    ])
@pytest.mark.parametrize("overint", [True, False])
@pytest.mark.parametrize("tpe", [False, True])
def test_divergence(actx_factory, form, dim, order, vectorize, nested,
                    overint, tpe, visualize=False):
    actx = actx_factory()
    if dim == 1 and tpe:
        pytest.skip("Skipping 1D test for tensor product elements.")

    from pytools.convergence import EOCRecorder
    eoc_rec = EOCRecorder()

    for n in [4, 6, 8]:
        group_cls = TensorProductElementGroup if tpe else None
        mesh = mgen.generate_regular_rect_mesh(
            a=(-1,)*dim, b=(1,)*dim,
            nelements_per_axis=(n,)*dim,
            group_cls=group_cls)

        dcoll = make_discretization_collection(
            actx, mesh, discr_tag_to_group_factory={
                DISCR_TAG_BASE:
                InterpolatoryEdgeClusteredGroupFactory(order),
                DISCR_TAG_QUAD: QuadratureGroupFactory(3 * order)
            })

        if overint:
            quad_discr_tag = DISCR_TAG_QUAD
        else:
            quad_discr_tag = DISCR_TAG_BASE

        allfaces_dd_quad = as_dofdesc(FACE_RESTR_ALL, quad_discr_tag)
        vol_dd_base = as_dofdesc(DTAG_VOLUME_ALL)
        vol_dd_quad = vol_dd_base.with_discr_tag(quad_discr_tag)

        x_base = actx.thaw(dcoll.nodes())

        def f(x, dcoll=dcoll):
            zeros = 0 * x[0]
            result = make_obj_array([zeros + (i+1) for i in range(dim)])
            for i in range(dim-1):
                result = result * actx.np.sin(np.pi*x[i])
            result = result * actx.np.cos(np.pi/2*x[dim-1])
            return result

        def div_f(x, dcoll=dcoll):
            zeros = 0*x[0]
            result = 1*zeros
            for i in range(dim-1):
                deriv = 1*zeros + (i+1)
                for j in range(i):
                    deriv = deriv * actx.np.sin(np.pi*x[j])
                deriv = deriv * np.pi*actx.np.cos(np.pi*x[i])
                for j in range(i+1, dim-1):
                    deriv = deriv * actx.np.sin(np.pi*x[j])
                deriv = deriv * actx.np.cos(np.pi/2*x[dim-1])
                result = result + deriv
            deriv = 1*zeros + dim
            for j in range(dim-1):
                deriv = deriv * actx.np.sin(np.pi*x[j])
            deriv = deriv * (-np.pi/2*actx.np.sin(np.pi/2*x[dim-1]))
            result = result + deriv
            return result

        if vectorize:
            u_base = make_obj_array([(i+1)*f(x_base) for i in range(dim)])
            if not nested:
                u_base = np.stack(u_base, axis=0)
        else:
            u_base = f(x_base)

        def get_flux(u_tpair, dcoll=dcoll):
            dd = u_tpair.dd
            dd_allfaces = dd.with_domain_tag(
                BoundaryDomainTag(FACE_RESTR_ALL, VTAG_ALL)
                )
            normal = geometry.normal(actx, dcoll, dd)
            flux = u_tpair.avg @ normal
            return op.project(dcoll, dd, dd_allfaces, flux)

        if form == "strong":
            div_u = (
                op.local_div(dcoll, u_base)
                # No flux terms because u doesn't have inter-el jumps
                )
        elif form == "weak":
            div_u = op.inverse_mass(
                dcoll, vol_dd_quad,
                -op.weak_local_div(
                    dcoll, vol_dd_quad,
                    op.project(dcoll, vol_dd_base, vol_dd_quad, u_base))
                +
                op.face_mass(
                    dcoll, allfaces_dd_quad,
                    # Note: no boundary flux terms here because
                    # u_ext == u_int == 0
                    sum(get_flux(op.project_tracepair(
                        dcoll, allfaces_dd_quad, utpair))
                        for utpair in op.interior_trace_pairs(dcoll, u_base))
                )
            )
        else:
            raise ValueError("Invalid form argument.")

        if vectorize:
            expected_div_u = make_obj_array([(i+1)*div_f(x_base)
                                             for i in range(dim)])
        else:
            expected_div_u = div_f(x_base)

        if visualize:
            from grudge.shortcuts import make_visualizer
            vis = make_visualizer(dcoll, vis_order=order
                                  if dim == 3 else dim+3)

            filename = (f"test_divergence_{form}_{dim}_{order}"
                f"{'_vec' if vectorize else ''}{'_nested' if nested else ''}.vtu")
            vis.write_vtk_file(filename, [
                ("u", u_base),
                ("div_u", div_u),
                ("expected_div_u", expected_div_u),
                ], overwrite=True)

        rel_linf_err = actx.to_numpy(
            op.norm(dcoll, div_u - expected_div_u, np.inf)
            / op.norm(dcoll, expected_div_u, np.inf))
        eoc_rec.add_data_point(1./n, rel_linf_err)

    print("L^inf error:")
    print(eoc_rec)
    assert (eoc_rec.order_estimate() >= order - 0.5
                or eoc_rec.max_error() < 1e-11)

# }}}


# You can test individual routines by typing
# $ python test_grudge.py 'test_routine()'

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        pytest.main([__file__])

# vim: fdm=marker
