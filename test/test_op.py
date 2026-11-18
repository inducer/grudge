from __future__ import annotations


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
from typing import TYPE_CHECKING

import numpy as np
import pytest

import meshmode.mesh.generation as mgen
from arraycontext import ArrayContextFactory, pytest_generate_tests_for_array_contexts
from meshmode.discretization.poly_element import (
    InterpolatoryEdgeClusteredGroupFactory,
    QuadratureGroupFactory,
)
from meshmode.mesh import BTAG_ALL
from pytools import obj_array

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


if TYPE_CHECKING:
    from meshmode.dof_array import DOFArray

    from grudge.discretization import DiscretizationCollection


logger = logging.getLogger(__name__)
pytest_generate_tests = pytest_generate_tests_for_array_contexts(
        [PytestPyOpenCLArrayContextFactory])


# {{{ gradient

@pytest.mark.parametrize("form", ["strong", "weak", "weak-overint"])
@pytest.mark.parametrize("dim", [1, 2, 3])
@pytest.mark.parametrize("order", [2, 3])
@pytest.mark.parametrize("warp_mesh", [False, True])
@pytest.mark.parametrize(("vectorize", "nested"), [
    (False, False),
    (True, False),
    (True, True)
    ])
def test_gradient(
            actx_factory: ArrayContextFactory,
            form,
            dim,
            order,
            vectorize,
            nested,
            warp_mesh,
            visualize=False):
    actx = actx_factory()

    from pytools.convergence import EOCRecorder
    eoc_rec = EOCRecorder()

    for n in [8, 12, 16] if warp_mesh else [4, 6, 8]:
        if warp_mesh:
            if dim == 1:
                pytest.skip("warped mesh in 1D not implemented")
            mesh = mgen.generate_warped_rect_mesh(
                          dim=dim, order=order, nelements_side=n)
        else:
            mesh = mgen.generate_regular_rect_mesh(
                    a=(-1,)*dim, b=(1,)*dim,
                    nelements_per_axis=(n,)*dim)

        dcoll = make_discretization_collection(
                   actx, mesh,
                   discr_tag_to_group_factory={
                       DISCR_TAG_BASE: InterpolatoryEdgeClusteredGroupFactory(order),
                       DISCR_TAG_QUAD: QuadratureGroupFactory(3 * order)
                   })

        def f(x):
            result = 1
            for i in range(dim-1):
                result = result * actx.np.sin(np.pi*x[i])

            return result * actx.np.cos(np.pi/2*x[dim-1])

        def grad_f(x):
            result = obj_array.new_1d([1 for _ in range(dim)])
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

        def vectorize_if_requested(vec):
            if vectorize:
                return obj_array.new_1d([(i+1)*vec for i in range(dim)])
            else:
                return vec

        def get_flux(u_tpair, dcoll=dcoll):
            dd = u_tpair.dd
            dd_allfaces = dd.with_domain_tag(
                BoundaryDomainTag(FACE_RESTR_ALL, VTAG_ALL)
                )
            normal = geometry.normal(actx, dcoll, dd)
            u_avg = u_tpair.avg
            if vectorize:
                if nested:
                    flux = obj_array.new_1d([u_avg_i * normal for u_avg_i in u_avg])
                else:
                    flux = np.outer(u_avg, normal)
            else:
                flux = u_avg * normal
            return op.project(dcoll, dd, dd_allfaces, flux)

        x = actx.thaw(dcoll.nodes())
        u = vectorize_if_requested(f(x))

        bdry_dd_base = as_dofdesc(BTAG_ALL)
        bdry_x = actx.thaw(dcoll.nodes(bdry_dd_base))
        bdry_u = vectorize_if_requested(f(bdry_x))

        if form == "strong":
            grad_u = (
                op.local_grad(dcoll, u, nested=nested)
                # No flux terms because u doesn't have inter-el jumps
                )
        elif form.startswith("weak"):
            assert form in ["weak", "weak-overint"]
            if "overint" in form:
                quad_discr_tag = DISCR_TAG_QUAD
            else:
                quad_discr_tag = DISCR_TAG_BASE

            allfaces_dd_base = as_dofdesc(FACE_RESTR_ALL, quad_discr_tag)
            vol_dd_base = as_dofdesc(DTAG_VOLUME_ALL)
            vol_dd_quad = vol_dd_base.with_discr_tag(quad_discr_tag)
            bdry_dd_quad = bdry_dd_base.with_discr_tag(quad_discr_tag)
            allfaces_dd_quad = allfaces_dd_base.with_discr_tag(quad_discr_tag)

            grad_u = op.inverse_mass(dcoll,
                -op.weak_local_grad(dcoll, vol_dd_quad,
                        op.project(dcoll, vol_dd_base, vol_dd_quad, u),
                        nested=nested)
                +
                op.face_mass(dcoll,
                    allfaces_dd_quad,
                    sum(get_flux(
                        op.project_tracepair(dcoll, allfaces_dd_quad, utpair))
                        for utpair in op.interior_trace_pairs(
                                      dcoll, u, volume_dd=vol_dd_base))
                    + get_flux(
                        op.project_tracepair(dcoll, bdry_dd_quad,
                                   bv_trace_pair(dcoll, bdry_dd_base, u, bdry_u)))
                )
            )
        else:
            raise ValueError("Invalid form argument.")

        if vectorize:
            expected_grad_u = obj_array.new_1d(
                [(i+1)*grad_f(x) for i in range(dim)])
            if not nested:
                expected_grad_u = obj_array.stack(expected_grad_u, axis=0)
        else:
            expected_grad_u = grad_f(x)

        if visualize:
            # the code below does not handle the vectorized case
            assert not vectorize

            from grudge.shortcuts import make_visualizer
            vis = make_visualizer(dcoll, vis_order=order if dim == 3 else dim+3)

            filename = (f"test_gradient_{form}_{dim}_{order}"
                f"{'_vec' if vectorize else ''}{'_nested' if nested else ''}.vtu")
            vis.write_vtk_file(filename, [
                ("u", u),
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
def test_divergence(
            actx_factory: ArrayContextFactory,
            form: str,
            dim: int,
            order: int,
            vectorize: bool,
            nested: bool,
            visualize: bool = False) -> None:
    actx = actx_factory()

    from pytools.convergence import EOCRecorder
    eoc_rec = EOCRecorder()

    for n in [4, 6, 8]:
        mesh = mgen.generate_regular_rect_mesh(
                a=(-1,)*dim, b=(1,)*dim,
                nelements_per_axis=(n,)*dim)

        dcoll = make_discretization_collection(actx, mesh, order=order)

        def f(
                x: obj_array.ObjectArray1D[DOFArray],
                dcoll: DiscretizationCollection = dcoll,
        ) -> obj_array.ObjectArray1D[DOFArray]:
            result = obj_array.new_1d([dcoll.zeros(actx) + (i+1) for i in range(dim)])
            for i in range(dim-1):
                result = result * actx.np.sin(np.pi*x[i])

            return result * actx.np.cos(np.pi/2*x[dim-1])

        def div_f(
                x: obj_array.ObjectArray1D[DOFArray],
                dcoll: DiscretizationCollection = dcoll,
        ) -> DOFArray:
            result = dcoll.zeros(actx)
            for i in range(dim-1):
                deriv = dcoll.zeros(actx) + (i+1)
                for j in range(i):
                    deriv = deriv * actx.np.sin(np.pi*x[j])
                deriv = deriv * np.pi*actx.np.cos(np.pi*x[i])
                for j in range(i+1, dim-1):
                    deriv = deriv * actx.np.sin(np.pi*x[j])
                deriv = deriv * actx.np.cos(np.pi/2*x[dim-1])
                result = result + deriv

            deriv = dcoll.zeros(actx) + dim
            for j in range(dim-1):
                deriv = deriv * actx.np.sin(np.pi*x[j])
            deriv = deriv * (-np.pi/2*actx.np.sin(np.pi/2*x[dim-1]))

            return result + deriv

        x = actx.thaw(dcoll.nodes())

        if vectorize:
            u = obj_array.new_1d([(i+1)*f(x) for i in range(dim)])
            if not nested:
                u = obj_array.stack(u, axis=0)
        else:
            u = f(x)

        def get_flux(u_tpair, dcoll=dcoll):
            dd = u_tpair.dd
            dd_allfaces = dd.with_domain_tag(
                BoundaryDomainTag(FACE_RESTR_ALL, VTAG_ALL)
                )
            normal = geometry.normal(actx, dcoll, dd)
            flux = u_tpair.avg @ normal
            return op.project(dcoll, dd, dd_allfaces, flux)

        dd_allfaces = as_dofdesc(FACE_RESTR_ALL)

        if form == "strong":
            div_u = (
                op.local_div(dcoll, u)
                # No flux terms because u doesn't have inter-el jumps
                )
        elif form == "weak":
            div_u = op.inverse_mass(dcoll,
                -op.weak_local_div(dcoll, u)
                +
                op.face_mass(dcoll,
                    dd_allfaces,
                    # Note: no boundary flux terms here because u_ext == u_int == 0
                    sum(get_flux(utpair)
                        for utpair in op.interior_trace_pairs(dcoll, u))
                )
            )
        else:
            raise ValueError("Invalid form argument.")

        if vectorize:
            expected_div_u = obj_array.new_1d([(i+1)*div_f(x) for i in range(dim)])
        else:
            expected_div_u = div_f(x)

        if visualize:
            from grudge.shortcuts import make_visualizer
            vis = make_visualizer(dcoll, vis_order=order if dim == 3 else dim+3)

            filename = (f"test_divergence_{form}_{dim}_{order}"
                f"{'_vec' if vectorize else ''}{'_nested' if nested else ''}.vtu")
            vis.write_vtk_file(filename, [
                ("u", u),
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


@pytest.mark.parametrize("operator_name", [
    "local_d_dx", "weak_local_d_dx", "face_mass",
])
def test_operator_rejects_scalar_leaves(
        actx_factory: ArrayContextFactory, operator_name: str) -> None:
    actx = actx_factory()
    mesh = mgen.generate_regular_rect_mesh(
        a=(-1,), b=(1,), nelements_per_axis=(1,))
    dcoll = make_discretization_collection(actx, mesh, order=2)
    operator = getattr(op, operator_name)
    args = () if operator_name == "face_mass" else (0,)

    for scalar in (1, 1.0, 1j, np.float64(1)):
        for value in (scalar, obj_array.new_1d([
                obj_array.new_1d([scalar])])):
            with pytest.raises(TypeError, match="scalars not allowed"):
                operator(dcoll, *args, value)


@pytest.mark.parametrize("dim", [1, 2, 3])
@pytest.mark.parametrize("tensor_product", [False, True])
@pytest.mark.parametrize("first_enabled", [False, True])
def test_sum_factorization_escape_hatch(
        actx_factory: ArrayContextFactory, dim, tensor_product, first_enabled):
    from meshmode.mesh import SimplexElementGroup, TensorProductElementGroup

    from grudge.bilinear_forms import (
        make_inverse_mass_operator,
        make_mass_operator,
        make_strong_differentiation_operator,
    )
    from grudge.dof_desc import DD_VOLUME_ALL

    actx = actx_factory()
    mesh = mgen.generate_regular_rect_mesh(
        a=(-1,)*dim, b=(1,)*dim, nelements_per_axis=(2,)*dim,
        group_cls=(TensorProductElementGroup if tensor_product
                   else SimplexElementGroup))
    dcoll = make_discretization_collection(
        actx, mesh, order=3,
        discr_tag_to_group_factory={DISCR_TAG_QUAD: QuadratureGroupFactory(5)})
    discr = dcoll.discr_from_dd(DD_VOLUME_ALL)
    x = actx.thaw(discr.nodes())
    u = 1 + sum((axis+1)*x[axis]**2 for axis in range(dim))

    # Exercise both cache insertion orders within the same array context.
    for enabled in (first_enabled, not first_enabled):
        for group in discr.groups:
            mass = make_mass_operator(
                actx, group, group, enable_sum_factorization=enabled)
            inverse = make_inverse_mass_operator(
                actx, group, enable_sum_factorization=enabled)
            derivatives = make_strong_differentiation_operator(
                actx, group, group, enable_sum_factorization=enabled)
            assert isinstance(mass, tuple) == (tensor_product and enabled)
            assert isinstance(inverse, tuple) == (tensor_product and enabled)
            assert len(derivatives) == dim
            assert all(isinstance(d, tuple) == (tensor_product and enabled)
                       for d in derivatives)
            if isinstance(mass, tuple):
                assert len(mass) == dim
                assert all(factor is mass[0] for factor in mass)
            else:
                assert mass.shape == (group.nunit_dofs, group.nunit_dofs)
            assert mass is make_mass_operator(
                actx, group, group, enable_sum_factorization=enabled)
            assert inverse is make_inverse_mass_operator(
                actx, group, enable_sum_factorization=enabled)
            assert derivatives is make_strong_differentiation_operator(
                actx, group, group, enable_sum_factorization=enabled)
            if enabled:
                assert mass is make_mass_operator(actx, group, group)

    def check_close(actual, expected):
        assert actx.to_numpy(op.norm(dcoll, actual - expected, np.inf)) < 1e-11

    # Containers must propagate the flag to every component.
    fields = obj_array.new_1d([u, 2*u])
    results = {}
    for enabled in (first_enabled, not first_enabled):
        mass = op.mass(dcoll, fields, enable_sum_factorization=enabled)
        inverse = op.inverse_mass(dcoll, mass, enable_sum_factorization=enabled)
        grad = op.local_grad(
            dcoll, DD_VOLUME_ALL, u, enable_sum_factorization=enabled)
        nested = op.local_grad(
            dcoll, fields, nested=True, enable_sum_factorization=enabled)
        for component in range(2):
            check_close(inverse[component], fields[component])
            for axis in range(dim):
                check_close(nested[component][axis], (component+1)*grad[axis])
        for axis in range(dim):
            check_close(grad[axis], 2*(axis+1)*x[axis])
        results[enabled] = mass
    for component in range(2):
        check_close(results[False][component], results[True][component])

    # Rectangular quadrature-to-base operators also support the dense fallback.
    dd_quad = DD_VOLUME_ALL.with_discr_tag(DISCR_TAG_QUAD)
    u_quad = op.project(dcoll, DD_VOLUME_ALL, dd_quad, u)
    check_close(
        op.mass(dcoll, dd_quad, u_quad, enable_sum_factorization=False),
        op.mass(dcoll, dd_quad, u_quad, enable_sum_factorization=True))

    if tensor_product:
        quad_discr = dcoll.discr_from_dd(dd_quad)
        for group, quad_group in zip(discr.groups, quad_discr.groups, strict=True):
            # Dense construction bypasses the unfinished non-matching TP path.
            derivatives = make_strong_differentiation_operator(
                actx, group, quad_group, enable_sum_factorization=False)
            assert all(d.shape == (quad_group.nunit_dofs, group.nunit_dofs)
                       for d in derivatives)
            with pytest.raises(NotImplementedError, match="between different"):
                make_strong_differentiation_operator(
                    actx, group, quad_group, enable_sum_factorization=True)


@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("face_first", [False, True])
@pytest.mark.parametrize("nodes", ["lobatto", "gauss"])
@pytest.mark.parametrize("quadrature_order", [1, 3, 5])
def test_face_mass_factor_reuse(
        actx_factory: ArrayContextFactory, dim: int, face_first: bool,
        nodes: str, quadrature_order: int) -> None:
    import modepy as mp
    from meshmode.discretization.poly_element import InterpolatoryQuadratureGroupFactory
    from meshmode.dof_array import DOFArray
    from meshmode.mesh import TensorProductElementGroup

    from grudge.bilinear_forms import (
        make_face_mass_operator,
        make_mass_operator,
        make_stiffness_t_operator,
    )
    from grudge.dof_desc import DD_VOLUME_ALL

    actx = actx_factory()
    mesh = mgen.generate_regular_rect_mesh(
        a=(-1,)*dim, b=(1,)*dim, nelements_per_axis=(2,)*dim,
        group_cls=TensorProductElementGroup)
    base_factory = (InterpolatoryEdgeClusteredGroupFactory(3) if nodes == "lobatto"
                    else InterpolatoryQuadratureGroupFactory(3))
    dcoll = make_discretization_collection(actx, mesh,
        discr_tag_to_group_factory={
            DISCR_TAG_BASE: base_factory,
            DISCR_TAG_QUAD: QuadratureGroupFactory(quadrature_order),
        })
    vol_group, = dcoll.discr_from_dd(DD_VOLUME_ALL).groups
    rng = np.random.default_rng(31)

    for discr_tag in (DISCR_TAG_BASE, DISCR_TAG_QUAD):
        dd = DD_VOLUME_ALL.trace(FACE_RESTR_ALL).with_discr_tag(discr_tag)
        face_group, = dcoll.discr_from_dd(dd).groups
        if not face_first:
            make_mass_operator(actx, vol_group, vol_group)
        faces = make_face_mass_operator(actx, face_group, vol_group)
        mass = make_mass_operator(actx, vol_group, vol_group)
        stiffness = make_stiffness_t_operator(actx, vol_group, vol_group)
        assert isinstance(mass, tuple)
        assert isinstance(stiffness, tuple)
        assert stiffness[0][1] is mass[0]
        assert faces is make_face_mass_operator(actx, face_group, vol_group)

        matching = (face_group.order == vol_group.order
                    and np.array_equal(face_group.unit_nodes_1d,
                                       vol_group.unit_nodes_1d))
        for face, factors in zip(mp.faces_for_shape(vol_group.shape), faces,
                                 strict=True):
            assert isinstance(factors, tuple)
            mapped = face.map_to_volume(np.column_stack((
                np.zeros(dim-1), np.eye(dim-1))))
            directions = mapped[:, 1:] - mapped[:, :1]
            for axis, factor in enumerate(factors):
                tangential_axes = np.flatnonzero(directions[axis])
                if not len(tangential_axes):
                    assert factor.shape == (vol_group.order+1,)
                    assert factor is not mass[0]
                elif matching and directions[axis, tangential_axes[0]] == 1:
                    assert factor is mass[0]
                else:
                    assert factor is not mass[0]

        values = rng.standard_normal((face_group.nelements, face_group.nunit_dofs))
        values = values + 1j*rng.standard_normal(values.shape)
        vec = DOFArray(actx, (actx.from_numpy(values),))
        dense = op.face_mass(dcoll, dd, vec, enable_sum_factorization=False)
        factorized = op.face_mass(dcoll, dd, vec)
        np.testing.assert_allclose(actx.to_numpy(factorized[0]),
                                   actx.to_numpy(dense[0]), rtol=1e-11, atol=1e-12)


# You can test individual routines by typing
# $ python test_grudge.py 'test_routine()'

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        pytest.main([__file__])

# vim: fdm=marker
