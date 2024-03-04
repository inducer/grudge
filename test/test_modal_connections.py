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


from grudge.array_context import PytestPyOpenCLArrayContextFactory
from arraycontext import pytest_generate_tests_for_array_contexts

from grudge.discretization import make_discretization_collection
pytest_generate_tests = pytest_generate_tests_for_array_contexts(
        [PytestPyOpenCLArrayContextFactory])

from meshmode.discretization.poly_element import (
    # Simplex group factories
    InterpolatoryQuadratureSimplexGroupFactory,
    ModalGroupFactory,
    PolynomialWarpAndBlend2DRestrictingGroupFactory,
    PolynomialEquidistantSimplexGroupFactory,
    # Tensor product group factories
    LegendreGaussLobattoTensorProductGroupFactory,
    # Quadrature-based (non-interpolatory) group factories
    QuadratureSimplexGroupFactory
)
from meshmode.dof_array import flat_norm
import meshmode.mesh.generation as mgen

import grudge.dof_desc as dof_desc

import pytest


@pytest.mark.parametrize("nodal_group_factory", [
    InterpolatoryQuadratureSimplexGroupFactory,
    PolynomialWarpAndBlend2DRestrictingGroupFactory,
    PolynomialEquidistantSimplexGroupFactory,
    LegendreGaussLobattoTensorProductGroupFactory,
    ]
)
def test_inverse_modal_connections(actx_factory, nodal_group_factory):
    actx = actx_factory()
    order = 4

    def f(x):
        return 2*actx.np.sin(20*x) + 0.5*actx.np.cos(10*x)

    # Make a regular rectangle mesh
    mesh = mgen.generate_regular_rect_mesh(
        a=(0, 0), b=(5, 3), npoints_per_axis=(10, 6), order=order,
        group_cls=nodal_group_factory.mesh_group_class
    )

    dcoll = make_discretization_collection(
        actx, mesh,
        discr_tag_to_group_factory={
            dof_desc.DISCR_TAG_BASE: nodal_group_factory(order),
            dof_desc.DISCR_TAG_MODAL: ModalGroupFactory(order),
        }
    )

    dd_modal = dof_desc.DD_VOLUME_ALL_MODAL
    dd_volume = dof_desc.DD_VOLUME_ALL

    x_nodal = actx.thaw(dcoll.discr_from_dd(dd_volume).nodes()[0])
    nodal_f = f(x_nodal)

    # Map nodal coefficients of f to modal coefficients
    forward_conn = dcoll.connection_from_dds(dd_volume, dd_modal)
    modal_f = forward_conn(nodal_f)
    # Now map the modal coefficients back to nodal
    backward_conn = dcoll.connection_from_dds(dd_modal, dd_volume)
    nodal_f_2 = backward_conn(modal_f)

    # This error should be small since we composed a map with
    # its inverse
    err = flat_norm(nodal_f - nodal_f_2)

    assert err <= 1e-13


def test_inverse_modal_connections_quadgrid(actx_factory):
    actx = actx_factory()
    order = 5

    def f(x):
        return 1 + 2*x + 3*x**2

    # Make a regular rectangle mesh
    mesh = mgen.generate_regular_rect_mesh(
        a=(0, 0), b=(5, 3), npoints_per_axis=(10, 6), order=order,
        group_cls=QuadratureSimplexGroupFactory.mesh_group_class
    )

    dcoll = make_discretization_collection(
        actx, mesh,
        discr_tag_to_group_factory={
            dof_desc.DISCR_TAG_BASE:
            PolynomialWarpAndBlend2DRestrictingGroupFactory(order),
            dof_desc.DISCR_TAG_QUAD: QuadratureSimplexGroupFactory(2*order),
            dof_desc.DISCR_TAG_MODAL: ModalGroupFactory(order),
        }
    )

    # Use dof descriptors on the quadrature grid
    dd_modal = dof_desc.DD_VOLUME_ALL_MODAL
    dd_quad = dof_desc.DOFDesc(dof_desc.DTAG_VOLUME_ALL,
                               dof_desc.DISCR_TAG_QUAD)

    x_quad = actx.thaw(dcoll.discr_from_dd(dd_quad).nodes()[0])
    quad_f = f(x_quad)

    # Map nodal coefficients of f to modal coefficients
    forward_conn = dcoll.connection_from_dds(dd_quad, dd_modal)
    modal_f = forward_conn(quad_f)
    # Now map the modal coefficients back to nodal
    backward_conn = dcoll.connection_from_dds(dd_modal, dd_quad)
    quad_f_2 = backward_conn(modal_f)

    # This error should be small since we composed a map with
    # its inverse
    err = flat_norm(quad_f - quad_f_2)

    assert err <= 1e-11
