from __future__ import annotations


__copyright__ = """
Copyright (C) 2015 Andreas Kloeckner
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

import logging

import mesh_data
import numpy as np
import numpy.linalg as la
import pytest

import meshmode.mesh.generation as mgen
import pytools.obj_array as obj_array
from arraycontext import ArrayContextFactory, pytest_generate_tests_for_array_contexts
from meshmode import _acf  # noqa: F401
from meshmode.discretization.poly_element import (
    InterpolatoryEdgeClusteredGroupFactory,
    QuadratureGroupFactory,
)
from meshmode.dof_array import flat_norm
from meshmode.mesh import TensorProductElementGroup

from grudge import dof_desc, geometry, op
from grudge.array_context import PytestPyOpenCLArrayContextFactory
from grudge.discretization import make_discretization_collection


logger = logging.getLogger(__name__)
pytest_generate_tests = pytest_generate_tests_for_array_contexts(
        [PytestPyOpenCLArrayContextFactory])


# {{{ mass operator trig integration

@pytest.mark.parametrize("ambient_dim", [1, 2, 3])
@pytest.mark.parametrize("discr_tag", [dof_desc.DISCR_TAG_BASE,
                                       dof_desc.DISCR_TAG_QUAD])
def test_mass_mat_trig(actx_factory: ArrayContextFactory, ambient_dim, discr_tag):
    """Check the integral of some trig functions on an interval using the mass
    matrix.
    """
    actx = actx_factory()

    nel_1d = 16
    order = 4

    a = -4.0 * np.pi
    b = +9.0 * np.pi
    true_integral = 13*np.pi/2 * (b - a)**(ambient_dim - 1)

    from meshmode.discretization.poly_element import QuadratureSimplexGroupFactory

    dd_quad = dof_desc.DOFDesc(dof_desc.DTAG_VOLUME_ALL, discr_tag)
    if discr_tag is dof_desc.DISCR_TAG_BASE:
        discr_tag_to_group_factory = {}
    else:
        discr_tag_to_group_factory = {
            discr_tag: QuadratureSimplexGroupFactory(order=2*order)
        }

    mesh = mgen.generate_regular_rect_mesh(
            a=(a,)*ambient_dim, b=(b,)*ambient_dim,
            nelements_per_axis=(nel_1d,)*ambient_dim, order=1)
    dcoll = make_discretization_collection(
        actx, mesh, order=order,
        discr_tag_to_group_factory=discr_tag_to_group_factory
    )

    def f(x):
        return actx.np.sin(x[0])**2

    volm_disc = dcoll.discr_from_dd(dof_desc.DD_VOLUME_ALL)
    x_volm = actx.thaw(volm_disc.nodes())
    f_volm = f(x_volm)
    ones_volm = volm_disc.zeros(actx) + 1

    quad_disc = dcoll.discr_from_dd(dd_quad)
    x_quad = actx.thaw(quad_disc.nodes())
    f_quad = f(x_quad)
    ones_quad = quad_disc.zeros(actx) + 1

    mop_1 = op.mass(dcoll, dd_quad, f_quad)
    num_integral_1 = op.nodal_sum(
        dcoll, dof_desc.DD_VOLUME_ALL, ones_volm * mop_1
    )

    err_1 = abs(num_integral_1 - true_integral)
    assert err_1 < 2e-9, err_1

    mop_2 = op.mass(dcoll, dd_quad, ones_quad)
    num_integral_2 = op.nodal_sum(dcoll, dof_desc.DD_VOLUME_ALL, f_volm * mop_2)

    err_2 = abs(num_integral_2 - true_integral)
    assert err_2 < 2e-9, err_2

    if discr_tag is dof_desc.DISCR_TAG_BASE:
        # NOTE: `integral` always makes a square mass matrix and
        # `QuadratureSimplexGroupFactory` does not have a `mass_matrix` method.
        num_integral_3 = op.nodal_sum(dcoll, dof_desc.DD_VOLUME_ALL, f_quad * mop_2)
        err_3 = abs(num_integral_3 - true_integral)
        assert err_3 < 5e-10, err_3

# }}}


# {{{ mass operator on surface

def _ellipse_surface_area(radius, aspect_ratio):
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.ellipe.html
    eccentricity = 1.0 - (1/aspect_ratio)**2

    if abs(aspect_ratio - 2.0) < 1.0e-14:
        # NOTE: hardcoded value so we don't need scipy for the test
        ellip_e = 1.2110560275684594
    else:
        from scipy.special import ellipe  # pylint: disable=no-name-in-module

        ellip_e = ellipe(eccentricity)

    return 4.0 * radius * ellip_e


def _spheroid_surface_area(radius, aspect_ratio):
    # https://en.wikipedia.org/wiki/Ellipsoid#Surface_area
    a = 1.0
    c = aspect_ratio

    if a < c:
        e = np.sqrt(1.0 - (a/c)**2)
        return 2.0 * np.pi * radius**2 * (1.0 + (c/a) / e * np.arcsin(e))
    else:
        e = np.sqrt(1.0 - (c/a)**2)
        return 2.0 * np.pi * radius**2 * (1 + (c/a)**2 / e * np.arctanh(e))


@pytest.mark.parametrize("name", [
    "2-1-ellipse", "spheroid", "box2d", "box3d"
    ])
def test_mass_surface_area(actx_factory: ArrayContextFactory, name):
    actx = actx_factory()

    # {{{ cases

    order = 4

    if name == "2-1-ellipse":
        builder = mesh_data.EllipseMeshBuilder(radius=3.1, aspect_ratio=2.0)
        surface_area = _ellipse_surface_area(builder.radius, builder.aspect_ratio)
    elif name == "spheroid":
        builder = mesh_data.SpheroidMeshBuilder()
        surface_area = _spheroid_surface_area(builder.radius, builder.aspect_ratio)
    elif name == "box2d":
        builder = mesh_data.BoxMeshBuilder2D()
        surface_area = 1.0
    elif name == "box3d":
        builder = mesh_data.BoxMeshBuilder3D()
        surface_area = 1.0
    else:
        raise ValueError(f"unknown geometry name: {name}")

    # }}}

    # {{{ convergence

    from pytools.convergence import EOCRecorder

    eoc = EOCRecorder()

    for resolution in builder.resolutions:
        mesh = builder.get_mesh(resolution, order)
        dcoll = make_discretization_collection(actx, mesh, order=order)
        volume_discr = dcoll.discr_from_dd(dof_desc.DD_VOLUME_ALL)

        logger.info("ndofs:     %d", volume_discr.ndofs)
        logger.info("nelements: %d", volume_discr.mesh.nelements)

        # {{{ compute surface area

        dd = dof_desc.DD_VOLUME_ALL
        ones_volm = volume_discr.zeros(actx) + 1
        approx_surface_area = actx.to_numpy(op.integral(dcoll, dd, ones_volm))

        logger.info(
            f"surface: got {approx_surface_area:.5e} / expected {surface_area:.5e}")  # noqa: G004
        area_error = abs(approx_surface_area - surface_area) / abs(surface_area)

        # }}}

        # compute max element size
        from grudge.dt_utils import h_max_from_volume

        h_max = h_max_from_volume(dcoll)

        eoc.add_data_point(actx.to_numpy(h_max), area_error)

    # }}}

    logger.info("surface area error\n%s", eoc)

    assert eoc.max_error() < 3e-13 or eoc.order_estimate() > order

# }}}


# {{{ mass inverse

@pytest.mark.parametrize("name", [
    "2-1-ellipse",
    "spheroid",
    "warped_rect2",
    "warped_rect3",
    "gh-339-1",
    "gh-339-4",
    ])
def test_mass_operator_inverse(actx_factory: ArrayContextFactory, name):
    actx = actx_factory()

    # {{{ cases

    order = 4
    overintegrate = False

    if name == "2-1-ellipse":
        # curve
        builder = mesh_data.EllipseMeshBuilder(radius=3.1, aspect_ratio=2.0)
    elif name == "spheroid":
        # surface
        builder = mesh_data.SpheroidMeshBuilder()
    elif name.startswith("warped_rect"):
        builder = mesh_data.WarpedRectMeshBuilder(dim=int(name[-1]))
    elif name == "gh-339-1":
        builder = mesh_data.GmshMeshBuilder3D("gh-339.msh")
        order = 1
        # NOTE: We're definitely not evaluating the bilinear forms accurately
        # in that case, the mappings are very non-constant.
        # It's kind of surprising that WADG manages to make a 15-digit inverse,
        # but empirically it seems to.
    elif name == "gh-339-1-overint":
        builder = mesh_data.GmshMeshBuilder3D("gh-339.msh")
        order = 1
        overintegrate = True
    elif name == "gh-339-4":
        builder = mesh_data.GmshMeshBuilder3D("gh-339.msh")
    else:
        raise ValueError(f"unknown geometry name: {name}")

    # }}}

    # {{{ inv(m) @ m == id

    from pytools.convergence import EOCRecorder

    eoc = EOCRecorder()

    for resolution in builder.resolutions:
        mesh = builder.get_mesh(resolution)
        dcoll = make_discretization_collection(
                       actx, mesh, discr_tag_to_group_factory={
                           dof_desc.DISCR_TAG_BASE: (
                               InterpolatoryEdgeClusteredGroupFactory(order)),
                           dof_desc.DISCR_TAG_QUAD: (
                               QuadratureGroupFactory(order))
                       })
        volume_discr = dcoll.discr_from_dd(dof_desc.DD_VOLUME_ALL)

        logger.info("ndofs:     %d", volume_discr.ndofs)
        logger.info("nelements: %d", volume_discr.mesh.nelements)

        # {{{ compute inverse mass

        def f(x):
            return actx.np.cos(4.0 * x[0])

        x_volm = actx.thaw(volume_discr.nodes())
        f_volm = f(x_volm)
        if not overintegrate:
            dd = dof_desc.DD_VOLUME_ALL
            f_inv = op.inverse_mass(
                dcoll, op.mass(dcoll, dd, f_volm)
            )
        else:
            dd_base_vol = dof_desc.as_dofdesc(
                                dof_desc.DTAG_VOLUME_ALL, dof_desc.DISCR_TAG_BASE)
            dd_quad_vol = dof_desc.as_dofdesc(
                                dof_desc.DTAG_VOLUME_ALL, dof_desc.DISCR_TAG_QUAD)
            f_inv = op.inverse_mass(
                dcoll, op.mass(dcoll, dd_quad_vol,
                               op.project(dcoll, dd_base_vol, dd_quad_vol, f_volm)))

        inv_error = actx.to_numpy(
                op.norm(dcoll, f_volm - f_inv, 2) / op.norm(dcoll, f_volm, 2))

        # }}}

        # compute max element size
        from grudge.dt_utils import h_max_from_volume

        h_max = h_max_from_volume(dcoll)

        eoc.add_data_point(actx.to_numpy(h_max), inv_error)

    logger.info("inverse mass error\n%s", eoc)

    # NOTE: both cases give 1.0e-16-ish at the moment, but just to be on the
    # safe side, choose a slightly larger tolerance
    assert eoc.max_error() < 1.0e-14

    # }}}

# }}}


# {{{ surface face normal orthogonality

@pytest.mark.parametrize("mesh_name", ["2-1-ellipse", "spheroid"])
def test_face_normal_surface(actx_factory: ArrayContextFactory, mesh_name):
    """Check that face normals are orthogonal to the surface normal"""
    actx = actx_factory()

    # {{{ geometry

    if mesh_name == "2-1-ellipse":
        builder = mesh_data.EllipseMeshBuilder(radius=3.1, aspect_ratio=2.0)
    elif mesh_name == "spheroid":
        builder = mesh_data.SpheroidMeshBuilder()
    else:
        raise ValueError(f"unknown mesh name: {mesh_name}")

    order = 4
    mesh = builder.get_mesh(builder.resolutions[0], order)
    dcoll = make_discretization_collection(actx, mesh, order=order)

    volume_discr = dcoll.discr_from_dd(dof_desc.DD_VOLUME_ALL)
    logger.info("ndofs:    %d", volume_discr.ndofs)
    logger.info("nelements: %d", volume_discr.mesh.nelements)

    # }}}

    # {{{ Compute surface and face normals
    from meshmode.discretization.connection import FACE_RESTR_INTERIOR

    dv = dof_desc.DD_VOLUME_ALL
    df = dof_desc.as_dofdesc(FACE_RESTR_INTERIOR)

    ambient_dim = mesh.ambient_dim

    surf_normal = op.project(
        dcoll, dv, df,
        geometry.normal(actx, dcoll, dd=dv)
    )
    surf_normal = surf_normal / actx.np.sqrt(sum(surf_normal**2))

    face_normal_i = geometry.normal(actx, dcoll, df)
    face_normal_e = dcoll.opposite_face_connection(
            dof_desc.BoundaryDomainTag(
                dof_desc.FACE_RESTR_INTERIOR, dof_desc.VTAG_ALL)
            )(face_normal_i)

    if ambient_dim == 3:
        # NOTE: there's only one face tangent in 3d
        face_tangent = (
            geometry.pseudoscalar(actx, dcoll, dd=df)
            / geometry.area_element(actx, dcoll, dd=df)
        ).as_vector(dtype=object)

    # }}}

    # {{{ checks

    def _eval_error(x):
        return op.norm(dcoll, x, np.inf, dd=df)

    rtol = 1.0e-14

    # check interpolated surface normal is orthogonal to face normal
    error = _eval_error(surf_normal.dot(face_normal_i))
    logger.info("error[n_dot_i]:    %.5e", error)
    assert error < rtol

    # check angle between two neighboring elements
    error = _eval_error(face_normal_i.dot(face_normal_e) + 1.0)
    logger.info("error[i_dot_e]:    %.5e", error)
    assert error > rtol

    # check orthogonality with face tangent
    if ambient_dim == 3:
        error = _eval_error(face_tangent.dot(face_normal_i))
        logger.info("error[t_dot_i]:  %.5e", error)
        assert error < 5 * rtol

    # }}}

# }}}


# {{{ diff operator

@pytest.mark.parametrize("dim", [1, 2, 3])
def test_tri_diff_mat(actx_factory: ArrayContextFactory, dim, order=4):
    """Check differentiation matrix along the coordinate axes on a disk

    Uses sines as the function to differentiate.
    """

    actx = actx_factory()

    from pytools.convergence import EOCRecorder
    axis_eoc_recs = [EOCRecorder() for axis in range(dim)]

    def f(x, axis):
        return actx.np.sin(3*x[axis])

    def df(x, axis):
        return 3*actx.np.cos(3*x[axis])

    for n in [4, 8, 16]:
        mesh = mgen.generate_regular_rect_mesh(a=(-0.5,)*dim, b=(0.5,)*dim,
                nelements_per_axis=(n,)*dim, order=4)

        dcoll = make_discretization_collection(actx, mesh, order=4)
        volume_discr = dcoll.discr_from_dd(dof_desc.DD_VOLUME_ALL)
        x = actx.thaw(volume_discr.nodes())

        for axis in range(dim):
            df_num = op.local_grad(dcoll, f(x, axis))[axis]
            df_volm = df(x, axis)

            linf_error = flat_norm(df_num - df_volm, ord=np.inf)
            axis_eoc_recs[axis].add_data_point(1/n, actx.to_numpy(linf_error))

    for axis, eoc_rec in enumerate(axis_eoc_recs):
        logger.info("axis %d\n%s", axis, eoc_rec)
        assert eoc_rec.order_estimate() > order - 0.25

# }}}


# {{{ divergence theorem

@pytest.mark.parametrize(
             "case", ["circle", "tp_box2", "tp_box3", "gh-403", "gh-339"])
def test_gauss_theorem(actx_factory: ArrayContextFactory, case, visualize=False):
    """Verify Gauss's theorem explicitly on a mesh"""

    pytest.importorskip("meshpy")

    order = 2
    use_overint = False

    if case == "circle":
        from meshpy.geometry import GeometryBuilder, make_circle
        from meshpy.triangle import MeshInfo, build

        geob = GeometryBuilder()
        geob.add_geometry(*make_circle(1))
        mesh_info = MeshInfo()
        geob.set(mesh_info)

        mesh_info = build(mesh_info)

        from meshmode.mesh.io import from_meshpy
        mesh = from_meshpy(mesh_info, order=1)

    elif case == "gh-403":
        # https://github.com/inducer/meshmode/issues/403
        from meshmode.mesh.io import read_gmsh
        mesh = read_gmsh("gh-403.msh")

    elif case == "gh-339":
        # https://github.com/inducer/grudge/issues/339
        from meshmode.mesh.io import read_gmsh
        mesh = read_gmsh("gh-339.msh")
        order = 1
        use_overint = True

    elif case.startswith("tp_box"):
        dim = int(case[-1])
        mesh = mgen.generate_regular_rect_mesh(
                a=(-0.5,)*dim,
                b=(0.5,)*dim,
                nelements_per_axis=(4,)*dim,
                group_cls=TensorProductElementGroup)

    else:
        raise ValueError(f"unknown case: {case}")

    from meshmode.mesh import BTAG_ALL

    actx = actx_factory()

    dcoll = make_discretization_collection(
               actx, mesh, discr_tag_to_group_factory={
                   dof_desc.DISCR_TAG_BASE: (
                       InterpolatoryEdgeClusteredGroupFactory(order)),
                   dof_desc.DISCR_TAG_QUAD: (
                       QuadratureGroupFactory(order))
               })
    volm_disc = dcoll.discr_from_dd(dof_desc.DD_VOLUME_ALL)
    x_volm = actx.thaw(volm_disc.nodes())

    def f(x):
        if len(x) == 3:
            x0, x1, x2 = x
        elif len(x) == 2:
            x0, x1 = x
            x2 = 0
        else:
            raise ValueError("unsupported dimensionality")

        return obj_array.flat(
            actx.np.sin(3*x0) + actx.np.cos(3*x1) + 2*actx.np.cos(2*x2),
            actx.np.sin(2*x0) + actx.np.cos(x1) + 4*actx.np.cos(0.5*x2),
            actx.np.sin(1*x0) + actx.np.cos(2*x1) + 3*actx.np.cos(0.8*x2),
        )[:dcoll.ambient_dim]

    f_volm = f(x_volm)

    if not use_overint:
        div_f = op.local_div(dcoll, f_volm)
        int_1 = op.integral(dcoll, "vol", div_f)

        prj_f = op.project(dcoll, "vol", BTAG_ALL, f_volm)
        normal = geometry.normal(actx, dcoll, BTAG_ALL)
        f_dot_n = prj_f.dot(normal)
        int_2 = op.integral(dcoll, BTAG_ALL, f_dot_n)
    else:
        dd_base_vol = dof_desc.as_dofdesc(
                            dof_desc.DTAG_VOLUME_ALL, dof_desc.DISCR_TAG_BASE)
        dd_quad_vol = dof_desc.as_dofdesc(
                            dof_desc.DTAG_VOLUME_ALL, dof_desc.DISCR_TAG_QUAD)
        dd_quad_bd = dof_desc.as_dofdesc(BTAG_ALL, dof_desc.DISCR_TAG_QUAD)

        div_f = op.local_div(
                 dcoll, dd_quad_vol,
                 op.project(dcoll, dd_base_vol, dd_quad_vol, f_volm))
        int_1 = op.integral(dcoll, dd_quad_vol, div_f)

        prj_f = op.project(dcoll, "vol", dd_quad_bd, f_volm)
        normal = geometry.normal(actx, dcoll, dd_quad_bd)
        f_dot_n = prj_f.dot(normal)
        int_2 = op.integral(dcoll, dd_quad_bd, f_dot_n)

    if visualize:
        from grudge.shortcuts import make_boundary_visualizer, make_visualizer

        vis = make_visualizer(dcoll)
        bvis = make_boundary_visualizer(dcoll)

        vis.write_vtk_file(
            f"gauss-thm-{case}-vol.vtu", [("div_f", div_f),])
        bvis.write_vtk_file(
            f"gauss-thm-{case}-bdry.vtu", [
                ("f_dot_n", f_dot_n),
                ("normal", normal),
            ])

    assert abs(int_1 - int_2) < 1e-13


@pytest.mark.parametrize("mesh_name", ["2-1-ellipse", "2-1-spheroid"])
def test_surface_divergence_theorem(
            actx_factory: ArrayContextFactory,
            mesh_name,
            visualize=False):
    r"""Check the surface divergence theorem.

        .. math::

            \int_Sigma \phi \nabla_i f_i =
            \int_\Sigma \nabla_i \phi f_i +
            \int_\Sigma \kappa \phi f_i n_i +
            \int_{\partial \Sigma} \phi f_i m_i

        where :math:`n_i` is the surface normal and :class:`m_i` is the
        face normal (which should be orthogonal to both the surface normal
        and the face tangent).
    """
    actx = actx_factory()

    # {{{ cases

    if mesh_name == "2-1-ellipse":
        builder = mesh_data.EllipseMeshBuilder(radius=3.1, aspect_ratio=2.0)
    elif mesh_name == "2-1-spheroid":
        builder = mesh_data.SpheroidMeshBuilder(radius=3.1, aspect_ratio=2.0)
    elif mesh_name == "circle":
        builder = mesh_data.EllipseMeshBuilder(radius=1.0, aspect_ratio=1.0)
    elif mesh_name == "starfish":
        builder = mesh_data.StarfishMeshBuilder()
    elif mesh_name == "sphere":
        builder = mesh_data.SphereMeshBuilder(radius=1.0)
    else:
        raise ValueError(f"unknown mesh name: {mesh_name}")

    # }}}

    # {{{ convergence

    def f(x):
        return obj_array.flat(
            actx.np.sin(3*x[1]) + actx.np.cos(3*x[0]) + 1.0,
            actx.np.sin(2*x[0]) + actx.np.cos(x[1]),
            3.0 * actx.np.cos(x[0] / 2) + actx.np.cos(x[1]),
        )[:ambient_dim]

    from pytools.convergence import EOCRecorder
    eoc_global = EOCRecorder()
    eoc_local = EOCRecorder()

    theta = np.pi / 3.33
    ambient_dim = builder.ambient_dim
    if ambient_dim == 2:
        mesh_rotation = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)],
            ])
    else:
        mesh_rotation = np.array([
            [1.0, 0.0, 0.0],
            [0.0, np.cos(theta), -np.sin(theta)],
            [0.0, np.sin(theta), np.cos(theta)],
            ])

    order = 4

    mesh_offset = np.array([0.33, -0.21, 0.0])[:ambient_dim]

    for i, resolution in enumerate(builder.resolutions):
        from meshmode.discretization.connection import FACE_RESTR_ALL
        from meshmode.mesh.processing import affine_map

        mesh = builder.get_mesh(resolution, order)
        mesh = affine_map(mesh, A=mesh_rotation, b=mesh_offset)

        from meshmode.discretization.poly_element import QuadratureSimplexGroupFactory

        qtag = dof_desc.DISCR_TAG_QUAD
        dcoll = make_discretization_collection(
            actx, mesh, order=order,
            discr_tag_to_group_factory={
                qtag: QuadratureSimplexGroupFactory(2 * order)
            }
        )

        volume = dcoll.discr_from_dd(dof_desc.DD_VOLUME_ALL)
        logger.info("ndofs:     %d", volume.ndofs)
        logger.info("nelements: %d", volume.mesh.nelements)

        dd = dof_desc.DD_VOLUME_ALL
        dq = dd.with_discr_tag(qtag)
        df = dof_desc.as_dofdesc(FACE_RESTR_ALL)
        ambient_dim = dcoll.ambient_dim

        # variables
        f_num = f(actx.thaw(dcoll.nodes(dd=dd)))
        f_quad_num = f(actx.thaw(dcoll.nodes(dd=dq)))

        kappa = geometry.summed_curvature(actx, dcoll, dd=dq)
        normal = geometry.normal(actx, dcoll, dd=dq)
        face_normal = geometry.normal(actx, dcoll, df)
        face_f = op.project(dcoll, dd, df, f_num)

        # operators
        stiff = op.mass(dcoll, sum(op.local_d_dx(dcoll, i, f_num_i)
                                   for i, f_num_i in enumerate(f_num)))
        stiff_t = sum(op.weak_local_d_dx(dcoll, i, f_num_i)
                      for i, f_num_i in enumerate(f_num))
        kterm = op.mass(dcoll, dq, kappa * f_quad_num.dot(normal))
        flux = op.face_mass(dcoll, face_f.dot(face_normal))

        # sum everything up
        op_global = op.nodal_sum(dcoll, dd, stiff - (stiff_t + kterm))
        op_local = op.elementwise_sum(dcoll, dd, stiff - (stiff_t + kterm + flux))

        # compute max element size
        from grudge.dt_utils import h_max_from_volume

        h_max = actx.to_numpy(h_max_from_volume(dcoll))
        err_global = actx.to_numpy(abs(op_global))
        err_local = actx.to_numpy(op.norm(dcoll, op_local, np.inf))
        logger.info("errors: h_max %.5e global %.5e local %.5e",
                h_max, err_global, err_local)

        eoc_global.add_data_point(h_max, err_global)
        eoc_local.add_data_point(h_max, err_local)

        if visualize:
            from grudge.shortcuts import make_visualizer

            vis = make_visualizer(dcoll)

            filename = f"surface_divergence_theorem_{mesh_name}_{i:04d}.vtu"
            vis.write_vtk_file(filename, [
                ("r", actx.np.log10(op_local))
                ], overwrite=True)

    # }}}

    exp_order = order - 0.5
    logger.info("eoc_global:\n%s", eoc_global)
    logger.info("eoc_local:\n%s", eoc_local)

    assert eoc_global.max_error() < 1.0e-12 \
            or eoc_global.order_estimate() > exp_order - 0.5

    assert eoc_local.max_error() < 1.0e-12 \
            or eoc_local.order_estimate() > exp_order - 0.5

# }}}


# {{{ models: advection

@pytest.mark.parametrize(("mesh_name", "mesh_pars"), [
    ("segment", [8, 16, 32]),
    ("disk", [0.07, 0.02, 0.01]),
    ("rect2", [4, 8]),
    ("rect3", [4, 6]),
    ("warped2", [4, 8]),
    ])
@pytest.mark.parametrize("op_type", ["strong", "weak"])
@pytest.mark.parametrize("flux_type", ["central"])
@pytest.mark.parametrize("order", [3, 4, 5])
# test: 'test_convergence_advec(cl._csc, "disk", [0.1, 0.05], "strong", "upwind", 3)'
def test_convergence_advec(
            actx_factory: ArrayContextFactory,
            mesh_name,
            mesh_pars,
            op_type,
            flux_type,
            order,
            visualize=False):
    """Test whether 2D advection actually converges"""

    actx = actx_factory()

    from pytools.convergence import EOCRecorder
    eoc_rec = EOCRecorder()

    for mesh_par in mesh_pars:
        if mesh_name == "segment":
            mesh = mgen.generate_box_mesh(
                [np.linspace(-1.0, 1.0, mesh_par)],
                order=order)

            dim = 1
            dt_factor = 1.0
        elif mesh_name == "disk":
            pytest.importorskip("meshpy")

            from meshpy.geometry import GeometryBuilder, make_circle
            from meshpy.triangle import MeshInfo, build

            geob = GeometryBuilder()
            geob.add_geometry(*make_circle(1))
            mesh_info = MeshInfo()
            geob.set(mesh_info)

            mesh_info = build(mesh_info, max_volume=mesh_par)

            from meshmode.mesh.io import from_meshpy
            mesh = from_meshpy(mesh_info, order=1)
            dim = 2
            dt_factor = 4
        elif mesh_name.startswith("rect"):
            dim = int(mesh_name[-1:])
            mesh = mgen.generate_regular_rect_mesh(a=(-0.5,)*dim, b=(0.5,)*dim,
                    nelements_per_axis=(mesh_par,)*dim, order=4)

            if dim == 2:
                dt_factor = 4
            elif dim == 3:
                dt_factor = 2
            else:
                raise ValueError(f"dt_factor not known for {dim}d")
        elif mesh_name.startswith("warped"):
            dim = int(mesh_name[-1:])
            mesh = mgen.generate_warped_rect_mesh(dim, order=order,
                    nelements_side=mesh_par)

            if dim == 2:
                dt_factor = 4
            elif dim == 3:
                dt_factor = 2
            else:
                raise ValueError(f"dt_factor not known for {dim}d")
        else:
            raise ValueError("invalid mesh name: " + mesh_name)

        v = np.array([0.27, 0.31, 0.1])[:dim]
        norm_v = la.norm(v)

        def f(x):
            return actx.np.sin(10*x)

        def u_analytic(x, t=0, v=v, norm_v=norm_v):
            return f(-v.dot(x)/norm_v + t*norm_v)

        from meshmode.mesh import BTAG_ALL

        from grudge.models.advection import (
            StrongAdvectionOperator,
            WeakAdvectionOperator,
        )

        dcoll = make_discretization_collection(actx, mesh, order=order)
        op_class = {"strong": StrongAdvectionOperator,
                    "weak": WeakAdvectionOperator}[op_type]
        adv_operator = op_class(dcoll, v,
                                inflow_u=lambda t, dcoll=dcoll: u_analytic(
                                    actx.thaw(dcoll.nodes(dd=BTAG_ALL)),
                                    t=t
                                ),
                                flux_type=flux_type)

        nodes = actx.thaw(dcoll.nodes())
        u = u_analytic(nodes, t=0)

        def rhs(t, u, adv_operator=adv_operator):
            return adv_operator.operator(t, u)

        compiled_rhs = actx.compile(rhs)

        if dim == 3:
            final_time = 0.1
        else:
            final_time = 0.2

        from grudge.dt_utils import h_max_from_volume

        h_max = h_max_from_volume(dcoll, dim=dcoll.ambient_dim)
        dt = actx.to_numpy(dt_factor * h_max/order**2)
        nsteps = (final_time // dt) + 1
        tol = 1e-14
        dt = final_time/nsteps + tol

        from grudge.shortcuts import compiled_lsrk45_step, make_visualizer

        vis = make_visualizer(dcoll)

        step = 0
        t = 0

        while t < final_time - tol:
            step += 1
            logger.debug("[%04d] t = %.5f", step, t)

            u = compiled_lsrk45_step(actx, u, t, dt, compiled_rhs)

            if visualize:
                vis.write_vtk_file(
                    f"fld-{mesh_par}-{step:04d}vtu" % (mesh_par, step),
                    [("u", u)]
                )

            t += dt

            if t + dt >= final_time - tol:
                dt = final_time-t

        error_l2 = op.norm(
            dcoll,
            u - u_analytic(nodes, t=t),
            2
        )
        logger.info("h_max %.5e error %.5e", actx.to_numpy(h_max), error_l2)
        eoc_rec.add_data_point(actx.to_numpy(h_max), actx.to_numpy(error_l2))

    logger.info("\n%s", eoc_rec.pretty_print(
        abscissa_label="h",
        error_label="L2 Error"))

    if mesh_name.startswith("warped"):
        # NOTE: curvilinear meshes are hard
        assert eoc_rec.order_estimate() > order - 0.5
    else:
        assert eoc_rec.order_estimate() > order


@pytest.mark.parametrize("order", [4])
@pytest.mark.parametrize("order_sbp", [6])
@pytest.mark.parametrize("spacing_factor", [2])
@pytest.mark.parametrize("c", [np.array([0.27, 0.31]), np.array([-0.27, 0.31])])
def test_convergence_sbp_advec(actx_factory, order, order_sbp, spacing_factor, c,
                               visualize=False):

    actx = actx_factory()

    # Domain: x = [-1,1], y = [-1,1]
    # SBP Subdomain: x = [-1,0], y = [0,1] (structured mesh)
    # DG Subdomain: x = [0,1], y = [0,1] (unstructured mesh)

    # DG Half.
    dim = 2

    from pytools.convergence import EOCRecorder
    eoc_rec = EOCRecorder()
    from sbp_operators import (sbp21, sbp42, sbp63)
    from projection import sbp_dg_projection

    nelems = [20, 40, 60]
    for nelem in nelems:

        mesh = mgen.generate_regular_rect_mesh(a=(0, -1), b=(1, 1),
                                               n=(nelem, nelem), order=order,
                                               boundary_tag_to_face={
                                                  "btag_sbp": ["-x"],
                                                  "btag_std": ["+y", "+x",
                                                          "-y"]})

        # Check to make sure this actually worked.
        from meshmode.mesh import is_boundary_tag_empty
        assert not is_boundary_tag_empty(mesh, "btag_sbp")
        assert not is_boundary_tag_empty(mesh, "btag_std")

        # Check this new isolating discretization, as well as the inverse.
        dcoll = DiscretizationCollection(actx, mesh, order=order)
        sbp_bdry_discr = dcoll.discr_from_dd(dof_desc.DTAG_BOUNDARY("btag_sbp"))
        sbp_bdry_discr.cl_context = actx

        dt_factor = 4
        h = 1/nelem

        norm_c = la.norm(c)

        flux_type = "upwind"

        def f(x):
            return actx.np.sin(10*x)

        def u_analytic(x, t=0):
            return f(-c.dot(x)/norm_c+t*norm_c)

        from grudge.models.advection import WeakAdvectionSBPOperator

        std_tag = dof_desc.DTAG_BOUNDARY("btag_std")
        adv_operator = WeakAdvectionSBPOperator(dcoll, c,
                                                inflow_u=lambda t: u_analytic(
                                                    thaw(dcoll.nodes(dd=std_tag),
                                                        actx),
                                                    t=t
                                                    ),
                                                flux_type=flux_type)

        nodes = thaw(dcoll.nodes(), actx)
        u = u_analytic(nodes, t=0)

        # Count the number of DG nodes - will need this later
        ngroups = len(mesh.groups)
        nnodes_grp = np.ones(ngroups)
        for i in range(0, dim):
            for j in range(0, ngroups):
                nnodes_grp[j] = nnodes_grp[j] * \
                        mesh.groups[j].nodes.shape[i*-1 - 1]

        nnodes = int(sum(nnodes_grp))

        final_time = 0.2

        dt = dt_factor * h/order**2
        nsteps = (final_time // dt) + 1
        dt = final_time/nsteps + 1e-15

        # SBP Half.

        # First, need to set up the structured mesh.
        n_sbp_x = nelem
        n_sbp_y = nelem*spacing_factor
        x_sbp = np.linspace(-1, 0, n_sbp_x, endpoint=True)
        y_sbp = np.linspace(-1, 1, n_sbp_y, endpoint=True)
        dx = x_sbp[1] - x_sbp[0]
        dy = y_sbp[1] - y_sbp[0]

        # Set up solution vector:
        # For now, timestep is the same as DG.
        u_sbp = np.zeros(int(n_sbp_x*n_sbp_y))

        # Initial condition
        for j in range(0, n_sbp_y):
            for i in range(0, n_sbp_x):
                u_sbp[i + j*(n_sbp_x)] = np.sin(10*(-c.dot(
                                                    [x_sbp[i],
                                                     y_sbp[j]])/norm_c))

        # obtain P and Q
        if order_sbp == 2:
            [p_x, q_x] = sbp21(n_sbp_x)
            [p_y, q_y] = sbp21(n_sbp_y)
        elif order_sbp == 4:
            [p_x, q_x] = sbp42(n_sbp_x)
            [p_y, q_y] = sbp42(n_sbp_y)
        elif order_sbp == 6:
            [p_x, q_x] = sbp63(n_sbp_x)
            [p_y, q_y] = sbp63(n_sbp_y)

        tau_l = 1
        tau_r = 1

        # for the boundaries
        el_x = np.zeros(n_sbp_x)
        er_x = np.zeros(n_sbp_x)
        el_x[0] = 1
        er_x[n_sbp_x-1] = 1
        e_l_matx = np.zeros((n_sbp_x, n_sbp_x,))
        e_r_matx = np.zeros((n_sbp_x, n_sbp_x,))

        for i in range(0, n_sbp_x):
            for j in range(0, n_sbp_x):
                e_l_matx[i, j] = el_x[i]*el_x[j]
                e_r_matx[i, j] = er_x[i]*er_x[j]

        el_y = np.zeros(n_sbp_y)
        er_y = np.zeros(n_sbp_y)
        el_y[0] = 1
        er_y[n_sbp_y-1] = 1
        e_l_maty = np.zeros((n_sbp_y, n_sbp_y,))
        e_r_maty = np.zeros((n_sbp_y, n_sbp_y,))

        for i in range(0, n_sbp_y):
            for j in range(0, n_sbp_y):
                e_l_maty[i, j] = el_y[i]*el_y[j]
                e_r_maty[i, j] = er_y[i]*er_y[j]

        # construct the spatial operators
        d_x = np.linalg.inv(dx*p_x).dot(q_x - 0.5*e_l_matx + 0.5*e_r_matx)
        d_y = np.linalg.inv(dy*p_y).dot(q_y - 0.5*e_l_maty + 0.5*e_r_maty)

        # for the boundaries
        c_l_x = np.kron(tau_l, (np.linalg.inv(dx*p_x).dot(el_x)))
        c_r_x = np.kron(tau_r, (np.linalg.inv(dx*p_x).dot(er_x)))
        c_l_y = np.kron(tau_l, (np.linalg.inv(dy*p_y).dot(el_y)))
        c_r_y = np.kron(tau_r, (np.linalg.inv(dy*p_y).dot(er_y)))

        # For speed...
        dudx_mat = -np.kron(np.eye(n_sbp_y), d_x)
        dudy_mat = -np.kron(d_y, np.eye(n_sbp_x))

        # Number of nodes in our SBP-DG boundary discretization
        from meshmode.dof_array import flatten, unflatten
        sbp_nodes_y = flatten(thaw(sbp_bdry_discr.nodes(), actx)[1])
        # When projecting, we use nodes sorted in y, but we will have to unsort
        # afterwards to make sure projected solution is injected into DG BC
        # in the correct way.
        nodesort = np.argsort(sbp_nodes_y)
        nodesortlist = nodesort.tolist()
        rangex = np.array(range(sbp_nodes_y.shape[0]))
        unsort_args = [nodesortlist.index(x) for x in rangex]

        west_nodes = np.sort(np.array(sbp_nodes_y))

        # Make element-aligned glue grid.
        dg_side_gg = np.zeros(int(west_nodes.shape[0]/(order+1))+1)
        counter = 0
        for i in range(0, west_nodes.shape[0]):
            west_nodes[i] = west_nodes[i].get()
            if i % (order+1) == 0:
                dg_side_gg[counter] = west_nodes[i]
                counter += 1

        dg_side_gg[-1] = west_nodes[-1]
        n_west_elements = int(west_nodes.shape[0] / (order + 1))
        sbp2dg, dg2sbp = sbp_dg_projection(n_sbp_y-1, n_west_elements,
                                           order_sbp, order, dg_side_gg,
                                           west_nodes)

        # Get mapping for western face
        base_nodes = flatten(thaw(dcoll._volume_discr.nodes(), actx)[0])
        nsbp_nodes = sbp_bdry_discr.ndofs
        nodes_per_element = mesh.groups[0].nodes.shape[2]
        west_indices = np.zeros(nsbp_nodes)
        count = 0
        # Sweep through first block of indices in the box.
        for i in range(0, (nelem-1)*2*nodes_per_element):
            if base_nodes[i] < 1e-10:
                # Make sure we're actually at the edge faces.
                if i % (2*nodes_per_element) < nodes_per_element:
                    west_indices[count] = i
                    count += 1

        def rhs(t, u):
            # Initialize the entire RHS to 0.
            rhs_out = np.zeros(int(n_sbp_x*n_sbp_y) + int(nnodes))

            # Fill the first part with the SBP half of the domain.

            # Pull the SBP vector out of device array for now.
            u_sbp_ts = u[0:int(n_sbp_x*n_sbp_y)]

            dudx = np.zeros((n_sbp_x*n_sbp_y))
            dudy = np.zeros((n_sbp_x*n_sbp_y))

            dudx = dudx_mat.dot(u_sbp_ts)
            dudy = dudy_mat.dot(u_sbp_ts)

            # Boundary condition
            dl_x = np.zeros(n_sbp_x*n_sbp_y)
            dr_x = np.zeros(n_sbp_x*n_sbp_y)
            dl_y = np.zeros(n_sbp_x*n_sbp_y)
            dr_y = np.zeros(n_sbp_x*n_sbp_y)

            # Pull DG solution at western face to project.
            u_dg_ts = u[int(n_sbp_x*n_sbp_y):]

            dg_west = np.zeros(nsbp_nodes)
            for i in range(0, nsbp_nodes):
                dg_west[i] = u_dg_ts[int(west_indices[i])]

            # Project DG to SBP:
            dg_proj = dg2sbp.dot(dg_west)

            # Need to fill this by looping through each segment.
            # X-boundary conditions:
            for j in range(0, n_sbp_y):
                u_bcx = u_sbp_ts[j*n_sbp_x:((j+1)*n_sbp_x)]
                v_l_x = np.transpose(el_x).dot(u_bcx)
                v_r_x = np.transpose(er_x).dot(u_bcx)
                left_bcx = np.sin(10*(-c.dot(
                                      [x_sbp[0], y_sbp[j]])/norm_c
                                      + norm_c*t))
                # Incorporate DG here.
                right_bcx = dg_proj[j]
                dl_xbc = c_l_x*(v_l_x - left_bcx)
                dr_xbc = c_r_x*(v_r_x - right_bcx)
                dl_x[j*n_sbp_x:((j+1)*n_sbp_x)] = dl_xbc
                dr_x[j*n_sbp_x:((j+1)*n_sbp_x)] = dr_xbc
            # Y-boundary conditions:
            for i in range(0, n_sbp_x):
                u_bcy = u_sbp_ts[i::n_sbp_x]
                v_l_y = np.transpose(el_y).dot(u_bcy)
                v_r_y = np.transpose(er_y).dot(u_bcy)
                left_bcy = np.sin(10*(-c.dot(
                                      [x_sbp[i], y_sbp[0]])/norm_c + norm_c*t))
                right_bcy = np.sin(10*(-c.dot(
                                       [x_sbp[i],
                                        y_sbp[n_sbp_y-1]])/norm_c + norm_c*t))
                dl_ybc = c_l_y*(v_l_y - left_bcy)
                dr_ybc = c_r_y*(v_r_y - right_bcy)
                dl_y[i::n_sbp_x] = dl_ybc
                dr_y[i::n_sbp_x] = dr_ybc

            # Add these at each point on the SBP half to get the SBP RHS.
            rhs_sbp = c[0]*dudx + c[1]*dudy - dl_x - dr_x - dl_y - dr_y

            rhs_out[0:int(n_sbp_x*n_sbp_y)] = rhs_sbp

            sbp_east = np.zeros(n_sbp_y)
            # Pull SBP domain values off of east face.
            counter = 0
            for i in range(0, n_sbp_x*n_sbp_y):
                if i == n_sbp_x - 1:
                    sbp_east[counter] = u_sbp_ts[i]
                    counter += 1
                elif i % n_sbp_x == n_sbp_x - 1:
                    sbp_east[counter] = u_sbp_ts[i]
                    counter += 1

            # Projection from SBP to DG is now a two-step process.
            # First: SBP-to-DG.
            sbp_proj = sbp2dg.dot(sbp_east)
            # Second: Fix the ordering.
            sbp_proj = sbp_proj[unsort_args]
            sbp_tag = dof_desc.DTAG_BOUNDARY("btag_sbp")

            u_dg_in = unflatten(actx, dcoll.discr_from_dd("vol"),
                                actx.from_numpy(u[int(n_sbp_x*n_sbp_y):]))
            u_sbp_in = unflatten(actx, dcoll.discr_from_dd(sbp_tag),
                                actx.from_numpy(sbp_proj))

            # Grudge DG RHS.
            # Critical step - now need to apply projected SBP state to the
            # proper nodal locations in u_dg.
            dg_rhs = adv_operator.operator(
                    t=t,
                    u=u_dg_in,
                    state_from_sbp=u_sbp_in,
                    sbp_tag=sbp_tag, std_tag=std_tag)
            dg_rhs = flatten(dg_rhs)
            rhs_out[int(n_sbp_x*n_sbp_y):] = dg_rhs.get()

            return rhs_out

        # Timestepper.
        from grudge.shortcuts import set_up_rk4

        # Make a combined u with the SBP and the DG parts.
        u_comb = np.zeros(int(n_sbp_x*n_sbp_y) + nnodes)
        u_comb[0:int(n_sbp_x*n_sbp_y)] = u_sbp
        u_flat = flatten(u)
        for i in range(int(n_sbp_x*n_sbp_y), int(n_sbp_x*n_sbp_y) + nnodes):
            u_comb[i] = u_flat[i - int(n_sbp_x*n_sbp_y)].get()
        dt_stepper = set_up_rk4("u", dt, u_comb, rhs)

        from grudge.shortcuts import make_visualizer
        vis = make_visualizer(dcoll, vis_order=order)

        step = 0

        # Create mesh for structured grid output
        sbp_mesh = np.zeros((2, n_sbp_y, n_sbp_x,))
        for j in range(0, n_sbp_y):
            sbp_mesh[0, j, :] = x_sbp
        for i in range(0, n_sbp_x):
            sbp_mesh[1, :, i] = y_sbp

        for event in dt_stepper.run(t_end=final_time):
            if isinstance(event, dt_stepper.StateComputed):

                step += 1

                last_t = event.t
                u_sbp = event.state_component[0:int(n_sbp_x*n_sbp_y)]
                u_dg = unflatten(actx, dcoll.discr_from_dd("vol"),
                        actx.from_numpy(
                            event.state_component[int(n_sbp_x*n_sbp_y):]))

                error_l2_dg = op.norm(
                    dcoll,
                    u_dg - u_analytic(nodes, t=last_t),
                    2
                )

                sbp_error = np.zeros((n_sbp_x*n_sbp_y))
                error_l2_sbp = 0
                for j in range(0, n_sbp_y):
                    for i in range(0, n_sbp_x):
                        sbp_error[i + j*n_sbp_x] = \
                                u_sbp[i + j*n_sbp_x] - \
                                np.sin(10*(-c.dot([x_sbp[i], y_sbp[j]])
                                       / norm_c + last_t*norm_c))
                        error_l2_sbp = error_l2_sbp + \
                            dx*dy*(sbp_error[i + j*n_sbp_x]) ** 2

                error_l2_sbp = np.sqrt(error_l2_sbp)

                # Write out the DG data
                if visualize:
                    vis.write_vtk_file("eoc_dg-%s-%04d.vtu" %
                                       (nelem, step),
                                       [("u", u_dg)], overwrite=True)

                # Write out the SBP data.
                from pyvisfile.vtk import write_structured_grid
                if visualize:
                    filename = "eoc_sbp_%s-%04d.vts" % (nelem, step)
                    write_structured_grid(filename, sbp_mesh,
                                          point_data=[("u", u_sbp)])

        if c[0] > 0:
            eoc_rec.add_data_point(h, error_l2_dg.get())
        else:
            eoc_rec.add_data_point(h, error_l2_sbp)

    assert eoc_rec.order_estimate() > (order_sbp/2 + 1) * 0.95

# }}}


# {{{ models: maxwell

@pytest.mark.parametrize("order", [3, 4, 5])
def test_convergence_maxwell(actx_factory: ArrayContextFactory,  order: int) -> None:
    """Test whether 3D Maxwell's actually converges"""

    actx = actx_factory()

    from pytools.convergence import EOCRecorder
    eoc_rec = EOCRecorder()

    dims = 3
    ns = [4, 6, 8]
    for n in ns:
        mesh = mgen.generate_regular_rect_mesh(
                a=(0.0,)*dims,
                b=(1.0,)*dims,
                nelements_per_axis=(n,)*dims)

        dcoll = make_discretization_collection(actx, mesh, order=order)

        epsilon = 1
        mu = 1

        from grudge.models.em import Vector, get_rectangular_cavity_mode

        def analytic_sol(x: Vector, t: float = 0) -> Vector:
            return get_rectangular_cavity_mode(actx, x, t, 1, (1, 2, 2))

        nodes = actx.thaw(dcoll.nodes())
        fields = analytic_sol(nodes, t=0)

        from grudge.models.em import MaxwellOperator

        maxwell_operator = MaxwellOperator(
            dcoll,
            epsilon,
            mu,
            flux_type=0.5,
            dimensions=dims
        )
        maxwell_operator.check_bc_coverage(mesh)

        def rhs(t: float,
                w: Vector,
                maxwell_operator: MaxwellOperator = maxwell_operator) -> Vector:
            return maxwell_operator.operator(t, w)

        dt = actx.to_numpy(maxwell_operator.estimate_rk4_timestep(actx, dcoll)).item()
        final_t = dt * 5
        nsteps = int(final_t/dt)

        from grudge.shortcuts import set_up_rk4

        dt_stepper = set_up_rk4("w", dt, fields, rhs)

        logger.info("dt %.5e nsteps %5d", dt, nsteps)

        esc = None

        step = 0
        for event in dt_stepper.run(t_end=final_t):
            if isinstance(event, dt_stepper.StateComputed):
                assert event.component_id == "w"
                esc = event.state_component

                step += 1
                logger.debug("[%04d] t = %.5e", step, event.t)

        assert esc is not None

        sol = analytic_sol(nodes, t=step * dt)
        total_error = op.norm(dcoll, esc - sol, 2)
        eoc_rec.add_data_point(1.0/n, actx.to_numpy(total_error).item())

    logger.info("\n%s", eoc_rec.pretty_print(
        abscissa_label="h",
        error_label="L2 Error"))

    assert eoc_rec.order_estimate() > order

# }}}


# {{{ models: variable coefficient advection oversampling

@pytest.mark.parametrize("order", [2, 3, 4])
def test_improvement_quadrature(actx_factory: ArrayContextFactory, order):
    """Test whether quadrature improves things and converges"""
    from meshmode.discretization.poly_element import QuadratureSimplexGroupFactory
    from meshmode.mesh import BTAG_ALL
    from pytools.convergence import EOCRecorder

    from grudge.models.advection import VariableCoefficientAdvectionOperator

    actx = actx_factory()

    dims = 2

    def gaussian_mode(x):
        source_width = 0.1
        return actx.np.exp(-np.dot(x, x) / source_width**2)

    def conv_test(descr, use_quad):
        logger.info("-" * 75)
        logger.info(descr)
        logger.info("-" * 75)
        eoc_rec = EOCRecorder()

        if use_quad:
            qtag = dof_desc.DISCR_TAG_QUAD
        else:
            qtag = None

        ns = [20, 25]
        for n in ns:
            mesh = mgen.generate_regular_rect_mesh(
                a=(-0.5,)*dims,
                b=(0.5,)*dims,
                nelements_per_axis=(n,)*dims,
                order=order)

            if use_quad:
                discr_tag_to_group_factory = {
                    qtag: QuadratureSimplexGroupFactory(order=4*order)
                }
            else:
                discr_tag_to_group_factory = {}

            dcoll = make_discretization_collection(
                actx, mesh, order=order,
                discr_tag_to_group_factory=discr_tag_to_group_factory
            )

            nodes = actx.thaw(dcoll.nodes())

            def zero_inflow(dtag, t=0, dcoll=dcoll):
                dd = dof_desc.as_dofdesc(dtag, qtag)
                return dcoll.discr_from_dd(dd).zeros(actx)

            adv_op = VariableCoefficientAdvectionOperator(
                dcoll,
                obj_array.flat(-1*nodes[1], nodes[0]),
                inflow_u=lambda t: zero_inflow(BTAG_ALL, t=t),
                flux_type="upwind",
                quad_tag=qtag
            )

            total_error = op.norm(
                dcoll, adv_op.operator(0, gaussian_mode(nodes)), 2
            )
            eoc_rec.add_data_point(1.0/n, actx.to_numpy(total_error))

        logger.info("\n%s", eoc_rec.pretty_print(
            abscissa_label="h",
            error_label="L2 Error"))

        return eoc_rec.order_estimate(), np.array([x[1] for x in eoc_rec.history])

    eoc, errs = conv_test("no quadrature", False)
    q_eoc, q_errs = conv_test("with quadrature", True)

    assert q_eoc > eoc
    assert (q_errs < errs).all()
    assert q_eoc > order - 0.1

# }}}


# {{{ bessel

@pytest.mark.xfail
def test_bessel(actx_factory: ArrayContextFactory):
    actx = actx_factory()

    dims = 2

    mesh = mgen.generate_regular_rect_mesh(
            a=(0.1,)*dims,
            b=(1.0,)*dims,
            nelements_per_axis=(8,)*dims)

    dcoll = make_discretization_collection(actx, mesh, order=3)

    nodes = actx.thaw(dcoll.nodes())
    r = actx.np.sqrt(nodes[0]**2 + nodes[1]**2)

    # FIXME: Bessel functions need to brought out of the symbolic
    # layer. Related issue: https://github.com/inducer/grudge/issues/93
    def bessel_j(actx, n, r):
        from grudge import bind, sym  # pylint: disable=no-name-in-module
        return bind(dcoll, sym.bessel_j(n, sym.var("r")))(actx, r=r)

    # https://dlmf.nist.gov/10.6.1
    n = 3
    bessel_zero = (bessel_j(actx, n+1, r)
                   + bessel_j(actx, n-1, r)
                   - 2*n/r * bessel_j(actx, n, r))

    z = op.norm(dcoll, bessel_zero, 2)

    assert z < 1e-15

# }}}


# {{{ test norms

@pytest.mark.parametrize("p", [2, np.inf])
def test_norm_real(actx_factory: ArrayContextFactory, p):
    actx = actx_factory()

    dim = 2
    mesh = mgen.generate_regular_rect_mesh(
            a=(0,)*dim, b=(1,)*dim,
            nelements_per_axis=(8,)*dim, order=1)
    dcoll = make_discretization_collection(actx, mesh, order=4)
    nodes = actx.thaw(dcoll.nodes())

    norm = op.norm(dcoll, nodes[0], p)
    if p == 2:
        ref_norm = (1/3)**0.5
    elif p == np.inf:
        ref_norm = 1
    else:
        raise AssertionError("unsupported p")

    logger.info("norm: %.5e %.5e", norm, ref_norm)
    assert abs(norm-ref_norm) / abs(ref_norm) < 1e-13


@pytest.mark.parametrize("p", [2, np.inf])
def test_norm_complex(actx_factory: ArrayContextFactory, p):
    actx = actx_factory()

    dim = 2
    mesh = mgen.generate_regular_rect_mesh(
            a=(0,)*dim, b=(1,)*dim,
            nelements_per_axis=(8,)*dim, order=1)
    dcoll = make_discretization_collection(actx, mesh, order=4)
    nodes = actx.thaw(dcoll.nodes())

    norm = op.norm(dcoll, (1 + 1j)*nodes[0], p)
    if p == 2:
        ref_norm = (2/3)**0.5
    elif p == np.inf:
        ref_norm = 2**0.5
    else:
        raise AssertionError("unsupported p")

    logger.info("norm: %.5e %.5e", norm, ref_norm)
    assert abs(norm-ref_norm) / abs(ref_norm) < 1e-13


@pytest.mark.parametrize("p", [2, np.inf])
def test_norm_obj_array(actx_factory: ArrayContextFactory, p):
    actx = actx_factory()

    dim = 2
    mesh = mgen.generate_regular_rect_mesh(
            a=(0,)*dim, b=(1,)*dim,
            nelements_per_axis=(8,)*dim, order=1)
    dcoll = make_discretization_collection(actx, mesh, order=4)
    nodes = actx.thaw(dcoll.nodes())

    norm = op.norm(dcoll, nodes, p)

    if p == 2:
        ref_norm = (dim/3)**0.5
    elif p == np.inf:
        ref_norm = 1
    else:
        raise AssertionError("unsupported p")

    logger.info("norm: %.5e %.5e", norm, ref_norm)
    assert abs(norm-ref_norm) / abs(ref_norm) < 1e-14

# }}}


# {{{ empty boundaries

def test_empty_boundary(actx_factory: ArrayContextFactory):
    # https://github.com/inducer/grudge/issues/54

    from meshmode.mesh import BTAG_NONE

    actx = actx_factory()

    dim = 2
    mesh = mgen.generate_regular_rect_mesh(
            a=(-0.5,)*dim, b=(0.5,)*dim,
            nelements_per_axis=(8,)*dim, order=4)
    dcoll = make_discretization_collection(actx, mesh, order=4)
    normal = geometry.normal(actx, dcoll, BTAG_NONE)
    from meshmode.dof_array import DOFArray
    for component in normal:
        assert isinstance(component, DOFArray)
        assert len(component) == len(dcoll.discr_from_dd(BTAG_NONE).groups)

# }}}


# {{{ multi-volume

def test_multiple_independent_volumes(actx_factory: ArrayContextFactory):
    dim = 2
    actx = actx_factory()

    mesh1 = mgen.generate_regular_rect_mesh(
            a=(-2,)*dim, b=(-1,)*dim,
            nelements_per_axis=(4,)*dim, order=4)

    mesh2 = mgen.generate_regular_rect_mesh(
            a=(1,)*dim, b=(2,)*dim,
            nelements_per_axis=(8,)*dim, order=4)

    volume_to_mesh = {
        "vol1": mesh1,
        "vol2": mesh2}

    make_discretization_collection(actx, volume_to_mesh, order=4)

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
