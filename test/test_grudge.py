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

import numpy as np
import numpy.linalg as la

from meshmode import _acf       # noqa: F401
from meshmode.dof_array import flatten, thaw
import meshmode.mesh.generation as mgen

from pytools.obj_array import flat_obj_array, make_obj_array

from grudge import sym, bind, DiscretizationCollection

import grudge.dof_desc as dof_desc
import grudge.op as op


import pytest
from meshmode.array_context import (  # noqa
        pytest_generate_tests_for_pyopencl_array_context
        as pytest_generate_tests)

import logging

logger = logging.getLogger(__name__)


# {{{ inverse metric

@pytest.mark.parametrize("dim", [2, 3])
def test_inverse_metric(actx_factory, dim):
    actx = actx_factory()

    mesh = mgen.generate_regular_rect_mesh(a=(-0.5,)*dim, b=(0.5,)*dim,
            nelements_per_axis=(6,)*dim, order=4)

    def m(x):
        result = np.empty_like(x)
        result[0] = (
                1.5*x[0] + np.cos(x[0])
                + 0.1*np.sin(10*x[1]))
        result[1] = (
                0.05*np.cos(10*x[0])
                + 1.3*x[1] + np.sin(x[1]))
        if len(x) == 3:
            result[2] = x[2]
        return result

    from meshmode.mesh.processing import map_mesh
    mesh = map_mesh(mesh, m)

    dcoll = DiscretizationCollection(actx, mesh, order=4)

    from grudge.geometry import \
        forward_metric_derivative_mat, inverse_metric_derivative_mat

    mat = forward_metric_derivative_mat(actx, dcoll).dot(
        inverse_metric_derivative_mat(actx, dcoll))

    for i in range(mesh.dim):
        for j in range(mesh.dim):
            tgt = 1 if i == j else 0

            err = actx.np.linalg.norm(mat[i, j] - tgt, ord=np.inf)
            logger.info("error[%d, %d]: %.5e", i, j, err)
            assert err < 1.0e-12, (i, j, err)

# }}}


# {{{ mass operator trig integration

@pytest.mark.parametrize("ambient_dim", [1, 2, 3])
@pytest.mark.parametrize("discr_tag", [dof_desc.DISCR_TAG_BASE,
                                       dof_desc.DISCR_TAG_QUAD])
def test_mass_mat_trig(actx_factory, ambient_dim, discr_tag):
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
    dcoll = DiscretizationCollection(
        actx, mesh, order=order,
        discr_tag_to_group_factory=discr_tag_to_group_factory
    )

    def f(x):
        return actx.np.sin(x[0])**2

    volm_disc = dcoll.discr_from_dd(dof_desc.DD_VOLUME)
    x_volm = thaw(actx, volm_disc.nodes())
    f_volm = f(x_volm)
    ones_volm = volm_disc.zeros(actx) + 1

    quad_disc = dcoll.discr_from_dd(dd_quad)
    x_quad = thaw(actx, quad_disc.nodes())
    f_quad = f(x_quad)
    ones_quad = quad_disc.zeros(actx) + 1

    mop_1 = op.mass(dcoll, dd_quad, f_quad)
    num_integral_1 = np.dot(actx.to_numpy(flatten(ones_volm)),
                            actx.to_numpy(flatten(mop_1)))

    err_1 = abs(num_integral_1 - true_integral)
    assert err_1 < 1e-9, err_1

    mop_2 = op.mass(dcoll, dd_quad, ones_quad)
    num_integral_2 = np.dot(actx.to_numpy(flatten(f_volm)),
                            actx.to_numpy(flatten(mop_2)))

    err_2 = abs(num_integral_2 - true_integral)
    assert err_2 < 1.0e-9, err_2

    if discr_tag is dof_desc.DISCR_TAG_BASE:
        # NOTE: `integral` always makes a square mass matrix and
        # `QuadratureSimplexGroupFactory` does not have a `mass_matrix` method.
        num_integral_3 = np.dot(actx.to_numpy(flatten(f_quad)),
                                actx.to_numpy(flatten(mop_2)))
        err_3 = abs(num_integral_3 - true_integral)
        assert err_3 < 5.0e-10, err_3

# }}}


# {{{ mass operator on surface

def _ellipse_surface_area(radius, aspect_ratio):
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.ellipe.html
    eccentricity = 1.0 - (1/aspect_ratio)**2

    if abs(aspect_ratio - 2.0) < 1.0e-14:
        # NOTE: hardcoded value so we don't need scipy for the test
        ellip_e = 1.2110560275684594
    else:
        from scipy.special import ellipe        # pylint: disable=no-name-in-module
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
def test_mass_surface_area(actx_factory, name):
    actx = actx_factory()

    # {{{ cases

    if name == "2-1-ellipse":
        from mesh_data import EllipseMeshBuilder
        builder = EllipseMeshBuilder(radius=3.1, aspect_ratio=2.0)
        surface_area = _ellipse_surface_area(builder.radius, builder.aspect_ratio)
    elif name == "spheroid":
        from mesh_data import SpheroidMeshBuilder
        builder = SpheroidMeshBuilder()
        surface_area = _spheroid_surface_area(builder.radius, builder.aspect_ratio)
    elif name == "box2d":
        from mesh_data import BoxMeshBuilder
        builder = BoxMeshBuilder(ambient_dim=2)
        surface_area = 1.0
    elif name == "box3d":
        from mesh_data import BoxMeshBuilder
        builder = BoxMeshBuilder(ambient_dim=3)
        surface_area = 1.0
    else:
        raise ValueError("unknown geometry name: %s" % name)

    # }}}

    # {{{ convergence

    from pytools.convergence import EOCRecorder
    eoc = EOCRecorder()

    for resolution in builder.resolutions:
        mesh = builder.get_mesh(resolution, builder.mesh_order)
        dcoll = DiscretizationCollection(actx, mesh, order=builder.order)
        volume_discr = dcoll.discr_from_dd(dof_desc.DD_VOLUME)

        logger.info("ndofs:     %d", volume_discr.ndofs)
        logger.info("nelements: %d", volume_discr.mesh.nelements)

        # {{{ compute surface area

        dd = dof_desc.DD_VOLUME
        ones_volm = volume_discr.zeros(actx) + 1
        flattened_mass_weights = flatten(op.mass(dcoll, dd, ones_volm))
        approx_surface_area = np.dot(actx.to_numpy(flatten(ones_volm)),
                                     actx.to_numpy(flattened_mass_weights))

        logger.info("surface: got {:.5e} / expected {:.5e}".format(
            approx_surface_area, surface_area))
        area_error = abs(approx_surface_area - surface_area) / abs(surface_area)

        # }}}

        # {{{ compute h_max using mass weights

        h_max = actx.np.max(flattened_mass_weights) ** (1/dcoll.dim)

        # }}}

        eoc.add_data_point(h_max, area_error)

    # }}}

    logger.info("surface area error\n%s", str(eoc))

    assert eoc.max_error() < 3e-13 or eoc.order_estimate() > builder.order

# }}}


# {{{ mass inverse on surfaces

@pytest.mark.parametrize("name", ["2-1-ellipse", "spheroid"])
def test_surface_mass_operator_inverse(actx_factory, name):
    actx = actx_factory()

    # {{{ cases

    if name == "2-1-ellipse":
        from mesh_data import EllipseMeshBuilder
        builder = EllipseMeshBuilder(radius=3.1, aspect_ratio=2.0)
    elif name == "spheroid":
        from mesh_data import SpheroidMeshBuilder
        builder = SpheroidMeshBuilder()
    else:
        raise ValueError("unknown geometry name: %s" % name)

    # }}}

    # {{{ convergence

    from pytools.convergence import EOCRecorder
    eoc = EOCRecorder()

    for resolution in builder.resolutions:
        mesh = builder.get_mesh(resolution, builder.mesh_order)
        dcoll = DiscretizationCollection(actx, mesh, order=builder.order)
        volume_discr = dcoll.discr_from_dd(dof_desc.DD_VOLUME)

        logger.info("ndofs:     %d", volume_discr.ndofs)
        logger.info("nelements: %d", volume_discr.mesh.nelements)

        # {{{ compute inverse mass

        def f(x):
            return actx.np.cos(4.0 * x[0])

        dd = dof_desc.DD_VOLUME
        x_volm = thaw(actx, volume_discr.nodes())
        f_volm = f(x_volm)

        res = op.inverse_mass(
            dcoll, op.mass(dcoll, dd, f_volm)
        )

        inv_error = actx.np.linalg.norm(res - f_volm, ord=2)

        # }}}

        # {{{ compute h_max from mass weights

        ones_volm = volume_discr.zeros(actx) + 1
        flattened_mass_weights = flatten(op.mass(dcoll, dd, ones_volm))
        h_max = actx.np.max(flattened_mass_weights) ** (1/dcoll.dim)

        # }}}

        eoc.add_data_point(h_max, inv_error)

    # }}}

    logger.info("inverse mass error\n%s", str(eoc))

    assert eoc.max_error() < 5e-13

# }}}


# {{{ surface face normal orthogonality

@pytest.mark.parametrize("mesh_name", ["2-1-ellipse", "spheroid"])
def test_face_normal_surface(actx_factory, mesh_name):
    """Check that face normals are orthogonal to the surface normal"""
    actx = actx_factory()

    # {{{ geometry

    if mesh_name == "2-1-ellipse":
        from mesh_data import EllipseMeshBuilder
        builder = EllipseMeshBuilder(radius=3.1, aspect_ratio=2.0)
    elif mesh_name == "spheroid":
        from mesh_data import SpheroidMeshBuilder
        builder = SpheroidMeshBuilder()
    else:
        raise ValueError("unknown mesh name: %s" % mesh_name)

    mesh = builder.get_mesh(builder.resolutions[0], builder.mesh_order)
    dcoll = DiscretizationCollection(actx, mesh, order=builder.order)

    volume_discr = dcoll.discr_from_dd(dof_desc.DD_VOLUME)
    logger.info("ndofs:    %d", volume_discr.ndofs)
    logger.info("nelements: %d", volume_discr.mesh.nelements)

    # }}}

    # {{{ Compute surface and face normals
    from meshmode.discretization.connection import FACE_RESTR_INTERIOR
    from grudge.geometry import surface_normal

    dv = dof_desc.DD_VOLUME
    df = dof_desc.as_dofdesc(FACE_RESTR_INTERIOR)

    ambient_dim = mesh.ambient_dim
    dim = mesh.dim

    surf_normal = op.project(
        dcoll, dv, df,
        surface_normal(actx, dcoll,
                       dim=dim, dd=dv).as_vector(dtype=object)
    )
    surf_normal = surf_normal / op.norm(dcoll, surf_normal, 2)

    face_normal_i = thaw(actx, op.normal(dcoll, df))
    face_normal_e = dcoll.opposite_face_connection()(face_normal_i)

    if mesh.ambient_dim == 3:
        from grudge.geometry import pseudoscalar, area_element
        # NOTE: there's only one face tangent in 3d
        face_tangent = (
            pseudoscalar(actx, dcoll, dim=dim-1, dd=df)
            / area_element(actx, dcoll, dim=dim-1, dd=df)
        ).as_vector(dtype=object)

    # }}}

    # {{{ checks

    def _eval_error(x):
        return op.norm(dcoll, x, np.inf)

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
def test_tri_diff_mat(actx_factory, dim, order=4):
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

        dcoll = DiscretizationCollection(actx, mesh, order=4)
        volume_discr = dcoll.discr_from_dd(dof_desc.DD_VOLUME)
        x = thaw(actx, volume_discr.nodes())

        for axis in range(dim):
            df_num = op.local_grad(dcoll, f(x, axis))[axis]
            df_volm = df(x, axis)

            linf_error = actx.np.linalg.norm(df_num - df_volm, ord=np.inf)
            axis_eoc_recs[axis].add_data_point(1/n, linf_error)

    for axis, eoc_rec in enumerate(axis_eoc_recs):
        logger.info("axis %d\n%s", axis, eoc_rec)
        assert eoc_rec.order_estimate() > order - 0.25

# }}}


# {{{ divergence theorem

def test_2d_gauss_theorem(actx_factory):
    """Verify Gauss's theorem explicitly on a mesh"""

    pytest.importorskip("meshpy")

    from meshpy.geometry import make_circle, GeometryBuilder
    from meshpy.triangle import MeshInfo, build

    geob = GeometryBuilder()
    geob.add_geometry(*make_circle(1))
    mesh_info = MeshInfo()
    geob.set(mesh_info)

    mesh_info = build(mesh_info)

    from meshmode.mesh.io import from_meshpy
    from meshmode.mesh import BTAG_ALL

    mesh = from_meshpy(mesh_info, order=1)

    actx = actx_factory()

    dcoll = DiscretizationCollection(actx, mesh, order=2)
    volm_disc = dcoll.discr_from_dd(dof_desc.DD_VOLUME)
    x_volm = thaw(actx, volm_disc.nodes())

    def f(x):
        return flat_obj_array(
            actx.np.sin(3*x[0]) + actx.np.cos(3*x[1]),
            actx.np.sin(2*x[0]) + actx.np.cos(x[1])
        )

    f_volm = f(x_volm)
    int_1 = op.integral(dcoll, op.local_div(dcoll, f_volm))

    prj_f = op.project(dcoll, "vol", BTAG_ALL, f_volm)
    normal = thaw(actx, op.normal(dcoll, BTAG_ALL))
    int_2 = op.integral(dcoll, prj_f.dot(normal), dd=BTAG_ALL)

    assert abs(int_1 - int_2) < 1e-13


@pytest.mark.parametrize("mesh_name", ["2-1-ellipse", "spheroid"])
def test_surface_divergence_theorem(actx_factory, mesh_name, visualize=False):
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
        from mesh_data import EllipseMeshBuilder
        builder = EllipseMeshBuilder(radius=3.1, aspect_ratio=2.0)
    elif mesh_name == "spheroid":
        from mesh_data import SpheroidMeshBuilder
        builder = SpheroidMeshBuilder()
    elif mesh_name == "circle":
        from mesh_data import EllipseMeshBuilder
        builder = EllipseMeshBuilder(radius=1.0, aspect_ratio=1.0)
    elif mesh_name == "starfish":
        from mesh_data import StarfishMeshBuilder
        builder = StarfishMeshBuilder()
    elif mesh_name == "sphere":
        from mesh_data import SphereMeshBuilder
        builder = SphereMeshBuilder(radius=1.0, mesh_order=16)
    else:
        raise ValueError("unknown mesh name: %s" % mesh_name)

    # }}}

    # {{{ convergene

    def f(x):
        return flat_obj_array(
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

    mesh_offset = np.array([0.33, -0.21, 0.0])[:ambient_dim]

    for i, resolution in enumerate(builder.resolutions):
        from meshmode.mesh.processing import affine_map
        from meshmode.discretization.connection import FACE_RESTR_ALL

        mesh = builder.get_mesh(resolution, builder.mesh_order)
        mesh = affine_map(mesh, A=mesh_rotation, b=mesh_offset)

        from meshmode.discretization.poly_element import \
                QuadratureSimplexGroupFactory

        qtag = dof_desc.DISCR_TAG_QUAD
        dcoll = DiscretizationCollection(
            actx, mesh, order=builder.order,
            discr_tag_to_group_factory={
                qtag: QuadratureSimplexGroupFactory(2 * builder.order)
            }
        )

        volume = dcoll.discr_from_dd(dof_desc.DD_VOLUME)
        logger.info("ndofs:     %d", volume.ndofs)
        logger.info("nelements: %d", volume.mesh.nelements)

        dd = dof_desc.DD_VOLUME
        dq = dd.with_discr_tag(qtag)
        df = dof_desc.as_dofdesc(FACE_RESTR_ALL)
        ambient_dim = dcoll.ambient_dim
        dim = dcoll.dim

        # variables
        f_num = f(thaw(actx, op.nodes(dcoll, dd=dd)))
        f_quad_num = f(thaw(actx, op.nodes(dcoll, dd=dq)))

        from grudge.geometry import surface_normal, summed_curvature

        kappa = summed_curvature(actx, dcoll, dim=dim, dd=dq)
        normal = surface_normal(actx, dcoll,
                                dim=dim, dd=dq).as_vector(dtype=object)
        face_normal = thaw(actx, op.normal(dcoll, df))
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

        err_global = abs(op_global)
        err_local = bind(dcoll, sym.norm(np.inf, sym.var("x")))(actx, x=op_local)
        logger.info("errors: global %.5e local %.5e", err_global, err_local)

        # compute max element size
        ones_volm = volume.zeros(actx) + 1
        sum_mass_weights = op.elementwise_sum(dcoll, op.mass(dcoll, dd, ones_volm))
        h_max = op.nodal_max(dcoll, dd, sum_mass_weights) ** (1/dcoll.dim)

        eoc_global.add_data_point(h_max, err_global)
        eoc_local.add_data_point(h_max, err_local)

        if visualize:
            from grudge.shortcuts import make_visualizer
            vis = make_visualizer(dcoll, vis_order=builder.order)

            filename = f"surface_divergence_theorem_{mesh_name}_{i:04d}.vtu"
            vis.write_vtk_file(filename, [
                ("r", actx.np.log10(op_local))
                ], overwrite=True)

    # }}}

    order = min(builder.order, builder.mesh_order) - 0.5
    logger.info("\n%s", str(eoc_global))
    logger.info("\n%s", str(eoc_local))

    assert eoc_global.max_error() < 1.0e-12 \
            or eoc_global.order_estimate() > order - 0.5

    assert eoc_local.max_error() < 1.0e-12 \
            or eoc_local.order_estimate() > order - 0.5

# }}}


# {{{ models: advection

@pytest.mark.parametrize(("mesh_name", "mesh_pars"), [
    ("segment", [8, 16, 32]),
    ("disk", [0.1, 0.05]),
    ("rect2", [4, 8]),
    ("rect3", [4, 6]),
    ("warped2", [4, 8]),
    ])
@pytest.mark.parametrize("op_type", ["strong", "weak"])
@pytest.mark.parametrize("flux_type", ["central"])
@pytest.mark.parametrize("order", [3, 4, 5])
# test: 'test_convergence_advec(cl._csc, "disk", [0.1, 0.05], "strong", "upwind", 3)'
def test_convergence_advec(actx_factory, mesh_name, mesh_pars, op_type, flux_type,
        order, visualize=False):
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

            from meshpy.geometry import make_circle, GeometryBuilder
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
                raise ValueError("dt_factor not known for %dd" % dim)
        elif mesh_name.startswith("warped"):
            dim = int(mesh_name[-1:])
            mesh = mgen.generate_warped_rect_mesh(dim, order=order,
                    nelements_side=mesh_par)

            if dim == 2:
                dt_factor = 4
            elif dim == 3:
                dt_factor = 2
            else:
                raise ValueError("dt_factor not known for %dd" % dim)
        else:
            raise ValueError("invalid mesh name: " + mesh_name)

        v = np.array([0.27, 0.31, 0.1])[:dim]
        norm_v = la.norm(v)

        def f(x):
            return sym.sin(10*x)

        def u_analytic(x):
            return f(
                    -v.dot(x)/norm_v
                    + sym.var("t", dof_desc.DD_SCALAR)*norm_v)

        from grudge.models.advection import (
                StrongAdvectionOperator, WeakAdvectionOperator)
        from meshmode.mesh import BTAG_ALL

        discr = DiscretizationCollection(actx, mesh, order=order)
        op_class = {
                "strong": StrongAdvectionOperator,
                "weak": WeakAdvectionOperator,
                }[op_type]
        op = op_class(v,
                inflow_u=u_analytic(sym.nodes(dim, BTAG_ALL)),
                flux_type=flux_type)

        bound_op = bind(discr, op.sym_operator())

        u = bind(discr, u_analytic(sym.nodes(dim)))(actx, t=0)

        def rhs(t, u):
            return bound_op(t=t, u=u)

        if dim == 3:
            final_time = 0.1
        else:
            final_time = 0.2

        h_max = bind(discr, sym.h_max_from_volume(discr.ambient_dim))(actx)
        dt = dt_factor * h_max/order**2
        nsteps = (final_time // dt) + 1
        dt = final_time/nsteps + 1e-15

        from grudge.shortcuts import set_up_rk4
        dt_stepper = set_up_rk4("u", dt, u, rhs)

        last_u = None

        from grudge.shortcuts import make_visualizer
        vis = make_visualizer(discr, vis_order=order)

        step = 0

        for event in dt_stepper.run(t_end=final_time):
            if isinstance(event, dt_stepper.StateComputed):
                step += 1
                logger.debug("[%04d] t = %.5f", step, event.t)

                last_t = event.t
                last_u = event.state_component

                if visualize:
                    vis.write_vtk_file("fld-%s-%04d.vtu" % (mesh_par, step),
                            [("u", event.state_component)])

        error_l2 = bind(discr,
            sym.norm(2, sym.var("u")-u_analytic(sym.nodes(dim))))(
                t=last_t, u=last_u)
        logger.info("h_max %.5e error %.5e", h_max, error_l2)
        eoc_rec.add_data_point(h_max, error_l2)

    logger.info("\n%s", eoc_rec.pretty_print(
        abscissa_label="h",
        error_label="L2 Error"))

    if mesh_name.startswith("warped"):
        # NOTE: curvilinear meshes are hard
        assert eoc_rec.order_estimate() > order - 0.5
    else:
        assert eoc_rec.order_estimate() > order

# }}}


# {{{ models: maxwell

@pytest.mark.parametrize("order", [3, 4, 5])
def test_convergence_maxwell(actx_factory,  order):
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

        discr = DiscretizationCollection(actx, mesh, order=order)

        epsilon = 1
        mu = 1

        from grudge.models.em import get_rectangular_cavity_mode
        sym_mode = get_rectangular_cavity_mode(1, (1, 2, 2))

        analytic_sol = bind(discr, sym_mode)
        fields = analytic_sol(actx, t=0, epsilon=epsilon, mu=mu)

        from grudge.models.em import MaxwellOperator
        op = MaxwellOperator(epsilon, mu, flux_type=0.5, dimensions=dims)
        op.check_bc_coverage(mesh)
        bound_op = bind(discr, op.sym_operator())

        def rhs(t, w):
            return bound_op(t=t, w=w)

        dt = 0.002
        final_t = dt * 5
        nsteps = int(final_t/dt)

        from grudge.shortcuts import set_up_rk4
        dt_stepper = set_up_rk4("w", dt, fields, rhs)

        logger.info("dt %.5e nsteps %5d", dt, nsteps)

        norm = bind(discr, sym.norm(2, sym.var("u")))

        step = 0
        for event in dt_stepper.run(t_end=final_t):
            if isinstance(event, dt_stepper.StateComputed):
                assert event.component_id == "w"
                esc = event.state_component

                step += 1
                logger.debug("[%04d] t = %.5e", step, event.t)

        sol = analytic_sol(actx, mu=mu, epsilon=epsilon, t=step * dt)
        vals = [norm(u=(esc[i] - sol[i])) / norm(u=sol[i]) for i in range(5)] # noqa E501
        total_error = sum(vals)
        eoc_rec.add_data_point(1.0/n, total_error)

    logger.info("\n%s", eoc_rec.pretty_print(
        abscissa_label="h",
        error_label="L2 Error"))

    assert eoc_rec.order_estimate() > order

# }}}


# {{{ models: variable coefficient advection oversampling

@pytest.mark.parametrize("order", [2, 3, 4])
def test_improvement_quadrature(actx_factory, order):
    """Test whether quadrature improves things and converges"""
    from grudge.models.advection import VariableCoefficientAdvectionOperator
    from pytools.convergence import EOCRecorder
    from meshmode.discretization.poly_element import QuadratureSimplexGroupFactory

    actx = actx_factory()

    dims = 2
    sym_nds = sym.nodes(dims)
    advec_v = flat_obj_array(-1*sym_nds[1], sym_nds[0])

    flux = "upwind"
    op = VariableCoefficientAdvectionOperator(advec_v, 0, flux_type=flux)

    def gaussian_mode():
        source_width = 0.1
        sym_x = sym.nodes(2)
        return sym.exp(-np.dot(sym_x, sym_x) / source_width**2)

    def conv_test(descr, use_quad):
        logger.info("-" * 75)
        logger.info(descr)
        logger.info("-" * 75)
        eoc_rec = EOCRecorder()

        ns = [20, 25]
        for n in ns:
            mesh = mgen.generate_regular_rect_mesh(
                a=(-0.5,)*dims,
                b=(0.5,)*dims,
                nelements_per_axis=(n,)*dims,
                order=order)

            if use_quad:
                discr_tag_to_group_factory = {
                    "product": QuadratureSimplexGroupFactory(order=4*order)
                }
            else:
                discr_tag_to_group_factory = {"product": None}

            discr = DiscretizationCollection(
                actx, mesh, order=order,
                discr_tag_to_group_factory=discr_tag_to_group_factory
            )

            bound_op = bind(discr, op.sym_operator())
            fields = bind(discr, gaussian_mode())(actx, t=0)
            norm = bind(discr, sym.norm(2, sym.var("u")))

            esc = bound_op(u=fields)
            total_error = norm(u=esc)
            eoc_rec.add_data_point(1.0/n, total_error)

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


# {{{ operator collector determinism

def test_op_collector_order_determinism():
    class TestOperator(sym.Operator):

        def __init__(self):
            sym.Operator.__init__(self, dof_desc.DD_VOLUME, dof_desc.DD_VOLUME)

        mapper_method = "map_test_operator"

    from grudge.symbolic.mappers import BoundOperatorCollector

    class TestBoundOperatorCollector(BoundOperatorCollector):

        def map_test_operator(self, expr):
            return self.map_operator(expr)

    v0 = sym.var("v0")
    ob0 = sym.OperatorBinding(TestOperator(), v0)

    v1 = sym.var("v1")
    ob1 = sym.OperatorBinding(TestOperator(), v1)

    # The output order isn't significant, but it should always be the same.
    assert list(TestBoundOperatorCollector(TestOperator)(ob0 + ob1)) == [ob0, ob1]

# }}}


# {{{ bessel

def test_bessel(actx_factory):
    actx = actx_factory()

    dims = 2

    mesh = mgen.generate_regular_rect_mesh(
            a=(0.1,)*dims,
            b=(1.0,)*dims,
            nelements_per_axis=(8,)*dims)

    discr = DiscretizationCollection(actx, mesh, order=3)

    nodes = sym.nodes(dims)
    r = sym.cse(sym.sqrt(nodes[0]**2 + nodes[1]**2))

    # https://dlmf.nist.gov/10.6.1
    n = 3
    bessel_zero = (
            sym.bessel_j(n+1, r)
            + sym.bessel_j(n-1, r)
            - 2*n/r * sym.bessel_j(n, r))

    z = bind(discr, sym.norm(2, bessel_zero))(actx)

    assert z < 1e-15

# }}}


# {{{ function symbol

def test_external_call(actx_factory):
    actx = actx_factory()

    def double(queue, x):
        return 2 * x

    dims = 2

    mesh = mgen.generate_regular_rect_mesh(
            a=(0,) * dims, b=(1,) * dims, nelements_per_axis=(4,) * dims)
    discr = DiscretizationCollection(actx, mesh, order=1)

    ones = sym.Ones(dof_desc.DD_VOLUME)
    op = (
            ones * 3
            + sym.FunctionSymbol("double")(ones))

    from grudge.function_registry import (
            base_function_registry, register_external_function)

    freg = register_external_function(
            base_function_registry,
            "double",
            implementation=double,
            dd=dof_desc.DD_VOLUME)

    bound_op = bind(discr, op, function_registry=freg)

    result = bound_op(actx, double=double)
    assert actx.to_numpy(flatten(result) == 5).all()


@pytest.mark.parametrize("array_type", ["scalar", "vector"])
def test_function_symbol_array(actx_factory, array_type):
    """Test if `FunctionSymbol` distributed properly over object arrays."""

    actx = actx_factory()

    dim = 2
    mesh = mgen.generate_regular_rect_mesh(
            a=(-0.5,)*dim, b=(0.5,)*dim,
            nelements_per_axis=(8,)*dim, order=4)
    discr = DiscretizationCollection(actx, mesh, order=4)
    volume_discr = discr.discr_from_dd(dof_desc.DD_VOLUME)

    if array_type == "scalar":
        sym_x = sym.var("x")
        x = thaw(actx, actx.np.cos(volume_discr.nodes()[0]))
    elif array_type == "vector":
        sym_x = sym.make_sym_array("x", dim)
        x = thaw(actx, volume_discr.nodes())
    else:
        raise ValueError("unknown array type")

    norm = bind(discr, sym.norm(2, sym_x))(x=x)
    assert isinstance(norm, float)

# }}}


@pytest.mark.parametrize("p", [2, np.inf])
def test_norm_obj_array(actx_factory, p):
    """Test :func:`grudge.symbolic.operators.norm` for object arrays."""

    actx = actx_factory()

    dim = 2
    mesh = mgen.generate_regular_rect_mesh(
            a=(-0.5,)*dim, b=(0.5,)*dim,
            nelements_per_axis=(8,)*dim, order=1)
    discr = DiscretizationCollection(actx, mesh, order=4)

    w = make_obj_array([1.0, 2.0, 3.0])[:dim]

    # {{ scalar

    sym_w = sym.var("w")
    norm = bind(discr, sym.norm(p, sym_w))(actx, w=w[0])

    norm_exact = w[0]
    logger.info("norm: %.5e %.5e", norm, norm_exact)
    assert abs(norm - norm_exact) < 1.0e-14

    # }}}

    # {{{ vector

    sym_w = sym.make_sym_array("w", dim)
    norm = bind(discr, sym.norm(p, sym_w))(actx, w=w)

    norm_exact = np.sqrt(np.sum(w**2)) if p == 2 else np.max(w)
    logger.info("norm: %.5e %.5e", norm, norm_exact)
    assert abs(norm - norm_exact) < 1.0e-14

    # }}}


def test_map_if(actx_factory):
    """Test :meth:`grudge.symbolic.execution.ExecutionMapper.map_if` handling
    of scalar conditions.
    """

    actx = actx_factory()

    dim = 2
    mesh = mgen.generate_regular_rect_mesh(
            a=(-0.5,)*dim, b=(0.5,)*dim,
            nelements_per_axis=(8,)*dim, order=4)
    discr = DiscretizationCollection(actx, mesh, order=4)

    sym_if = sym.If(sym.Comparison(2.0, "<", 1.0e-14), 1.0, 2.0)
    bind(discr, sym_if)(actx)


def test_empty_boundary(actx_factory):
    # https://github.com/inducer/grudge/issues/54

    from meshmode.mesh import BTAG_NONE

    actx = actx_factory()

    dim = 2
    mesh = mgen.generate_regular_rect_mesh(
            a=(-0.5,)*dim, b=(0.5,)*dim,
            nelements_per_axis=(8,)*dim, order=4)
    discr = DiscretizationCollection(actx, mesh, order=4)
    normal = bind(discr,
            sym.normal(BTAG_NONE, dim, dim=dim - 1))(actx)
    from meshmode.dof_array import DOFArray
    for component in normal:
        assert isinstance(component, DOFArray)
        assert len(component) == len(discr.discr_from_dd(BTAG_NONE).groups)


def test_operator_compiler_overwrite(actx_factory):
    """Tests that the same expression in ``eval_code`` and ``discr_code``
    does not confuse the OperatorCompiler in grudge/symbolic/compiler.py.
    """

    actx = actx_factory()

    ambient_dim = 2
    target_order = 4

    from meshmode.mesh.generation import generate_regular_rect_mesh
    mesh = generate_regular_rect_mesh(
            a=(-0.5,)*ambient_dim, b=(0.5,)*ambient_dim,
            n=(8,)*ambient_dim, order=1)
    discr = DiscretizationCollection(actx, mesh, order=target_order)

    # {{{ test

    sym_u = sym.nodes(ambient_dim)
    sym_div_u = sum(d(u) for d, u in zip(sym.nabla(ambient_dim), sym_u))

    div_u = bind(discr, sym_div_u)(actx)
    error = bind(discr, sym.norm(2, sym.var("x")))(actx, x=div_u - discr.dim)
    logger.info("error: %.5e", error)

    # }}}


@pytest.mark.parametrize("ambient_dim", [
    2,
    # FIXME, cf. https://github.com/inducer/grudge/pull/78/
    pytest.param(3, marks=pytest.mark.xfail)
    ])
def test_incorrect_assignment_aggregation(actx_factory, ambient_dim):
    """Tests that the greedy assignemnt aggregation code works on a non-trivial
    expression (on which it didn't work at the time of writing).
    """

    actx = actx_factory()

    target_order = 4

    from meshmode.mesh.generation import generate_regular_rect_mesh
    mesh = generate_regular_rect_mesh(
            a=(-0.5,)*ambient_dim, b=(0.5,)*ambient_dim,
            n=(8,)*ambient_dim, order=1)
    discr = DiscretizationCollection(actx, mesh, order=target_order)

    # {{{ test with a relative norm

    from grudge.dof_desc import DD_VOLUME
    dd = DD_VOLUME
    sym_x = sym.make_sym_array("y", ambient_dim, dd=dd)
    sym_y = sym.make_sym_array("y", ambient_dim, dd=dd)

    sym_norm_y = sym.norm(2, sym_y, dd=dd)
    sym_norm_d = sym.norm(2, sym_x - sym_y, dd=dd)
    sym_op = sym_norm_d / sym_norm_y
    logger.info("%s", sym.pretty(sym_op))

    # FIXME: this shouldn't raise a RuntimeError
    with pytest.raises(RuntimeError):
        bind(discr, sym_op)(actx, x=1.0, y=discr.discr_from_dd(dd).nodes())

    # }}}

    # {{{ test with repeated mass inverses

    sym_minv_y = sym.cse(sym.InverseMassOperator()(sym_y), "minv_y")

    sym_u = make_obj_array([0.5 * sym.Ones(dd), 0.0, 0.0])[:ambient_dim]
    sym_div_u = sum(d(u) for d, u in zip(sym.nabla(ambient_dim), sym_u))

    sym_op = sym.MassOperator(dd)(sym_u) \
            + sym.MassOperator(dd)(sym_minv_y * sym_div_u)
    logger.info("%s", sym.pretty(sym_op))

    # FIXME: this shouldn't raise a RuntimeError either
    bind(discr, sym_op)(actx, y=discr.discr_from_dd(dd).nodes())

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
