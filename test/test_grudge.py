from __future__ import division, absolute_import, print_function

__copyright__ = "Copyright (C) 2015 Andreas Kloeckner"

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

import pyopencl as cl
import pyopencl.array
import pyopencl.clmath

from pytools.obj_array import join_fields, make_obj_array
from grudge import sym, bind, DGDiscretizationWithBoundaries

import pytest
from pyopencl.tools import (  # noqa
        pytest_generate_tests_for_pyopencl
        as pytest_generate_tests)

import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


# {{{ inverse metric

@pytest.mark.parametrize("dim", [2, 3])
def test_inverse_metric(ctx_factory, dim):
    cl_ctx = ctx_factory()
    queue = cl.CommandQueue(cl_ctx)

    from meshmode.mesh.generation import generate_regular_rect_mesh
    mesh = generate_regular_rect_mesh(a=(-0.5,)*dim, b=(0.5,)*dim,
            n=(6,)*dim, order=4)

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

    discr = DGDiscretizationWithBoundaries(cl_ctx, mesh, order=4)

    sym_op = (
            sym.forward_metric_derivative_mat(mesh.dim)
            .dot(
                sym.inverse_metric_derivative_mat(mesh.dim)
                )
            .reshape(-1))

    op = bind(discr, sym_op)
    mat = op(queue).reshape(mesh.dim, mesh.dim)

    for i in range(mesh.dim):
        for j in range(mesh.dim):
            tgt = 1 if i == j else 0

            err = np.max(np.abs((mat[i, j] - tgt).get(queue=queue)))
            logger.info("error[%d, %d]: %.5e", i, j, err)
            assert err < 1.0e-12, (i, j, err)

# }}}


# {{{ mass operator trig integration

@pytest.mark.parametrize("ambient_dim", [1, 2, 3])
@pytest.mark.parametrize("quad_tag", [sym.QTAG_NONE, "OVSMP"])
def test_mass_mat_trig(ctx_factory, ambient_dim, quad_tag):
    """Check the integral of some trig functions on an interval using the mass
    matrix.
    """
    cl_ctx = ctx_factory()
    queue = cl.CommandQueue(cl_ctx)

    nelements = 17
    order = 4

    a = -4.0 * np.pi
    b = +9.0 * np.pi
    true_integral = 13*np.pi/2 * (b - a)**(ambient_dim - 1)

    from meshmode.discretization.poly_element import QuadratureSimplexGroupFactory
    dd_quad = sym.DOFDesc(sym.DTAG_VOLUME_ALL, quad_tag)
    if quad_tag is sym.QTAG_NONE:
        quad_tag_to_group_factory = {}
    else:
        quad_tag_to_group_factory = {
                quad_tag: QuadratureSimplexGroupFactory(order=2*order)
                }

    from meshmode.mesh.generation import generate_regular_rect_mesh
    mesh = generate_regular_rect_mesh(
            a=(a,)*ambient_dim, b=(b,)*ambient_dim,
            n=(nelements,)*ambient_dim, order=1)
    discr = DGDiscretizationWithBoundaries(cl_ctx, mesh, order=order,
            quad_tag_to_group_factory=quad_tag_to_group_factory)

    def _get_variables_on(dd):
        sym_f = sym.var("f", dd=dd)
        sym_x = sym.nodes(ambient_dim, dd=dd)
        sym_ones = sym.Ones(dd)

        return sym_f, sym_x, sym_ones

    sym_f, sym_x, sym_ones = _get_variables_on(sym.DD_VOLUME)
    f_volm = bind(discr, sym.cos(sym_x[0])**2)(queue).get()
    ones_volm = bind(discr, sym_ones)(queue).get()

    sym_f, sym_x, sym_ones = _get_variables_on(dd_quad)
    f_quad = bind(discr, sym.cos(sym_x[0])**2)(queue)
    ones_quad = bind(discr, sym_ones)(queue)

    mass_op = bind(discr, sym.MassOperator(dd_quad, sym.DD_VOLUME)(sym_f))

    num_integral_1 = np.dot(ones_volm, mass_op(queue, f=f_quad).get())
    err_1 = abs(num_integral_1 - true_integral)
    assert err_1 < 5.0e-10, err_1

    num_integral_2 = np.dot(f_volm, mass_op(queue, f=ones_quad).get())
    err_2 = abs(num_integral_2 - true_integral)
    assert err_2 < 5.0e-10, err_2

    if quad_tag is sym.QTAG_NONE:
        # NOTE: `integral` always makes a square mass matrix and
        # `QuadratureSimplexGroupFactory` does not have a `mass_matrix` method.
        num_integral_3 = bind(discr,
                sym.integral(sym_f, dd=dd_quad))(queue, f=f_quad)
        err_3 = abs(num_integral_3 - true_integral)
        assert err_3 < 5.0e-10, err_3

# }}}


# {{{ mass operator surface area

def _ellipse_surface_area(radius, aspect_ratio):
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.ellipe.html
    eccentricity = 1.0 - (1/aspect_ratio)**2

    if abs(aspect_ratio - 2.0) < 1.0e-14:
        # NOTE: hardcoded value so we don't need scipy for the test
        ellip_e = 1.2110560275684594
    else:
        from scipy.special import ellipe
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
def test_mass_surface_area(ctx_factory, name):
    cl_ctx = cl.create_some_context()
    queue = cl.CommandQueue(cl_ctx)

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
        discr = DGDiscretizationWithBoundaries(cl_ctx, mesh, order=builder.order)
        volume_discr = discr.discr_from_dd(sym.DD_VOLUME)

        x = discr.discr_from_dd("vol").nodes().get(queue)
        logger.info("nnodes:    %d", volume_discr.nnodes)
        logger.info("nelements: %d", volume_discr.mesh.nelements)
        logger.info("bbox:      %s",
                [(np.min(x[n]), np.max(x[n])) for n in range(x.shape[0])])

        # {{{ compute surface area

        dd = sym.DD_VOLUME
        sym_op = sym.NodalSum(dd)(sym.MassOperator(dd, dd)(sym.Ones(dd)))
        approx_surface_area = bind(discr, sym_op)(queue)

        logger.info("surface: got {:.5e} / expected {:.5e}".format(
            approx_surface_area, surface_area))
        area_error = abs(approx_surface_area - surface_area) / abs(surface_area)

        # }}}

        h_max = bind(discr, sym.h_max_from_volume(discr.ambient_dim, dd=dd))(queue)
        eoc.add_data_point(h_max, area_error)

    # }}}

    logger.info("surface area error\n%s", str(eoc))

    assert eoc.max_error() < 1.0e-14 \
            or eoc.order_estimate() >= (builder.order - 1.0)

# }}}


# {{{ surface mass inverse

@pytest.mark.parametrize("name", ["2-1-ellipse", "spheroid"])
def test_surface_mass_operator_inverse(ctx_factory, name):
    cl_ctx = cl.create_some_context()
    queue = cl.CommandQueue(cl_ctx)

    # {{{ cases

    if name == "2-1-ellipse":
        from mesh_data import EllipseMeshBuilder
        builder = EllipseMeshBuilder(radius=3.1, aspect_ratio=2.0)
        surface_area = _ellipse_surface_area(builder.radius, builder.aspect_ratio)
    elif name == "spheroid":
        from mesh_data import SpheroidMeshBuilder
        builder = SpheroidMeshBuilder()
        surface_area = _spheroid_surface_area(builder.radius, builder.aspect_ratio)
    else:
        raise ValueError("unknown geometry name: %s" % name)

    # }}}

    # {{{ convergence

    from pytools.convergence import EOCRecorder
    eoc = EOCRecorder()

    for resolution in builder.resolutions:
        mesh = builder.get_mesh(resolution, builder.mesh_order)
        discr = DGDiscretizationWithBoundaries(cl_ctx, mesh, order=builder.order)
        volume_discr = discr.discr_from_dd(sym.DD_VOLUME)

        x = discr.discr_from_dd("vol").nodes().get(queue)
        logger.info("nnodes:    %d", volume_discr.nnodes)
        logger.info("nelements: %d", volume_discr.mesh.nelements)
        logger.info("bbox:      %s",
                [(np.min(x[n]), np.max(x[n])) for n in range(x.shape[0])])

        # {{{ compute inverse mass

        dd = sym.DD_VOLUME
        sym_f = sym.cos(4.0 * sym.nodes(mesh.ambient_dim, dd)[0])
        sym_op = sym.InverseMassOperator(dd, dd)(
                sym.MassOperator(dd, dd)(sym.var("f")))

        f = bind(discr, sym_f)(queue)
        f_inv = bind(discr, sym_op)(queue, f=f)

        logger.info("inverse: got {:.5e} / expected {:.5e}".format(
            cl.array.max(f - f_inv).get(queue), 1.0))
        inv_error = la.norm(f.get(queue) - f_inv.get(queue)) \
                / la.norm(f.get(queue))

        # }}}

        h_max = bind(discr, sym.h_max_from_volume(discr.ambient_dim, dd=dd))(queue)
        eoc.add_data_point(h_max, inv_error)

    # }}}

    logger.info("inverse mass error\n%s", str(eoc))

    assert eoc.max_error() < 5.0e-09 \
            or eoc.order_estimate() >= (builder.order - 1.0)

# }}}


# {{{ surface face normal orthogonality

def _avg_face_normal(dd, ambient_dim, dim=None):
    dd = sym.as_dofdesc(dd)
    assert dd.is_trace()

    if dim is None:
        dim = ambient_dim - 1

    face_normal_i = sym.normal(dd, ambient_dim, dim=dim)
    face_normal_e = sym.OppositeInteriorFaceSwap()(face_normal_i)
    face_normal = (face_normal_i - face_normal_e) / 2.0

    return sym.cse(
            face_normal / sym.sqrt(face_normal.dot(face_normal)),
            "avg_face_normal",
            sym.cse_scope.DISCRETIZATION)


@pytest.mark.parametrize("mesh_name", ["2-1-ellipse", "spheroid"])
def test_face_normal_surface(ctx_factory, mesh_name):
    """Check that face normals are orthogonal to the surface normal"""

    cl_ctx = ctx_factory()
    queue = cl.CommandQueue(cl_ctx)

    # {{{ geometry

    if mesh_name == "2-1-ellipse":
        from mesh_data import EllipseMeshBuilder
        builder = EllipseMeshBuilder(radius=3.1, aspect_ratio=2.0)
    elif mesh_name == "spheroid":
        from mesh_data import SpheroidMeshBuilder
        builder = SpheroidMeshBuilder()
    else:
        raise ValueError("unknown mesh name: %s" % mesh_name)

    mesh = builder.get_mesh(builder.resolutions[1], builder.mesh_order)
    discr = DGDiscretizationWithBoundaries(cl_ctx, mesh, order=builder.order)

    # }}}

    # {{{ symbolic

    dv = sym.DD_VOLUME
    df = sym.as_dofdesc(sym.FACE_RESTR_INTERIOR)

    ambient_dim = mesh.ambient_dim
    surf_dim = mesh.dim
    face_dim = surf_dim - 1

    sym_surf_normal = sym.interp(dv, df)(
            sym.surface_normal(ambient_dim, dim=surf_dim, dd=dv).as_vector()
            )
    sym_surf_normal = sym_surf_normal / sym.sqrt(sum(sym_surf_normal**2))

    sym_face_normal_i = sym.normal(df, ambient_dim, dim=face_dim)
    sym_face_normal_e = sym.OppositeInteriorFaceSwap(df)(sym_face_normal_i)

    sym_face_normal_avg = _avg_face_normal(df, ambient_dim, dim=face_dim)
    sym_face_normal_op = sym.OppositeInteriorFaceSwap(df)(sym_face_normal_avg)

    if mesh.ambient_dim == 3:
        # NOTE: there's only one face tangent in 3d
        sym_face_tangent = (sym.pseudoscalar(ambient_dim, face_dim, dd=df)
                / sym.area_element(ambient_dim, face_dim, dd=df)).as_vector()

    # }}}

    # {{{ checks

    rtol = 1.0e-14

    surf_normal = bind(discr, sym_surf_normal)(queue)

    face_normal_i = bind(discr, sym_face_normal_i)(queue)
    face_normal_e = bind(discr, sym_face_normal_e)(queue)

    face_normal_avg = bind(discr, sym_face_normal_avg)(queue)
    face_normal_op = bind(discr, sym_face_normal_op)(queue)

    # check interpolated surface normal is orthogonal to face normal
    error = la.norm(surf_normal.dot(face_normal_i).get(queue), np.inf)
    logger.info("error[n_dot_i]:    %.5e", error)
    assert error < rtol

    # check angle between two neighboring elements
    error = la.norm(face_normal_i.dot(face_normal_e).get(queue) + 1.0, np.inf)
    logger.info("error[i_dot_e]:    %.5e", error)
    assert error > rtol

    # check uniqueness of normal on the two sides
    error = la.norm(sum(face_normal_avg + face_normal_op).get(queue), np.inf)
    logger.info("error[a_plus_o]:   %.5e", error)
    assert error < rtol

    # check orthogonality with face tangent
    if ambient_dim == 3:
        face_tangent = bind(discr, sym_face_tangent)(queue)

        error = la.norm(face_tangent.dot(face_normal_avg).get(queue), np.inf)
        logger.info("error[t_dot_avg]:  %.5e", error)
        assert error < 5 * rtol

    # }}}

# }}}


# {{{ diff operator

@pytest.mark.parametrize("dim", [1, 2, 3])
def test_tri_diff_mat(ctx_factory, dim, order=4):
    """Check differentiation matrix along the coordinate axes on a disk

    Uses sines as the function to differentiate.
    """

    cl_ctx = ctx_factory()
    queue = cl.CommandQueue(cl_ctx)

    from meshmode.mesh.generation import generate_regular_rect_mesh

    from pytools.convergence import EOCRecorder
    axis_eoc_recs = [EOCRecorder() for axis in range(dim)]

    for n in [10, 20]:
        mesh = generate_regular_rect_mesh(a=(-0.5,)*dim, b=(0.5,)*dim,
                n=(n,)*dim, order=4)

        discr = DGDiscretizationWithBoundaries(cl_ctx, mesh, order=4)
        nabla = sym.nabla(dim)

        for axis in range(dim):
            x = sym.nodes(dim)

            f = bind(discr, sym.sin(3*x[axis]))(queue)
            df = bind(discr, 3*sym.cos(3*x[axis]))(queue)

            sym_op = nabla[axis](sym.var("f"))
            bound_op = bind(discr, sym_op)
            df_num = bound_op(queue, f=f)

            linf_error = la.norm((df_num-df).get(), np.Inf)
            axis_eoc_recs[axis].add_data_point(1/n, linf_error)

    for axis, eoc_rec in enumerate(axis_eoc_recs):
        logger.info("axis %d\n%s", axis, eoc_rec)
        assert eoc_rec.order_estimate() >= order

# }}}


# {{{ divergence theorem

def test_2d_gauss_theorem(ctx_factory):
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
    mesh = from_meshpy(mesh_info, order=1)

    cl_ctx = ctx_factory()
    queue = cl.CommandQueue(cl_ctx)

    discr = DGDiscretizationWithBoundaries(cl_ctx, mesh, order=2)

    def f(x):
        return join_fields(
                sym.sin(3*x[0])+sym.cos(3*x[1]),
                sym.sin(2*x[0])+sym.cos(x[1]))

    gauss_err = bind(discr,
            sym.integral((
                sym.nabla(2) * f(sym.nodes(2))
                ).sum())
            -  # noqa: W504
            sym.integral(
                sym.interp("vol", sym.BTAG_ALL)(f(sym.nodes(2)))
                .dot(sym.normal(sym.BTAG_ALL, 2)),
                dd=sym.BTAG_ALL)
            )(queue)

    assert abs(gauss_err) < 1e-13

# }}}


# {{{ models: advection

@pytest.mark.parametrize(("mesh_name", "mesh_pars"), [
    ("segment", [8, 16, 32]),
    ("disk", [0.1, 0.05]),
    ("rect2", [4, 8]),
    ("rect3", [4, 6]),
    ])
@pytest.mark.parametrize("op_type", ["strong", "weak"])
@pytest.mark.parametrize("flux_type", ["central"])
@pytest.mark.parametrize("order", [3, 4, 5])
# test: 'test_convergence_advec(cl._csc, "disk", [0.1, 0.05], "strong", "upwind", 3)'
def test_convergence_advec(ctx_factory, mesh_name, mesh_pars, op_type, flux_type,
        order, visualize=False):
    """Test whether 2D advection actually converges"""

    cl_ctx = ctx_factory()
    queue = cl.CommandQueue(cl_ctx)

    from pytools.convergence import EOCRecorder
    eoc_rec = EOCRecorder()

    for mesh_par in mesh_pars:
        if mesh_name == "segment":
            from meshmode.mesh.generation import generate_box_mesh
            mesh = generate_box_mesh(
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
            dim = int(mesh_name[4:])
            from meshmode.mesh.generation import generate_regular_rect_mesh
            mesh = generate_regular_rect_mesh(a=(-0.5,)*dim, b=(0.5,)*dim,
                    n=(mesh_par,)*dim, order=4)

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
                    + sym.var("t", sym.DD_SCALAR)*norm_v)

        from grudge.models.advection import (
                StrongAdvectionOperator, WeakAdvectionOperator)
        discr = DGDiscretizationWithBoundaries(cl_ctx, mesh, order=order)
        op_class = {
                "strong": StrongAdvectionOperator,
                "weak": WeakAdvectionOperator,
                }[op_type]
        op = op_class(v,
                inflow_u=u_analytic(sym.nodes(dim, sym.BTAG_ALL)),
                flux_type=flux_type)

        bound_op = bind(discr, op.sym_operator())

        u = bind(discr, u_analytic(sym.nodes(dim)))(queue, t=0)

        def rhs(t, u):
            return bound_op(queue, t=t, u=u)

        if dim == 3:
            final_time = 0.1
        else:
            final_time = 0.2

        h_max = bind(discr, sym.h_max_from_volume(discr.ambient_dim))(queue)
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
                queue, t=last_t, u=last_u)
        logger.info("h_max %.5e error %.5e", h_max, error_l2)
        eoc_rec.add_data_point(h_max, error_l2)

    logger.info("\n%s", eoc_rec.pretty_print(
        abscissa_label="h",
        error_label="L2 Error"))

    assert eoc_rec.order_estimate() > order

# }}}


# {{{ models: maxwell

@pytest.mark.parametrize("order", [3, 4, 5])
def test_convergence_maxwell(ctx_factory,  order):
    """Test whether 3D Maxwell's actually converges"""

    cl_ctx = ctx_factory()
    queue = cl.CommandQueue(cl_ctx)

    from pytools.convergence import EOCRecorder
    eoc_rec = EOCRecorder()

    dims = 3
    ns = [4, 6, 8]
    for n in ns:
        from meshmode.mesh.generation import generate_regular_rect_mesh
        mesh = generate_regular_rect_mesh(
                a=(0.0,)*dims,
                b=(1.0,)*dims,
                n=(n,)*dims)

        discr = DGDiscretizationWithBoundaries(cl_ctx, mesh, order=order)

        epsilon = 1
        mu = 1

        from grudge.models.em import get_rectangular_cavity_mode
        sym_mode = get_rectangular_cavity_mode(1, (1, 2, 2))

        analytic_sol = bind(discr, sym_mode)
        fields = analytic_sol(queue, t=0, epsilon=epsilon, mu=mu)

        from grudge.models.em import MaxwellOperator
        op = MaxwellOperator(epsilon, mu, flux_type=0.5, dimensions=dims)
        op.check_bc_coverage(mesh)
        bound_op = bind(discr, op.sym_operator())

        def rhs(t, w):
            return bound_op(queue, t=t, w=w)

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

        sol = analytic_sol(queue, mu=mu, epsilon=epsilon, t=step * dt)
        vals = [norm(queue, u=(esc[i] - sol[i])) / norm(queue, u=sol[i]) for i in range(5)] # noqa E501
        total_error = sum(vals)
        eoc_rec.add_data_point(1.0/n, total_error)

    logger.info("\n%s", eoc_rec.pretty_print(
        abscissa_label="h",
        error_label="L2 Error"))

    assert eoc_rec.order_estimate() > order

# }}}


# {{{ models: variable coefficient advection oversampling

@pytest.mark.parametrize("order", [2, 3, 4])
def test_improvement_quadrature(ctx_factory, order):
    """Test whether quadrature improves things and converges"""
    from meshmode.mesh.generation import generate_regular_rect_mesh
    from grudge.models.advection import VariableCoefficientAdvectionOperator
    from pytools.convergence import EOCRecorder
    from meshmode.discretization.poly_element import QuadratureSimplexGroupFactory

    cl_ctx = ctx_factory()
    queue = cl.CommandQueue(cl_ctx)

    dims = 2
    sym_nds = sym.nodes(dims)
    advec_v = join_fields(-1*sym_nds[1], sym_nds[0])

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
            mesh = generate_regular_rect_mesh(
                a=(-0.5,)*dims,
                b=(0.5,)*dims,
                n=(n,)*dims,
                order=order)

            if use_quad:
                quad_tag_to_group_factory = {
                    "product": QuadratureSimplexGroupFactory(order=4*order)
                    }
            else:
                quad_tag_to_group_factory = {"product": None}

            discr = DGDiscretizationWithBoundaries(cl_ctx, mesh, order=order,
                    quad_tag_to_group_factory=quad_tag_to_group_factory)

            bound_op = bind(discr, op.sym_operator())
            fields = bind(discr, gaussian_mode())(queue, t=0)
            norm = bind(discr, sym.norm(2, sym.var("u")))

            esc = bound_op(queue, u=fields)
            total_error = norm(queue, u=esc)
            eoc_rec.add_data_point(1.0/n, total_error)

        logger.info("\n%s", eoc_rec.pretty_print(
            abscissa_label="h",
            error_label="L2 Error"))

        return eoc_rec.order_estimate(), np.array([x[1] for x in eoc_rec.history])

    eoc, errs = conv_test("no quadrature", False)
    q_eoc, q_errs = conv_test("with quadrature", True)

    assert q_eoc > eoc
    assert (q_errs < errs).all()
    assert q_eoc > order

# }}}


# {{{ foreign points

def test_foreign_points(ctx_factory):
    pytest.importorskip("sumpy")
    import sumpy.point_calculus as pc

    cl_ctx = ctx_factory()
    queue = cl.CommandQueue(cl_ctx)

    dim = 2
    cp = pc.CalculusPatch(np.zeros(dim))

    from grudge.discretization import PointsDiscretization
    pdiscr = PointsDiscretization(cl.array.to_device(queue, cp.points))

    bind(pdiscr, sym.nodes(dim)**2)(queue)

# }}}


# {{{ operator collector determinism

def test_op_collector_order_determinism():
    class TestOperator(sym.Operator):

        def __init__(self):
            sym.Operator.__init__(self, sym.DD_VOLUME, sym.DD_VOLUME)

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

def test_bessel(ctx_factory):
    cl_ctx = ctx_factory()
    queue = cl.CommandQueue(cl_ctx)

    dims = 2

    from meshmode.mesh.generation import generate_regular_rect_mesh
    mesh = generate_regular_rect_mesh(
            a=(0.1,)*dims,
            b=(1.0,)*dims,
            n=(8,)*dims)

    discr = DGDiscretizationWithBoundaries(cl_ctx, mesh, order=3)

    nodes = sym.nodes(dims)
    r = sym.cse(sym.sqrt(nodes[0]**2 + nodes[1]**2))

    # https://dlmf.nist.gov/10.6.1
    n = 3
    bessel_zero = (
            sym.bessel_j(n+1, r)
            + sym.bessel_j(n-1, r)
            - 2*n/r * sym.bessel_j(n, r))

    z = bind(discr, sym.norm(2, bessel_zero))(queue)

    assert z < 1e-15

# }}}


# {{{ function symbol

def test_external_call(ctx_factory):
    cl_ctx = ctx_factory()
    queue = cl.CommandQueue(cl_ctx)

    def double(queue, x):
        return 2 * x

    from meshmode.mesh.generation import generate_regular_rect_mesh

    dims = 2

    mesh = generate_regular_rect_mesh(a=(0,) * dims, b=(1,) * dims, n=(4,) * dims)
    discr = DGDiscretizationWithBoundaries(cl_ctx, mesh, order=1)

    ones = sym.Ones(sym.DD_VOLUME)
    op = (
            ones * 3
            + sym.FunctionSymbol("double")(ones))

    from grudge.function_registry import (
            base_function_registry, register_external_function)

    freg = register_external_function(
            base_function_registry,
            "double",
            implementation=double,
            dd=sym.DD_VOLUME)

    bound_op = bind(discr, op, function_registry=freg)

    result = bound_op(queue, double=double)
    assert (result == 5).get().all()


@pytest.mark.parametrize("array_type", ["scalar", "vector"])
def test_function_symbol_array(ctx_factory, array_type):
    ctx = ctx_factory()
    queue = cl.CommandQueue(ctx)

    from meshmode.mesh.generation import generate_regular_rect_mesh
    dim = 2
    mesh = generate_regular_rect_mesh(
            a=(-0.5,)*dim, b=(0.5,)*dim,
            n=(8,)*dim, order=4)
    discr = DGDiscretizationWithBoundaries(ctx, mesh, order=4)
    nnodes = discr.discr_from_dd(sym.DD_VOLUME).nnodes

    import pyopencl.clrandom        # noqa: F401
    if array_type == "scalar":
        sym_x = sym.var("x")
        x = cl.clrandom.rand(queue, nnodes, dtype=np.float)
    elif array_type == "vector":
        sym_x = sym.make_sym_array("x", dim)
        x = make_obj_array([
            cl.clrandom.rand(queue, nnodes, dtype=np.float)
            for _ in range(dim)
            ])
    else:
        raise ValueError("unknown array type")

    norm = bind(discr, sym.norm(2, sym_x))(queue, x=x)
    assert isinstance(norm, float)

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
