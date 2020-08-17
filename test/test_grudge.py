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
import pytest  # noqa

from pytools.obj_array import flat_obj_array, make_obj_array

import pyopencl as cl

from meshmode.array_context import PyOpenCLArrayContext
from meshmode.dof_array import unflatten, flatten, flat_norm

from pyopencl.tools import (  # noqa
        pytest_generate_tests_for_pyopencl as pytest_generate_tests)

import logging

logger = logging.getLogger(__name__)
logging.basicConfig()
logger.setLevel(logging.INFO)

from grudge import sym, bind, DGDiscretizationWithBoundaries


@pytest.mark.parametrize("dim", [2, 3])
def test_inverse_metric(ctx_factory, dim):
    cl_ctx = ctx_factory()
    queue = cl.CommandQueue(cl_ctx)
    actx = PyOpenCLArrayContext(queue)

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

    discr = DGDiscretizationWithBoundaries(actx, mesh, order=4)

    sym_op = (
            sym.forward_metric_derivative_mat(mesh.dim)
            .dot(
                sym.inverse_metric_derivative_mat(mesh.dim)
                )
            .reshape(-1))

    op = bind(discr, sym_op)
    mat = op(actx).reshape(mesh.dim, mesh.dim)

    for i in range(mesh.dim):
        for j in range(mesh.dim):
            tgt = 1 if i == j else 0

            err = flat_norm(mat[i, j] - tgt, np.inf)
            logger.info("error[%d, %d]: %.5e", i, j, err)
            assert err < 1.0e-12, (i, j, err)


@pytest.mark.parametrize("ambient_dim", [1, 2, 3])
@pytest.mark.parametrize("quad_tag", [sym.QTAG_NONE, "OVSMP"])
def test_mass_mat_trig(ctx_factory, ambient_dim, quad_tag):
    """Check the integral of some trig functions on an interval using the mass
    matrix.
    """
    cl_ctx = ctx_factory()
    queue = cl.CommandQueue(cl_ctx)
    actx = PyOpenCLArrayContext(queue)

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
    discr = DGDiscretizationWithBoundaries(actx, mesh, order=order,
            quad_tag_to_group_factory=quad_tag_to_group_factory)

    def _get_variables_on(dd):
        sym_f = sym.var("f", dd=dd)
        sym_x = sym.nodes(ambient_dim, dd=dd)
        sym_ones = sym.Ones(dd)

        return sym_f, sym_x, sym_ones

    sym_f, sym_x, sym_ones = _get_variables_on(sym.DD_VOLUME)
    f_volm = actx.to_numpy(flatten(bind(discr, sym.cos(sym_x[0])**2)(actx)))
    ones_volm = actx.to_numpy(flatten(bind(discr, sym_ones)(actx)))

    sym_f, sym_x, sym_ones = _get_variables_on(dd_quad)
    f_quad = bind(discr, sym.cos(sym_x[0])**2)(actx)
    ones_quad = bind(discr, sym_ones)(actx)

    mass_op = bind(discr, sym.MassOperator(dd_quad, sym.DD_VOLUME)(sym_f))

    num_integral_1 = np.dot(ones_volm, actx.to_numpy(flatten(mass_op(f=f_quad))))
    err_1 = abs(num_integral_1 - true_integral)
    assert err_1 < 1e-9, err_1

    num_integral_2 = np.dot(f_volm, actx.to_numpy(flatten(mass_op(f=ones_quad))))
    err_2 = abs(num_integral_2 - true_integral)
    assert err_2 < 1.0e-9, err_2

    if quad_tag is sym.QTAG_NONE:
        # NOTE: `integral` always makes a square mass matrix and
        # `QuadratureSimplexGroupFactory` does not have a `mass_matrix` method.
        num_integral_3 = bind(discr,
                sym.integral(sym_f, dd=dd_quad))(f=f_quad)
        err_3 = abs(num_integral_3 - true_integral)
        assert err_3 < 5.0e-10, err_3


@pytest.mark.parametrize("dim", [1, 2, 3])
def test_tri_diff_mat(ctx_factory, dim, order=4):
    """Check differentiation matrix along the coordinate axes on a disk

    Uses sines as the function to differentiate.
    """

    cl_ctx = ctx_factory()
    queue = cl.CommandQueue(cl_ctx)
    actx = PyOpenCLArrayContext(queue)

    from meshmode.mesh.generation import generate_regular_rect_mesh

    from pytools.convergence import EOCRecorder
    axis_eoc_recs = [EOCRecorder() for axis in range(dim)]

    for n in [10, 20]:
        mesh = generate_regular_rect_mesh(a=(-0.5,)*dim, b=(0.5,)*dim,
                n=(n,)*dim, order=4)

        discr = DGDiscretizationWithBoundaries(actx, mesh, order=4)
        nabla = sym.nabla(dim)

        for axis in range(dim):
            x = sym.nodes(dim)

            f = bind(discr, sym.sin(3*x[axis]))(actx)
            df = bind(discr, 3*sym.cos(3*x[axis]))(actx)

            sym_op = nabla[axis](sym.var("f"))
            bound_op = bind(discr, sym_op)
            df_num = bound_op(f=f)

            linf_error = flat_norm(df_num-df, np.Inf)
            axis_eoc_recs[axis].add_data_point(1/n, linf_error)

    for axis, eoc_rec in enumerate(axis_eoc_recs):
        logger.info("axis %d\n%s", axis, eoc_rec)
        assert eoc_rec.order_estimate() >= order


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
    actx = PyOpenCLArrayContext(queue)

    discr = DGDiscretizationWithBoundaries(actx, mesh, order=2)

    def f(x):
        return flat_obj_array(
                sym.sin(3*x[0])+sym.cos(3*x[1]),
                sym.sin(2*x[0])+sym.cos(x[1]))

    gauss_err = bind(discr,
            sym.integral((
                sym.nabla(2) * f(sym.nodes(2))
                ).sum())
            -  # noqa: W504
            sym.integral(
                sym.project("vol", sym.BTAG_ALL)(f(sym.nodes(2)))
                .dot(sym.normal(sym.BTAG_ALL, 2)),
                dd=sym.BTAG_ALL)
            )(actx)

    assert abs(gauss_err) < 1e-13


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
    actx = PyOpenCLArrayContext(queue)

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
        discr = DGDiscretizationWithBoundaries(actx, mesh, order=order)
        op_class = {
                "strong": StrongAdvectionOperator,
                "weak": WeakAdvectionOperator,
                }[op_type]
        op = op_class(v,
                inflow_u=u_analytic(sym.nodes(dim, sym.BTAG_ALL)),
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

    assert eoc_rec.order_estimate() > order


@pytest.mark.parametrize("order", [3, 4, 5])
def test_convergence_maxwell(ctx_factory,  order):
    """Test whether 3D Maxwell's actually converges"""

    cl_ctx = ctx_factory()
    queue = cl.CommandQueue(cl_ctx)
    actx = PyOpenCLArrayContext(queue)

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

        discr = DGDiscretizationWithBoundaries(actx, mesh, order=order)

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


@pytest.mark.parametrize("order", [2, 3, 4])
def test_improvement_quadrature(ctx_factory, order):
    """Test whether quadrature improves things and converges"""
    from meshmode.mesh.generation import generate_regular_rect_mesh
    from grudge.models.advection import VariableCoefficientAdvectionOperator
    from pytools.convergence import EOCRecorder
    from meshmode.discretization.poly_element import QuadratureSimplexGroupFactory

    cl_ctx = ctx_factory()
    queue = cl.CommandQueue(cl_ctx)
    actx = PyOpenCLArrayContext(queue)

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

            discr = DGDiscretizationWithBoundaries(actx, mesh, order=order,
                    quad_tag_to_group_factory=quad_tag_to_group_factory)

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
    assert q_eoc > order


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


def test_bessel(ctx_factory):
    cl_ctx = ctx_factory()
    queue = cl.CommandQueue(cl_ctx)
    actx = PyOpenCLArrayContext(queue)

    dims = 2

    from meshmode.mesh.generation import generate_regular_rect_mesh
    mesh = generate_regular_rect_mesh(
            a=(0.1,)*dims,
            b=(1.0,)*dims,
            n=(8,)*dims)

    discr = DGDiscretizationWithBoundaries(actx, mesh, order=3)

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


def test_external_call(ctx_factory):
    cl_ctx = ctx_factory()
    queue = cl.CommandQueue(cl_ctx)
    actx = PyOpenCLArrayContext(queue)

    def double(queue, x):
        return 2 * x

    from meshmode.mesh.generation import generate_regular_rect_mesh

    dims = 2

    mesh = generate_regular_rect_mesh(a=(0,) * dims, b=(1,) * dims, n=(4,) * dims)
    discr = DGDiscretizationWithBoundaries(actx, mesh, order=1)

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

    result = bound_op(actx, double=double)
    assert actx.to_numpy(flatten(result) == 5).all()


@pytest.mark.parametrize("array_type", ["scalar", "vector"])
def test_function_symbol_array(ctx_factory, array_type):
    """Test if `FunctionSymbol` distributed properly over object arrays."""

    ctx = ctx_factory()
    queue = cl.CommandQueue(ctx)
    actx = PyOpenCLArrayContext(queue)

    from meshmode.mesh.generation import generate_regular_rect_mesh
    dim = 2
    mesh = generate_regular_rect_mesh(
            a=(-0.5,)*dim, b=(0.5,)*dim,
            n=(8,)*dim, order=4)
    discr = DGDiscretizationWithBoundaries(actx, mesh, order=4)
    volume_discr = discr.discr_from_dd(sym.DD_VOLUME)
    ndofs = sum(grp.ndofs for grp in volume_discr.groups)

    import pyopencl.clrandom        # noqa: F401
    if array_type == "scalar":
        sym_x = sym.var("x")
        x = unflatten(actx, volume_discr,
                cl.clrandom.rand(queue, ndofs, dtype=np.float))
    elif array_type == "vector":
        sym_x = sym.make_sym_array("x", dim)
        x = make_obj_array([
            unflatten(actx, volume_discr,
                cl.clrandom.rand(queue, ndofs, dtype=np.float))
            for _ in range(dim)
            ])
    else:
        raise ValueError("unknown array type")

    norm = bind(discr, sym.norm(2, sym_x))(x=x)
    assert isinstance(norm, float)


@pytest.mark.parametrize("p", [2, np.inf])
def test_norm_obj_array(ctx_factory, p):
    """Test :func:`grudge.symbolic.operators.norm` for object arrays."""

    ctx = ctx_factory()
    queue = cl.CommandQueue(ctx)
    actx = PyOpenCLArrayContext(queue)

    from meshmode.mesh.generation import generate_regular_rect_mesh
    dim = 2
    mesh = generate_regular_rect_mesh(
            a=(-0.5,)*dim, b=(0.5,)*dim,
            n=(8,)*dim, order=1)
    discr = DGDiscretizationWithBoundaries(actx, mesh, order=4)

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


def test_map_if(ctx_factory):
    """Test :meth:`grudge.symbolic.execution.ExecutionMapper.map_if` handling
    of scalar conditions.
    """

    ctx = ctx_factory()
    queue = cl.CommandQueue(ctx)
    actx = PyOpenCLArrayContext(queue)

    from meshmode.mesh.generation import generate_regular_rect_mesh
    dim = 2
    mesh = generate_regular_rect_mesh(
            a=(-0.5,)*dim, b=(0.5,)*dim,
            n=(8,)*dim, order=4)
    discr = DGDiscretizationWithBoundaries(actx, mesh, order=4)

    sym_if = sym.If(sym.Comparison(2.0, "<", 1.0e-14), 1.0, 2.0)
    bind(discr, sym_if)(actx)


# You can test individual routines by typing
# $ python test_grudge.py 'test_routine()'

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        from pytest import main
        main([__file__])

# vim: fdm=marker
