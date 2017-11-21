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


import numpy as np  # noqa
import numpy.linalg as la  # noqa
import pyopencl as cl  # noqa
import pyopencl.array  # noqa
import pyopencl.clmath  # noqa

import pytest  # noqa

from pyopencl.tools import (  # noqa
        pytest_generate_tests_for_pyopencl as pytest_generate_tests)

import logging
logger = logging.getLogger(__name__)

from grudge import sym, bind, DGDiscretizationWithBoundaries


@pytest.mark.parametrize("dim", [2, 3])
def test_inverse_metric(ctx_factory, dim):
    cl_ctx = cl.create_some_context()
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
            print(i, j, err)
            assert err < 1e-12, (i, j, err)


def test_1d_mass_mat_trig(ctx_factory):
    """Check the integral of some trig functions on an interval using the mass
    matrix
    """

    cl_ctx = cl.create_some_context()
    queue = cl.CommandQueue(cl_ctx)

    from meshmode.mesh.generation import generate_regular_rect_mesh

    mesh = generate_regular_rect_mesh(a=(-4*np.pi,), b=(9*np.pi,),
            n=(17,), order=1)

    discr = DGDiscretizationWithBoundaries(cl_ctx, mesh, order=8)

    x = sym.nodes(1)
    f = bind(discr, sym.cos(x[0])**2)(queue)
    ones = bind(discr, sym.Ones(sym.DD_VOLUME))(queue)

    mass_op = bind(discr, sym.MassOperator()(sym.var("f")))

    num_integral_1 = np.dot(ones.get(), mass_op(queue, f=f))
    num_integral_2 = np.dot(f.get(), mass_op(queue, f=ones))
    num_integral_3 = bind(discr, sym.integral(sym.var("f")))(queue, f=f)

    true_integral = 13*np.pi/2
    err_1 = abs(num_integral_1-true_integral)
    err_2 = abs(num_integral_2-true_integral)
    err_3 = abs(num_integral_3-true_integral)

    assert err_1 < 1e-10
    assert err_2 < 1e-10
    assert err_3 < 1e-10


@pytest.mark.parametrize("dim", [1, 2, 3])
def test_tri_diff_mat(ctx_factory, dim, order=4):
    """Check differentiation matrix along the coordinate axes on a disk

    Uses sines as the function to differentiate.
    """

    cl_ctx = cl.create_some_context()
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
        print(axis)
        print(eoc_rec)
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

    cl_ctx = cl.create_some_context()
    queue = cl.CommandQueue(cl_ctx)

    discr = DGDiscretizationWithBoundaries(cl_ctx, mesh, order=2)

    def f(x):
        return sym.join_fields(
                sym.sin(3*x[0])+sym.cos(3*x[1]),
                sym.sin(2*x[0])+sym.cos(x[1]))

    gauss_err = bind(discr,
            sym.integral((
                sym.nabla(2) * f(sym.nodes(2))
                ).sum())
            -
            sym.integral(
                sym.interp("vol", sym.BTAG_ALL)(f(sym.nodes(2)))
                .dot(sym.normal(sym.BTAG_ALL, 2)),
                dd=sym.BTAG_ALL)
            )(queue)

    assert abs(gauss_err) < 1e-13


@pytest.mark.parametrize(("mesh_name", "mesh_pars"), [
    ("disk", [0.1, 0.05]),
    ("rect2", [4, 8]),
    ("rect3", [4, 6]),
    ])
@pytest.mark.parametrize("op_type", ["strong", "weak"])
@pytest.mark.parametrize("flux_type", ["upwind"])
@pytest.mark.parametrize("order", [3, 4, 5])
# test: 'test_convergence_advec(cl._csc, "disk", [0.1, 0.05], "strong", "upwind", 3)'
def test_convergence_advec(ctx_factory, mesh_name, mesh_pars, op_type, flux_type,
        order, visualize=False):
    """Test whether 2D advection actually converges"""

    cl_ctx = cl.create_some_context()
    queue = cl.CommandQueue(cl_ctx)

    from pytools.convergence import EOCRecorder
    eoc_rec = EOCRecorder()

    for mesh_par in mesh_pars:
        if mesh_name == "disk":
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
            h = np.sqrt(mesh_par)
            dim = 2
            dt_factor = 4

        elif mesh_name.startswith("rect"):
            dim = int(mesh_name[4:])
            from meshmode.mesh.generation import generate_regular_rect_mesh
            mesh = generate_regular_rect_mesh(a=(-0.5,)*dim, b=(0.5,)*dim,
                    n=(mesh_par,)*dim, order=4)

            h = 1/mesh_par
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
        #print(bound_op)
        #1/0

        u = bind(discr, u_analytic(sym.nodes(dim)))(queue, t=0)

        def rhs(t, u):
            return bound_op(queue, t=t, u=u)

        if dim == 3:
            final_time = 0.1
        else:
            final_time = 0.2

        dt = dt_factor * h/order**2
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
                print(event.t)

                last_t = event.t
                last_u = event.state_component

                if visualize:
                    vis.write_vtk_file("fld-%s-%04d.vtu" % (mesh_par, step),
                            [("u", event.state_component)])

        error_l2 = bind(discr,
            sym.norm(2, sym.var("u")-u_analytic(sym.nodes(dim))))(
                queue, t=last_t, u=last_u)
        print(h, error_l2)
        eoc_rec.add_data_point(h, error_l2)

    print(eoc_rec.pretty_print(abscissa_label="h",
            error_label="L2 Error"))

    assert eoc_rec.order_estimate() > order


@pytest.mark.parametrize("order", [3, 4, 5])
# test: 'test_convergence_advec(cl._csc, "disk", [0.1, 0.05], "strong", "upwind", 3)'
def test_convergence_maxwell(ctx_factory,  order, visualize=False):
    """Test whether 3D maxwells actually converges"""

    cl_ctx = cl.create_some_context()
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

        discr = Discretization(cl_ctx, mesh, order=order)

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

        print("dt=%g nsteps=%d" % (dt, nsteps))

        norm = bind(discr, sym.norm(2, sym.var("u")))

        step = 0
        for event in dt_stepper.run(t_end=final_t):
            if isinstance(event, dt_stepper.StateComputed):
                assert event.component_id == "w"
                esc = event.state_component

                step += 1
                print(step)

        sol = analytic_sol(queue, mu=mu, epsilon=epsilon, t=step * dt)
        vals = [norm(queue, u=(esc[i] - sol[i])) / norm(queue, u=sol[i]) for i in range(5)] # noqa E501
        total_error = sum(vals)
        eoc_rec.add_data_point(1.0/n, total_error)

    print(eoc_rec.pretty_print(abscissa_label="h",
            error_label="L2 Error"))

    assert eoc_rec.order_estimate() > order


def test_foreign_points(ctx_factory):
    pytest.importorskip("sumpy")
    import sumpy.point_calculus as pc

    cl_ctx = cl.create_some_context()
    queue = cl.CommandQueue(cl_ctx)

    dim = 2
    cp = pc.CalculusPatch(np.zeros(dim))

    from grudge.discretization import PointsDiscretization
    pdiscr = PointsDiscretization(cl.array.to_device(queue, cp.points))

    bind(pdiscr, sym.nodes(dim)**2)(queue)


# You can test individual routines by typing
# $ python test_grudge.py 'test_routine()'

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        from py.test.cmdline import main
        main([__file__])

# vim: fdm=marker
