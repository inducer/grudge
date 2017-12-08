import pyopencl as cl
import numpy as np

from grudge import sym, bind, DGDiscretizationWithBoundaries
from pytools.obj_array import join_fields


def test_convergence_maxwell(order, visualize=False):
    """Test whether 3D maxwells actually converges"""

    cl_ctx = cl.create_some_context()
    queue = cl.CommandQueue(cl_ctx)

    from pytools.convergence import EOCRecorder
    eoc_rec = EOCRecorder()

    dims = 2
    ns = [8, 10, 12]
    for n in ns:
        from meshmode.mesh.generation import generate_regular_rect_mesh
        mesh = generate_regular_rect_mesh(
                a=(-0.5,)*dims,
                b=(0.5,)*dims,
                n=(n,)*dims,
                order=order)

        discr = DGDiscretizationWithBoundaries(cl_ctx, mesh,
            order=order, quad_min_degrees={"product": 4*order})

        sym_nds = sym.nodes(dims)
        advec_v = join_fields(-1*sym_nds[1], sym_nds[0])

        def angular_vel(radius):
            return 1/radius

        def gaussian_mode():
            time = sym.var("t")
            source_center = np.array([sym.cos(time), sym.sin(time)]) * 0.1
            source_width = 0.05

            sym_x = sym.nodes(2)
            sym_source_center_dist = sym_x - source_center

            return sym.exp(
                -np.dot(sym_source_center_dist, sym_source_center_dist)
                / source_width**2)

        def u_analytic(x):
            return 0

        from grudge.models.advection import VariableCoefficientAdvectionOperator
        op = VariableCoefficientAdvectionOperator(2, advec_v,
            u_analytic(sym.nodes(dims, sym.BTAG_ALL)), quad_tag="product",
            flux_type="central")

        nsteps = 40
        dt = 0.015

        bound_op = bind(discr, op.sym_operator())
        analytic_sol = bind(discr, gaussian_mode())
        fields = analytic_sol(queue, t=0)

        def rhs(t, u):
            return bound_op(queue, t=t, u=u)

        from grudge.shortcuts import set_up_rk4
        dt_stepper = set_up_rk4("u", dt, fields, rhs)

        print("dt=%g nsteps=%d" % (dt, nsteps))

        norm = bind(discr, sym.norm(2, sym.var("u")))

        #from grudge.shortcuts import make_visualizer
        #vis = make_visualizer(discr, vis_order=order)
        #vis.write_vtk_file("fld-start.vtu",
        #[("u", fields)])

        step = 0
        final_t = nsteps * dt
        for event in dt_stepper.run(t_end=final_t):
            if isinstance(event, dt_stepper.StateComputed):
                assert event.component_id == "u"
                esc = event.state_component

                step += 1
                #vis.write_vtk_file("fld-%04d.vtu" % step,
                #[("u", esc)])

                if step % 10 == 0:
                    print(step)

        #vis.write_vtk_file("fld-end.vtu",
                    #[("u", analytic_sol(queue, t=step * dt))])

        #vis.write_vtk_file("fld-actual-end.vtu",
                    #[("u", esc)])

        sol = analytic_sol(queue, t=step * dt)
        vals = [norm(queue, u=(esc - sol)) / norm(queue, u=sol)] # noqa E501
        total_error = sum(vals)
        print(total_error)
        eoc_rec.add_data_point(1.0/n, total_error)

    print(eoc_rec.pretty_print(abscissa_label="h",
            error_label="L2 Error"))

    #assert eoc_rec.order_estimate() > order


if __name__ == "__main__":
    test_convergence_maxwell(4)
