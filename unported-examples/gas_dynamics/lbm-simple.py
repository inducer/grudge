__copyright__ = "Copyright (C) 2011 Andreas Kloeckner"

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




from __future__ import division
from __future__ import absolute_import
from __future__ import print_function
import numpy as np
import numpy.linalg as la
from six.moves import range




def main(write_output=True, dtype=np.float32):
    from grudge.backends import guess_run_context
    rcon = guess_run_context()

    from grudge.mesh.generator import make_rect_mesh
    if rcon.is_head_rank:
        h_fac = 1
        mesh = make_rect_mesh(a=(0,0),b=(1,1), max_area=h_fac**2*1e-4,
                periodicity=(True,True),
                subdivisions=(int(70/h_fac), int(70/h_fac)))

    from grudge.models.gas_dynamics.lbm import \
            D2Q9LBMMethod, LatticeBoltzmannOperator

    op = LatticeBoltzmannOperator(
            D2Q9LBMMethod(), lbm_delta_t=0.001, nu=1e-4)

    if rcon.is_head_rank:
        print("%d elements" % len(mesh.elements))
        mesh_data = rcon.distribute_mesh(mesh)
    else:
        mesh_data = rcon.receive_mesh()

    discr = rcon.make_discretization(mesh_data, order=3,
            default_scalar_type=dtype,
            debug=["cuda_no_plan"])
    from grudge.timestep.runge_kutta import LSRK4TimeStepper
    stepper = LSRK4TimeStepper(dtype=dtype,
            #vector_primitive_factory=discr.get_vector_primitive_factory()
            )

    from grudge.visualization import VtkVisualizer
    if write_output:
        vis = VtkVisualizer(discr, rcon, "fld")

    from grudge.data import CompiledExpressionData
    def ic_expr(t, x, fields):
        from grudge.symbolic import FunctionSymbol
        from pymbolic.primitives import IfPositive
        from pytools.obj_array import make_obj_array

        tanh = FunctionSymbol("tanh")
        sin = FunctionSymbol("sin")

        rho = 1
        u0 = 0.05
        w = 0.05
        delta = 0.05

        from grudge.symbolic.primitives import make_common_subexpression as cse
        u = cse(make_obj_array([
            IfPositive(x[1]-1/2,
                u0*tanh(4*(3/4-x[1])/w),
                u0*tanh(4*(x[1]-1/4)/w)),
            u0*delta*sin(2*np.pi*(x[0]+1/4))]),
            "u")

        return make_obj_array([
            op.method.f_equilibrium(rho, alpha, u)
            for alpha in range(len(op.method))
            ])


    # timestep loop -----------------------------------------------------------
    stream_rhs = op.bind_rhs(discr)
    collision_update = op.bind(discr, op.collision_update)
    get_rho = op.bind(discr, op.rho)
    get_rho_u = op.bind(discr, op.rho_u)


    f_bar = CompiledExpressionData(ic_expr).volume_interpolant(0, discr)

    from grudge.discretization import ExponentialFilterResponseFunction
    from grudge.symbolic.operators import FilterOperator
    mode_filter = FilterOperator(
            ExponentialFilterResponseFunction(min_amplification=0.9, order=4))\
                    .bind(discr)

    final_time = 1000
    try:
        lbm_dt = op.lbm_delta_t
        dg_dt = op.estimate_timestep(discr, stepper=stepper)
        print(dg_dt)

        dg_steps_per_lbm_step = int(np.ceil(lbm_dt / dg_dt))
        dg_dt = lbm_dt / dg_steps_per_lbm_step

        lbm_steps = int(final_time // op.lbm_delta_t)
        for step in range(lbm_steps):
            t = step*lbm_dt

            if step % 100 == 0 and write_output:
                visf = vis.make_file("fld-%04d" % step)

                rho = get_rho(f_bar)
                rho_u = get_rho_u(f_bar)
                vis.add_data(visf,
                        [ ("fbar%d" %i,
                            discr.convert_volume(f_bar_i, "numpy")) for i, f_bar_i in enumerate(f_bar)]+
                        [
                            ("rho", discr.convert_volume(rho, "numpy")),
                            ("rho_u", discr.convert_volume(rho_u, "numpy")),
                        ],
                        time=t,
                        step=step)
                visf.close()

            print("step=%d, t=%f" % (step, t))

            f_bar = collision_update(f_bar)

            for substep in range(dg_steps_per_lbm_step):
                f_bar = stepper(f_bar, t + substep*dg_dt, dg_dt, stream_rhs)

            #f_bar = mode_filter(f_bar)

    finally:
        if write_output:
            vis.close()

        discr.close()




if __name__ == "__main__":
    main(True)
