__copyright__ = "Copyright (C) 2008 Andreas Kloeckner"

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
import numpy
import numpy.linalg as la




class SineWave:
    def __init__(self):
        self.gamma = 1.4
        self.mu = 0
        self.prandtl = 0.72
        self.spec_gas_const = 287.1

    def __call__(self, t, x_vec):
        rho = 2 + numpy.sin(x_vec[0] + x_vec[1] + x_vec[2] - 2 * t)
        velocity = numpy.array([1, 1, 0])
        p = 1
        e = p/(self.gamma-1) + rho/2 * numpy.dot(velocity, velocity)
        rho_u = rho * velocity[0]
        rho_v = rho * velocity[1]
        rho_w = rho * velocity[2]

        from grudge.tools import join_fields
        return join_fields(rho, e, rho_u, rho_v, rho_w)

    def properties(self):
        return(self.gamma, self.mu, self.prandtl, self.spec_gas_const)

    def volume_interpolant(self, t, discr):
        return discr.convert_volume(
                        self(t, discr.nodes.T),
                        kind=discr.compute_kind)

    def boundary_interpolant(self, t, discr, tag):
        return discr.convert_boundary(
                        self(t, discr.get_boundary(tag).nodes.T),
                         tag=tag, kind=discr.compute_kind)




def main(final_time=1, write_output=False):
    from grudge.backends import guess_run_context
    rcon = guess_run_context()

    from grudge.tools import EOCRecorder, to_obj_array
    eoc_rec = EOCRecorder()

    if rcon.is_head_rank:
        from grudge.mesh import make_box_mesh
        mesh = make_box_mesh((0,0,0), (10,10,10), max_volume=0.5)
        mesh_data = rcon.distribute_mesh(mesh)
    else:
        mesh_data = rcon.receive_mesh()

    for order in [3, 4, 5]:
        discr = rcon.make_discretization(mesh_data, order=order,
                        default_scalar_type=numpy.float64)

        from grudge.visualization import SiloVisualizer, VtkVisualizer
        vis = VtkVisualizer(discr, rcon, "sinewave-%d" % order)
        #vis = SiloVisualizer(discr, rcon)

        sinewave = SineWave()
        fields = sinewave.volume_interpolant(0, discr)
        gamma, mu, prandtl, spec_gas_const = sinewave.properties()

        from grudge.mesh import BTAG_ALL
        from grudge.models.gas_dynamics import GasDynamicsOperator
        op = GasDynamicsOperator(dimensions=mesh.dimensions, gamma=gamma, mu=mu,
                prandtl=prandtl, spec_gas_const=spec_gas_const,
                bc_inflow=sinewave, bc_outflow=sinewave, bc_noslip=sinewave,
                inflow_tag=BTAG_ALL, source=None)

        euler_ex = op.bind(discr)

        max_eigval = [0]
        def rhs(t, q):
            ode_rhs, speed = euler_ex(t, q)
            max_eigval[0] = speed
            return ode_rhs
        rhs(0, fields)

        if rcon.is_head_rank:
            print("---------------------------------------------")
            print("order %d" % order)
            print("---------------------------------------------")
            print("#elements=", len(mesh.elements))

        from grudge.timestep import RK4TimeStepper
        stepper = RK4TimeStepper()

        # diagnostics setup ---------------------------------------------------
        from logpyle import LogManager, add_general_quantities, \
                add_simulation_quantities, add_run_info

        if write_output:
            log_name = ("euler-sinewave-%(order)d-%(els)d.dat"
                    % {"order":order, "els":len(mesh.elements)})
        else:
            log_name = False
        logmgr = LogManager(log_name, "w", rcon.communicator)
        add_run_info(logmgr)
        add_general_quantities(logmgr)
        add_simulation_quantities(logmgr)
        discr.add_instrumentation(logmgr)
        stepper.add_instrumentation(logmgr)

        logmgr.add_watches(["step.max", "t_sim.max", "t_step.max"])

        # timestep loop -------------------------------------------------------
        try:
            from grudge.timestep import times_and_steps
            step_it = times_and_steps(
                    final_time=final_time, logmgr=logmgr,
                    max_dt_getter=lambda t: op.estimate_timestep(discr,
                        stepper=stepper, t=t, max_eigenvalue=max_eigval[0]))

            for step, t, dt in step_it:
                #if step % 10 == 0:
                if write_output:
                    visf = vis.make_file("sinewave-%d-%04d" % (order, step))

                    #from pyvisfile.silo import DB_VARTYPE_VECTOR
                    vis.add_data(visf,
                            [
                                ("rho", discr.convert_volume(op.rho(fields), kind="numpy")),
                                ("e", discr.convert_volume(op.e(fields), kind="numpy")),
                                ("rho_u", discr.convert_volume(op.rho_u(fields), kind="numpy")),
                                ("u", discr.convert_volume(op.u(fields), kind="numpy")),

                                #("true_rho", op.rho(true_fields)),
                                #("true_e", op.e(true_fields)),
                                #("true_rho_u", op.rho_u(true_fields)),
                                #("true_u", op.u(true_fields)),

                                #("rhs_rho", op.rho(rhs_fields)),
                                #("rhs_e", op.e(rhs_fields)),
                                #("rhs_rho_u", op.rho_u(rhs_fields)),
                                ],
                            #expressions=[
                                #("diff_rho", "rho-true_rho"),
                                #("diff_e", "e-true_e"),
                                #("diff_rho_u", "rho_u-true_rho_u", DB_VARTYPE_VECTOR),

                                #("p", "0.4*(e- 0.5*(rho_u*u))"),
                                #],
                            time=t, step=step
                            )
                    visf.close()

                fields = stepper(fields, t, dt, rhs)

        finally:
            vis.close()
            logmgr.close()
            discr.close()

        true_fields = sinewave.volume_interpolant(t, discr)
        eoc_rec.add_data_point(order, discr.norm(fields-true_fields))
        print()
        print(eoc_rec.pretty_print("P.Deg.", "L2 Error"))




if __name__ == "__main__":
    main()




# entry points for py.test ----------------------------------------------------
from pytools.test import mark_test
@mark_test.long
def test_euler_sine_wave():
    main(final_time=0.1, write_output=False)

