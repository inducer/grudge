from __future__ import absolute_import
from __future__ import print_function

__copyright__ = "Copyright (C) 2007 Andreas Kloeckner"

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




import numpy
import numpy.linalg as la




def main(write_output=True) :
    from math import sin, cos, pi, exp, sqrt
    from grudge.data import TimeConstantGivenFunction, \
            ConstantGivenFunction

    from grudge.backends import guess_run_context
    rcon = guess_run_context()

    dim = 2

    def boundary_tagger(fvi, el, fn, all_v):
        if el.face_normals[fn][0] > 0:
            return ["dirichlet"]
        else:
            return ["neumann"]

    if dim == 2:
        if rcon.is_head_rank:
            from grudge.mesh.generator import make_disk_mesh
            mesh = make_disk_mesh(r=0.5, boundary_tagger=boundary_tagger)
    elif dim == 3:
        if rcon.is_head_rank:
            from grudge.mesh.generator import make_ball_mesh
            mesh = make_ball_mesh(max_volume=0.001)
    else:
        raise RuntimeError("bad number of dimensions")

    if rcon.is_head_rank:
        print("%d elements" % len(mesh.elements))
        mesh_data = rcon.distribute_mesh(mesh)
    else:
        mesh_data = rcon.receive_mesh()

    discr = rcon.make_discretization(mesh_data, order=3,
            debug=["cuda_no_plan"],
            default_scalar_type=numpy.float64)

    if write_output:
        from grudge.visualization import  VtkVisualizer
        vis = VtkVisualizer(discr, rcon, "fld")

    def u0(x, el):
        if la.norm(x) < 0.2:
            return 1
        else:
            return 0

    def coeff(x, el):
        if x[0] < 0:
            return 0.25
        else:
            return 1

    def dirichlet_bc(t, x):
        return 0

    def neumann_bc(t, x):
        return 2

    from grudge.models.diffusion import DiffusionOperator
    op = DiffusionOperator(discr.dimensions,
            #coeff=coeff,
            dirichlet_tag="dirichlet",
            dirichlet_bc=TimeConstantGivenFunction(ConstantGivenFunction(0)),
            neumann_tag="neumann",
            neumann_bc=TimeConstantGivenFunction(ConstantGivenFunction(1))
            )
    u = discr.interpolate_volume_function(u0)

    # diagnostics setup -------------------------------------------------------
    from pytools.log import LogManager, \
            add_general_quantities, \
            add_simulation_quantities, \
            add_run_info

    if write_output:
        log_file_name = "heat.dat"
    else:
        log_file_name = None

    logmgr = LogManager(log_file_name, "w", rcon.communicator)
    add_run_info(logmgr)
    add_general_quantities(logmgr)
    add_simulation_quantities(logmgr)
    discr.add_instrumentation(logmgr)

    from grudge.log import LpNorm
    u_getter = lambda: u
    logmgr.add_quantity(LpNorm(u_getter, discr, 1, name="l1_u"))
    logmgr.add_quantity(LpNorm(u_getter, discr, name="l2_u"))

    logmgr.add_watches(["step.max", "t_sim.max", "l2_u", "t_step.max"])

    # timestep loop -----------------------------------------------------------
    from grudge.timestep.runge_kutta import LSRK4TimeStepper, ODE45TimeStepper
    from grudge.timestep.dumka3 import Dumka3TimeStepper
    #stepper = LSRK4TimeStepper()
    stepper = Dumka3TimeStepper(3, rtol=1e-6, rcon=rcon,
            vector_primitive_factory=discr.get_vector_primitive_factory(),
            dtype=discr.default_scalar_type)
    #stepper = ODE45TimeStepper(rtol=1e-6, rcon=rcon,
            #vector_primitive_factory=discr.get_vector_primitive_factory(),
            #dtype=discr.default_scalar_type)
    stepper.add_instrumentation(logmgr)

    rhs = op.bind(discr)
    try:
        next_dt = op.estimate_timestep(discr,
                stepper=LSRK4TimeStepper(), t=0, fields=u)

        from grudge.timestep import times_and_steps
        step_it = times_and_steps(
                final_time=0.1, logmgr=logmgr,
                max_dt_getter=lambda t: next_dt,
                taken_dt_getter=lambda: taken_dt)

        for step, t, dt in step_it:
            if step % 10 == 0 and write_output:
                visf = vis.make_file("fld-%04d" % step)
                vis.add_data(visf, [
                    ("u", discr.convert_volume(u, kind="numpy")),
                    ], time=t, step=step)
                visf.close()

            u, t, taken_dt, next_dt = stepper(u, t, next_dt, rhs)
            #u = stepper(u, t, dt, rhs)

        assert discr.norm(u) < 1
    finally:
        if write_output:
            vis.close()

        logmgr.close()
        discr.close()




if __name__ == "__main__":
    main()




# entry points for py.test ----------------------------------------------------
from pytools.test import mark_test
@mark_test.long
def test_heat():
    main(write_output=False)
