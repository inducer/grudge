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

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function
import numpy
import numpy.linalg as la




def main(write_output=True, flux_type_arg="upwind"):
    from grudge.tools import mem_checkpoint
    from math import sin, cos, pi, sqrt
    from math import floor

    from grudge.backends import guess_run_context
    rcon = guess_run_context()

    def f(x):
        return sin(pi*x)

    def u_analytic(x, el, t):
        return f((-numpy.dot(v, x)/norm_v+t*norm_v))

    def boundary_tagger(vertices, el, face_nr, all_v):
        if numpy.dot(el.face_normals[face_nr], v) < 0:
            return ["inflow"]
        else:
            return ["outflow"]

    dim = 2

    if dim == 1:
        v = numpy.array([1])
        if rcon.is_head_rank:
            from grudge.mesh.generator import make_uniform_1d_mesh
            mesh = make_uniform_1d_mesh(0, 2, 10, periodic=True)
    elif dim == 2:
        v = numpy.array([2,0])
        if rcon.is_head_rank:
            from grudge.mesh.generator import make_disk_mesh
            mesh = make_disk_mesh(boundary_tagger=boundary_tagger)
    elif dim == 3:
        v = numpy.array([0,0,1])
        if rcon.is_head_rank:
            from grudge.mesh.generator import make_cylinder_mesh, make_ball_mesh, make_box_mesh

            mesh = make_cylinder_mesh(max_volume=0.04, height=2, boundary_tagger=boundary_tagger,
                    periodic=False, radial_subdivisions=32)
    else:
        raise RuntimeError("bad number of dimensions")

    norm_v = la.norm(v)

    if rcon.is_head_rank:
        mesh_data = rcon.distribute_mesh(mesh)
    else:
        mesh_data = rcon.receive_mesh()

    if dim != 1:
        mesh_data = mesh_data.reordered_by("cuthill")

    discr = rcon.make_discretization(mesh_data, order=4)
    vis_discr = discr

    from grudge.visualization import VtkVisualizer
    if write_output:
        vis = VtkVisualizer(vis_discr, rcon, "fld")

    # operator setup ----------------------------------------------------------
    from grudge.data import \
            ConstantGivenFunction, \
            TimeConstantGivenFunction, \
            TimeDependentGivenFunction
    from grudge.models.advection import StrongAdvectionOperator, WeakAdvectionOperator
    op = WeakAdvectionOperator(v,
            inflow_u=TimeDependentGivenFunction(u_analytic),
            flux_type=flux_type_arg)

    u = discr.interpolate_volume_function(lambda x, el: u_analytic(x, el, 0))

    # timestep setup ----------------------------------------------------------
    from grudge.timestep.runge_kutta import LSRK4TimeStepper
    stepper = LSRK4TimeStepper()

    if rcon.is_head_rank:
        print("%d elements" % len(discr.mesh.elements))

    # diagnostics setup -------------------------------------------------------
    from pytools.log import LogManager, \
            add_general_quantities, \
            add_simulation_quantities, \
            add_run_info

    if write_output:
        log_file_name = "advection.dat"
    else:
        log_file_name = None

    logmgr = LogManager(log_file_name, "w", rcon.communicator)
    add_run_info(logmgr)
    add_general_quantities(logmgr)
    add_simulation_quantities(logmgr)
    discr.add_instrumentation(logmgr)

    stepper.add_instrumentation(logmgr)

    from grudge.log import Integral, LpNorm
    u_getter = lambda: u
    logmgr.add_quantity(Integral(u_getter, discr, name="int_u"))
    logmgr.add_quantity(LpNorm(u_getter, discr, p=1, name="l1_u"))
    logmgr.add_quantity(LpNorm(u_getter, discr, name="l2_u"))

    logmgr.add_watches(["step.max", "t_sim.max", "l2_u", "t_step.max"])

    # timestep loop -----------------------------------------------------------
    rhs = op.bind(discr)

    try:
        from grudge.timestep import times_and_steps
        step_it = times_and_steps(
                final_time=3, logmgr=logmgr,
                max_dt_getter=lambda t: op.estimate_timestep(discr,
                    stepper=stepper, t=t, fields=u))

        for step, t, dt in step_it:
            if step % 5 == 0 and write_output:
                visf = vis.make_file("fld-%04d" % step)
                vis.add_data(visf, [
                    ("u", discr.convert_volume(u, kind="numpy")),
                    ], time=t, step=step)
                visf.close()

            u = stepper(u, t, dt, rhs)

        true_u = discr.interpolate_volume_function(lambda x, el: u_analytic(x, el, t))
        print(discr.norm(u-true_u))
        assert discr.norm(u-true_u) < 1e-2
    finally:
        if write_output:
            vis.close()

        logmgr.close()
        discr.close()




if __name__ == "__main__":
    main()




# entry points for py.test ----------------------------------------------------
def test_advection():
    from pytools.test import mark_test
    mark_long = mark_test.long

    for flux_type in ["upwind", "central", "lf"]:
        yield "advection with %s flux" % flux_type, \
                mark_long(main), False, flux_type
