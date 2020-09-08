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

import os
import numpy as np
import numpy.linalg as la

import pyopencl as cl

from meshmode.array_context import PyOpenCLArrayContext
from meshmode.dof_array import thaw, flatten

from grudge import bind, sym

import logging
logger = logging.getLogger(__name__)


# {{{ plotting (keep in sync with `var-velocity.py`)

class Plotter:
    def __init__(self, actx, discr, order, visualize=True, ylim=None):
        self.actx = actx
        self.dim = discr.ambient_dim

        self.visualize = visualize
        if not self.visualize:
            return

        if self.dim == 1:
            import matplotlib.pyplot as pt
            self.fig = pt.figure(figsize=(8, 8), dpi=300)
            self.ylim = ylim

            volume_discr = discr.discr_from_dd(sym.DD_VOLUME)
            self.x = actx.to_numpy(flatten(thaw(actx, volume_discr.nodes()[0])))
        else:
            from grudge.shortcuts import make_visualizer
            self.vis = make_visualizer(discr, vis_order=order)

    def __call__(self, evt, basename, overwrite=True):
        if not self.visualize:
            return

        if self.dim == 1:
            u = self.actx.to_numpy(flatten(evt.state_component))

            filename = "%s.png" % basename
            if not overwrite and os.path.exists(filename):
                from meshmode import FileExistsError
                raise FileExistsError("output file '%s' already exists" % filename)

            ax = self.fig.gca()
            ax.plot(self.x, u, "-")
            ax.plot(self.x, u, "k.")
            if self.ylim is not None:
                ax.set_ylim(self.ylim)

            ax.set_xlabel("$x$")
            ax.set_ylabel("$u$")
            ax.set_title(f"t = {evt.t:.2f}")

            self.fig.savefig(filename)
            self.fig.clf()
        else:
            self.vis.write_vtk_file("%s.vtu" % basename, [
                ("u", evt.state_component)
                ], overwrite=overwrite)

# }}}


def main(ctx_factory, dim=2, order=4, visualize=False):
    cl_ctx = ctx_factory()
    queue = cl.CommandQueue(cl_ctx)
    actx = PyOpenCLArrayContext(queue)

    # {{{ parameters

    # domain [-d/2, d/2]^dim
    d = 1.0
    # number of points in each dimension
    npoints = 20
    # grid spacing
    h = d / npoints

    # cfl
    dt_factor = 2.0
    # final time
    final_time = 1.0
    # compute number of steps
    dt = dt_factor * h/order**2
    nsteps = int(final_time // dt) + 1
    dt = final_time/nsteps + 1.0e-15

    # velocity field
    c = np.array([0.5] * dim)
    norm_c = la.norm(c)
    # flux
    flux_type = "central"

    # }}}

    # {{{ discretization

    from meshmode.mesh.generation import generate_box_mesh
    mesh = generate_box_mesh(
            [np.linspace(-d/2, d/2, npoints) for _ in range(dim)],
            order=order)

    from grudge import DGDiscretizationWithBoundaries
    discr = DGDiscretizationWithBoundaries(actx, mesh, order=order)

    # }}}

    # {{{ symbolic operators

    def f(x):
        return sym.sin(3 * x)

    def u_analytic(x):
        t = sym.var("t", sym.DD_SCALAR)
        return f(-np.dot(c, x) / norm_c + t * norm_c)

    from grudge.models.advection import WeakAdvectionOperator
    op = WeakAdvectionOperator(c,
        inflow_u=u_analytic(sym.nodes(dim, sym.BTAG_ALL)),
        flux_type=flux_type)

    bound_op = bind(discr, op.sym_operator())
    u = bind(discr, u_analytic(sym.nodes(dim)))(actx, t=0)

    def rhs(t, u):
        return bound_op(t=t, u=u)

    # }}}

    # {{{ time stepping

    from grudge.shortcuts import set_up_rk4
    dt_stepper = set_up_rk4("u", dt, u, rhs)
    plot = Plotter(actx, discr, order, visualize=visualize,
            ylim=[-1.1, 1.1])

    norm = bind(discr, sym.norm(2, sym.var("u")))

    step = 0
    norm_u = 0.0
    for event in dt_stepper.run(t_end=final_time):
        if not isinstance(event, dt_stepper.StateComputed):
            continue

        if step % 10 == 0:
            norm_u = norm(u=event.state_component)
            plot(event, "fld-weak-%04d" % step)

        step += 1
        logger.info("[%04d] t = %.5f |u| = %.5e", step, event.t, norm_u)

    # }}}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dim", default=2, type=int)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    main(cl.create_some_context,
            dim=args.dim)
