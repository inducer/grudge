"""Minimal example of a grudge driver."""

__copyright__ = """
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

import os
import numpy as np

import pyopencl as cl

from arraycontext import PyOpenCLArrayContext, thaw

from meshmode.dof_array import flatten

from pytools.obj_array import make_obj_array

import grudge.dof_desc as dof_desc
import grudge.op as op

import logging
logger = logging.getLogger(__name__)


# {{{ plotting (keep in sync with `weak.py`)

class Plotter:
    def __init__(self, actx, dcoll, order, visualize=True, ylim=None):
        self.actx = actx
        self.dim = dcoll.ambient_dim

        self.visualize = visualize
        if not self.visualize:
            return

        if self.dim == 1:
            import matplotlib.pyplot as pt
            self.fig = pt.figure(figsize=(8, 8), dpi=300)
            self.ylim = ylim

            volume_discr = dcoll.discr_from_dd(dof_desc.DD_VOLUME)
            self.x = actx.to_numpy(flatten(thaw(volume_discr.nodes()[0], actx)))
        else:
            from grudge.shortcuts import make_visualizer
            self.vis = make_visualizer(dcoll)

    def __call__(self, evt, basename, overwrite=True):
        if not self.visualize:
            return

        if self.dim == 1:
            u = self.actx.to_numpy(flatten(evt.state_component[0]))

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


def main(ctx_factory, dim=1, order=4, visualize=False):
    cl_ctx = ctx_factory()
    queue = cl.CommandQueue(cl_ctx)
    actx = PyOpenCLArrayContext(queue)

    # {{{ parameters

    # domain [0, d]^dim
    d = 2*np.pi

    # number of points in each dimension
    npoints = 50

    # final time
    final_time = 2

    # flux
    flux_type = "lf"

    # }}}

    # {{{ discretization

    # bdry_names = ("x", "y", "z")

    # btag_to_face = {
    #     "lower": ["-" + bdry_names[dim-1]],
    #     "upper": ["+" + bdry_names[dim-1]]
    # }

    from meshmode.mesh.generation import generate_regular_rect_mesh
    mesh = generate_regular_rect_mesh(
        a=(0.0,)*dim,
        b=(d,)*dim,
        nelements_per_axis=(npoints,)*dim,
        # boundary_tag_to_face=btag_to_face
    )

    # from meshmode.mesh.processing import glue_mesh_boundaries
    # glued_mesh = glue_mesh_boundaries(mesh, glued_boundary_mappings=[
    #     ("lower", "upper", (np.eye(dim), (0,)*(dim-1) + (1,))),
    # ])

    discr_tag_to_group_factory = {}

    from grudge import DiscretizationCollection

    dcoll = DiscretizationCollection(
        actx,
        mesh,
        # glued_mesh,
        order=order,
        discr_tag_to_group_factory=discr_tag_to_group_factory
    )

    # }}}

    # {{{ Burgers operator

    from grudge.models.burgers import InviscidBurgers

    x = thaw(dcoll.nodes(), actx)

    # Initial velocity magnitudes
    if dim == 1:
        u_init = [actx.np.sin(x[0])]
    else:
        raise NotImplementedError(f"Example not implemented for d={dim}")

    u_init = make_obj_array(u_init)

    burgers_operator = InviscidBurgers(
        actx,
        dcoll,
        flux_type=flux_type
    )

    def rhs(t, u):
        return burgers_operator.operator(t, u)

    # }}}

    # {{{ time stepping

    dt = 1/2 * burgers_operator.estimate_rk4_timestep(actx, dcoll, fields=u_init)

    from grudge.shortcuts import set_up_rk4
    dt_stepper = set_up_rk4("u", dt, u_init, rhs)
    plot = Plotter(actx, dcoll, order, visualize=visualize)

    step = 0
    for event in dt_stepper.run(t_end=final_time):
        if not isinstance(event, dt_stepper.StateComputed):
            continue

        norm_u = op.norm(dcoll, event.state_component, 2)
        assert norm_u < 5

        if step % 10 == 0:
            plot(event, "fld-burgers-%04d" % step)

        step += 1
        logger.info("[%04d] t = %.5f |u| = %.5e", step, event.t, norm_u)

    # }}}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dim", choices=[1, 2, 3], default=1, type=int)
    parser.add_argument("--order", default=4, type=int)
    parser.add_argument("--visualize", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    main(cl.create_some_context,
         dim=args.dim,
         order=args.order,
         visualize=args.visualize)
