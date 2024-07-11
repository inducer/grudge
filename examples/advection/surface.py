"""Minimal example of a grudge driver for DG on surfaces."""

__copyright__ = """
Copyright (C) 2020 Alexandru Fikl
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

import logging
import os

import numpy as np

import pyopencl as cl
import pyopencl.tools as cl_tools
from arraycontext import flatten
from meshmode.discretization.connection import FACE_RESTR_INTERIOR
from pytools.obj_array import make_obj_array

import grudge.dof_desc as dof_desc
import grudge.geometry as geo
import grudge.op as op
from grudge.array_context import PyOpenCLArrayContext


logger = logging.getLogger(__name__)


# {{{ plotting (keep in sync with `var-velocity.py`)

class Plotter:
    def __init__(self, actx, dcoll, order, visualize=True):
        self.actx = actx
        self.ambient_dim = dcoll.ambient_dim
        self.dim = dcoll.dim

        self.visualize = visualize
        if not self.visualize:
            return

        if self.ambient_dim == 2:
            import matplotlib.pyplot as pt
            self.fig = pt.figure(figsize=(8, 8), dpi=300)

            x = actx.thaw(dcoll.discr_from_dd(dof_desc.DD_VOLUME_ALL).nodes())
            self.x = actx.to_numpy(flatten(actx.np.arctan2(x[1], x[0]), self.actx))
        elif self.ambient_dim == 3:
            from grudge.shortcuts import make_visualizer
            self.vis = make_visualizer(dcoll)
        else:
            raise ValueError("unsupported dimension")

    def __call__(self, evt, basename, overwrite=True):
        if not self.visualize:
            return

        if self.ambient_dim == 2:
            u = self.actx.to_numpy(flatten(evt.state_component, self.actx))

            filename = f"{basename}.png"
            if not overwrite and os.path.exists(filename):
                from meshmode import FileExistsError
                raise FileExistsError(f"output file '{filename}' already exists")

            ax = self.fig.gca()
            ax.grid()
            ax.plot(self.x, u, "-")
            ax.plot(self.x, u, "k.")
            ax.set_xlabel(r"$\theta$")
            ax.set_ylabel("$u$")
            ax.set_title(f"t = {evt.t:.2f}")

            self.fig.savefig(filename)
            self.fig.clf()
        elif self.ambient_dim == 3:
            self.vis.write_vtk_file(f"{basename}.vtu", [
                ("u", evt.state_component)
                ], overwrite=overwrite)
        else:
            raise ValueError("unsupported dimension")

# }}}


def main(ctx_factory, dim=2, order=4, use_quad=False, visualize=False):
    cl_ctx = ctx_factory()
    queue = cl.CommandQueue(cl_ctx)
    actx = PyOpenCLArrayContext(
        queue,
        allocator=cl_tools.MemoryPool(cl_tools.ImmediateAllocator(queue)),
        force_device_scalars=True,
    )

    # {{{ parameters

    # sphere radius
    radius = 1.0
    # sphere resolution
    resolution = 64 if dim == 2 else 1

    # final time
    final_time = np.pi

    # flux
    flux_type = "lf"

    # }}}

    # {{{ discretization

    if dim == 2:
        from meshmode.mesh.generation import ellipse, make_curve_mesh
        mesh = make_curve_mesh(
                lambda t: radius * ellipse(1.0, t),
                np.linspace(0.0, 1.0, resolution + 1),
                order)
    elif dim == 3:
        from meshmode.mesh.generation import generate_icosphere
        mesh = generate_icosphere(radius, order=4 * order,
                uniform_refinement_rounds=resolution)
    else:
        raise ValueError("unsupported dimension")

    discr_tag_to_group_factory = {}
    if use_quad:
        qtag = dof_desc.DISCR_TAG_QUAD
    else:
        qtag = None

    from meshmode.discretization.poly_element import (
        QuadratureSimplexGroupFactory,
        default_simplex_group_factory,
    )

    discr_tag_to_group_factory[dof_desc.DISCR_TAG_BASE] = \
        default_simplex_group_factory(base_dim=dim-1, order=order)

    if use_quad:
        discr_tag_to_group_factory[qtag] = \
            QuadratureSimplexGroupFactory(order=4*order)

    from grudge.discretization import make_discretization_collection

    dcoll = make_discretization_collection(
        actx, mesh,
        discr_tag_to_group_factory=discr_tag_to_group_factory
    )

    volume_discr = dcoll.discr_from_dd(dof_desc.DD_VOLUME_ALL)
    logger.info("ndofs:     %d", volume_discr.ndofs)
    logger.info("nelements: %d", volume_discr.mesh.nelements)

    # }}}

    # {{{ Surface advection operator

    # velocity field
    x = actx.thaw(dcoll.nodes())
    c = make_obj_array([-x[1], x[0], 0.0])[:dim]

    def f_initial_condition(x):
        return x[0]

    from grudge.models.advection import SurfaceAdvectionOperator
    adv_operator = SurfaceAdvectionOperator(
        dcoll,
        c,
        flux_type=flux_type,
        quad_tag=qtag
    )

    u0 = f_initial_condition(x)

    def rhs(t, u):
        return adv_operator.operator(t, u)

    # check velocity is tangential
    from grudge.geometry import normal

    surf_normal = normal(actx, dcoll, dd=dof_desc.DD_VOLUME_ALL)

    error = op.norm(dcoll, c.dot(surf_normal), 2)
    logger.info("u_dot_n:   %.5e", error)

    # }}}

    # {{{ time stepping

    # FIXME: dt estimate is not necessarily valid for surfaces
    dt = actx.to_numpy(
        0.45 * adv_operator.estimate_rk4_timestep(actx, dcoll, fields=u0))
    nsteps = int(final_time // dt) + 1

    logger.info("dt:        %.5e", dt)
    logger.info("nsteps:    %d", nsteps)

    from grudge.shortcuts import set_up_rk4
    dt_stepper = set_up_rk4("u", dt, u0, rhs)
    plot = Plotter(actx, dcoll, order, visualize=visualize)

    norm_u = actx.to_numpy(op.norm(dcoll, u0, 2))

    step = 0

    event = dt_stepper.StateComputed(0.0, 0, 0, u0)
    plot(event, "fld-surface-%04d" % 0)

    if visualize and dim == 3:
        from grudge.shortcuts import make_visualizer
        vis = make_visualizer(dcoll)
        vis.write_vtk_file(
            "fld-surface-velocity.vtu",
            [
                ("u", c),
                ("n", surf_normal)
            ],
            overwrite=True
        )

        df = dof_desc.as_dofdesc(FACE_RESTR_INTERIOR)
        face_discr = dcoll.discr_from_dd(df)
        face_normal = geo.normal(actx, dcoll, dd=df)

        from meshmode.discretization.visualization import make_visualizer
        vis = make_visualizer(actx, face_discr)
        vis.write_vtk_file("fld-surface-face-normals.vtu", [
            ("n", face_normal)
            ], overwrite=True)

    for event in dt_stepper.run(t_end=final_time, max_steps=nsteps + 1):
        if not isinstance(event, dt_stepper.StateComputed):
            continue

        step += 1
        if step % 10 == 0:
            norm_u = actx.to_numpy(op.norm(dcoll, event.state_component, 2))
            plot(event, "fld-surface-%04d" % step)

        logger.info("[%04d] t = %.5f |u| = %.5e", step, event.t, norm_u)

        # NOTE: These are here to ensure the solution is bounded for the
        # time interval specified
        assert norm_u < 3

    # }}}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dim", choices=[2, 3], default=2, type=int)
    parser.add_argument("--order", default=4, type=int)
    parser.add_argument("--use-quad", action="store_true")
    parser.add_argument("--visualize", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    main(cl.create_some_context,
            dim=args.dim,
            order=args.order,
            use_quad=args.use_quad,
            visualize=args.visualize)
