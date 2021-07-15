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
import pyopencl.tools as cl_tools

from arraycontext import thaw
from grudge.array_context import PyOpenCLArrayContext

from meshmode.dof_array import flatten

from grudge.models.burgers import \
    InviscidBurgers, EntropyConservativeInviscidBurgers

import grudge.dof_desc as dof_desc
import grudge.op as op

import logging
logger = logging.getLogger(__name__)


# {{{ plotting (keep in sync with `weak.py`)

class Plotter:
    def __init__(self, actx, dcoll, order, npoints, visualize=True, ylim=None):
        self.actx = actx
        self.dim = dcoll.ambient_dim
        self.order = order
        self.npoints = npoints

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
            ax.set_title(f"t = {evt.t:.2f} ($N={self.npoints}$, $k={self.order}$)")
            self.fig.savefig(filename)
            self.fig.clf()
        else:
            self.vis.write_vtk_file("%s.vtu" % basename, [
                ("u", evt.state_component)
                ], overwrite=overwrite)

# }}}


# {{{ SSPRK3 timestepper

def set_up_ssprk3(field_var_name, dt, fields, rhs, t_start=0):
    from leap.rk import SSPRK33MethodBuilder
    from dagrt.codegen import PythonCodeGenerator

    dt_method = SSPRK33MethodBuilder(component_id=field_var_name)
    dt_code = dt_method.generate()
    dt_stepper_class = PythonCodeGenerator("TimeStep").get_class(dt_code)
    dt_stepper = dt_stepper_class({"<func>"+dt_method.component_id: rhs})

    dt_stepper.set_up(t_start=t_start, dt_start=dt,
                      context={dt_method.component_id: fields})

    return dt_stepper

# }}}


# {{{ Quick plotting helper functions

def plot_entropy(time, integrated_entropy, exp_name, npoints, order):

    import matplotlib.pyplot as pt

    fig = pt.figure(figsize=(8, 8), dpi=300)
    ax = fig.gca()
    ax.plot(time, integrated_entropy, "b-")
    ax.plot(time, integrated_entropy, "b.")

    ax.set_xlabel("$t$")
    ax.set_ylabel(
        "$\\int_D\\frac{1}{2}u(t)^2-\\frac{1}{2}u(0)^2$/$\\int_D\\frac{1}{2}u(0)^2$"
    )
    ax.set_title(f"Rel. diff integrated entropy vs time ($N={npoints}$, $k={order}$)")

    ax.set_ylim([-1, 1])
    ax.set_yscale("log")

    fig.savefig("%s-entropy.png" % exp_name)
    fig.clf()


def plot_growth_factors(time, growth_factors, exp_name, npoints, order):

    import matplotlib.pyplot as pt

    fig = pt.figure(figsize=(8, 8), dpi=300)
    ax = fig.gca()
    ax.plot(time, growth_factors, "-")
    ax.plot(time, growth_factors, "k.")

    ax.set_xlabel("$t$")
    ax.set_ylabel(
        "$|| u(t) || / || u(0) ||$"
    )
    ax.set_title(f"Growth factor vs time ($N={npoints}$, $k={order}$)")
    ax.set_ylim([0.9, 1.5])

    fig.savefig("%s-factors.png" % exp_name)
    fig.clf()

# }}}


def main(ctx_factory, dim=1, order=4, npoints=30, vis_freq=10,
         strong_form=False, entropy_conservative=False, visualize=False):
    cl_ctx = ctx_factory()
    queue = cl.CommandQueue(cl_ctx)
    actx = PyOpenCLArrayContext(
        queue,
        allocator=cl_tools.MemoryPool(cl_tools.ImmediateAllocator(queue)),
        force_device_scalars=True,
    )

    # {{{ parameters

    # domain [0, d]^dim
    d = 2.0*np.pi

    # final time
    final_time = 1.5

    # flux
    flux_type = "central"

    if strong_form:
        dgtype = "strong"
    else:
        dgtype = "weak"

    # }}}

    # {{{ discretization

    from meshmode.mesh.generation import generate_regular_rect_mesh
    from meshmode.mesh.processing import glue_mesh_boundaries
    from meshmode.discretization.poly_element import (
        default_simplex_group_factory,
        QuadratureSimplexGroupFactory
    )
    from grudge import DiscretizationCollection

    bdry_names = ("x", "y", "z")

    btag_to_face = {
        "lower": ["-" + bdry_names[dim-1]],
        "upper": ["+" + bdry_names[dim-1]]
    }

    mesh = generate_regular_rect_mesh(
        a=(0.0,)*dim,
        b=(d,)*dim,
        nelements_per_axis=(npoints,)*dim,
        order=order,
        boundary_tag_to_face=btag_to_face
    )

    glued_mesh = glue_mesh_boundaries(
        mesh, glued_boundary_mappings=[
            ("lower", "upper", (np.eye(dim), (0,)*(dim-1) + (d,)), 1e-16),
        ]
    )

    dd_quad = dof_desc.DOFDesc(dof_desc.DTAG_VOLUME_ALL, dof_desc.DISCR_TAG_QUAD)

    dcoll = DiscretizationCollection(
        actx,
        glued_mesh,
        discr_tag_to_group_factory={
            dof_desc.DISCR_TAG_BASE: default_simplex_group_factory(base_dim=dim,
                                                                   order=order),
            dof_desc.DISCR_TAG_QUAD: default_simplex_group_factory(base_dim=dim,
                                                                   order=3*order + 1)
        }
    )

    # }}}

    # {{{ Burgers operator

    x = thaw(dcoll.nodes(), actx)

    # Initial velocity magnitudes
    if dim == 1:
        u_init = 0.5 + actx.np.sin(x[0])
    else:
        raise NotImplementedError(f"Example not implemented for d={dim}")

    if entropy_conservative:
        exp_name = "fld-burgers-esdg"
        burgers_op = EntropyConservativeInviscidBurgers(dcoll, flux_type=flux_type)
    else:
        exp_name = "fld-burgers-%s-%s" % (flux_type, dgtype)
        burgers_op = InviscidBurgers(dcoll, flux_type=flux_type,
                                     strong_form=strong_form)

    def rhs(t, u):
        return burgers_op.operator(t, u)

    # }}}

    # {{{ time stepping

    dt = 1/10 * burgers_op.estimate_rk4_timestep(actx, dcoll, fields=u_init)

    from grudge.shortcuts import set_up_rk4

    # dt_stepper = set_up_ssprk3("u", dt, u_init, rhs)
    dt_stepper = set_up_rk4("u", dt, u_init, rhs)
    plot = Plotter(actx, dcoll, order, npoints, visualize=visualize, ylim=[-1, 2])

    step = 0
    time = [0.0]

    init_ef = actx.to_numpy(
        op.integral(
            dcoll, dd_quad,
            burgers_op.entropy_function(
                op.project(dcoll, "vol", dd_quad, u_init)
            )
        )
    )
    integrated_entropy = [0.0]
    norm_init = actx.to_numpy(op.norm(dcoll, u_init, 2))
    sol_growth_factor = [1.0]

    for event in dt_stepper.run(t_end=final_time):
        if not isinstance(event, dt_stepper.StateComputed):
            continue

        u_h = event.state_component
        norm_u = actx.to_numpy(op.norm(dcoll, u_h, 2))

        if step % vis_freq == 0:
            plot(event, "%s-%04d" % (exp_name, step))

        time.append(event.t)

        ef_tn = actx.to_numpy(
            op.integral(
                dcoll, dd_quad,
                burgers_op.entropy_function(
                    op.project(dcoll, "vol", dd_quad, u_h)
                )
            )
        )

        integrated_entropy.append(
            (ef_tn - init_ef)/abs(init_ef)
        )
        gt = norm_u / norm_init
        sol_growth_factor.append(gt)

        # if norm_u > 5:
        #     plot_entropy(time, integrated_entropy, exp_name, npoints, order)
        #     plot_growth_factors(time, sol_growth_factor, exp_name, npoints, order)
        #     raise RuntimeError("Solution norm is growing")

        step += 1
        logger.info("[%04d] t = %.5f |u| = %.5e Growth factor = %.5f", step, event.t, norm_u, gt)

    # }}}

    plot_entropy(time, integrated_entropy, exp_name, npoints, order)
    plot_growth_factors(time, sol_growth_factor, exp_name, npoints, order)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dim", choices=[1, 2, 3], default=1, type=int)
    parser.add_argument("--order", default=4, type=int)
    parser.add_argument("--npoints", default=20, type=int)
    parser.add_argument("--visfreq", default=10, type=int)
    parser.add_argument("--strong", action="store_true")
    parser.add_argument("--esdg", action="store_true")
    parser.add_argument("--visualize", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    main(cl.create_some_context,
         dim=args.dim,
         order=args.order,
         npoints=args.npoints,
         vis_freq=args.visfreq,
         strong_form=args.strong,
         entropy_conservative=args.esdg,
         visualize=args.visualize)
