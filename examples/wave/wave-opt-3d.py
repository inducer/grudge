"""Minimal example of a grudge driver."""

__copyright__ = """
Copyright (C) 2015 Andreas Kloeckner
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


import numpy as np
import pyopencl as cl
import pyopencl.tools as cl_tools

from arraycontext import thaw, freeze
from grudge.array_context import PyOpenCLArrayContext
from meshmode.array_context import (
    PytatoPyOpenCLArrayContext as PytatoArrayContextBase)

from time import time
from grudge import DiscretizationCollection

from pytools.obj_array import flat_obj_array

import grudge.op as op

import logging
logger = logging.getLogger(__name__)


class PytatoArrayContext(PytatoArrayContextBase):
    def transform_dag(self, dag):
        import pytato as pt

        # {{{ collapse data wrappers

        data_wrapper_cache = {}

        def cached_data_wrapper_if_present(ary):
            if isinstance(ary, pt.DataWrapper):
                cache_key = (ary.data.data.int_ptr, ary.data.offset,
                             ary.shape, ary.data.strides)
                try:
                    result = data_wrapper_cache[cache_key]
                except KeyError:
                    result = ary
                    data_wrapper_cache[cache_key] = result

                return result
            else:
                return ary

        dag = pt.transform.map_and_copy(dag, cached_data_wrapper_if_present)

        # }}}

        # {{{ get rid of copies for different views of a cl-array

        def eliminate_reshapes_of_data_wrappers(ary):
            if (isinstance(ary, pt.Reshape)
                    and isinstance(ary.array, pt.DataWrapper)):
                return pt.make_data_wrapper(ary.array.data.reshape(ary.shape))
            else:
                return ary

        dag = pt.transform.map_and_copy(dag,
                                        eliminate_reshapes_of_data_wrappers)

        # }}}

        return dag


def rk4_step(y, t, h, f):
    k1 = f(t, y)
    k2 = f(t+h/2, y + h/2*k1)
    k3 = f(t+h/2, y + h/2*k2)
    k4 = f(t+h, y + h*k3)
    return y + h/6*(k1 + 2*k2 + 2*k3 + k4)


def main(ctx_factory, dim=2, order=4, visualize=False, lazy=False):
    cl_ctx = ctx_factory()
    queue = cl.CommandQueue(cl_ctx)
    if lazy:
        actx = PytatoArrayContext(
            queue,
            allocator=cl_tools.MemoryPool(cl_tools.ImmediateAllocator(queue)),
        )
    else:
        actx = PyOpenCLArrayContext(
            queue,
            allocator=cl_tools.MemoryPool(cl_tools.ImmediateAllocator(queue)),
            force_device_scalars=True,
        )

    from meshmode.mesh.generation import generate_regular_rect_mesh
    mesh = generate_regular_rect_mesh(
            a=(-0.5,)*dim,
            b=(0.5,)*dim,
            nelements_per_axis=(20,)*dim)

    dcoll = DiscretizationCollection(actx, mesh, order=order)

    def source_f(actx, dcoll, t=0):
        source_center = np.array([0.1, 0.22, 0.33])[:dcoll.dim]
        source_width = 0.05
        source_omega = 3
        nodes = thaw(dcoll.nodes(), actx)
        source_center_dist = flat_obj_array(
            [nodes[i] - source_center[i] for i in range(dcoll.dim)]
        )
        return (
            actx.np.sin(source_omega*t)
            * actx.np.exp(
                -np.dot(source_center_dist, source_center_dist)
                / source_width**2
            )
        )

    x = thaw(dcoll.nodes(), actx)
    ones = dcoll.zeros(actx) + 1
    c = actx.np.where(actx.np.less(np.dot(x, x), 0.15), 0.1 * ones, 0.2 * ones)

    from grudge.models.wave import VariableCoefficientWeakWaveOperator
    from meshmode.mesh import BTAG_ALL, BTAG_NONE

    wave_op = VariableCoefficientWeakWaveOperator(
        dcoll,
        c,
        source_f=source_f,
        dirichlet_tag=BTAG_NONE,
        neumann_tag=BTAG_NONE,
        radiation_tag=BTAG_ALL,
        flux_type="upwind"
    )

    fields = flat_obj_array(
        dcoll.zeros(actx),
        [dcoll.zeros(actx) for i in range(dcoll.dim)]
    )

    wave_op.check_bc_coverage(mesh)

    def rhs(t, w):
        return wave_op.operator(t, w)

    dt = 2/3 * wave_op.estimate_rk4_timestep(actx, dcoll, fields=fields)

    final_t = 1
    nsteps = int(final_t/dt) + 1

    logger.info(f"{mesh.elements} elements, dt={dt}, nsteps={nsteps}")

    from grudge.shortcuts import make_visualizer
    vis = make_visualizer(dcoll)

    t = 0.
    step = 1

    def norm(u):
        return op.norm(dcoll, u, 2)

    t_last_step = time()

    if visualize:
        u = fields[0]
        v = fields[1:]
        vis.write_vtk_file(
            f"fld-var-propogation-speed-{step:04d}.vtu",
            [
                ("u", u),
                ("v", v),
                ("c", c),
            ]
        )

    compiled_rhs = actx.compile(rhs)

    while t < final_t:
        # thaw+freeze to see similar expression graphs in rk4
        fields = thaw(freeze(fields, actx), actx)

        fields = rk4_step(fields, t, dt, compiled_rhs)

        if step % 10 == 0:
            actx.queue.finish()
            logger.info(f"step: {step} t: {time()-t_last_step} secs. "
                        f"L2: {norm(u=fields[0])}")

            if visualize:
                vis.write_vtk_file(
                    f"fld-var-propogation-speed-{step:04d}.vtu",
                    [
                        ("u", fields[0]),
                        ("v", fields[1:]),
                        ("c", c),
                    ]
                )

            assert norm(u=fields[0]) < 1
            t_last_step = time()

        t += dt
        step += 1


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dim", default=3, type=int)
    parser.add_argument("--order", default=4, type=int)
    parser.add_argument("--visualize", action="store_true", default=False)
    parser.add_argument("--lazy", action="store_true", default=False)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    main(cl.create_some_context,
         dim=args.dim,
         order=args.order,
         visualize=args.visualize,
         lazy=args.lazy)
