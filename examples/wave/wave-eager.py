__copyright__ = "Copyright (C) 2020 Andreas Kloeckner"

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
import numpy.linalg as la  # noqa
import pyopencl as cl

from pytools.obj_array import flat_obj_array

from meshmode.array_context import PyOpenCLArrayContext
from meshmode.dof_array import thaw

from meshmode.mesh import BTAG_ALL, BTAG_NONE  # noqa

from grudge.eager import EagerDGDiscretization, interior_trace_pair
from grudge.shortcuts import make_visualizer
from grudge.symbolic.primitives import TracePair


# {{{ wave equation bits

def wave_flux(discr, c, w_tpair):
    u = w_tpair[0]
    v = w_tpair[1:]

    normal = thaw(u.int.array_context, discr.normal(w_tpair.dd))

    flux_weak = flat_obj_array(
            np.dot(v.avg, normal),
            normal*u.avg,
            )

    # upwind
    flux_weak += flat_obj_array(
            0.5*(u.ext-u.int),
            0.5*normal*np.dot(normal, v.ext-v.int),
            )

    return discr.project(w_tpair.dd, "all_faces", c*flux_weak)


def wave_operator(discr, c, w):
    u = w[0]
    v = w[1:]

    dir_u = discr.project("vol", BTAG_ALL, u)
    dir_v = discr.project("vol", BTAG_ALL, v)
    dir_bval = flat_obj_array(dir_u, dir_v)
    dir_bc = flat_obj_array(-dir_u, dir_v)

    return (
            discr.inverse_mass(
                flat_obj_array(
                    -c*discr.weak_div(v),
                    -c*discr.weak_grad(u)
                    )
                +  # noqa: W504
                discr.face_mass(
                    wave_flux(discr, c=c, w_tpair=interior_trace_pair(discr, w))
                    + wave_flux(discr, c=c, w_tpair=TracePair(
                        BTAG_ALL, interior=dir_bval, exterior=dir_bc))
                    ))
                )

# }}}


def rk4_step(y, t, h, f):
    k1 = f(t, y)
    k2 = f(t+h/2, y + h/2*k1)
    k3 = f(t+h/2, y + h/2*k2)
    k4 = f(t+h, y + h*k3)
    return y + h/6*(k1 + 2*k2 + 2*k3 + k4)


def bump(actx, discr, t=0):
    source_center = np.array([0.2, 0.35, 0.1])[:discr.dim]
    source_width = 0.05
    source_omega = 3

    nodes = thaw(actx, discr.nodes())
    center_dist = flat_obj_array([
        nodes[i] - source_center[i]
        for i in range(discr.dim)
        ])

    return (
        np.cos(source_omega*t)
        * actx.np.exp(
            -np.dot(center_dist, center_dist)
            / source_width**2))


def main():
    cl_ctx = cl.create_some_context()
    queue = cl.CommandQueue(cl_ctx)
    actx = PyOpenCLArrayContext(queue)

    dim = 2
    nel_1d = 16
    from meshmode.mesh.generation import generate_regular_rect_mesh
    mesh = generate_regular_rect_mesh(
            a=(-0.5,)*dim,
            b=(0.5,)*dim,
            n=(nel_1d,)*dim)

    order = 3

    if dim == 2:
        # no deep meaning here, just a fudge factor
        dt = 0.75/(nel_1d*order**2)
    elif dim == 3:
        # no deep meaning here, just a fudge factor
        dt = 0.45/(nel_1d*order**2)
    else:
        raise ValueError("don't have a stable time step guesstimate")

    print("%d elements" % mesh.nelements)

    discr = EagerDGDiscretization(actx, mesh, order=order)

    fields = flat_obj_array(
            bump(actx, discr),
            [discr.zeros(actx) for i in range(discr.dim)]
            )

    vis = make_visualizer(discr, order+3 if dim == 2 else order)

    def rhs(t, w):
        return wave_operator(discr, c=1, w=w)

    t = 0
    t_final = 3
    istep = 0
    while t < t_final:
        fields = rk4_step(fields, t, dt, rhs)

        if istep % 10 == 0:
            print(f"step: {istep} t: {t} L2: {discr.norm(fields[0])} "
                    f"sol max: {discr.nodal_max('vol', fields[0])}")
            vis.write_vtk_file("fld-wave-eager-%04d.vtu" % istep,
                    [
                        ("u", fields[0]),
                        ("v", fields[1:]),
                        ])

        t += dt
        istep += 1


if __name__ == "__main__":
    main()

# vim: foldmethod=marker
