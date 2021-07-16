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

from grudge.grudge_array_context import GrudgeArrayContext, AutoTuningArrayContext
from meshmode.array_context import PyOpenCLArrayContext  # noqa F401
from meshmode.dof_array import thaw

from meshmode.mesh import BTAG_ALL, BTAG_NONE  # noqa

from grudge.discretization import DiscretizationCollection
import grudge.op as op
from grudge.shortcuts import make_visualizer
from grudge.symbolic.primitives import TracePair


# {{{ wave equation bits

def wave_flux(dcoll, c, w_tpair):
    u = w_tpair[0]
    v = w_tpair[1:]

    normal = thaw(u.int.array_context, op.normal(dcoll, w_tpair.dd))

    flux_weak = flat_obj_array(
            np.dot(v.avg, normal),
            normal*u.avg,
            )

    # upwind
    flux_weak += flat_obj_array(
            0.5*(u.ext-u.int),
            0.5*normal*np.dot(normal, v.ext-v.int),
            )

    return op.project(dcoll, w_tpair.dd, "all_faces", c*flux_weak)

#'''
def wave_operator(discr, c, w):
    from pyopencl import MemoryError
    from pyopencl.array import Array
    try:

        u = w[0]
        v = w[1:]

        dir_u = op.project(discr, "vol", BTAG_ALL, u)
        dir_v = op.project(discr, "vol", BTAG_ALL, v)
        dir_bval = flat_obj_array(dir_u, dir_v)
        neg_dir_u = -dir_u; del dir_u
        dir_bc = flat_obj_array(neg_dir_u, dir_v)
        #print(discr._discr_scoped_subexpr_name_to_value.keys())
        div = op.weak_local_div(discr,v)

        #print(discr._discr_scoped_subexpr_name_to_value.keys())

        neg_c_div = (-c)*div; del div

        #print(discr._discr_scoped_subexpr_name_to_value.keys())
        grad = op.weak_local_grad(discr,u)

        neg_c_grad = (-c)*grad; del grad
        obj_array = flat_obj_array(neg_c_div, neg_c_grad)

        trace_pair1 = op.interior_trace_pair(discr, w)
        wave_flux1 = wave_flux(discr, c=c, w_tpair=trace_pair1)
        del trace_pair1

        trace_pair2 = TracePair(BTAG_ALL, interior=dir_bval, exterior=dir_bc)
        wave_flux2 = wave_flux(discr, c=c, w_tpair=trace_pair2)
        del trace_pair2
        del dir_bc
        del neg_dir_u
        del dir_v
        del dir_bval

        wave_flux_sum = wave_flux1 + wave_flux2;
        """
        print("####################")
        print(type(wave_flux_sum))
        for entry in wave_flux_sum:
            print(type(entry))
            print(entry._data.shape)
        """

        del wave_flux1
        del wave_flux2

        face_mass = op.face_mass(discr, wave_flux_sum)
        del wave_flux_sum

        inverse_arg = obj_array + face_mass
        """
        print("@@@@@@@@@@@@@@@@@@@@@")
        print(type(inverse_arg))
        for entry in inverse_arg:
            print(type(entry))
            print(type(entry._data))
            print(len(entry._data))
            print(entry._data[0].shape)
        exit()
        """

        del obj_array
        del face_mass
        del neg_c_div
        del neg_c_grad

        result = op.inverse_mass(discr,inverse_arg)
        del inverse_arg

        """
        # Original version
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
                        wave_flux(discr, c=c, w_tpair=op.interior_trace_pair(discr, w))
                        + wave_flux(discr, c=c, w_tpair=TracePair(
                            BTAG_ALL, interior=dir_bval, exterior=dir_bc))
                        ))
                    )
        """

        from time import sleep
        sleep(3)
        #print_allocated_arrays() 

        scoped = discr._discr_scoped_subexpr_name_to_value
        print(len(scoped.items()))
        print(scoped.keys())
        sum = 0
        for value in scoped.values():
            #print(type(value))
            if isinstance(value._data, tuple):
                for entry in value._data:
                    print(entry.shape)
                    sum += entry.shape[0]*entry.shape[1]*8
            else:
                print(value._data.shape)
                sum += value._data.shape[0]*value_data.shape[1]*8
        print(sum / 1e9)
        #exit()

    except MemoryError:
        for key, value in Array.alloc_dict.items():
            print("{} {}".format(key, value[1]/1e9))
            for entry in value[0]:
                print(entry)
            print()
        exit() 


    return (result)
#'''

"""
def wave_operator(dcoll, c, w):
    u = w[0]
    v = w[1:]

    dir_u = op.project(dcoll, "vol", BTAG_ALL, u)
    dir_v = op.project(dcoll, "vol", BTAG_ALL, v)
    dir_bval = flat_obj_array(dir_u, dir_v)
    dir_bc = flat_obj_array(-dir_u, dir_v)

    return (
            op.inverse_mass(dcoll,
                flat_obj_array(
                    -c*op.weak_local_div(dcoll, v),
                    -c*op.weak_local_grad(dcoll, u)
                    )
                +  # noqa: W504
                op.face_mass(dcoll,
                    wave_flux(dcoll, c=c, w_tpair=op.interior_trace_pair(dcoll, w))
                    + wave_flux(dcoll, c=c, w_tpair=TracePair(
                        BTAG_ALL, interior=dir_bval, exterior=dir_bc))
                    ))
                )
"""
# }}}


def rk4_step(y, t, h, f):
    k1 = f(t, y)
    kSum = k1
    h2k1 = (h/2)*k1
    del k1
    yph2k1 = y + h2k1
    del h2k1
    k2 = f(t+h/2, y + yph2k1)
    #k2 = f(t+h/2, y + h/2*k1)
    twok2 = 2*k2
    kSum = kSum + twok2
    del twok2
    h2k2 = (h/2)*k2
    del k2
    yph2k2 = y + h2k2
    k3 = f(t+h/2, yph2k2)
    #k3 = f(t+h/2, y + h/2*k2)
    twok3 = 2*k3
    kSum = kSum + twok3
    del twok3
    hk3 = h*k3
    del k3
    yphk3 = y + hk3
    del hk3
    k4 = f(t+h, yphk3)
    kSum = kSum + k4
    del k4
    h6kSum = (h/6)*kSum
    del kSum
    return y + h6kSum
    #return y + h/6*(k1 + 2*k2 + 2*k3 + k4)


def bump(actx, dcoll, t=0):
    source_center = np.array([0.2, 0.35, 0.1])[:dcoll.dim]
    source_width = 0.05
    source_omega = 3

    nodes = thaw(actx, op.nodes(dcoll))
    center_dist = flat_obj_array([
        nodes[i] - source_center[i]
        for i in range(dcoll.dim)
        ])

    return (
        np.cos(source_omega*t)
        * actx.np.exp(
            -np.dot(center_dist, center_dist)
            / source_width**2))


def main():
    cl_ctx = cl.create_some_context()
    queue = cl.CommandQueue(cl_ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)
    from pyopencl.tools import ImmediateAllocator
    actx = AutoTuningArrayContext(queue, allocator=ImmediateAllocator(queue))

    dim = 3
    nel_1d = 2**5 # Order 6 runs out of memory with 2**5
    from meshmode.mesh.generation import generate_regular_rect_mesh
    mesh = generate_regular_rect_mesh(
            coord_dtype=np.float64,
            a=(-0.5,)*dim,
            b=(0.5,)*dim,
            nelements_per_axis=(nel_1d,)*dim)

    order = 4

    if dim == 2:
        # no deep meaning here, just a fudge factor
        dt = 0.7/(nel_1d*order**2)
    elif dim == 3:
        # no deep meaning here, just a fudge factor
        dt = 0.45/(nel_1d*order**2)
    else:
        raise ValueError("don't have a stable time step guesstimate")

    print("%d elements" % mesh.nelements)

    dcoll = DiscretizationCollection(actx, mesh, order=order)

    fields = flat_obj_array(
            bump(actx, dcoll),
            [dcoll.zeros(actx) for i in range(dcoll.dim)]
            )

    vis = make_visualizer(dcoll)

    def rhs(t, w):
        return wave_operator(dcoll, c=1, w=w)

    t = 0
    t_final = (21)*dt
    istep = 0
    while t < t_final:

        print(f"===========TIME STEP {istep}===========")
        fields = rk4_step(fields, t, dt, rhs)

        if istep % 10 == 0:
            print(f"step: {istep} t: {t} L2: {op.norm(dcoll, fields[0], 2)} "
                  f"sol max: {op.nodal_max(dcoll, 'vol', fields[0])}")
            vis.write_vtk_file("fld-wave-eager-%04d.vtu" % istep,
                    [
                        ("u", fields[0]),
                        ("v", fields[1:]),
                        ])

        print(f"===========END TIME STEP {istep}===========")
        istep += 1
        t = istep*dt

        # Should compare against base version at some point
        #assert op.norm(dcoll, fields[0], 2) < 1


if __name__ == "__main__":
    main()

# vim: foldmethod=marker
