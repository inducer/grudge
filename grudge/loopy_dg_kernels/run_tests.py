import numpy as np

import pyopencl as cl
import pyopencl.array
import pyopencl.clrandom

import loopy as lp
#from loopy.version import LOOPY_USE_LANGUAGE_VERSION_2018_2
#from loopy.kernel.data import AddressSpace

import pycuda.gpuarray as cuarray
import pycuda.driver as drv
import pycuda.tools
import pycuda.autoinit
from pycuda.compiler import SourceModule
from pycuda.curandom import rand as curand

from modepy import equidistant_nodes

from bs4 import UnicodeDammit
import hjson
import time
#from math import ceil
import sys

# setup
# -----
lp.set_caching_enabled(False)
import loopy.options
loopy.options.ALLOW_TERMINAL_COLORS = False

from grudge.loopy_dg_kernels import (gen_diff_knl, gen_diff_knl_fortran2,
    generate_transformation_list, apply_transformation_list, gen_elwise_linear_knl, gen_face_mass_knl, gen_face_mass_knl_merged)

def testBandwidth(fp_format=np.float32, nruns=100):

    from pyopencl.array import sum as clsum
    platform = cl.get_platforms()
    my_gpu_devices = platform[0].get_devices(device_type=cl.device_type.GPU)
    #ctx = cl.Context(devices=my_gpu_devices)
    ctx = cl.create_some_context(interactive=True)
    queue = cl.CommandQueue(ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)

    from pyopencl.tools import ImmediateAllocator
    allocator = ImmediateAllocator(queue)


    knl = lp.make_copy_kernel("c,c", old_dim_tags="c,c")
    knl = lp.add_dtypes(knl, {"input": fp_format, "output": fp_format})
    knl = knl.copy(target=lp.PyOpenCLTarget(my_gpu_devices[0]))
    n0 = 1
    #knl = lp.split_iname(knl, "i1", 1024//2, inner_tag="l.0", outer_tag="g.0", slabs=(0,1))
    knl = lp.split_iname(knl, "i1", 1024, inner_tag="l.0", outer_tag="g.0", slabs=(0,1))
    #knl = lp.split_iname(knl, "i1", 6*16, outer_tag="g.0") 
    #knl = lp.split_iname(knl, "i1_inner", 16, outer_tag="ilp", inner_tag="l.0", slabs=(0,1)) 
    #knl = lp.split_iname(knl, "i0", n0, inner_tag="l.1", outer_tag="g.1", slabs=(0,0))

    fp_bytes = 8 if fp_format == np.float64 else 4

    # This assumes fp32
    len_list = []
    float_count = 1
    max_floats = 2**29
    while float_count <= max_floats:
        len_list.append(float_count)
        float_count = int(np.ceil(float_count*1.5))
    for i in range(29):
        len_list.append(2**i)
    len_list = sorted(list(set(len_list)))

    #data = np.random.randint(-127, 128, (1,max_bytes), dtype=np.int8)
    #inpt = cl.array.to_device(queue, data, allocator=allocator)

    print(len_list)

    for n in len_list:
    #for i in range(29):

        #n = 2**i
        kern = lp.fix_parameters(knl, n0=n0, n1=n)
        #data = np.random.randint(-127, 128, (1,n), dtype=np.int8)
        #inpt = cl.array.to_device(queue, data, allocator=allocator)
        inpt = cl.clrandom.rand(queue, (n0, n), dtype=fp_format)
        outpt = cl.array.Array(queue, (n0, n), dtype=fp_format, allocator=allocator)
     
        #kern = lp.set_options(kern, "write_code")  # Output code before editing it

        for j in range(2):
            kern(queue, input=inpt, output=outpt)
        dt = 0
        events = []
        for j in range(nruns):
            evt, _ = kern(queue, input=inpt, output=outpt)
            events.append(evt)

        cl.wait_for_events(events)
        for evt in events:
            dt += evt.profile.end - evt.profile.start 
        #queue.finish()
        dt = dt / nruns / 1e9

        nbytes_transferred = 2*fp_bytes*n*n0
        bandwidth = nbytes_transferred / dt / 1e9
        print("{} {}".format(nbytes_transferred, bandwidth))

        #print((inpt - outpt)) 
        diff = (inpt - outpt)
        if  clsum(inpt - outpt) != 0:
            print("INCORRECT COPY")

# There seems to be a pattern here. Maybe pass in a function that returns the arguments
def test_face_mass(queue, kern, backend="OPENCL", nruns=10, warmup=True):

    kern = lp.set_options(kern, "no_numpy")
    kern = lp.set_options(kern, "return_dict")
    for arg in kern.args:
        if arg.name == "vec":
            fp_format = arg.dtype.dtype
            fp_bytes = fp_format.itemsize
            nfaces, nelements, nface_nodes = arg.shape
        elif arg.name == "mat":
            nvol_nodes, _, _ = arg.shape

    CUDA = (backend == "CUDA")
    OPENCL = not CUDA

    if CUDA:
        print("CUDA mode not currently supported")
        exit()
    elif OPENCL:
        #platform = cl.get_platforms()
        #my_gpu_devices = platform[0].get_devices(device_type=cl.device_type.GPU)
        ##ctx = cl.Context(devices=my_gpu_devices)
        #ctx = cl.create_some_context(interactive=True)
        #queue = cl.CommandQueue(ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)

        #kern = lp.set_options(kern, edit_code=False) #Only works for OpenCL?
        kern = lp.set_options(kern, "write_code")  # Output code before editing it
        # Print the Code
        kern = kern.copy(target=lp.PyOpenCLTarget(my_gpu_devices[0]))
        code = lp.generate_code_v2(kern).device_code()
        prog = cl.Program(ctx, code)
        prog = prog.build()
        ptx = prog.get_info(cl.program_info.BINARIES)[0]#.decode(
        #errors="ignore") #Breaks pocl
        dammit = UnicodeDammit(ptx)
        #print(dammit.unicode_markup)
        f = open("ptx.ptx", "w")
        f.write(dammit.unicode_markup)
        f.close()

        from pyopencl.tools import ImmediateAllocator
        allocator = ImmediateAllocator(queue)

        A_dev = cl.clrandom.rand(queue, (nvol_nodes, nfaces, nface_nodes), dtype=fp_format)
        # Might try strides=(nelements*nface_nodes, 1, nelements)
        strides = (fp_bytes*nelements, fp_bytes*1, fp_bytes*nelements*nfaces) #original
        #strides = (fp_bytes*nelements*nface_nodes, 1*fp_bytes, fp_bytes*nelements)
        X_dev = cl.array.Array(queue, (nfaces, nelements, nface_nodes), dtype=fp_format, 
                                strides=strides, allocator=allocator)
        cl.clrandom.fill_rand(X_dev, queue=queue)
        B_dev = cl.array.Array(queue, (nelements, nvol_nodes), dtype=fp_format, allocator=allocator,order="F")

        if warmup:
            for i in range(2):
                kern(queue, result=B_dev, mat=A_dev, vec=X_dev)
            queue.finish()

        sum_time = 0.0
        events = []
        for i in range(nruns):
            evt, _ = kern(queue, result=B_dev, mat=A_dev, vec=X_dev)
            events.append(evt)

        cl.wait_for_events(events)
        for evt in events:
            sum_time += evt.profile.end - evt.profile.start
        sum_time = sum_time / 1e9        
        #queue.finish()

    avg_time = sum_time / nruns

    return (B_dev, A_dev, X_dev), avg_time

def test_face_mass_merged(kern, backend="OPENCL", nruns=10, warmup=True):
    #kern = gen_diff_knl(n_elem, n_in, n_out, k_inner_outer, k_inner_inner,
    #    i_inner_outer, i_inner_inner, j_inner)
    kern = lp.set_options(kern, "no_numpy")
    kern = lp.set_options(kern, "return_dict")
    for arg in kern.args:
        if arg.name == "vec":
            fp_format = arg.dtype
            n_elem, n_in = arg.shape
        elif arg.name == "mat":
            n_out, _ = arg.shape

    CUDA = (backend == "CUDA")
    OPENCL = not CUDA

    if CUDA:
        print("Not supported")
        exit()
    elif OPENCL:
        platform = cl.get_platforms()
        my_gpu_devices = platform[0].get_devices(device_type=cl.device_type.GPU)
        #ctx = cl.Context(devices=my_gpu_devices)
        ctx = cl.create_some_context(interactive=True)
        queue = cl.CommandQueue(ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)

        #kern = lp.set_options(kern, edit_code=False) #Only works for OpenCL?
        kern = lp.set_options(kern, "write_code")  # Output code before editing it
        # Print the Code
        kern = kern.copy(target=lp.PyOpenCLTarget(my_gpu_devices[0]))
        code = lp.generate_code_v2(kern).device_code()
        prog = cl.Program(ctx, code)
        prog = prog.build()
        ptx = prog.get_info(cl.program_info.BINARIES)[0]#.decode(
        #errors="ignore") #Breaks pocl
        dammit = UnicodeDammit(ptx)
        #print(dammit.unicode_markup)
        f = open("ptx.ptx", "w")
        f.write(dammit.unicode_markup)
        f.close()

        from pyopencl.tools import ImmediateAllocator
        allocator = ImmediateAllocator(queue)

        X_dev = cl.array.Array(queue, (n_elem, n_in), dtype=fp_format, order="F", allocator=allocator)
        cl.clrandom.fill_rand(X_dev, queue=queue)
        B_dev = cl.array.Array(queue, (n_elem, n_out), dtype=fp_format, allocator=allocator,order="F")
        A_dev = cl.clrandom.rand(queue, (n_out, n_in), dtype=fp_format)

        if warmup:
            for i in range(2):
                kern(queue, result=B_dev, mat=A_dev, vec=X_dev)
            queue.finish()

        sum_time = 0.0
        events = []
        for i in range(nruns):
            evt, _ = kern(queue, result=B_dev, mat=A_dev, vec=X_dev)
            events.append(evt)

        cl.wait_for_events(events)
        for evt in events:
            sum_time += evt.profile.end - evt.profile.start
        sum_time = sum_time / 1e9        
        #queue.finish()

    avg_time = sum_time / nruns

    return (B_dev, A_dev, X_dev), avg_time


# Maybe the queue could also be a cuda stream? Could use the type of that to
# distinguish between CUDA and OpenCL possibly
def test_elwise_linear(queue, kern, backend="OPENCL", nruns=10, warmup=True):

    kern = lp.set_options(kern, "no_numpy")
    kern = lp.set_options(kern, "return_dict")
    for arg in kern.args:
        if arg.name == "vec":
            fp_format = arg.dtype.dtype
            n_elem, n_in = arg.shape
        elif arg.name == "mat":
            n_out, _ = arg.shape

    CUDA = (backend == "CUDA")
    OPENCL = not CUDA

    if CUDA:
        stream = drv.Stream()
        (ygrid, xgrid), (yblk, xblk) = kern.get_grid_size_upper_bounds_as_exprs()
        kern = kern.copy(target=lp.CudaTarget())
        #kern = lp.set_options(kern, edit_code=True)
        code = lp.generate_code_v2(kern).device_code()
        mod = SourceModule(code, keep=True)
        diff_fn = mod.get_function("elwise_linear")
        A_dev = curand((n_out, n_in),  dtype=fp_format, stream=stream)
        X_dev = curand((n_in, n_elem), dtype=fp_format, stream=stream)
        B_dev = cuarray.GPUArray((n_out, n_elem), fp_format)
        print(code)

        if warmup:
            for i in range(2):
                diff_fn(B_dev, A_dev, X_dev, block=(yblk, xblk, 1), grid=(ygrid, xgrid))
            pycuda.autoinit.context.synchronize()

        start = time.time()
        for i in range(nruns):
            diff_fn(B_dev, A_dev, X_dev, block=(yblk, xblk, 1), grid=(ygrid, xgrid))

        pycuda.autoinit.context.synchronize()
        sum_time = time.time() - start

    elif OPENCL:
        #platform = cl.get_platforms()
        #my_gpu_devices = platform[0].get_devices(device_type=cl.device_type.GPU)
        ##ctx = cl.Context(devices=my_gpu_devices)
        #ctx = cl.create_some_context(interactive=True)
        #queue = cl.CommandQueue(ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)

        #kern = lp.set_options(kern, edit_code=False) #Only works for OpenCL?
        kern = lp.set_options(kern, "write_code")  # Output code before editing it
        # Print the Code
        kern = kern.copy(target=lp.PyOpenCLTarget(my_gpu_devices[0]))
        code = lp.generate_code_v2(kern).device_code()
        prog = cl.Program(ctx, code)
        prog = prog.build()
        ptx = prog.get_info(cl.program_info.BINARIES)[0]#.decode(
        #errors="ignore") #Breaks pocl
        dammit = UnicodeDammit(ptx)
        #print(dammit.unicode_markup)
        f = open("ptx.ptx", "w")
        f.write(dammit.unicode_markup)
        f.close()

        from pyopencl.tools import ImmediateAllocator
        allocator = ImmediateAllocator(queue)

        X_dev = cl.array.Array(queue, (n_elem, n_in), fp_format, order="F", allocator=allocator)
        cl.clrandom.fill_rand(X_dev, queue) 
        B_dev = cl.array.Array(queue, (n_elem, n_out), dtype=fp_format, allocator=allocator,order="F")
        A_dev = cl.clrandom.rand(queue, (n_out, n_in), dtype=fp_format)

        if warmup:
            for i in range(2):
                kern(queue, result=B_dev, mat=A_dev, vec=X_dev)
            queue.finish()

        sum_time = 0.0
        events = []
        for i in range(nruns):
            evt, _ = kern(queue, result=B_dev, mat=A_dev, vec=X_dev)
            events.append(evt)

        cl.wait_for_events(events)
        for evt in events:
            sum_time += evt.profile.end - evt.profile.start
        sum_time = sum_time / 1e9        
        #queue.finish()

    avg_time = sum_time / nruns

    return (B_dev, A_dev, X_dev), avg_time


def test_diff(queue, kern, backend="OPENCL", nruns=10, warmup=True):
    for arg in kern.args:
        if arg.name == "diff_mat":
            n_mat, n_out, n_in = arg.shape
        elif arg.name == "vec":
            n_elem, _ = arg.shape
            fp_format = arg.dtype.dtype

    kern = lp.set_options(kern, "no_numpy")
    kern = lp.set_options(kern, "return_dict")

    # Can probably set this based on type of queue object
    CUDA = (backend == "CUDA")
    OPENCL = not CUDA

    if CUDA:
        stream = drv.Stream()
        (ygrid, xgrid), (yblk, xblk) = kern.get_grid_size_upper_bounds_as_exprs()
        kern = kern.copy(target=lp.CudaTarget())
        #kern = lp.set_options(kern, edit_code=True)
        code = lp.generate_code_v2(kern).device_code()
        mod = SourceModule(code, keep=True)
        diff_fn = mod.get_function("opt_diff")
        A_dev1 = curand((n_out, n_in),  dtype=fp_format, stream=stream)
        A_dev2 = curand((n_out, n_in),  dtype=fp_format, stream=stream)
        A_dev3 = curand((n_out, n_in),  dtype=fp_format, stream=stream)
        X_dev = curand((n_in, n_elem), dtype=fp_format, stream=stream)
        B_dev1 = cuarray.GPUArray((n_out, n_elem), fp_format)
        B_dev2 = cuarray.GPUArray((n_out, n_elem), fp_format)
        B_dev3 = cuarray.GPUArray((n_out, n_elem), fp_format)
        print(code)
    elif OPENCL:

        #kern = lp.set_options(kern, edit_code=False) #Only works for OpenCL?
        kern = lp.set_options(kern, "write_code")  # Output code before editing it

        # Print the Code
        """
        kern = kern.copy(target=lp.PyOpenCLTarget(my_gpu_devices[0]))
        code = lp.generate_code_v2(kern).device_code()
        prog = cl.Program(ctx, code)
        prog = prog.build()
        ptx = prog.get_info(cl.program_info.BINARIES)[0]#.decode(
        #errors="ignore") #Breaks pocl
        dammit = UnicodeDammit(ptx)
        print(dammit.unicode_markup)
        """

        from pyopencl.tools import ImmediateAllocator
        allocator = ImmediateAllocator(queue)
        X_dev = cl.array.Array(queue, (n_elem, n_in), fp_format, order="F", allocator=allocator)
        cl.clrandom.fill_rand(X_dev, queue)
        from pytools.obj_array import make_obj_array
        B_dev, A_dev = [], []
        for i in range(n_mat):
            B_dev.append(cl.array.Array(queue, (n_elem, n_out), dtype=fp_format, allocator=allocator,order="F"))
            A_dev.append(cl.clrandom.rand(queue, (n_out, n_in), dtype=fp_format))
        B_dev = make_obj_array(B_dev)
        A_dev = make_obj_array(A_dev)
 
        # Code for running binaries
        #device = queue.get_info(cl.command_queue_info.DEVICE)
        #with open("cuda_32x32.ptx", "rb") as f:
        #    binary = f.read()
        #prg = cl.Program(ctx, [device], [binary])
        #prg = prg.build()
        #diff_knl = prg.diff

    if warmup==True:
        for i in range(2):
            if OPENCL:
                #diff_knl.set_args(B_dev.data, A_dev.data, X_dev.data)
                #evt = cl.enqueue_nd_range_kernel(queue, diff_knl, 
                kern(queue, result=B_dev, diff_mat=A_dev, vec=X_dev)
            elif CUDA:
                diff_fn(B_dev1, B_dev2, B_dev3, A_dev1, A_dev2, A_dev3, X_dev,
                    block=(yblk, xblk, 1), grid=(ygrid, xgrid))

    if OPENCL:
        queue.finish()
    elif CUDA:
        pycuda.autoinit.context.synchronize()

    sum_time = 0.0
    events = []
    start = time.time()
    for i in range(nruns):
        if OPENCL:
            evt, _ = kern(queue, result=B_dev, diff_mat=A_dev, vec=X_dev)
            events.append(evt)
        elif CUDA:
            diff_fn(B_dev1, B_dev2, B_dev3, A_dev1, A_dev2, A_dev3, X_dev,
                block=(yblk, xblk, 1), grid=(ygrid, xgrid))
    if OPENCL:
        cl.wait_for_events(events)
        for evt in events:
            sum_time += evt.profile.end - evt.profile.start
        sum_time = sum_time / 1e9
    elif CUDA:
        pycuda.autoinit.context.synchronize()
        sum_time = time.time() - start

    avg_time = sum_time / nruns

    return (B_dev, A_dev, X_dev), avg_time


def analyze_knl_bandwidth(knl, avg_time):
    nbytes = 0
    for arg in knl.args:
        print(arg.name)
        print(arg.shape)
        print(type(arg.dtype))
        entries = np.prod((arg.shape))
        fp_bytes = 8 if arg.dtype.dtype == np.float64 else 4
        nbytes += fp_bytes * entries
    bw = nbytes / avg_time / 1e9
    print("Time: {}, Bytes: {}, Bandwidth: {} GB/s".format(avg_time, nbytes, bw))


def analyzeResult(n_out, n_in, n_elem, peak_gflops, device_memory_bandwidth,
                    avg_time, fp_bytes=4):

    flops = 3*2*(n_out * n_in * n_elem)
    gflop_rate = (flops / avg_time) * 1e-9
    #peak_gflop_rate = 12288
    #peak_gflop_rate = 368 # (8 flops / cycle) * 2.3 Hertz * 20 cores
    frac_peak_gflops = gflop_rate / peak_gflops
    print("GFLOP/s: " + str(gflop_rate))
    print("Peak GFLOP/s: " + str(peak_gflops))
    print("Percent peak: " + str(100*(frac_peak_gflops)))
    print()

    # Calculate bandwidth
    # Assumes each element only read once
    ideal_total_bytes_transferred = fp_bytes*(3*(n_out * n_elem) + (n_in * n_elem)
                                                + 3*(n_out * n_in))
    GBps = (ideal_total_bytes_transferred / avg_time) / 1e9
    frac_peak_GBps = GBps / device_memory_bandwidth
    print("GB/s: " + str(GBps))
    print("Peak GB/s: " + str(device_memory_bandwidth))
    print("Percent peak: " + str(100*(frac_peak_GBps)))
    print()
    return frac_peak_gflops, frac_peak_GBps


def verifyResult(B_dev1, B_dev2, B_dev3, A_dev1, A_dev2, A_dev3, X_dev):
    A_host1 = A_dev1.get()
    A_host2 = A_dev2.get()
    A_host3 = A_dev3.get()
    X_host = X_dev.get()
    B_host1 = B_dev1.get()
    B_host2 = B_dev2.get()
    B_host3 = B_dev3.get()
    np.set_printoptions(threshold=sys.maxsize)
    errMat = ((A_host1 @ X_host) - B_host1) / np.linalg.norm(A_host1 @ X_host)
    print("Fraction Nonzero: " + str(np.count_nonzero(errMat)/(n_out*n_elem)))
    print("Norm1: " + str(np.linalg.norm((A_host1 @ X_host) - B_host1)
            / np.linalg.norm(A_host1 @ X_host)))
    print("Norm2: " + str(np.linalg.norm((A_host2 @ X_host) - B_host2)
            / np.linalg.norm(A_host2 @ X_host)))
    print("Norm3: " + str(np.linalg.norm((A_host3 @ X_host) - B_host3)
            / np.linalg.norm(A_host3 @ X_host)))


def verifyResultFortran(B_dev1, B_dev2, B_dev3, A_dev1, A_dev2, A_dev3, X_dev):
    A_host1 = A_dev1.get()
    A_host2 = A_dev2.get()
    A_host3 = A_dev3.get()
    X_host = X_dev.get().T
    B_host1 = B_dev1.get()
    B_host2 = B_dev2.get()
    B_host3 = B_dev3.get()
    np.set_printoptions(threshold=sys.maxsize)
    errMat = ((A_host1 @ X_host).T - B_host1) / np.linalg.norm(A_host1 @ X_host)
    print("Fraction Nonzero: " + str(np.count_nonzero(errMat)/(n_out*n_elem)))
    print("Norm1: " + str(np.linalg.norm((A_host1 @ X_host).T - B_host1)
            / np.linalg.norm(A_host1 @ X_host)))
    print("Norm2: " + str(np.linalg.norm((A_host2 @ X_host).T - B_host2)
            / np.linalg.norm(A_host2 @ X_host)))
    print("Norm3: " + str(np.linalg.norm((A_host3 @ X_host).T - B_host3)
            / np.linalg.norm(A_host3 @ X_host)))


def k_inner_inner_options(reverse=True):
    return sorted([8, 16, 32], reverse=reverse)


def k_inner_outer_options(n_in, k_inner_inner, sm_size,
                            fp_bytes=4, reverse=False):
    # Possibilities limited by size of global memory
    options = np.arange(1, (sm_size // (fp_bytes*k_inner_inner*n_in)) + 1)
    #Arbitrarily limit to at max 12 inline to limit search space
    options = k_inner_inner*options[options <= 12]
    return sorted(options, reverse=reverse)


def i_inner_inner_options(n_out, k_inner_inner, max_work_group_size=1024, reverse=True):
    factors = np.arange(2, n_out+1)[(n_out % np.arange(2, n_out+1)) == 0]
    # Ensure total number of workitems is less than maximum
    usable_factors = factors[factors*k_inner_inner <= max_work_group_size]
    return sorted(usable_factors, reverse=reverse)


def i_inner_outer_options(n_out, i_inner_inner, reverse=False):
    # Select a number of inline blocks such that n_out % outer*inner == 0
    inline = np.arange(1, (n_out // i_inner_inner) + 1)
    options = i_inner_inner*inline[n_out % (inline*i_inner_inner) == 0]
    return sorted(options, reverse=reverse)


def j_inner_options(n_in, reverse=False):
    factors = list(np.arange(1, n_in + 1)[(n_in % np.arange(1, n_in + 1)) == 0])
    return sorted(factors, reverse=reverse)

def exhaustive_search(queue, knl, test_fn, time_limit=float("inf"), max_gflops=None, 
        device_memory_bandwidth=None, gflops_cutoff=0.95, bandwidth_cutoff=0.95):

    # Imports
    from random import choice
    from grudge.grudge_tags import ParameterValue
    from grudge.grudge_array_context import set_memory_layout

    local_mem_size = queue.device.local_mem_size
    max_work_group_size = queue.device.max_work_group_size    

    avg_time_saved = float("inf")
    result_saved = None

    # Kernel specific logic 
    if "diff" in knl.name:
        for arg in knl.args:
            if arg.name == "vec":
                n_elem, n_out = arg.shape
                fp_bytes = arg.dtype.dtype.itemsize
            elif arg.name == "diff_mat":
                n_mat, n_out, n_in = arg.shape
        knl = lp.tag_inames(knl, "imatrix: ilp")

    elif knl.name == "elwise_linear":
        for arg in knl.args:
            if arg.name == "vec":
                n_elem, n_out = arg.shape
                fp_bytes = arg.dtype.dtype.itemsize
            elif arg.name == "mat":
                n_out, n_in = arg.shape      

    elif knl.name == "face_mass":
        for arg in knl.args:
            if arg.name == "result":
                n_elem, n_out = arg.shape
                fp_bytes = arg.dtype.dtype.itemsize
            elif arg.name == "vec":
                nfaces, n_elem, n_in = arg.shape        

    # Also fixes the parameters    
    knl = set_memory_layout(knl)

    tested = []

    k_inner_inner_opt = k_inner_inner_options()
    j_inner_opt = j_inner_options(n_in)
    knl_base = knl.copy()

    avg_time_saved = float("inf")
    result_saved = None
    
    # Iterate over five search dimensions
    start = time.time()
    for kii in k_inner_inner_opt:
        # This prevents shared memory from overflowing when running with the face mass kernel
        if knl.name == "face_mass":
            n_in_2 = n_in * nfaces
        else:
            n_in_2 = n_in
        for kio in k_inner_outer_options(n_in_2, kii, local_mem_size, fp_bytes=fp_bytes):
            for iii in i_inner_inner_options(n_out, kii,
                    max_work_group_size=max_work_group_size):
                for iio in i_inner_outer_options(n_out, iii):
                    for ji in j_inner_opt:

                        # Transform and run
                        knl = knl_base.copy()
                        knl = lp.split_iname(knl, "iel", kio, outer_tag="g.0", slabs=(0,1))
                        knl = lp.split_iname(knl, "iel_inner", kii, outer_tag="ilp", inner_tag="l.0", slabs=(0,1))
                        knl = lp.split_iname(knl, "idof", iio, outer_tag="g.1", slabs=(0,0))
                        knl = lp.split_iname(knl, "idof_inner", iii, outer_tag="ilp", inner_tag="l.1", slabs=(0,0))        

                        if knl.name == "face_mass":
                            knl = lp.add_prefetch(knl, "vec", "f,j,iel_inner_outer,iel_inner_inner", temporary_name="vecf", default_tag="l.auto")
                            knl = lp.tag_array_axes(knl, "vecf", "N1,N0,N2")
                        else:   
                            knl = lp.add_prefetch(knl, "vec", "j,iel_inner_outer,iel_inner_inner", temporary_name="vecf", default_tag="l.auto")
                            knl = lp.tag_array_axes(knl, "vecf", "f,f")

                        knl = lp.split_iname(knl, "j", ji, outer_tag="for", inner_tag="for")

                        dev_arrays, avg_time = test_fn(queue, knl)

                        # Analyze the results
                        if max_gflops is not None and device_memory_bandwidth is not None:  # noqa
                            frac_peak_gflops, frac_peak_GBps = analyzeResult(n_out,
                                n_in, n_elem, max_gflops, device_memory_bandwidth,
                                avg_time)
                            if frac_peak_gflops >= gflops_cutoff or frac_peak_GBps  >= bandwidth_cutoff:  # noqa
                                # Should validate result here
                                print("Performance is within tolerance of peak bandwith or flop rate. Terminating search")  # noqa
                                return (kio, kii, iio, iii, ji)
                        if avg_time < avg_time_saved:
                            avg_time_saved = avg_time
                            result_saved = (kio, kii, iio, iii, ji)
                        if time.time() - start > time_limit:
                            return result_saved

    print("Suggested loop splittings")
    print(f"iel: {kio}")
    print(f"iel_inner: {kii}")
    print(f"idof: {iio}")
    print(f"idof_inner: {iii}")
    print(f"j: {ji}")
 
    return result_saved

def random_search(queue, knl, test_fn, time_limit=float("inf"), max_gflops=None, 
        device_memory_bandwidth=None, gflops_cutoff=0.95, bandwidth_cutoff=0.95):

    # Imports
    from random import choice
    from grudge.grudge_tags import ParameterValue
    from grudge.grudge_array_context import set_memory_layout

    local_mem_size = queue.device.local_mem_size
    max_work_group_size = queue.device.max_work_group_size    

    avg_time_saved = float("inf")
    result_saved = None

    # Kernel specific logic 
    if "diff" in knl.name:
        for arg in knl.args:
            if arg.name == "vec":
                n_elem, n_out = arg.shape
                fp_bytes = arg.dtype.dtype.itemsize
            elif arg.name == "diff_mat":
                n_mat, n_out, n_in = arg.shape
        knl = lp.tag_inames(knl, "imatrix: ilp")

    elif knl.name == "elwise_linear":
        for arg in knl.args:
            if arg.name == "vec":
                n_elem, n_out = arg.shape
                fp_bytes = arg.dtype.dtype.itemsize
            elif arg.name == "mat":
                n_out, n_in = arg.shape      

    elif knl.name == "face_mass":
        for arg in knl.args:
            if arg.name == "result":
                n_elem, n_out = arg.shape
                fp_bytes = arg.dtype.dtype.itemsize
            elif arg.name == "vec":
                nfaces, n_elem, n_in = arg.shape        

    # Also fixes the parameters    
    knl = set_memory_layout(knl)

    tested = []

    k_inner_inner_opt = k_inner_inner_options()
    j_inner_opt = j_inner_options(n_in)
    knl_base = knl.copy()

    start = time.time()
    while(time.time() - start < time_limit):
        # Can be more intelligent by ensuring choices are not run multiple times
        # Maybe could use expressions
        kii = choice(k_inner_inner_opt)
        if knl.name == "face_mass":
            kio = choice(k_inner_outer_options(n_in*nfaces, kii, local_mem_size, fp_bytes=fp_bytes))
        else:
            kio = choice(k_inner_outer_options(n_in, kii, local_mem_size, fp_bytes=fp_bytes))
        iii = choice(i_inner_inner_options(n_out, kii, max_work_group_size=max_work_group_size))
        iio = choice(i_inner_outer_options(n_out, iii))
        ji = choice(j_inner_opt)
        choices = (kio, kii, iio, iii, ji)

        if choices not in tested:
            print(choices)
            knl = knl_base.copy()
            knl = lp.split_iname(knl, "iel", kio, outer_tag="g.0", slabs=(0,1))
            knl = lp.split_iname(knl, "iel_inner", kii, outer_tag="ilp", inner_tag="l.0", slabs=(0,1))
            knl = lp.split_iname(knl, "idof", iio, outer_tag="g.1", slabs=(0,0))
            knl = lp.split_iname(knl, "idof_inner", iii, outer_tag="ilp", inner_tag="l.1", slabs=(0,0))        

            if knl.name == "face_mass":
                knl = lp.add_prefetch(knl, "vec", "f,j,iel_inner_outer,iel_inner_inner", temporary_name="vecf", default_tag="l.auto")
                knl = lp.tag_array_axes(knl, "vecf", "N1,N0,N2")
            else:   
                knl = lp.add_prefetch(knl, "vec", "j,iel_inner_outer,iel_inner_inner", temporary_name="vecf", default_tag="l.auto")
                knl = lp.tag_array_axes(knl, "vecf", "f,f")

            knl = lp.split_iname(knl, "j", ji, outer_tag="for", inner_tag="for")

            dev_arrays, avg_time = test_fn(queue, knl)
            tested.append(choices)

            if max_gflops is not None and device_memory_bandwidth is not None:
                frac_peak_gflops, frac_peak_GBps = analyzeResult(n_out, n_in, n_elem,
                    max_gflops, device_memory_bandwidth, avg_time)
                if frac_peak_gflops >= gflops_cutoff or frac_peak_GBps >= bandwidth_cutoff:  # noqa
                    # Should validate result here
                    print("Performance is within tolerance of peak bandwith or flop rate. Terminating search")  # noqa
                    return choices
            if avg_time < avg_time_saved:
                avg_time_saved = avg_time
                result_saved = choices

    print("Time limit exceeded: returning current best result")

    print("Suggested loop splittings")
    print(f"iel: {kio}")
    print(f"iel_inner: {kii}")
    print(f"idof: {iio}")
    print(f"idof_inner: {iii}")
    print(f"j: {ji}")
    
    return result_saved

def get_transformation_id(device_id):
    hjson_file = open("device_mappings.hjson") 
    hjson_text = hjson_file.read()
    hjson_file.close()
    od = hjson.loads(hjson_text)
    return od[device_id]

if __name__ == "__main__": 
    from __init__ import gen_diff_knl, load_transformations_from_file, apply_transformation_list

    # Test existing optimizations
    platform = cl.get_platforms()
    my_gpu_devices = platform[0].get_devices(device_type=cl.device_type.GPU)
    #ctx = cl.Context(devices=my_gpu_devices)
    ctx = cl.create_some_context(interactive=True)
    queue = cl.CommandQueue(ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)
    
    # Testing code
    device_id = "NVIDIA Titan V"
    tid = get_transformation_id("NVIDIA Titan V")
    fp_format = np.float64
    fp_format_dict = {np.float32: (4, "FP32"), np.float64: (8, "FP64"),
                        np.complex128: (16, "C128")}
    fp_bytes, fp_string = (8, "FP64") if fp_format == np.float64 else (4, "FP32")

    """
    to_test = True
    if to_test:
        n_elem = 2**22#2**15  # 2**21
        pn = 5
        print(len(equidistant_nodes(pn, 3)[1]))
        n_out = len(equidistant_nodes(pn, 3)[1])
        n_in = len(equidistant_nodes(pn, 3)[1])

        #settings = exhaustiveSearch(n_in, n_out, n_elem, 4*12*1024, fp_bytes=fp_bytes,
        #               max_gflops=12288, device_memory_bandwidth=540)
        settings = randomSearch(n_in, n_out, n_elem, 4*12*1024, time_limit=120,
                        fp_format=fp_format, max_gflops=12288//2,
                        device_memory_bandwidth=540)
        #settings = noSearch(n_in, n_out, n_elem, 4*12*1024, time_limit=180,1
        #                       fp_bytes=fp_bytes, max_gflops=12288,
        #                       device_memory_bandwidth=540)
        print("FINAL RESULTS")
        print(settings)
    # Add functionality to write transformations to file
    """ 
    """
    dim_to_file = {1: "diff_1d_transform.hjson", 
                   2: "diff_2d_transform.hjson",
                   3: "diff_3d_transform.hjson"}

    for dim in range(3,4):
        hjson_file = open(dim_to_file[dim])
        #for i in range(2,8):
        pn = 3
        n_out = len(equidistant_nodes(pn, 3)[1])
        n_in = len(equidistant_nodes(pn, 3)[1]) 
        n_elem = 178746 # 2**20
        knl = gen_diff_knl_fortran2(dim, n_elem, n_out, n_in, fp_format=fp_format)
        trans = load_transformations_from_file(hjson_file, [tid, fp_string, str(pn)])
        knl = apply_transformation_list(knl, trans)
        #print(lp.generate_code_v2(knl).device_code())

        dev_arrays, avg_time = test_diff(queue, knl, nruns=10, warmup=True)
        #dev_arrays, avg_time = runTest(n_elem, n_in, n_out, kio, kii, iio, iii, ji)
        analyze_knl_bandwidth(knl, avg_time)
        #analyzeResult(n_out, n_in, n_elem, 12288//2, 540, avg_time, fp_bytes=fp_bytes)
        print(avg_time)
        #verifyResult(*dev_arrays)

    """
    #testBandwidth()
    """
    # Test elwise linear
    pn = 3
    n_out = len(equidistant_nodes(pn,3)[1])
    n_in = n_out
    n_elem = 178746
    fp_format = np.float64
    fp_string = "FP64" if fp_format == np.float64 else "FP32" 
    knl = gen_elwise_linear_knl(n_elem, n_in, n_out, fp_format)

    hjson_file = open("elwise_linear_transform.hjson")
    trans = load_transformations_from_file(hjson_file, [tid, fp_string, str(pn)])
    knl = apply_transformation_list(knl, trans)
    #print(knl)
    _, avg_time = test_elwise_linear(queue, knl, backend="OPENCL", nruns=10, warmup=True)
    print(avg_time)
    analyze_knl_bandwidth(knl, avg_time)
    """
    """
    # Test face_mass            
    pn = 3
    nvol_nodes = len(equidistant_nodes(pn,3)[1])
    nface_nodes = 10
    #nelements = 2**22
    nelements = 178746
    nfaces = 4
    fp_format = np.float64
    fp_string = "FP64" if fp_format == np.float64 else "FP32" 
    knl = gen_face_mass_knl(nelements, nfaces, nvol_nodes, nface_nodes, fp_format)
    #knl = gen_face_mass_knl_merged(nelements, nfaces, nvol_nodes, nface_nodes, fp_format)
    #hjson_file = open("elwise_linear_transform.hjson")
    #trans = load_transformations_from_file(hjson_file, [tid, fp_string, str(pn)])
    #knl = apply_transformation_list(knl, trans)
    print(knl)
    _, avg_time = test_face_mass(knl, backend="OPENCL", nruns=10, warmup=True)
    #_, avg_time = test_face_mass_merged(knl, backend="OPENCL", nruns=10, warmup=True)
    print(avg_time)
    analyze_knl_bandwidth(knl, avg_time)
    """

    # Test autotuner
    from grudge.execution import diff_prg, elwise_linear_prg, face_mass_prg
    knl = diff_prg(1, 178746, 20, np.float64)
    knl = elwise_linear_prg(178746, 20, np.float64)
    # Figure out the actual dimensions
    #knl = face_mass_prg(178746, 4, 20, 20, np.float64)
    #print(knl)
    #exit()
    exhaustive_search(queue, knl, test_elwise_linear, time_limit=60, max_gflops=6144, device_memory_bandwidth=580, gflops_cutoff=0.95, bandwidth_cutoff=0.95)    
