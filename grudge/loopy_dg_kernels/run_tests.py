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

#from bs4 import UnicodeDammit
#import hjson
import time
#from math import ceil
import sys

# setup
# -----
lp.set_caching_enabled(False)
import loopy.options
loopy.options.ALLOW_TERMINAL_COLORS = False

from grudge.loopy_dg_kernels import (gen_diff_knl, generate_transformation_list,
        apply_transformation_list)


def runTestFortran(n_elem, n_in, n_out, k_inner_outer, k_inner_inner, i_inner_outer,
        i_inner_inner, j_inner, backend="CUDA", fp_format=np.float32, nruns=10):

    kern = gen_diff_knl(n_elem, n_in, n_out, k_inner_outer, k_inner_inner,
        i_inner_outer, i_inner_inner, j_inner)
    kern = lp.set_options(kern, "no_numpy")

    CUDA = (backend == "CUDA")
    OPENCL = not CUDA

    if CUDA:
        stream = drv.Stream()
        (ygrid, xgrid), (yblk, xblk) = kern.get_grid_size_upper_bounds_as_exprs()
        kern = kern.copy(target=lp.CudaTarget())
        #kern = lp.set_options(kern, edit_code=True)
        code = lp.generate_code_v2(kern).device_code()
        mod = SourceModule(code, keep=True)
        diff_fn = mod.get_function("diff")
        A_dev1 = curand((n_out, n_in),  dtype=fp_format, stream=stream)
        A_dev2 = curand((n_out, n_in),  dtype=fp_format, stream=stream)
        A_dev3 = curand((n_out, n_in),  dtype=fp_format, stream=stream)
        X_dev = curand((n_in, n_elem), dtype=fp_format, stream=stream)
        B_dev1 = cuarray.GPUArray((n_out, n_elem), fp_format)
        B_dev2 = cuarray.GPUArray((n_out, n_elem), fp_format)
        B_dev3 = cuarray.GPUArray((n_out, n_elem), fp_format)
        print(code)
    elif OPENCL:

        platform = cl.get_platforms()
        my_gpu_devices = platform[0].get_devices(device_type=cl.device_type.GPU)
        ctx = cl.Context(devices=my_gpu_devices)
        #ctx = cl.create_some_context(interactive=True)
        queue = cl.CommandQueue(ctx)

        #kern = lp.set_options(kern, edit_code=False) #Only works for OpenCL?
        kern = lp.set_options(kern, "write_code")  # Output code before editing it
        kern = kern.copy(target=lp.PyOpenCLTarget(my_gpu_devices[0]))
        #code = lp.generate_code_v2(kern).device_code()
        # Print the Code
        #prog = cl.Program(ctx, code)
        #prog = prog.build()
        #ptx = prog.get_info(cl.program_info.BINARIES)[0]#.decode(
        #errors="ignore") #Breaks pocl
        #dammit = UnicodeDammit(ptx)
        #print(dammit.unicode_markup)

        #kern = lp.set_options(kern, "write_cl")
        #A_dev = cl.clrandom.rand(queue, (n_out, n_in), dtype=fp_format)
        #X_dev = cl.clrandom.rand(queue, (n_elem, n_in), dtype=fp_format)
        #B_dev = cl.array.Array(queue, (n_elem, n_out), dtype=fp_format)
        A_dev1 = cl.clrandom.rand(queue, (n_out, n_in), dtype=fp_format)
        A_dev2 = cl.clrandom.rand(queue, (n_out, n_in), dtype=fp_format)
        A_dev3 = cl.clrandom.rand(queue, (n_out, n_in), dtype=fp_format)
        X_dev = cl.clrandom.rand(queue, (n_in, n_elem), dtype=fp_format)
        B_dev1 = cl.array.Array(queue, (n_out, n_elem), dtype=fp_format)
        B_dev2 = cl.array.Array(queue, (n_out, n_elem), dtype=fp_format)
        B_dev3 = cl.array.Array(queue, (n_out, n_elem), dtype=fp_format)
        # Code for running binaries

        #device = queue.get_info(cl.command_queue_info.DEVICE)
        #with open("pocl_32x32.ptx","rb") as f:
        #with open("ptx_binary_nvopencl","rb") as f:
        #with open("ptx_binary", "rb") as f:
        #with open("cuda_32x32.ptx", "rb") as f:
        #    binary = f.read()
        #prg = cl.Program(ctx, [device], [binary])
        #prg = prg.build()
        #diff_knl = prg.diff

    events = []
    for i in range(2):
        if OPENCL:
            #diff_knl.set_args(B_dev.data, A_dev.data, X_dev.data)
            #evt = cl.enqueue_nd_range_kernel(queue, diff_knl,
            #    X_dev.shape, (32,32),None)
            evt, (B_dev1, B_dev2, B_dev3) = kern(queue, mat1=A_dev1, mat2=A_dev2,
                mat3=A_dev3, vec=X_dev)
            events.append(evt)
        elif CUDA:
            diff_fn(B_dev1, B_dev2, B_dev3, A_dev1, A_dev2, A_dev3, X_dev,
                block=(yblk, xblk, 1), grid=(ygrid, xgrid))
    if OPENCL:
        cl.wait_for_events(events)
    elif CUDA:
        pycuda.autoinit.context.synchronize()

    sum_time = 0.0
    events = []
    start = time.time()
    for i in range(nruns):
        if OPENCL:
            #diff_knl.set_args(B_dev.data, A_dev.data, X_dev.data)
            #evt = cl.enqueue_nd_range_kernel(queue, diff_knl,
            #    X_dev.shape, (32,32),None)
            evt, (B_dev1, B_dev2, B_dev3) = kern(queue, mat1=A_dev1, mat2=A_dev2,
                mat3=A_dev3, vec=X_dev)
            events.append(evt)
        elif CUDA:
            diff_fn(B_dev1, B_dev2, B_dev3, A_dev1, A_dev2, A_dev3, X_dev,
                block=(yblk, xblk, 1), grid=(ygrid, xgrid))
    if OPENCL:
        cl.wait_for_events(events)
    elif CUDA:
        pycuda.autoinit.context.synchronize()

    sum_time = time.time() - start
    avg_time = sum_time / nruns

    return (B_dev1, B_dev2, B_dev3, A_dev1, A_dev2, A_dev3, X_dev), avg_time


def runTest(n_elem, n_in, n_out, k_inner_outer, k_inner_inner, i_inner_outer,
                i_inner_inner, j_inner, backend="CUDA", fp_format=np.float32,
                nruns=10):

    n_mat = 3
    print(fp_format)
    kern = gen_diff_knl(n_mat, n_elem, n_in, n_out, fp_format=fp_format)
    print(kern)

    transformations = generate_transformation_list(k_inner_outer, k_inner_inner,
                                                    i_inner_outer, i_inner_inner,
                                                    j_inner)
    print(transformations)
    kern = apply_transformation_list(kern, transformations)

    kern = lp.set_options(kern, "no_numpy")
    # For some reason compyte will not install when using local pip
    backend = "OPENCL"

    CUDA = backend == "CUDA"
    OPENCL = not CUDA

    if CUDA:
        stream = drv.Stream()
        (ygrid, xgrid), (yblk, xblk) = kern.get_grid_size_upper_bounds_as_exprs()
        kern = kern.copy(target=lp.CudaTarget())
        #kern = lp.set_options(kern, edit_code=True)
        code = lp.generate_code_v2(kern).device_code()
        mod = SourceModule(code, keep=True)
        diff_fn = mod.get_function("diff")
        A_dev = curand((n_mat, n_out, n_in),  dtype=fp_format, stream=stream)
        X_dev = curand((n_in, n_elem), dtype=fp_format, stream=stream)
        B_dev = cuarray.GPUArray((n_mat, n_out, n_elem), fp_format)

        #for i in range(3):
        #    B_dev.append(cuarray.GPUArray((n_out, n_elem), fp_format)
        print(code)
    elif OPENCL:

        platform = cl.get_platforms()
        my_gpu_devices = platform[0].get_devices(device_type=cl.device_type.GPU)
        ctx = cl.Context(devices=my_gpu_devices)
        #ctx = cl.create_some_context(interactive=True)
        queue = cl.CommandQueue(ctx)

        #kern = lp.set_options(kern, edit_code=False)  # Only works for OpenCL?
        # Outputs the code before editing it
        kern = lp.set_options(kern, "write_code")
        kern = kern.copy(target=lp.PyOpenCLTarget(my_gpu_devices[0]))
        #code = lp.generate_code_v2(kern).device_code()
        # Print the Code
        #prog = cl.Program(ctx, code)
        #prog = prog.build()
        #  Breaks pocl
        #ptx = prog.get_info(cl.program_info.BINARIES)[0]#.decode(errors="ignore")
        #dammit = UnicodeDammit(ptx)
        #print(dammit.unicode_markup)

        #kern = lp.set_options(kern, "write_cl")
        #A_dev = cl.clrandom.rand(queue, (n_out, n_in), dtype=fp_format)
        #X_dev = cl.clrandom.rand(queue, (n_elem, n_in), dtype=fp_format)
        #B_dev = cl.array.Array(queue, (n_elem, n_out), dtype=fp_format)
        A_dev = cl.clrandom.rand(queue, (n_mat, n_out, n_in), dtype=fp_format)
        X_dev = cl.clrandom.rand(queue, (n_in, n_elem), dtype=fp_format)
        B_dev = cl.array.Array(queue, (n_mat, n_out, n_elem), dtype=fp_format)
        # Code for running binaries
        #device = queue.get_info(cl.command_queue_info.DEVICE)
        #with open("pocl_32x32.ptx","rb") as f:
        #with open("ptx_binary_nvopencl","rb") as f:
        #with open("ptx_binary", "rb") as f:
        #with open("cuda_32x32.ptx", "rb") as f:
        #    binary = f.read()
        #prg = cl.Program(ctx, [device], [binary])
        #prg = prg.build()
        #diff_knl = prg.diff

    events = []
    for i in range(2):
        if OPENCL:
            #diff_knl.set_args(B_dev.data, A_dev.data, X_dev.data)
            #evt = cl.enqueue_nd_range_kernel(queue, diff_knl, X_dev.shape, (32,32),
            #                                   None)
            evt, B_dev = kern(queue, diff_mat=A_dev, vec=X_dev)
            events.append(evt)
        elif CUDA:
            diff_fn(B_dev, A_dev, block=(yblk, xblk, 1), grid=(ygrid, xgrid))
    if OPENCL:
        cl.wait_for_events(events)
    elif CUDA:
        pycuda.autoinit.context.synchronize()
    #exit()

    sum_time = 0.0
    events = []
    start = time.time()
    for i in range(nruns):
        if OPENCL:
            #diff_knl.set_args(B_dev.data, A_dev.data, X_dev.data)
            #evt = cl.enqueue_nd_range_kernel(queue, diff_knl, X_dev.shape, (32,32),
            #                                   None)
            evt, B_dev = kern(queue, diff_mat=A_dev, vec=X_dev)
            events.append(evt)
        elif CUDA:
            diff_fn(B_dev, A_dev, X_dev, block=(yblk, xblk, 1), grid=(ygrid, xgrid))
    if OPENCL:
        cl.wait_for_events(events)
    elif CUDA:
        pycuda.autoinit.context.synchronize()

    sum_time = time.time() - start
    avg_time = sum_time / nruns

    return (B_dev, A_dev, X_dev), avg_time


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


def i_inner_inner_options(n_out, k_inner_inner, max_workitems=1024, reverse=True):
    factors = np.arange(2, n_out+1)[(n_out % np.arange(2, n_out+1)) == 0]
    # Ensure total number of workitems is less than maximum
    usable_factors = factors[factors*k_inner_inner <= max_workitems]
    return sorted(usable_factors, reverse=reverse)


def i_inner_outer_options(n_out, i_inner_inner, reverse=False):
    # Select a number of inline blocks such that n_out % outer*inner == 0
    inline = np.arange(1, (n_out // i_inner_inner) + 1)
    options = i_inner_inner*inline[n_out % (inline*i_inner_inner) == 0]
    return sorted(options, reverse=reverse)


def j_inner_options(n_in, reverse=False):
    factors = list(np.arange(1, n_in + 1)[(n_in % np.arange(1, n_in + 1)) == 0])
    return sorted(factors, reverse=reverse)


def exhaustiveSearch(n_in, n_out, n_elem, sm_size, time_limit=float("inf"),
                        fp_bytes=4, max_gflops=None, device_memory_bandwidth=None,
                        max_workitems=1024, gflops_cutoff=.95,
                        bandwidth_cutoff=0.95):
    k_inner_inner_opt = k_inner_inner_options()
    j_inner_opt = j_inner_options(n_in)
    start = time.time()

    avg_time_saved = float("inf")
    result_saved = None
    for kii in k_inner_inner_opt:
        for kio in k_inner_outer_options(n_in, kii, sm_size, fp_bytes=fp_bytes):
            for iii in i_inner_inner_options(n_out, kii,
                    max_workitems=max_workitems):
                for iio in i_inner_outer_options(n_out, iii):
                    for ji in j_inner_opt:
                        dev_arrays, avg_time = runTest(int(n_elem), int(n_in),
                            int(n_out), int(kio), int(kii), int(iio), int(iii),
                            int(ji))
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
    return result_saved


def randomSearch(n_in, n_out, n_elem, sm_size, time_limit=float("inf"),
                    fp_format=np.float32,
                    max_gflops=None, device_memory_bandwidth=None,
                    max_workitems=1024, gflops_cutoff=.95, bandwidth_cutoff=0.95):
    # Should probably make this a function
    fp_bytes, fp_string = (8, "FP64") if fp_format == np.float64 else (4, "FP32")
    from random import choice
    avg_time_saved = float("inf")
    result_saved = None
    start = time.time()

    k_inner_inner_opt = k_inner_inner_options()
    j_inner_opt = j_inner_options(n_in)

    tested = []
    # Assuming time is in milliseconds
    while(time.time() - start < time_limit):
        # Can be more intelligent by ensuring choices are not run multiple times
        kii = choice(k_inner_inner_opt)
        kio = choice(k_inner_outer_options(n_in, kii, sm_size, fp_bytes=fp_bytes))
        iii = choice(i_inner_inner_options(n_out, kii, max_workitems=max_workitems))
        iio = choice(i_inner_outer_options(n_out, iii))
        ji = choice(j_inner_opt)
        choices = (kio, kii, iio, iii, ji)
        if choices not in tested:
            print(choices)
            dev_arrays, avg_time = runTest(int(n_elem), int(n_in), int(n_out),
                int(kio), int(kii), int(iio), int(iii), int(ji), fp_format=fp_format)
            #exit()
            tested.append(choices)
            if max_gflops is not None and device_memory_bandwidth is not None:
                frac_peak_gflops, frac_peak_GBps = analyzeResult(n_out, n_in, n_elem,
                    max_gflops, device_memory_bandwidth, avg_time)
                if frac_peak_gflops >= gflops_cutoff or frac_peak_GBps >= bandwidth_cutoff:  # noqa
                    # Should validate result here
                    print("Performance is within tolerance of peak bandwith or flop rate. Terminating search")  # noqa
                    return (kio, kii, iio, iii, ji)
            if avg_time < avg_time_saved:
                avg_time_saved = avg_time
                result_saved = (kio, kii, iio, iii, ji)

    print("Time limit exceeded: returning current best result")
    return result_saved


# Testing code
file_name = "transform.hjson"
device_id = "NVIDIA Titan V"
fp_format = np.float64
fp_format_dict = {np.float32: (4, "FP32"), np.float64: (8, "FP64"),
                    np.complex128: (16, "C128")}
fp_bytes, fp_string = (8, "FP64") if fp_format == np.float64 else (4, "FP32")

to_test = True
if to_test:
    n_elem = 2**15  # 2**21
    pn = 6
    print(len(equidistant_nodes(pn, 3)[1]))
    n_out = len(equidistant_nodes(pn, 3)[1])
    n_in = len(equidistant_nodes(pn, 3)[1])

    #settings = exhaustiveSearch(n_in, n_out, n_elem, 4*12*1024, fp_bytes=fp_bytes,
    #               max_gflops=12288, device_memory_bandwidth=540)
    settings = randomSearch(n_in, n_out, n_elem, 4*12*1024, time_limit=60,
                    fp_format=fp_format, max_gflops=12288//2,
                    device_memory_bandwidth=540)
    #settings = noSearch(n_in, n_out, n_elem, 4*12*1024, time_limit=180,1
    #                       fp_bytes=fp_bytes, max_gflops=12288,
    #                       device_memory_bandwidth=540)
    print("FINAL RESULTS")
    print(settings)
# Add functionality to write transformations to file

#for i in range(10):
#  dev_arrays, avg_time = runTest(n_elem, n_in, n_out, kio, kii, iio, iii, ji)
#  analyzeResult(n_out, n_in, n_elem, 12288, 540, avg_time)
#  verifyResult(*dev_arrays)
