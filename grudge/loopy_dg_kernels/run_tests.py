import numpy as np

import pyopencl as cl
import pyopencl.array
import pyopencl.clrandom

import loopy as lp
from loopy.version import LOOPY_USE_LANGUAGE_VERSION_2018_2
from grudge.loopy_dg_kernels import apply_transformation_list
#from loopy.kernel.data import AddressSpace

"""
import pycuda.gpuarray as cuarray
import pycuda.driver as drv
import pycuda.tools
import pycuda.autoinit
from pycuda.compiler import SourceModule
from pycuda.curandom import rand as curand
"""

from modepy import equidistant_nodes
from pytools.obj_array import make_obj_array

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
    apply_transformation_list, gen_elwise_linear_knl, gen_face_mass_knl, gen_face_mass_knl_merged)
from grudge.grudge_tags import (IsDOFArray, IsSepVecDOFArray, IsOpArray,
    IsSepVecOpArray, IsFaceDOFArray, IsFaceMassOpArray, IsVecDOFArray, IsVecOpArray, IsFourAxisDOFArray)
import  grudge.grudge_array_context as gac#import set_memory_layout

def testBandwidth(fp_format=np.float32, nruns=100):

    from pyopencl.array import sum as clsum
    platform = cl.get_platforms()
    my_gpu_devices = platform[0].get_devices(device_type=cl.device_type.GPU)
    #ctx = cl.Context(devices=my_gpu_devices)
    ctx = cl.create_some_context(interactive=True)
    queue = cl.CommandQueue(ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)

    from pyopencl.tools import ImmediateAllocator, MemoryPool
    allocator = ImmediateAllocator(queue)
    mem_pool = MemoryPool(allocator) 


    knl = lp.make_copy_kernel("c,c", old_dim_tags="c,c")
    knl = lp.add_dtypes(knl, {"input": fp_format, "output": fp_format})
    knl = knl.copy(target=lp.PyOpenCLTarget(my_gpu_devices[0]))
    n0 = 2
    #knl = lp.split_iname(knl, "i1", 1024//2, inner_tag="l.0", outer_tag="g.0", slabs=(0,1))
    knl = lp.split_iname(knl, "i1", 256, inner_tag="l.0", outer_tag="g.0", slabs=(0,1))
    #knl = lp.split_iname(knl, "i1", 6*16, outer_tag="g.0") 
    #knl = lp.split_iname(knl, "i1_inner", 16, outer_tag="ilp", inner_tag="l.0", slabs=(0,1)) 
    #knl = lp.split_iname(knl, "i0", n0, inner_tag="l.1", outer_tag="g.1", slabs=(0,0))

    fp_bytes = 8 if fp_format == np.float64 else 4

    # This assumes fp32
    len_list = []
    float_count = 1
    max_floats = 2**28
    while float_count <= max_floats:
        len_list.append(float_count)
        float_count = int(np.ceil(float_count*1.5))
    for i in range(29):
        len_list.append(2**i)
    len_list = sorted(list(set(len_list)))

    #data = np.random.randint(-127, 128, (1,max_bytes), dtype=np.int8)
    #inpt = cl.array.to_device(queue, data, allocator=mem_pool)

    print(len_list)

    for n in len_list:
    #for i in range(29):

        #n = 2**i
        kern = lp.fix_parameters(knl, n0=n0, n1=n)
        #data = np.random.randint(-127, 128, (1,n), dtype=np.int8)
        #inpt = cl.array.to_device(queue, data, allocator=mem_pool)
        inpt = cl.clrandom.rand(queue, (n0, n), dtype=fp_format)
        outpt = cl.array.Array(queue, (n0, n), dtype=fp_format, allocator=mem_pool)
     
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
        from bs4 import UnicodeDammit
        dammit = UnicodeDammit(ptx)
        #print(dammit.unicode_markup)
        f = open("ptx.ptx", "w")
        f.write(dammit.unicode_markup)
        f.close()

        from pyopencl.tools import ImmediateAllocator, MemoryPool
        allocator = ImmediateAllocator(queue)
        mem_pool = MemoryPool(allocator)

        X_dev = cl.array.Array(queue, (n_elem, n_in), dtype=fp_format, order="F", allocator=mem_pool)
        cl.clrandom.fill_rand(X_dev, queue=queue)
        B_dev = cl.array.Array(queue, (n_elem, n_out), dtype=fp_format, allocator=mem_pool,order="F")
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
# This hardcodes the memory layout, should probably instead retrieve it from somewhere on a per
# tag basis
def generic_test(queue, kern, backend="OPENCL", nruns=10, warmup=True):

    kern = lp.set_options(kern, "no_numpy")
    kern = lp.set_options(kern, "return_dict")

    CUDA = (backend == "CUDA")
    OPENCL = not CUDA

    if CUDA:
        print("CUDA not supported")
        exit()
    elif OPENCL:
        """
        platform = cl.get_platforms()
        my_gpu_devices = platform[0].get_devices(device_type=cl.device_type.GPU)
        ctx = cl.Context(devices=my_gpu_devices)
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
        print(dammit.unicode_markup)
        f = open("ptx.ptx", "w")
        f.write(dammit.unicode_markup)
        f.close()
        """

        from pyopencl.tools import ImmediateAllocator, MemoryPool
        allocator = ImmediateAllocator(queue)
        mem_pool = MemoryPool(allocator)

        arg_dict = {}

        # Fill arrays with random data
        # Could probably just read the strides from the kernel to get ordering
        for arg in kern.default_entrypoint.args:
            if IsDOFArray() in arg.tags:
                arg_dict[arg.name] = cl.array.Array(queue, arg.shape, arg.dtype, order="F", allocator=mem_pool)
                if not arg.is_output:
                    cl.clrandom.fill_rand(arg_dict[arg.name], queue)
            elif IsSepVecDOFArray() in arg.tags:
                if arg.is_output:
                    obj_array = [cl.array.Array(queue, arg.shape[1:], dtype=arg.dtype, allocator=mem_pool, order="F") for i in range(arg.shape[0])]
                    arg_dict[arg.name] = make_obj_array(obj_array)
                else:
                    print("Input SepVecDOFArrays are not currently supported")
                    exit()
            elif IsSepVecOpArray() in arg.tags:
                obj_array = [cl.clrandom.rand(queue, arg.shape[1:], dtype=arg.dtype) for i in range(arg.shape[0])]
                arg_dict[arg.name] = make_obj_array(obj_array)
            elif IsOpArray() in arg.tags or IsVecOpArray() in arg.tags:
                arg_dict[arg.name] = cl.clrandom.rand(queue, arg.shape, dtype=arg.dtype)
            elif IsFaceDOFArray() in arg.tags:
                fp_bytes = arg.dtype.numpy_dtype.itemsize
                nfaces, nelements, nface_nodes = arg.shape
                strides = (fp_bytes*nelements, fp_bytes*1, fp_bytes*nelements*nfaces) #original
                arg_dict[arg.name] = cl.array.Array(queue, arg.shape, dtype=arg.dtype, 
                    strides=strides, allocator=mem_pool)
                cl.clrandom.fill_rand(arg_dict[arg.name], queue=queue)
            elif IsFaceMassOpArray() in arg.tags:
                # Are these strides correct?
                arg_dict[arg.name] = cl.clrandom.rand(queue, arg.shape, dtype=arg.dtype)
            elif IsVecDOFArray() in arg.tags:
                fp_bytes = arg.dtype.numpy_dtype.itemsize
                nr, nelements, ndofs = arg.shape
                strides = (fp_bytes*nelements*ndofs, fp_bytes, fp_bytes*nelements) #original
                arg_dict[arg.name] = cl.array.Array(queue, arg.shape, dtype=arg.dtype, 
                    strides=strides, allocator=mem_pool)
                cl.clrandom.fill_rand(arg_dict[arg.name], queue=queue)
            elif IsFourAxisDOFArray() in arg.tags:
                fp_bytes = arg.dtype.numpy_dtype.itemsize
                nx, nr, nelements, ndofs = arg.shape
                strides = (fp_bytes*nelements*ndofs*nr, fp_bytes*nelements*ndofs,
                            fp_bytes, fp_bytes*nelements)
                arg_dict[arg.name] = cl.array.Array(queue, arg.shape, dtype=arg.dtype, 
                    strides=strides, allocator=mem_pool)
                cl.clrandom.fill_rand(arg_dict[arg.name], queue=queue)
            elif isinstance(arg, lp.ArrayArg):
                print("No tags recognized. Assuming default data layout")
                print(arg.name)
                # Assume default layout
                arg_dict[arg.name] = cl.clrandom.rand(queue, arg.shape, dtype=arg.dtype)
                #print(arg.name)
                #print(arg.tags)
                #print("Unknown Tag")
                #exit()

        if warmup:
            for i in range(2):
                kern(queue, **arg_dict)
            queue.finish()

        sum_time = 0.0
        events = []
        for i in range(nruns):
            evt, _ = kern(queue, **arg_dict)
            events.append(evt)

        cl.wait_for_events(events)
        for evt in events:
            sum_time += evt.profile.end - evt.profile.start
        sum_time = sum_time / 1e9        
        #queue.finish()

    avg_time = sum_time / nruns

    return arg_dict, avg_time


def analyze_knl_bandwidth(knl, avg_time):
    nbytes = 0
    # What if the output is not in the input arguments?
    #print(knl.default_entrypoint.args)
    # Would probably be better to use the memory footprint
    # if can get it to work.
    for arg in knl.default_entrypoint.args:
        print(arg.name)
        print(arg.shape)
        print(type(arg.dtype))
        entries = np.prod((arg.shape))
        fp_bytes = arg.dtype.dtype.itemsize
        nbytes += fp_bytes * entries
    bw = nbytes / avg_time / 1e9

    # Seems lp.gather_access_footprint_bytes breaks
    #footprint = lp.gather_access_footprint_bytes(knl)
    #footprint_bytes = 0
    #for val in footprint.values():
    #    footprint_bytes += val.eval_with_dict({})
    #footprint_bw =  footprint_bytes / avg_time / 1e9  
    #print(f"Time: {avg_time}, Bytes: {nbytes}, Bandwidth: {bw} GB/s Footprint BW: {footprint_bw} GB/s")

    print(f"Time: {avg_time}, Bytes: {nbytes}, Bandwidth: {bw} GB/s")
    return bw


def analyze_FLOPS(knl, avg_time, max_gflops=None):

    op_map = lp.get_op_map(knl, count_within_subscripts=False, subgroup_size=1)
    #print(op_map)
    map_flops = 0
    for val in op_map.values():
        map_flops += val.eval_with_dict({})
    gflop_rate = map_flops / avg_time / 1e9

    """
    n_mat = 1
    nfaces = 1
    for arg in knl.default_entrypoint.args:
        if IsDOFArray() in arg.tags:
            n_elem, n_out = arg.shape
            fp_bytes = arg.dtype.dtype.itemsize
        elif IsSepVecOpArray() in arg.tags or IsVecOpArray() in arg.tags:
            n_mat, n_out, n_in = arg.shape
        elif IsOpArray() in arg.tags:
            n_out, n_in = arg.shape
        elif IsFaceDOFArray() in arg.tags:
            nfaces, n_elem, n_in = arg.shape
    
    flops = nfaces*n_mat*2*(n_out * n_in * n_elem)
    """
    gflop_rate = (map_flops / avg_time) * 1e-9
    print("GFLOP/s: " + str(gflop_rate))

    #print("Map GFLOP/s: " + str(map_gflop_rate))
    #print(flops)
    #print(map_flops)

    frac_peak_gflops = None
    if max_gflops is not None:
        print("Peak GFLOP/s: " + str(max_gflops))
        frac_peak_gflops = gflop_rate / max_gflops
        print("Percent peak: " + str(100*(frac_peak_gflops)))
    
    print()

    # Calculate bandwidth
    # Assumes each element only read once
    #ideal_total_bytes_transferred = fp_bytes*(3*(n_out * n_elem) + (n_in * n_elem)
    #                                            + 3*(n_out * n_in))
    #GBps = (ideal_total_bytes_transferred / avg_time) / 1e9
    #frac_peak_GBps = GBps / device_memory_bandwidth
    #print("GB/s: " + str(GBps))
    #print("Peak GB/s: " + str(device_memory_bandwidth))
    #print("Percent peak: " + str(100*(frac_peak_GBps)))
    #print()

    return gflop_rate, frac_peak_gflops


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


# This can be removed eventually
def apply_transformations_and_run_test(queue, knl, test_fn, params, tgenerator, max_gflops=None,
	device_memory_bandwidth=None, gflops_cutoff=0.95, bandwidth_cutoff=0.95, start_param=None):
	
    kio, kii, iio, iii, ji = params

    # Transform and run
    knl = gac.set_memory_layout(knl)
    if applicator is not None:
        trans_list = tgenerator(params)
    else:
        # Should probably read in eligible transformations from a file instead of using if-statements
        trans_list = []
        if "diff" in knl.default_entrypoint.name:
            trans_list.append(["tag_inames", ["imatrix: ilp"]])

        trans_list.append(["split_iname", ["iel", kio], {"outer_tag": "g.0", "slabs":(0,1)}])
        trans_list.append(["split_iname", ["iel_inner", kii], 
            {"outer_tag": "ilp", "inner_tag":"l.0", "slabs":(0,1)}])
        trans_list.append(["split_iname", ["idof", iio], {"outer_tag": "g.1", "slabs":(0,0)}])
        trans_list.append(["split_iname", ["idof_inner", iii], 
            {"outer_tag": "ilp", "inner_tag":"l.1", "slabs":(0,1)}])

        if knl.default_entrypoint.name == "face_mass":
            pass
            #trans_list.append(["add_prefetch", ["vec", "f,j,iel_inner_outer,iel_inner_inner"],
            #    {"temporary_name":"vecf", "default_tag":"l.auto"}])
            #trans_list.append(["tag_array_axes", ["vecf", "N1,N0,N2"]])
        elif knl.default_entrypoint.name == "nodes":
            trans_list.append(["add_prefetch", ["nodes", "j,iel_inner_outer,iel_inner_inner"],
                {"temporary_name":"vecf", "default_tag":"l.auto"}])
            trans_list.append(["tag_array_axes", ["vecf", "f,f"]])
        elif "resample_by_mat" in knl.default_entrypoint.name:
            # Indirection may prevent prefetching
            pass
        else:
            trans_list.append(["add_prefetch", ["vec", "j,iel_inner_outer,iel_inner_inner"],
                {"temporary_name":"vecf", "default_tag":"l.auto"}])
            trans_list.append(["tag_array_axes", ["vecf", "f,f"]])

        trans_list.append(["split_iname", ["j", ji], {"outer_tag":"for", "inner_tag":"for"}])
        trans_list.append(["add_inames_for_unused_hw_axes"]) 

    knl = apply_transformation_list(knl, trans_list)


    #print(knl.default_entrypoint.name)
    #print(trans_list)

    # Execute and analyze the results
    dev_arrays, avg_time = test_fn(queue, knl)
    #avg_time = np.random.rand()

    return avg_time, trans_list

    """
    # The analysis should be done elsewhere
    bw = None
    flop_rate = None

    if device_memory_bandwidth is not None:  # noqa
	bw = analyze_knl_bandwidth(knl, avg_time)
	frac_peak_GBps = bw / device_memory_bandwidth
	if frac_peak_GBps  >= bandwidth_cutoff:  # noqa
	    # Should validate result here
	    print("Performance is within tolerance of peak bandwith. Terminating search")  # noqa
	    return avg_time, params

    # Einsum complicates this. This depends on the kernel being called.
    if max_gflops is not None:
	frac_peak_gflops = analyze_FLOPS(knl, max_gflops, avg_time)
	if frac_peak_gflops >= gflops_cutoff:
	    # Should validate result here
	    print("Performance is within tolerance of peak bandwith or flop rate. Terminating search")  # noqa
	    return choices

    if device_memory_bandwidth is not None and max_gflops is not None:
	data = (avg_time, 
			    frac_peak_GBps*device_memory_bandwidth, 
			    frac_peak_gflops*max_gflops, 
			    frac_peak_GBps, 
			    frac_peak_gflops, 
			    (kio, kii, iio, iii, ji))
	result_list.append(data)
	f.write(str(data) + "\n")

    if avg_time < avg_time_saved:
	avg_time_saved = avg_time
	result_saved = choices
	result_saved_list = trans_list
    if time.time() - start > time_limit: 
	result_list.sort()
	print("Avg_time, Peak_BW, Peak_GFLOPS, Frac_peak_bandwidth, Frac_peak_GFlops")
	for entry in result_list:
	    print(entry)
	print()


	#return result_saved_list
	return result_saved
    """

def run_single_param_set(queue, knl_base, tlist_generator, params, test_fn, max_gflops=None, device_memory_bandwidth=None):
    trans_list = tlist_generator(params, knl=knl_base)
    knl = apply_transformation_list(knl_base, trans_list)
    dev_arrays, avg_time = test_fn(queue, knl)

    # Should this return the fraction of peak of should that be calculated in this function?
    gflops, frac_peak_gflops = analyze_FLOPS(knl, avg_time, max_gflops=max_gflops)
    bw = analyze_knl_bandwidth(knl, avg_time)

    if device_memory_bandwidth is not None:  # noqa
        bw = analyze_knl_bandwidth(knl, avg_time)
        frac_peak_GBps = bw / device_memory_bandwidth
        if frac_peak_GBps  >= bandwidth_cutoff:  # noqa
            # Should validate result here
            print("Performance is within tolerance of peak bandwith. Terminating search")  # noqa
            return choices

    # This is incorrect for general einsum kernels
    if max_gflops is not None:
        frac_peak_gflops = analyze_FLOPS(knl, max_gflops, avg_time)
        if frac_peak_gflops >= gflops_cutoff:
            # Should validate result here
            print("Performance is within tolerance of peak bandwith or flop rate. Terminating search")  # noqa
            return choices

    data = None
    if device_memory_bandwidth is not None and max_gflops is not None:
        data = (frac_peak_GBps*device_memory_bandwidth, 
                frac_peak_gflops*max_gflops, 
                frac_peak_GBps, 
                frac_peak_gflops)

    return (avg_time, trans_list, data)


def exhaustive_search_v2(queue, knl, test_fn, pspace_generator, tlist_generator, time_limit=float("inf"), max_gflops=None, 
        device_memory_bandwidth=None, gflops_cutoff=0.95, bandwidth_cutoff=0.95, start_param=None):


    #param_list = gen_autotune_list(queue, knl, start_param=start_param)

    #Probably don't need all of these parameters
    #apply_transformations_and_run_test(queue, knl, test_fn, params, max_gflops=None,
	    #device_memory_bandwidth=None, gflops_cutoff=0.95, bandwidth_cutoff=0.95, start_param=None):
	
    # Should probably obtain device_memory_bandwidth from empirical tests

    # Also fixes the parameters. Maybe that should be a separate function   
    knl = gac.set_memory_layout(knl)

    knl_base = knl.copy()

    params_list = pspace_generator(queue, knl, start_param=start_param)
    #print(knl)
    #print(len(params_list))
    
    result_list = []
    start = time.time()

    # Iterate over parameter space coordinates
    # If serial run this otherwise, run the parallel autotuner
    # Should probably make separate function for each.
    for params in params_list:
        print(f"Currently testing: {params}")
        """
        trans_list = tlist_generator(params, knl=knl)
        knl = apply_transformation_list(knl_base, trans_list)
        dev_arrays, avg_time = test_fn(queue, knl)

        # Should this return the fraction of peak of should that be calculated in this function?
        gflops, frac_peak_gflops = analyze_FLOPS(knl, avg_time, max_gflops=max_gflops)
        bw = analyze_knl_bandwidth(knl, avg_time)

        if device_memory_bandwidth is not None:  # noqa
            bw = analyze_knl_bandwidth(knl, avg_time)
            frac_peak_GBps = bw / device_memory_bandwidth
            if frac_peak_GBps  >= bandwidth_cutoff:  # noqa
                # Should validate result here
                print("Performance is within tolerance of peak bandwith. Terminating search")  # noqa
                return choices

        # This is incorrect for general einsum kernels
        if max_gflops is not None:
            frac_peak_gflops = analyze_FLOPS(knl, max_gflops, avg_time)
            if frac_peak_gflops >= gflops_cutoff:
                # Should validate result here
                print("Performance is within tolerance of peak bandwith or flop rate. Terminating search")  # noqa
                return choices

        data = None
        if device_memory_bandwidth is not None and max_gflops is not None:
            data = (frac_peak_GBps*device_memory_bandwidth, 
                    frac_peak_gflops*max_gflops, 
                    frac_peak_GBps, 
                    frac_peak_gflops)
        """

        avg_time, trans_list, data = run_single_param_set(queue, knl_base, tlist_generator, params, test_fn, max_gflops=max_gflops, device_memory_bandwidth=device_memory_bandwidth)
        result_list.append((avg_time, trans_list, data))
        print(avg_time)
        #result_list.append(data)
        #f.write(str(data) + "\n")

        #if avg_time < avg_time_saved:
        #    avg_time_saved = avg_time
        #    result_saved = choices
        #    result_saved_list = trans_list
        
        if time.time() - start > time_limit: 
            break
            #result_list.sort()
            #print("Avg_time, Peak_BW, Peak_GFLOPS, Frac_peak_bandwidth, Frac_peak_GFlops")
            #for entry in result_list:
            #    print(entry)
            #print()

        #return result_saved_list
        #return result_saved

    #print("Avg_time, Peak_BW, Peak_GFLOPS, Frac_peak_bandwidth, Frac_peak_GFlops")
    #for entry in result_list:
    #    print(entry)
    #print()


    
    #print("Suggested loop splittings")
    #print(result_saved)
    #print(f"iel: {kio}")
    #print(f"iel_inner: {kii}")
    #print(f"idof: {iio}")
    #print(f"idof_inner: {iii}")
    #print(f"j: {ji}")
 
    #return result_saved_list
    #return result_saved

    # Could save the highest performing function, but often one wants to see the results
    # over the entire parameter space
    key_func = lambda result: result[0]
    sorted_results = sorted(result_list, key=key_func)
    return sorted_results[0]
  

def exhaustive_search(queue, knl, test_fn, time_limit=float("inf"), max_gflops=None, 
        device_memory_bandwidth=None, gflops_cutoff=0.95, bandwidth_cutoff=0.95, start_param=None):

    # Should probably obtain device_memory_bandwidth from empirical tests

    # Imports
    from grudge.grudge_tags import ParameterValue

    local_mem_size = queue.device.local_mem_size
    max_work_group_size = queue.device.max_work_group_size    

    avg_time_saved = float("inf")
    result_saved = None

    transform_list = []

    for arg in knl.default_entrypoint.args:
        if "resample_by_mat" not in knl.default_entrypoint.name:
            if IsDOFArray() in arg.tags:
                n_elem, n_out = arg.shape
                fp_bytes = arg.dtype.dtype.itemsize
                #n_in = n_out # Not true for non-square
            elif IsSepVecOpArray() in arg.tags:
                n_mat, n_out, n_in = arg.shape
            elif IsOpArray() in arg.tags:
                n_out, n_in = arg.shape
            elif IsFaceDOFArray() in arg.tags:
                nfaces, n_elem, n_in = arg.shape
        else:
            if IsOpArray() in arg.tags:
                n_out, n_in = arg.shape
                fp_bytes = arg.dtype.dtype.itemsize

    # Also fixes the parameters    
    knl = gac.set_memory_layout(knl)

    tested = []

    if start_param is not None:
        kio_s, kii_s, iio_s, iii_s, ji_s = start_param
    else:
        kio_s, kii_s, iio_s, iii_s, ji_s = (None, None, None, None, None)

    #k_inner_inner_opt = k_inner_inner_options(start_val=kii_s)
    #kii_s = None
    #j_inner_opt = j_inner_options(n_in)
    knl_base = knl.copy()

    avg_time_saved = float("inf")
    result_saved = None
    result_saved_list = []
    
    # Iterate over five search dimensions
    result_list = []
    start = time.time()
    with open("output.txt", "a") as f:
        for kii in k_inner_inner_options(start_val=kii_s):
            # This prevents shared memory from overflowing when running with the face mass kernel
            if knl.default_entrypoint.name == "face_mass":
                n_in_2 = n_in * nfaces
            else:
                n_in_2 = n_in
            for kio in k_inner_outer_options(n_in_2, kii, local_mem_size, fp_bytes=fp_bytes,start_val=kio_s):
                kio_s = None # Set to None so will form the full set the next time around
                for iii in i_inner_inner_options(n_out, kii,
                        max_work_group_size=max_work_group_size, start_val=iii_s):
                    iii_s = None
                    for iio in i_inner_outer_options(n_out, iii, start_val=iio_s):
                        iio_s = None
                        for ji in j_inner_options(n_in, start_val=ji_s):
                            ji_s = None
                            print((kio, kii, iio, iii, ji))
                            # Transform and run
                            knl = knl_base.copy()
                            knl = lp.split_iname(knl, "iel", kio, outer_tag="g.0", slabs=(0,1))
                            knl = lp.split_iname(knl, "iel_inner", kii, outer_tag="ilp", inner_tag="l.0", slabs=(0,1))
                            knl = lp.split_iname(knl, "idof", iio, outer_tag="g.1", slabs=(0,0))
                            knl = lp.split_iname(knl, "idof_inner", iii, outer_tag="ilp", inner_tag="l.1", slabs=(0,0))        

                            if knl.default_entrypoint.name == "face_mass":
                                knl = lp.add_prefetch(knl, "vec", "f,j,iel_inner_outer,iel_inner_inner", temporary_name="vecf", default_tag="l.auto")
                                #knl = lp.tag_array_axes(knl, "vecf", "N1,N0,N2") # Should be this but breaks
                            elif knl.default_entrypoint.name == "nodes":
                                knl = lp.add_prefetch(knl, "nodes", "j,iel_inner_outer,iel_inner_inner", temporary_name="vecf", default_tag="l.auto")
                                knl = lp.tag_array_axes(knl, "vecf", "f,f")
                            elif "resample_by_mat" in knl.default_entrypoint.name: # Reads are scattered so prefetching is difficult
                                pass
                                #knl = lp.add_prefetch(knl, "ary", "j,iel_inner_outer,iel_inner_inner", temporary_name="vecf", default_tag="l.auto")
                                #knl = lp.tag_array_axes(knl, "vecf", "f,f")                           
                            else:   
                                knl = lp.add_prefetch(knl, "vec", "j,iel_inner_outer,iel_inner_inner", temporary_name="vecf", default_tag="l.auto")
                                knl = lp.tag_array_axes(knl, "vecf", "f,f")

                            knl = lp.split_iname(knl, "j", ji, outer_tag="for", inner_tag="for")
                            knl = lp.add_inames_for_unused_hw_axes(knl)


                            # Change this to just use the transformation list instead of applying the transformations
                            # directly
                            trans_list = []
                            if "diff" in knl.default_entrypoint.name:
                                trans_list.append(["tag_inames", ["imatrix: ilp"]])
                            trans_list.append(["split_iname", ["iel", kio], {"outer_tag": "g.0", "slabs":(0,1)}])
                            trans_list.append(["split_iname", ["iel_inner", kii], 
                                {"outer_tag": "ilp", "inner_tag":"l.0", "slabs":(0,1)}])
                            trans_list.append(["split_iname", ["idof", iio], {"outer_tag": "g.1", "slabs":(0,0)}])
                            trans_list.append(["split_iname", ["idof_inner", iii], 
                                {"outer_tag": "ilp", "inner_tag":"l.1", "slabs":(0,1)}])

                            if knl.default_entrypoint.name == "face_mass":
                                pass
                                #trans_list.append(["add_prefetch", ["vec", "f,j,iel_inner_outer,iel_inner_inner"],
                                #    {"temporary_name":"vecf", "default_tag":"l.auto"}])
                                #trans_list.append(["tag_array_axes", ["vecf", "N1,N0,N2"]])
                            elif knl.default_entrypoint.name == "nodes":
                                trans_list.append(["add_prefetch", ["nodes", "j,iel_inner_outer,iel_inner_inner"],
                                    {"temporary_name":"vecf", "default_tag":"l.auto"}])
                                trans_list.append(["tag_array_axes", ["vecf", "f,f"]])
                            elif "resample_by_mat" in knl.default_entrypoint.name:
                                # Indirection may prevent prefetching
                                pass
                            else:
                                trans_list.append(["add_prefetch", ["vec", "j,iel_inner_outer,iel_inner_inner"],
                                    {"temporary_name":"vecf", "default_tag":"l.auto"}])
                                trans_list.append(["tag_array_axes", ["vecf", "f,f"]])

                            trans_list.append(["split_iname", ["j", ji], {"outer_tag":"for", "inner_tag":"for"}])
                            trans_list.append(["add_inames_for_unused_hw_axes"]) 

                            print(knl.default_entrypoint.name)
                            print(trans_list)

                            # Execute and analyze the results
                            dev_arrays, avg_time = test_fn(queue, knl)

                            choices = (kio, kii, iio, iii, ji)
                            """
                            if device_memory_bandwidth is not None:  # noqa
                                #frac_peak_gflops, frac_peak_GBps = analyzeResult(n_out,
                                #    n_in, n_elem, max_gflops, device_memory_bandwidth,
                                #    avg_time)
                                bw  = analyze_knl_bandwidth(knl, avg_time)
                                frac_peak_GBps = bw / device_memory_bandwidth
                                result_list.append((frac_peak_GBps, (kio, kii, iio, iii, ji)))
                                if frac_peak_GBps  >= bandwidth_cutoff:  # noqa
                                    # Should validate result here
                                    pass
                                    #print("Performance is within tolerance of peak bandwith. Terminating search")  # noqa
                                    #return (kio, kii, iio, iii, ji)
                            """
                            """
                            # TODO: Fix flop calculation
                            if max_gflops is not None and device_memory_bandwidth is not None:  # noqa
                                frac_peak_gflops, frac_peak_GBps = analyzeResult(n_out,
                                    n_in, n_elem, max_gflops, device_memory_bandwidth,
                                    avg_time)
                                if frac_peak_gflops >= gflops_cutoff or frac_peak_GBps  >= bandwidth_cutoff:  # noqa
                                    # Should validate result here
                                    print("Performance is within tolerance of peak bandwith or flop rate. Terminating search")  # noqa
                                    return (kio, kii, iio, iii, ji)
                            """
                            print(choices)
                            if device_memory_bandwidth is not None:  # noqa
                                bw = analyze_knl_bandwidth(knl, avg_time)
                                frac_peak_GBps = bw / device_memory_bandwidth
                                if frac_peak_GBps  >= bandwidth_cutoff:  # noqa
                                    # Should validate result here
                                    print("Performance is within tolerance of peak bandwith. Terminating search")  # noqa
                                    return choices
                    
                            if max_gflops is not None:
                                frac_peak_gflops = analyze_FLOPS(knl, max_gflops, avg_time)
                                if frac_peak_gflops >= gflops_cutoff:
                                    # Should validate result here
                                    print("Performance is within tolerance of peak bandwith or flop rate. Terminating search")  # noqa
                                    return choices

                            if device_memory_bandwidth is not None and max_gflops is not None:
                                data = (avg_time, 
                                                    frac_peak_GBps*device_memory_bandwidth, 
                                                    frac_peak_gflops*max_gflops, 
                                                    frac_peak_GBps, 
                                                    frac_peak_gflops, 
                                                    (kio, kii, iio, iii, ji))
                                result_list.append(data)
                                f.write(str(data) + "\n")
                                
                            if avg_time < avg_time_saved:
                                avg_time_saved = avg_time
                                result_saved = choices
                                result_saved_list = trans_list
                            if time.time() - start > time_limit: 
                                result_list.sort()
                                print("Avg_time, Peak_BW, Peak_GFLOPS, Frac_peak_bandwidth, Frac_peak_GFlops")
                                for entry in result_list:
                                    print(entry)
                                print()

       
                                #return result_saved_list
                                return result_saved


    result_list.sort()

    print("Avg_time, Peak_BW, Peak_GFLOPS, Frac_peak_bandwidth, Frac_peak_GFlops")
    for entry in result_list:
        print(entry)
    print()


    
    print("Suggested loop splittings")
    print(result_saved)
    #print(f"iel: {kio}")
    #print(f"iel_inner: {kii}")
    #print(f"idof: {iio}")
    #print(f"idof_inner: {iii}")
    #print(f"j: {ji}")
 
    return result_saved_list
    #return result_saved

def random_search(queue, knl, test_fn, time_limit=float("inf"), max_gflops=None, 
        device_memory_bandwidth=None, gflops_cutoff=0.95, bandwidth_cutoff=0.95):

    # Imports
    from random import choice
    from grudge.grudge_tags import ParameterValue

    local_mem_size = queue.device.local_mem_size
    max_work_group_size = queue.device.max_work_group_size    

    avg_time_saved = float("inf")
    result_saved = None
    result_saved_list = []

    # Get sizes
    for arg in knl.default_entrypoint.args:
        if "resample_by_mat" not in knl.default_entrypoint.name:
            if IsDOFArray() in arg.tags:
                n_elem, n_out = arg.shape
                fp_bytes = arg.dtype.dtype.itemsize
                #n_in = n_out
            elif IsSepVecOpArray() in arg.tags:
                n_mat, n_out, n_in = arg.shape
            elif IsOpArray() in arg.tags:
                n_out, n_in = arg.shape
            elif IsFaceDOFArray() in arg.tags:
                nfaces, n_elem, n_in = arg.shape
        else:
            if IsOpArray() in arg.tags:
                n_out, n_in = arg.shape
                fp_bytes = arg.dtype.dtype.itemsize

    # Also fixes the parameters
    knl = gac.set_memory_layout(knl)

    tested = []

    k_inner_inner_opt = k_inner_inner_options()
    j_inner_opt = j_inner_options(n_in)
    knl_base = knl.copy()
    result_list = []

    start = time.time()
    while(time.time() - start < time_limit):
        # Can be more intelligent by ensuring choices are not run multiple times
        # Maybe could use expressions
        kii = choice(k_inner_inner_opt)
        if knl.default_entrypoint.name == "face_mass":
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
            if "diff" in knl.default_entrypoint.name:
                knl = lp.tag_inames(knl, "imatrix: ilp")
            knl = lp.split_iname(knl, "iel", kio, outer_tag="g.0", slabs=(0,1))
            knl = lp.split_iname(knl, "iel_inner", kii, outer_tag="ilp", inner_tag="l.0", slabs=(0,1))
            knl = lp.split_iname(knl, "idof", iio, outer_tag="g.1", slabs=(0,0))
            knl = lp.split_iname(knl, "idof_inner", iii, outer_tag="ilp", inner_tag="l.1", slabs=(0,0))        

            if knl.default_entrypoint.name == "face_mass":
                knl = lp.add_prefetch(knl, "vec", "f,j,iel_inner_outer,iel_inner_inner", temporary_name="vecf", default_tag="l.auto")
                # Both N1,N0,N2 and N0,N1,N2 both seem to give memory errors..
                #knl = lp.tag_array_axes(knl, "vecf", "N1,N0,N2")
            elif knl.default_entrypoint.name == "nodes":
                knl = lp.add_prefetch(knl, "nodes", "j,iel_inner_outer,iel_inner_inner", temporary_name="vecf", default_tag="l.auto")
                knl = lp.tag_array_axes(knl, "vecf", "f,f")
            elif "resample_by_mat" in knl.default_entrypoint.name:
                pass
                # Indirection may prevent prefetching
                #knl = lp.add_prefetch(knl, "ary", "j,iel_inner_outer,iel_inner_inner", temporary_name="vecf", default_tag="l.auto")
                #knl = lp.tag_array_axes(knl, "vecf", "f,f")                           
            else:   
                knl = lp.add_prefetch(knl, "vec", "j,iel_inner_outer,iel_inner_inner", temporary_name="vecf", default_tag="l.auto")
                knl = lp.tag_array_axes(knl, "vecf", "f,f")

            knl = lp.split_iname(knl, "j", ji, outer_tag="for", inner_tag="for")
            knl = lp.add_inames_for_unused_hw_axes(knl)

            # Change this to just use the transformation list instead of applying the transformations
            # directly
            trans_list = []
            if "diff" in knl.default_entrypoint.name:
                trans_list.append(["tag_inames", ["imatrix: ilp"]])
            trans_list.append(["split_iname", ["iel", kio], {"outer_tag": "g.0", "slabs":(0,1)}])
            trans_list.append(["split_iname", ["iel_inner", kii], 
                {"outer_tag": "ilp", "inner_tag":"l.0", "slabs":(0,1)}])
            trans_list.append(["split_iname", ["idof", iio], {"outer_tag": "g.1", "slabs":(0,0)}])
            trans_list.append(["split_iname", ["idof_inner", iii], 
                {"outer_tag": "ilp", "inner_tag":"l.1", "slabs":(0,1)}])

            if knl.default_entrypoint.name == "face_mass":
                trans_list.append(["add_prefetch", ["vec", "f,j,iel_inner_outer,iel_inner_inner"],
                    {"temporary_name":"vecf", "default_tag":"l.auto"}])
                #trans_list.append(["tag_array_axes", ["vecf", "N1,N0,N2"]])
            elif knl.default_entrypoint.name == "nodes":
                trans_list.append(["add_prefetch", ["nodes", "j,iel_inner_outer,iel_inner_inner"],
                    {"temporary_name":"vecf", "default_tag":"l.auto"}])
                trans_list.append(["tag_array_axes", ["vecf", "f,f"]])
            elif "resample_by_mat" in knl.default_entrypoint.name:
                # Indirection may prevent prefetching
                pass
            else:
                trans_list.append(["add_prefetch", ["vec", "j,iel_inner_outer,iel_inner_inner"],
                    {"temporary_name":"vecf", "default_tag":"l.auto"}])
                trans_list.append(["tag_array_axes", ["vecf", "f,f"]])

            trans_list.append(["split_iname", ["j", ji], {"outer_tag":"for", "inner_tag":"for"}])
            trans_list.append(["add_inames_for_unused_hw_axes"]) 
            
            dev_arrays, avg_time = test_fn(queue, knl)
            tested.append(choices)

            print(choices)
            if device_memory_bandwidth is not None:  # noqa
                bw  = analyze_knl_bandwidth(knl, avg_time)
                frac_peak_GBps = bw / device_memory_bandwidth
                #result_list.append((frac_peak_GBps, (kio, kii, iio, iii, ji)))
                if frac_peak_GBps  >= bandwidth_cutoff:  # noqa
                    # Should validate result here
                    print("Performance is within tolerance of peak bandwith. Terminating search")  # noqa
                    return choices
    
            if max_gflops is not None:
                frac_peak_gflops = analyze_FLOPS(knl, max_gflops, avg_time)
                if frac_peak_gflops >= gflops_cutoff:
                    # Should validate result here
                    print("Performance is within tolerance of peak bandwith or flop rate. Terminating search")  # noqa
                    return choices

            if device_memory_bandwidth is not None and max_gflops is not None:
                result_list.append((avg_time, frac_peak_GBps*device_memory_bandwidth, frac_peak_gflops*max_gflops,
                                     frac_peak_GBps, frac_peak_gflops, (kio, kii, iio, iii, ji)))

            if avg_time < avg_time_saved:
                avg_time_saved = avg_time
                result_saved = choices
                result_saved_list = trans_list

    print("Time limit exceeded: returning current best result")

    """
    print("Suggested loop splittings")
    print(f"iel: {kio}")
    print(f"iel_inner: {kii}")
    print(f"idof: {iio}")
    print(f"idof_inner: {iii}")
    print(f"j: {ji}")
    """    

    result_list.sort()

    print("Avg_time, Peak_BW, Peak_GFLOPS, Frac_peak_bandwidth, Frac_peak_GFlops")
    #print("Avg time, Frac peak bandwidth, Frac peak GFlops")
    for entry in result_list:
        print(entry)
    print()
    #print(result_list)


    #return result_saved
    return result_saved_list

def convert(o):
    if isinstance(o, np.generic): return o.item()
    raise TypeError


def autotune_and_save(queue, search_fn, tlist_generator, pspace_generator,  hjson_file_str, time_limit=np.inf):
    from hjson import dump
    try:
        avg_time, transformations, data = search_fn(queue, program, generic_test, 
                                    pspace_generator, tlist_generator, time_limit=time_limit)
    except cl._cl.RuntimeError as e:
        print(e)
        print("Profiling is not enabled and the PID does not match any transformation file. Turn on profiling and run again.")

    od = {"transformations": transformations}
    out_file = open(hjson_file_str, "wt+")

    hjson.dump(od, out_file,default=convert)
    out_file.close()
    return transformations


def get_transformation_id(device_id):
    hjson_file = open("device_mappings.hjson") 
    hjson_text = hjson_file.read()
    hjson_file.close()
    od = hjson.loads(hjson_text)
    return od[device_id]

if __name__ == "__main__": 
    from __init__ import gen_diff_knl, load_transformations_from_file, apply_transformation_list
    from grudge.execution import diff_prg, elwise_linear_prg, face_mass_prg

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

    bandwidths = []
    from os import environ
    for nreg in range(57,58):#range(1, 61):
        environ['CU_JIT_MAX_REGISTERS'] = str(nreg)
        for dim in range(3,4):
            hjson_file = open(dim_to_file[dim])
            #for i in range(2,8):
            pn = 5
            n_out = len(equidistant_nodes(pn, 3)[1])
            n_in = len(equidistant_nodes(pn, 3)[1]) 
            n_elem = 178746 # 2**20
            knl = diff_prg(dim, n_elem, n_out, fp_format) 
            #knl = gen_diff_knl_fortran2(dim, n_elem, n_out, n_in, fp_format=fp_format)
            knl = set_memory_layout(knl)
            knl = lp.set_options(knl, "write_code")
            trans = load_transformations_from_file(hjson_file, [tid, fp_string, str(n_out)])
            knl = apply_transformation_list(knl, trans)
            #print(lp.generate_code_v2(knl).device_code())

            dev_arrays, avg_time = generic_test(queue, knl, nruns=10, warmup=True)
            #dev_arrays, avg_time = runTest(n_elem, n_in, n_out, kio, kii, iio, iii, ji)
            bw = analyze_knl_bandwidth(knl, avg_time)
            bandwidths.append(bw)
            #analyzeResult(n_out, n_in, n_elem, 12288//2, 540, avg_time, fp_bytes=fp_bytes)
            print(avg_time)
            #verifyResult(*dev_arrays)
    
    print(knl)
    for i, entry in enumerate(bandwidths):
        print(f"{i}, {entry}")
    #print(bandwidths)
    """
    #testBandwidth()
    #exit()
    """
    # Test elwise linear
    pn = 4
    n_out = len(equidistant_nodes(pn,3)[1])
    n_in = n_out
    n_elem = 178746
    fp_format = np.float64
    fp_string = "FP64" if fp_format == np.float64 else "FP32" 
    knl = elwise_linear_prg(n_elem, n_out, fp_format)
    #knl = gen_elwise_linear_knl(n_elem, n_in, n_out, fp_format)

    hjson_file = open("elwise_linear_transform.hjson")
    trans = load_transformations_from_file(hjson_file, [tid, fp_string, str(n_out)])

    knl = set_memory_layout(knl)
    knl = apply_transformation_list(knl, trans)
    #print(knl)
    _, avg_time = generic_test(queue, knl, backend="OPENCL", nruns=10, warmup=True)
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

    knl = face_mass_prg(178746, 4, 20, 20, np.float64)
    knl = set_memory_layout(knl)
    #knl = gen_face_mass_knl(nelements, nfaces, nvol_nodes, nface_nodes, fp_format)
    #knl = gen_face_mass_knl_merged(nelements, nfaces, nvol_nodes, nface_nodes, fp_format)
    # Need to load these from file
    #hjson_file = open("elwise_linear_transform.hjson")
    #trans = load_transformations_from_file(hjson_file, [tid, fp_string, str(pn)])
    #knl = apply_transformation_list(knl, trans)
    print(knl)
    _, avg_time = test_face_mass(queue, knl, backend="OPENCL", nruns=10, warmup=True)
    #_, avg_time = test_face_mass_merged(queue, knl, backend="OPENCL", nruns=10, warmup=True)
    print(avg_time)
    analyze_knl_bandwidth(knl, avg_time)
    """

    # Test order=4 copy
    """
    knl = lp.make_copy_kernel("f,f", old_dim_tags="f,f")
    knl = lp.add_dtypes(knl, {"input": np.float64, "output": np.float64})
    knl = lp.fix_parameters(knl, {"n0": 178746, "n1": 35})  
    knl = lp.split_iname(knl, "i0", 48, outer_tag="g.0")
    knl = lp.split_iname(knl, "i0_inner", 16, outer_tag="ilp", inner_tag="l.0")
    knl = lp.split_iname(knl, "i1", 35, outer_tag="g.1", inner_tag="l.1")
    for arg in knl.default_entrypoint.args:
        if arg.name == "input":
            arg.tags = IsDOFArray()
            arg.shape = (178746, 35)
        if arg.name == "output":
            arg.tags = IsDOFArray()
            arg.is_output = True 
            arg.shape = (178746, 35)

    print(knl)
    _, avg_time = generic_test(queue, knl)
    analyze_knl_bandwidth(knl, avg_time)
    #knl = lp.split_iname(knl, "i1", 1024//2, inner_tag="l.0", outer_tag="g.0", slabs=(0,1))
    #knl = lp.split_iname(knl, "i1", 1024, inner_tag="l.0", outer_tag="g.0", slabs=(0,1))
    #knl = lp.split_iname(knl, "i1", 6*16, outer_tag="g.0") 
    #knl = lp.split_iname(knl, "i1_inner", 16, outer_tag="ilp", inner_tag="l.0", slabs=(0,1)) 
    #knl = lp.split_iname(knl, "i0", n0, inner_tag="l.1", outer_tag="g.1", slabs=(0,0))
    """
   

    #"""
    # Test autotuner
    knl = diff_prg(3, 1000000, 3, np.float64)
    #print(knl)
    #print(knl.default_entrypoint.domains)
    #print(knl.default_entrypoint.instructions)
    #exit()
    #knl = diff_prg(3, 196608, 10, np.float64)
    #knl = elwise_linear_prg(24576, 120, np.float64)
    #dofs = 84
    #knl = elwise_linear_prg(1000000, 3*dofs, np.float64, nnodes_in=dofs)
    #start_param = (24, 4, 126, 9, 28)#(96, 32, 60, 2, 5)
    start_param = None
    ## Figure out the actual dimensions
    #knl = face_mass_prg(178746, 4, 20, 20, np.float64)

    # Spock
    #result = exhaustive_search(queue, knl, generic_test, time_limit=np.inf, max_gflops=11540, device_memory_bandwidth=1047, gflops_cutoff=0.95, bandwidth_cutoff=1.0, start_param=start_param)
    #pspace_generator = gen_autotune_list(queue, knl)
    #print(len(result))

    # Titan V
    #result = exhaustive_search(queue, knl, generic_test, time_limit=np.inf, max_gflops=6144, device_memory_bandwidth=580, gflops_cutoff=0.95, bandwidth_cutoff=1.0, start_param=start_param)
    #print(result)
    pspace_generator = gen_autotune_list
    tlist_generator = mxm_trans_list_generator 
    result = exhaustive_search_v2(queue, knl, generic_test, pspace_generator, tlist_generator, time_limit=np.inf, gflops_cutoff=0.95, bandwidth_cutoff=1.0, start_param=start_param)
 
    #result = exhaustive_search_v2(queue, knl, generic_test, pspace_generator, tlist_generator, time_limit=np.inf, max_gflops=6144, device_memory_bandwidth=580, gflops_cutoff=0.95, bandwidth_cutoff=1.0, start_param=start_param)
