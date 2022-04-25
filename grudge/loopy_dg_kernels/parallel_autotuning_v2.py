from charm4py import entry_method, chare, Chare, Array, Reducer, Future, charm
from charm4py.pool import PoolScheduler, Pool
from charm4py.charm import Charm, CharmRemote
#from charm4py.chare import GROUP, MAINCHARE, ARRAY, CHARM_TYPES, Mainchare, Group, ArrayMap
#from charm4py.sections import SectionManager
#import inspect
#import sys
import hjson
import pyopencl as cl
import numpy as np
import grudge.loopy_dg_kernels as dgk
import os
import grudge.grudge_array_context as gac
import loopy as lp
from os.path import exists
from grudge.loopy_dg_kernels.run_tests import run_single_param_set, generic_test
from grudge.grudge_array_context import convert
#from grudge.execution import diff_prg, elwise_linear

# Makes one PE inactive on each host so the number of workers is the same on all hosts as
# opposed to the basic PoolScheduler which has one fewer worker on the host with PE 0.
# This can be useful for running tasks on a GPU cluster for example.
class BalancedPoolScheduler(PoolScheduler):

    def __init__(self):
       super().__init__()
       n_pes = charm.numPes()
       n_hosts = charm.numHosts()
       pes_per_host = n_pes // n_hosts

       assert n_pes % n_hosts == 0 # Enforce constant number of pes per host
       assert pes_per_host > 1 # We're letting one pe on each host be unused

       self.idle_workers = set([i for i in range(n_pes) if not i % pes_per_host == 0 ])
       self.num_workers = len(self.idle_workers)

# Use all PEs including PE 0 
class AllPEsPoolScheduler(PoolScheduler):

    def __init__(self):
       super().__init__()
       n_pes = charm.numPes()
       n_hosts = charm.numHosts()

       self.idle_workers = set(range(n_pes))
       self.num_workers = len(self.idle_workers)


def get_queue(pe_num, platform_num):
    platforms = cl.get_platforms()
    gpu_devices = platforms[platform_num].get_devices(device_type=cl.device_type.GPU)
    ctx = cl.Context(devices=[gpu_devices[pe_num % len(gpu_devices)]])
    queue = cl.CommandQueue(ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)
    return queue


def do_work(args):
    params = args[0]
    knl = args[1]
    queue = get_queue(charm.myPe())
    print("PE: ", charm.myPe())
    avg_time, transform_list = dgk.run_tests.apply_transformations_and_run_test(queue, knl, dgk.run_tests.generic_test, params)
    return avg_time, params

def test(args):
    platform_id, knl, tlist_generator, params, test_fn = args
    queue = get_queue(charm.myPe(), platform_id)
    result = run_single_param_set(queue, knl, tlist_generator, params, test_fn) 
    return result



def unpickle_kernel(fname):
    from pickle import load
    f = open(fname, "rb")
    program = load(f)
    f.close()
    return program

def autotune_pickled_kernels(path, platform_id, actx_class, comm):
    from os import listdir
    dir_list = listdir(path)
    for f in dir_list:
        if f.endswith(".pickle"):
            fname = path + "/" + f
            print("===============================================")
            print("Autotuning", fname)
            knl = unpickle_kernel(fname)
            knl_id = f.split(".")[0]
            knl_id = knl_id.split("_")[-1]
            print("Kernel ID", knl_id)
            print("New kernel ID", gac.unique_program_id(knl))
            
            assert knl_id == gac.unique_program_id(knl)
            knl = lp.set_options(knl, lp.Options(no_numpy=True, return_dict=True))
            knl = gac.set_memory_layout(knl)
            assert knl_id == gac.unique_program_id(knl)

            print(knl)
            pid = gac.unique_program_id(knl)
            hjson_file_str = f"hjson/{knl.default_entrypoint.name}_{pid}.hjson"
            if not exists(hjson_file_str):
                parallel_autotune(knl, platform_id, actx_class, comm)
            else:
                print("hjson file exists, skipping")

def parallel_autotune(knl, platform_id, actx_class, comm):

    # Create queue, assume all GPUs on the machine are the same
    platforms = cl.get_platforms()
    gpu_devices = platforms[platform_id].get_devices(device_type=cl.device_type.GPU)
    n_gpus = len(gpu_devices)
    ctx = cl.Context(devices=[gpu_devices[charm.myPe() % n_gpus]])
    profiling = cl.command_queue_properties.PROFILING_ENABLE
    queue = cl.CommandQueue(ctx, properties=profiling)    


    import pyopencl.tools as cl_tools
    actx = actx_class(
        comm,
        queue,
        allocator=cl_tools.MemoryPool(cl_tools.ImmediateAllocator(queue)))

    #knl = gac.fix_program_parameters(knl)
    #knl = lp.set_options(knl, lp.Options(no_numpy=True, return_dict=True))
    knl = gac.set_memory_layout(knl)
    pid = gac.unique_program_id(knl)
    os.makedirs(os.getcwd() + "/hjson", exist_ok=True)
    hjson_file_str = f"hjson/{knl.default_entrypoint.name}_{pid}.hjson"


    assert charm.numPes() > 1
    #assert charm.numPes() - 1 <= charm.numHosts()*len(gpu_devices)
    #assert charm.numPes() <= charm.numHosts()*(len(gpu_devices) + 1)
    # Check that it can assign one PE to each GPU
    # The first PE is used for scheduling
    # Not certain how this will work with multiple nodes

    from run_tests import run_single_param_set
    
    tlist_generator, pspace_generator = actx.get_generators(knl)
    params_list = pspace_generator(actx.queue, knl)

    # Could make a massive list with all kernels and parameters
    args = [(platform_id, knl, tlist_generator, p, generic_test,) for p in params_list]


    # May help to balance workload
    # Should test if shuffling matters
    from random import shuffle
    shuffle(args)


    #a = Array(AutotuneTask, dims=(len(args)), args=args[0])
    #a.get_queue()
   
    #result = charm.pool.map(do_work, args)

    #pool_proxy = Chare(BalancedPoolScheduler, onPE=0) # Need to use own charm++ branch to make work

    pool_proxy = Chare(PoolScheduler, onPE=0)
    mypool = Pool(pool_proxy)
    if len(args) > 0: # Guard against empty list
        results = mypool.map(test, args)

        sort_key = lambda entry: entry[0]
        results.sort(key=sort_key)
        
        #for r in results:
        #    print(r)
        # Workaround for pocl CUDA bug
        # whereby times are imprecise
        ret_index = 0
        for i, result in enumerate(results):
            if result[0] > 1e-7:
                ret_index = i
                break

        avg_time, transformations, data = results[ret_index]
    else:
        transformations = {}
    
    od = {"transformations": transformations}
    out_file = open(hjson_file_str, "wt+")
    hjson.dump(od, out_file,default=convert)
    out_file.close()

    return transformations

"""
def main(args):

    # Create queue, assume all GPUs on the machine are the same
    platforms = cl.get_platforms()
    platform_id = 0
    gpu_devices = platforms[platform_id].get_devices(device_type=cl.device_type.GPU)
    n_gpus = len(gpu_devices)
    ctx = cl.Context(devices=[gpu_devices[charm.myPe() % n_gpus]])
    profiling = cl.command_queue_properties.PROFILING_ENABLE
    queue = cl.CommandQueue(ctx, properties=profiling)    
   
    assert charm.numPes() > 1
    #assert charm.numPes() - 1 <= charm.numHosts()*len(gpu_devices)
    assert charm.numPes() <= charm.numHosts()*(len(gpu_devices) + 1)
    # Check that it can assign one PE to each GPU
    # The first PE is used for scheduling
    # Not certain how this will work with multiple nodes
    
    from grudge.execution import diff_prg, elwise_linear_prg
    knl = diff_prg(3, 1000000, 3, np.float64)
    params = dgk.run_tests.gen_autotune_list(queue, knl)

    args = [[param, knl] for param in params]

    # May help to balance workload
    from random import shuffle
    shuffle(args)
    
    #a = Array(AutotuneTask, dims=(len(args)), args=args[0])
    #a.get_queue()
   
    #result = charm.pool.map(do_work, args)

    pool_proxy = Chare(BalancedPoolScheduler, onPE=0)
    mypool = Pool(pool_proxy)
    result = mypool.map(do_work, args)

    sort_key = lambda entry: entry[0]
    result.sort(key=sort_key)
    

    for r in result:
        print(r)
"""

def main(args):
    import mpi4py.MPI as MPI
    from mirgecom.array_context import MirgecomAutotuningArrayContext as Maac
    comm = MPI.COMM_WORLD
    
    autotune_pickled_kernels("./pickled_programs", 0, Maac, comm)
    print("DONE!")
    exit()

def charm_autotune():
    charm.start(main)
    print(result)
    charm.exit()
 
if __name__ == "__main__":
    charm.start(main)
    print(result)
    charm.exit()
