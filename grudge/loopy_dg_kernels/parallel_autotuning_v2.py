from charm4py import entry_method, chare, Chare, Array, Reducer, Future, charm
from charm4py.pool import PoolScheduler, Pool
from charm4py.charm import Charm, CharmRemote
#from charm4py.chare import GROUP, MAINCHARE, ARRAY, CHARM_TYPES, Mainchare, Group, ArrayMap
#from charm4py.sections import SectionManager
#import inspect
#import sys

import pyopencl as cl
import numpy as np
import grudge.loopy_dg_kernels as dgk
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


def get_queue(pe_num, platform_num=0):
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
    knl = diff_prg(3, 1000000, 20, np.float64)
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
    
if __name__ == "__main__":
    charm.start(main)
    print(result)
    charm.exit()
    
