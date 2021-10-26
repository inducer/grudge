from charm4py import charm, Chare, Array, Reducer, Future
import pyopencl as cl
import numpy as np
import grudge.loopy_dg_kernels as dgk
#from grudge.execution import diff_prg, elwise_linear

class AutotuneTask(Chare):

    def __init__(self, platform_id, params):
        self.platform_id = platform_id
        self.params = params

    def get_queue(self):
        platform = cl.get_platforms()
        gpu_devices = platform[self.platform_id].get_devices(device_type=cl.device_type.GPU)
        n_gpus = len(gpu_devices)
        ctx = cl.Context(devices=[gpu_devices[charm.myPe() % n_gpus]])
        profiling = cl.command_queue_properties.PROFILING_ENABLE
        queue = cl.CommandQueue(ctx, properties=profiling)    
        return queue

    def run(self):
        print([self.params, np.random.rand])


class Test(Chare):
    def start(self):
        print('I am element', self.thisIndex, 'on PE', charm.myPe(),
              'sending a msg to element 1')
        self.thisProxy[1].sayHi()

    #@coro
    def sayHi(self, future):
        rn = np.random.rand()
        print('Hello from element', self.thisIndex, 'on PE', charm.myPe(), 'random', rn)
        self.reduce(future, rn, Reducer.max)

def get_queue(pe_num, platform_num=0):
    platforms = cl.get_platforms()
    gpu_devices = platforms[platform_num].get_devices(device_type=cl.device_type.GPU)
    ctx = cl.Context(devices=[gpu_devices[pe_num % len(gpu_devices)]])
    queue = cl.CommandQueue(ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)
    return queue
    #return gpu_devices[pe_num % len(gpu_devices)].int_ptr

def do_work(args):
    params = args[0]
    knl = args[1]
    queue = get_queue(charm.myPe())
    result = dgk.run_tests.apply_transformations_and_run_test(queue, knl, dgk.run_tests.generic_test, params)
    return result

def square(x):
    return x**2


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
    assert charm.numPes() - 1 <= charm.numHosts()*len(gpu_devices)
    # Check that it can assign one PE to each GPU
    # The first PE is used for scheduling
    # Not certain how this will work with multiple nodes
    
    from grudge.execution import diff_prg, elwise_linear_prg
    knl = diff_prg(3, 100000, 10, np.float64)
    params = dgk.run_tests.gen_autotune_list(queue, knl)

    args = [[param, knl] for param in params]
    
    #a = Array(AutotuneTask, dims=(len(args)), args=args[0])
    #a.get_queue()
   
    result = charm.pool.map(do_work, args[:10])
    for r in result:
        print(r)
    
    #knl = diff_prg(3, 100000, 56, np.float64)
    #autotune_list = gen_autotune_list(queue, knl) 
    #print(autotune_list)

    #print(charm.numHosts())

    #f = Future()
    #a = Array(Test, a.numPes())
    #a.sayHi(f)
    #result = f.get()
    #print(result)
    #print("All finished")
    charm.exit()    

charm.start(main)
