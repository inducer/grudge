from charm4py import charm, Chare, Array, Reducer, Future
import pyopencl as cl
import numpy as np
#import grudge.loopy_dg_kernels as dgk
#from grudge.execution import diff_prg, elwise_linear

class AutotuneTask(Chare):

    #def __init__(self, params):
    #    pass

    def get_queue():
        platform = cl.get_platforms()
        gpu_devices = platform[0].get_devices(device_type=cl.device_type.GPU)
        n_gpus = len(gpu_devices)
        print(gpu_devices[charm.myPe() % n_gpus])
        #ctx = cl.Context(devices=[gpu_devices[charm.myPe() % n_gpus]])
        #queue = cl.CommandQueue(ctx, properties=cl.commmand_queue_properties.PROFILING_ENABLE)    

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
    platform = cl.get_platforms()
    gpu_devices = platform[platform_num].get_devices(device_type=cl.device_type.GPU)
    ctx = cl.Context(devices=[gpu_devices[pe_num % len(gpu_devices)]])
    #return queue = cl.CommandQueue(ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)
    return gpu_devices[pe_num % len(gpu_devices)].int_ptr

def do_work(params):
    queue = get_queue(charm.myPe())
    return queue 

def square(x):
    return x**2


def main(args):

    platform = cl.get_platforms()
    gpu_devices = platform[0].get_devices(device_type=cl.device_type.GPU)
    assert charm.numPes() == charm.numHosts()*len(gpu_devices)

    # Create queue, assume all GPUs on the machine are the same
    
    # Check that it can assign one PE to each GPU
    # The first PE is used for scheduling
    # Not certain how this will work with multiple nodes
    
    print(charm.numPes())

    a = Array(AutotuneTask(), charm.numPes)
    a.get_queue()
    

    """
    # 
    ctx = cl.Context(devices=[gpu_devices[0]])
    queue = cl.CommandQueue(ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)


    print(gpu_devices[0].int_ptr)
    print(gpu_devices[1].int_ptr)

    result = charm.pool.map(do_work, range(2))
    for r in result:
        print(r)
    
    #knl = diff_prg(3, 100000, 56, np.float64)
    #autotune_list = gen_autotune_list(queue, knl) 
    #print(autotune_list)

    print(charm.numHosts())

    f = Future()
    a = Array(Test, 4)
    a.sayHi(f)
    result = f.get()
    print(result)
    print("All finished")
    exit()    
    """

charm.start(main)
