from charm4py import charm
import pyopencl as cl
#import grudge.loopy_dg_kernels as dgk
#from grudge.execution import diff_prg, elwise_linear


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
    n = 100000
    CHUNK_SIZE = 16
    # apply function 'square' to elements in 0 to n-1 using the available
    # cores. Parallel tasks are formed by grouping elements into chunks
    # of size CHUNK_SIZE
    result = charm.pool.map(square, range(n), chunksize=CHUNK_SIZE)
    assert result == [square(i) for i in range(n)]

    # Create queue, assume all GPUs on the machine are the same
    platform = cl.get_platforms()
    gpu_devices = platform[0].get_devices(device_type=cl.device_type.GPU)
    
    # Check that it can assign one PE to each GPU
    # The first PE is used for scheduling
    # Not certain how this will work with multiple nodes
    assert charm.numPes() - 1 == charm.numHosts()*len(gpu_devices)

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

    exit()


charm.start(main)
