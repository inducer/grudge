from charm4py import charm
import grudge.loopy_dg_kernels as dgk
from grudge.execution import diff_prg, elwise_linear

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
    ctx = cl.Context(devices=gpu_devices)
    queue = cl.CommandQueue(ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)

    knl = diff_prg(3, 100000, 56, np.float64)
    autotune_list = gen_autotune_list(queue, knl) 
    print(autotune_list)


    exit()


charm.start(main)
