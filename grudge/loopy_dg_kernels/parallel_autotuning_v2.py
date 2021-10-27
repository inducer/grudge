from charm4py import entry_method, charm, chare, Chare, Array, Reducer, Future
#from charm4py.pool import PoolScheduler
#from charm4py.charm import Charm, CharmRemote
#from charm4py.chare import GROUP, MAINCHARE, ARRAY, CHARM_TYPES, Mainchare, Group, ArrayMap
#from charm4py.sections import SectionManager
#import inspect
#import sys

import pyopencl as cl
import numpy as np
import grudge.loopy_dg_kernels as dgk
#from grudge.execution import diff_prg, elwise_linear

# Balances the number of workers per host
'''
class BalancedPoolScheduler(PoolScheduler):
    
    def __init__(self):
       super().__init__()
       n_pes = charm.numPes()
       n_hosts = charm.numHosts()
       assert n_pes % n_hosts == 0 # Enforce constant number of pes per host
       pes_per_host = n_pes // n_hosts
       assert pes_per_host > 1 # We're letting one pe on each host be unused

       self.idle_workers = set([i for i in range(n_pes) if not i % per_per_host == 0 ])
       self.num_workers = len(self.idle_workers)

class MyCharm(Charm):
    def _registerInternalChares(self):
        global SectionManager
        from charm4py.sections import SectionManager
        self.register(SectionManager, (GROUP,))

        self.register(CharmRemote, (GROUP,))

        from charm4py.pool import Worker
        if self.interactive:
            if sys.version_info < (3, 0, 0):
                entry_method.coro(BalancedPoolScheduler.start.im_func)
                entry_method.coro(BalancedPoolScheduler.startSingleTask.im_func)
            else:
                entry_method.coro(BalancedPoolScheduler.start)
                entry_method.coro(BalancedPoolScheduler.startSingleTask)
        self.register(BalancedPoolScheduler, (ARRAY,))
        self.register(Worker, (GROUP,))

        if self.options.profiling:
            self.internalChareTypes.update({SectionManager, CharmRemote,
                                            BalancedPoolScheduler, Worker})

    def _createInternalChares(self):
        Group(CharmRemote)
        Group(SectionManager)

        from charm4py.pool import Pool, BalancedPoolScheduler
        pool_proxy = Chare(BalancedPoolScheduler, onPE=0)
        self.pool = Pool(pool_proxy)
        readonlies.charm_pool_proxy__h = pool_proxy


    def registerInCharm(self, C):
        C.idx = [None] * len(CHARM_TYPES)
        charm_types = self.registered[C]
        if Mainchare in charm_types:
            self.registerInCharmAs(C, Mainchare, self.lib.CkRegisterMainchare)
        if Group in charm_types:
            if ArrayMap in C.mro():
                self.registerInCharmAs(C, Group, self.lib.CkRegisterArrayMap)
            elif C == SectionManager:
                self.registerInCharmAs(C, Group, self.lib.CkRegisterSectionManager)
            else:
                self.registerInCharmAs(C, Group, self.lib.CkRegisterGroup)
        if Array in charm_types:
            self.registerInCharmAs(C, Array, self.lib.CkRegisterArray)


    def registerAs(self, C, charm_type_id):
        if charm_type_id == MAINCHARE:
            assert not self.mainchareRegistered, 'More than one entry point has been specified'
            self.mainchareRegistered = True
            # make mainchare constructor always a coroutine
            if sys.version_info < (3, 0, 0):
                entry_method.coro(C.__init__.im_func)
            else:
                entry_method.coro(C.__init__)
        charm_type = chare.charm_type_id_to_class[charm_type_id]
        # print("charm4py: Registering class " + C.__name__, "as", charm_type.__name__, "type_id=", charm_type_id, charm_type)
        profilingOn = self.options.profiling
        ems = [entry_method.EntryMethod(C, m, profilingOn) for m in charm_type.__baseEntryMethods__()]

        members = dir(C)
        if C == SectionManager:
            ems.append(entry_method.EntryMethod(C, 'sendToSection', profilingOn))
            members.remove('sendToSection')
        self.classEntryMethods[charm_type_id][C] = ems

        for m in members:
            m_obj = getattr(C, m)
            if not callable(m_obj) or inspect.isclass(m_obj):
                continue
            if m in chare.method_restrictions['reserved'] and m_obj != getattr(Chare, m):
                raise Charm4PyError(str(C) + " redefines reserved method '"  + m + "'")
            if m.startswith('__') and m.endswith('__'):
                continue  # filter out non-user methods
            if m in chare.method_restrictions['non_entry_method']:
                continue
            if charm_type_id != ARRAY and m in {'migrate', 'setMigratable'}:
                continue
            # print(m)
            em = entry_method.EntryMethod(C, m, profilingOn)
            self.classEntryMethods[charm_type_id][C].append(em)
        self.registered[C].add(charm_type)
'''

#charm object that uses the BalancedPoolScheduler
#charm = MyCharm()

##### To Delete

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

def square(x):
    return x**2

### End To Delete

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
    assert charm.numPes() == charm.numHosts()*(len(gpu_devices) + 1)
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
   
    result = charm.pool.map(do_work, args)
    sort_key = lambda entry: entry[0]
    result.sort(key=sort_key)
    

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

if __name__ == "__main__":
    charm.start(main)
