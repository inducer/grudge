import numpy as np
from grudge.grudge_tags import (IsDOFArray, IsSepVecDOFArray,
    IsOpArray, IsSepVecOpArray, IsFaceDOFArray, IsFaceMassOpArray,
    IsVecDOFArray, IsVecOpArray, IsFourAxisDOFArray)

def k_inner_inner_options(start_val=None):
    #options = [8, 16, 4, 32]
    options = [32, 16, 8]
    start_ind = 0 if start_val is None else options.index(start_val)
    options = options[start_ind:]
    return options


def k_inner_outer_options(n_in, k_inner_inner, sm_size,
                            fp_bytes=8, start_val=None, nelem=None):
    ilp_limit = min(nelem // k_inner_inner, 6) if nelem is not None else 6
    # Possibilities limited by size of local memory
    # Use sm_size - 1 because CUDA errors when all of local memory is used
    options = np.arange(1, ((sm_size - 1) // (fp_bytes*k_inner_inner*n_in)) + 1)
    #Arbitrarily limit to at max 6 inline to limit search space
    options = list(k_inner_inner*options[options <= ilp_limit])
    start_ind = 0 if start_val is None else options.index(start_val)
    options = options[start_ind:]
    return options

def i_inner_inner_options(n_out, k_inner_inner, max_work_group_size=1024, start_val=None):
    factors = np.arange(1, n_out+1)[(n_out % np.arange(1, n_out+1)) == 0]
    # Fix for AMD
    #factors = np.arange(3, n_out+1)[(n_out % np.arange(2, n_out+1)) == 0]
    # Ensure total number of workitems is less than maximum
    usable_factors = factors[factors*k_inner_inner <= max_work_group_size]
    options = sorted(usable_factors, reverse=True)
    start_ind = 0 if start_val is None else options.index(start_val)
    options = options[start_ind:]
    return options

def i_inner_outer_options(n_out, i_inner_inner, start_val=None):
    # Select a number of inline blocks such that n_out % outer*inner == 0
    # Bumping up the start of the range could reduce autotune time, but an empty
    # autotune set might be returned if i < start value
    
    # Loopy confused about the number of dimensions when 
    # i_outer, i_inner_outer, and i_inner_inner are all 1
    inline = [1] if n_out == 1 else np.arange(2, (n_out // i_inner_inner) + 1)
    options = list(i_inner_inner*inline[n_out % (inline*i_inner_inner) == 0])
    start_ind = 0 if start_val is None else options.index(start_val)
    options = options[start_ind:]
    return options


def j_inner_options(n_in, start_val=None):

    start = 1
    factors = list(np.arange(start, n_in + 1)[(n_in % np.arange(start, n_in + 1)) == 0])
    #factors = list(np.arange(1, n_in + 1)[(n_in % np.arange(1, n_in + 1)) == 0])
    # Should this be limited by the number of registers
    start_ind = 0 if start_val is None else factors.index(start_val)
    factors = factors[start_ind:]
    return factors

# Creates a list containing tuples of search space parameters.
# Will need to create separate ones of this for each einsum kernel
def gen_autotune_list(queue, knl, start_param=None):

    local_mem_size = queue.device.local_mem_size
    max_work_group_size = queue.device.max_work_group_size    
    nfaces = 1

    n_in = None
    for arg in knl.default_entrypoint.args:
        if "resample_by_mat" not in knl.default_entrypoint.name:
            if IsDOFArray() in arg.tags:
                n_elem, n_out = arg.shape
                fp_bytes = arg.dtype.dtype.itemsize
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

    if n_in is None:
        n_in = n_out

    n_in = n_in * nfaces #Prevents shared memory from overflowing in face mass kernel   

    if start_param is not None:
        kio_s, kii_s, iio_s, iii_s, ji_s = start_param
    else:
        kio_s, kii_s, iio_s, iii_s, ji_s = (None, None, None, None, None)

    # Iterate over five search dimensions
    parameter_list = []
    for kii in k_inner_inner_options(start_val=kii_s):
        # Should come up with a way to set the effective local memory size. It depends on the number of
        # arrays actually prefetched.
        for kio in k_inner_outer_options(n_in*nfaces, kii, local_mem_size, fp_bytes=fp_bytes,start_val=kio_s):
            kio_s = None # Set to None so will form the full set the next time around
            for iii in i_inner_inner_options(n_out, kii,
                    max_work_group_size=max_work_group_size, start_val=iii_s):
                iii_s = None
                for iio in i_inner_outer_options(n_out, iii, start_val=iio_s):
                    # Kernel does not reach here.
                    iio_s = None
                    for ji in j_inner_options(n_in, start_val=ji_s):
                        ji_s = None
                        choices = (kio, kii, iio, iii, ji)
                        parameter_list.append(choices)

    return parameter_list


# Should separate this so don't need to supply knl
def mxm_trans_list_generator(params, **kwargs):
    trans_list = []
    kio, kii, iio, iii, ji = params
    knl = kwargs["knl"]


    #if "diff" in knl.default_entrypoint.name:
    #    trans_list.append(["tag_inames", ["imatrix: ilp"]])

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
    #elif knl.default_entrypoint.name == "nodes":
    elif knl.default_entrypoint.name == "lp_nodes":
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
    return trans_list


def grudge_elementwise_sum_knl_tlist_generator(params, **kwargs):
    trans_list = []
    kio, kii, iio, iii, ji = params
    knl = kwargs["knl"]

    trans_list.append(["split_iname", ["iel", kio], {"outer_tag": "g.0", "slabs":(0,1)}])
    trans_list.append(["split_iname", ["iel_inner", kii], 
        {"outer_tag": "ilp", "inner_tag":"l.0", "slabs":(0,1)}])
    trans_list.append(["split_iname", ["idof", iio], {"outer_tag": "g.1", "slabs":(0,0)}])
    trans_list.append(["split_iname", ["idof_inner", iii], 
        {"outer_tag": "ilp", "inner_tag":"l.1", "slabs":(0,1)}])
    # Should the i loop have (0,1) slabs for both?

    #trans_list.append(["add_prefetch", ["operand", "iel_inner_outer,iel_inner_inner"],
    #    {"temporary_name":"operandf", "default_tag":"l.auto"}])
    #trans_list.append(["tag_array_axes", ["operandf", "f,f"]])

    # Realistically, splitting the j loop probably is not necessary for this.
    trans_list.append(["split_iname", ["jdof", ji], {"outer_tag":"for", "inner_tag":"for"}])
    trans_list.append(["add_inames_for_unused_hw_axes"]) 
    return trans_list 

def grudge_elementwise_sum_knl_pspace_generator(queue, knl, start_param=None):

    local_mem_size = queue.device.local_mem_size
    max_work_group_size = queue.device.max_work_group_size    

    for arg in knl.default_entrypoint.args:
        if IsDOFArray() in arg.tags:
            n_elem, n_out = arg.shape
            n_in = n_out
            fp_bytes = arg.dtype.dtype.itemsize

    if start_param is not None:
        kio_s, kii_s, iio_s, iii_s, ji_s = start_param
    else:
        kio_s, kii_s, iio_s, iii_s, ji_s = (None, None, None, None, None)

    # Iterate over five search dimensions. Could reduce this to 4 if ignore j-loop.
    parameter_list = []
    for kii in k_inner_inner_options(start_val=kii_s):
        # Both jac and vec are prefetched so the available local_memory per prefetched array is halved
        for kio in k_inner_outer_options(n_in, kii, local_mem_size, fp_bytes=fp_bytes,start_val=kio_s):
            kio_s = None # Set to None so will form the full set the next time around
            for iii in i_inner_inner_options(n_out, kii,
                    max_work_group_size=max_work_group_size, start_val=iii_s):
                iii_s = None
                for iio in i_inner_outer_options(n_out, iii, start_val=iio_s):
                    iio_s = None
                    for ji in j_inner_options(n_in, start_val=ji_s):
                        ji_s = None
                        choices = (kio, kii, iio, iii, ji)
                        parameter_list.append(choices)

    return parameter_list


def einsum3to2_kernel_tlist_generator(params, **kwargs):
    trans_list = []
    kio, kii, iio, iii, ji = params
    knl = kwargs["knl"]

    trans_list.append(["split_iname", ["e", kio], {"outer_tag": "g.0", "slabs":(0,1)}])
    trans_list.append(["split_iname", ["e_inner", kii], 
        {"outer_tag": "ilp", "inner_tag":"l.0", "slabs":(0,1)}])
    trans_list.append(["split_iname", ["i", iio], {"outer_tag": "g.1", "slabs":(0,0)}])
    trans_list.append(["split_iname", ["i_inner", iii], 
        {"outer_tag": "ilp", "inner_tag":"l.1", "slabs":(0,1)}])
    # Should the i loop have (0,1) slabs for both?

    for arg in knl.default_entrypoint.args:

        if "vec" == arg.name:
            trans_list.append(["add_prefetch", ["vec", "j,e_inner_outer,e_inner_inner"],
                {"temporary_name":"vecf", "default_tag":"l.auto"}])
            trans_list.append(["tag_array_axes", ["vecf", "f,f"]])
        elif "jac" == arg.name:
            trans_list.append(["add_prefetch", ["jac", "j,e_inner_outer,e_inner_inner"],
                {"temporary_name":"jacf", "default_tag":"l.auto"}])
            trans_list.append(["tag_array_axes", ["jacf", "f,f"]])
        elif "arg2" == arg.name and IsDOFArray() in arg.tags:
            trans_list.append(["add_prefetch", ["arg2", "j,e_inner_outer,e_inner_inner"],
                {"temporary_name":"arg2f", "default_tag":"l.auto"}])
            trans_list.append(["tag_array_axes", ["arg2f", "f,f"]])
        elif "arg1" == arg.name and IsDOFArray() in arg.tags:
            trans_list.append(["add_prefetch", ["arg1", "j,e_inner_outer,e_inner_inner"],
                {"temporary_name":"arg1f", "default_tag":"l.auto"}])
            trans_list.append(["tag_array_axes", ["arg1f", "f,f"]])
        elif "arg0" == arg.name and IsDOFArray() in arg.tags:
            trans_list.append(["add_prefetch",
                ["arg0", "i_inner_outer,i_inner_inner,e_inner_outer,e_inner_inner"],
                {"temporary_name":"arg0f", "default_tag":"l.auto"}])
            trans_list.append(["tag_array_axes", ["arg0f", "f,f"]])

    trans_list.append(["split_iname", ["j", ji], {"outer_tag":"for", "inner_tag":"for"}])
    trans_list.append(["add_inames_for_unused_hw_axes"]) 
    return trans_list 

def einsum3to2_kernel_pspace_generator(queue, knl, start_param=None):

    local_mem_size = queue.device.local_mem_size
    max_work_group_size = queue.device.max_work_group_size    

    for arg in knl.default_entrypoint.args:
        if IsDOFArray() in arg.tags:
            n_elem, n_out = arg.shape
            fp_bytes = arg.dtype.dtype.itemsize
        elif IsOpArray() in arg.tags:
            n_out, n_in = arg.shape

    if start_param is not None:
        kio_s, kii_s, iio_s, iii_s, ji_s = start_param
    else:
        kio_s, kii_s, iio_s, iii_s, ji_s = (None, None, None, None, None)

    # Iterate over five search dimensions
    parameter_list = []
    for kii in k_inner_inner_options(start_val=kii_s):
        # Both jac and vec are prefetched so the available local_memory per prefetched array is halved
        for kio in k_inner_outer_options(n_in, kii, local_mem_size // 2,
                    fp_bytes=fp_bytes,start_val=kio_s,nelem=n_elem):
            kio_s = None # Set to None so will form the full set the next time around
            for iii in i_inner_inner_options(n_out, kii,
                    max_work_group_size=max_work_group_size, start_val=iii_s):
                iii_s = None
                for iio in i_inner_outer_options(n_out, iii, start_val=iio_s):
                    iio_s = None
                    for ji in j_inner_options(n_in, start_val=ji_s):
                        ji_s = None
                        choices = (kio, kii, iio, iii, ji)
                        parameter_list.append(choices)

    return parameter_list


def einsum2to2_kernel_tlist_generator(params, **kwargs):
    trans_list = []
    kio, kii, iio, iii = params
    knl = kwargs["knl"]

    trans_list.append(["split_iname", ["e", kio], {"outer_tag": "g.0", "slabs":(0,1)}])
    trans_list.append(["split_iname", ["e_inner", kii], 
        {"outer_tag": "ilp", "inner_tag":"l.0", "slabs":(0,1)}])
    trans_list.append(["split_iname", ["i", iio], {"outer_tag": "g.1", "slabs":(0,0)}])
    trans_list.append(["split_iname", ["i_inner", iii], 
        {"outer_tag": "ilp", "inner_tag":"l.1", "slabs":(0,1)}])
    # Should the i loop have (0,1) slabs for both?

    # Prefetching probably matters not for this kernel
    #trans_list.append(["add_prefetch", ["arg1", "e_inner_outer,e_inner_inner,i_inner_outer,i_inner_inner"],
    #    {"temporary_name":"arg1f", "default_tag":"l.auto"}])
    #trans_list.append(["tag_array_axes", ["arg1f", "f,f"]])

    trans_list.append(["add_inames_for_unused_hw_axes"]) 
    return trans_list 


def einsum2to2_kernel_pspace_generator(queue, knl, start_param=None):

    local_mem_size = queue.device.local_mem_size
    max_work_group_size = queue.device.max_work_group_size    

    for arg in knl.default_entrypoint.args:
        if IsDOFArray() in arg.tags:
            n_elem, n_out = arg.shape
            n_in = n_out
            fp_bytes = arg.dtype.dtype.itemsize

    if start_param is not None:
        kio_s, kii_s, iio_s, iii_s = start_param
    else:
        kio_s, kii_s, iio_s, iii_s = (None, None, None, None)

    # Iterate over five search dimensions
    parameter_list = []
    for kii in k_inner_inner_options(start_val=kii_s):
        # Both jac and vec are prefetched so the available local_memory per prefetched array is halved
        for kio in k_inner_outer_options(n_in, kii, local_mem_size, fp_bytes=fp_bytes,start_val=kio_s):
            kio_s = None # Set to None so will form the full set the next time around
            for iii in i_inner_inner_options(n_out, kii,
                    max_work_group_size=max_work_group_size, start_val=iii_s):
                iii_s = None
                for iio in i_inner_outer_options(n_out, iii, start_val=iio_s):
                    iio_s = None
                    #for ji in j_inner_options(n_in, start_val=ji_s):
                    #    ji_s = None
                    choices = (kio, kii, iio, iii)
                    parameter_list.append(choices)

    return parameter_list


def einsum4to2_face_mass_kernel_tlist_generator(params, **kwargs):
    trans_list = []
    kio, kii, iio, iii, ji = params

    trans_list.append(["tag_inames", ["f: unr"]])
    trans_list.append(["split_iname", ["e", kio], {"outer_tag": "g.0", "slabs":(0,1)}])
    trans_list.append(["split_iname", ["e_inner", kii], 
        {"outer_tag": "ilp", "inner_tag":"l.0", "slabs":(0,1)}])
    trans_list.append(["split_iname", ["i", iio], {"outer_tag": "g.1", "slabs":(0,0)}])
    trans_list.append(["split_iname", ["i_inner", iii], 
        {"outer_tag": "ilp", "inner_tag":"l.1", "slabs":(0,1)}])
    # Should the i loop have (0,1) slabs for both?

    trans_list.append(["add_prefetch", ["vec", "f,j,e_inner_outer,e_inner_inner"],
        {"temporary_name":"vecf", "default_tag":"l.auto"}])
    trans_list.append(["tag_array_axes", ["vecf", "N2,N0,N1"]])

    trans_list.append(["add_prefetch", ["jac_surf", "f,j,e_inner_outer,e_inner_inner"],
        {"temporary_name":"jac_surff", "default_tag":"l.auto"}])
    trans_list.append(["tag_array_axes", ["jac_surff", "N2,N0,N1"]])

    trans_list.append(["split_iname", ["j", ji], {"outer_tag":"for", "inner_tag":"for"}])

    trans_list.append(["add_inames_for_unused_hw_axes"]) 

    return trans_list 

"""
def einsum4to2_face_mass_kernel_pspace_generator(queue, knl, start_param=None):

    local_mem_size = queue.device.local_mem_size
    max_work_group_size = queue.device.max_work_group_size    

    for arg in knl.default_entrypoint.args:
        if IsDOFArray() in arg.tags:
            n_elem, n_out = arg.shape
            fp_bytes = arg.dtype.dtype.itemsize
        elif IsVecDOFArray() in arg.tags:
            n_r, n_elem, n_out = arg.shape
            fp_bytes = arg.dtype.dtype.itemsize
        elif IsVecOpArray() in arg.tags:
            n_r, n_out, n_in = arg.shape
        elif IsFaceMassOpArray() in arg.tags:
            n_out, n_r, n_in = arg.shape

    if start_param is not None:
        kio_s, kii_s, iio_s, iii_s, ji_s = start_param
    else:
        kio_s, kii_s, iio_s, iii_s, ji_s = (None, None, None, None, None)

    # Iterate over five search dimensions
    parameter_list = []
    for kii in k_inner_inner_options(start_val=kii_s):
        # Both inv_jac_t and vec are prefetched so the amount of available local memory per array is reduced
        for kio in k_inner_outer_options(n_in, kii, local_mem_size // (n_r + 1), fp_bytes=fp_bytes,start_val=kio_s):
            kio_s = None # Set to None so will form the full set the next time around
            for iii in i_inner_inner_options(n_out, kii,
                    max_work_group_size=max_work_group_size, start_val=iii_s):
                iii_s = None
                for iio in i_inner_outer_options(n_out, iii, start_val=iio_s):
                    iio_s = None
                    for ji in j_inner_options(n_in, start_val=ji_s):
                        ji_s = None
                        choices = (kio, kii, iio, iii, ji)
                        parameter_list.append(choices)

    return parameter_list
"""


def einsum4to2_kernel_tlist_generator(params, **kwargs):
    trans_list = []
    kio, kii, iio, iii, ji = params
    knl = kwargs["knl"]
    arg_names = {arg.name for arg in knl.default_entrypoint.args}
    inames = knl.default_entrypoint.inames.keys()
    
    if "r" in inames:
        trans_list.append(["tag_inames", ["r: unr"]])
    if "f" in inames:
        trans_list.append(["tag_inames", ["f: unr"]])
    

    trans_list.append(["split_iname", ["e", kio], {"outer_tag": "g.0", "slabs":(0,1)}])
    trans_list.append(["split_iname", ["e_inner", kii], 
        {"outer_tag": "ilp", "inner_tag":"l.0", "slabs":(0,1)}])
    trans_list.append(["split_iname", ["i", iio], {"outer_tag": "g.1", "slabs":(0,0)}])
    trans_list.append(["split_iname", ["i_inner", iii], 
        {"outer_tag": "ilp", "inner_tag":"l.1", "slabs":(0,1)}])
    # Should the i loop have (0,1) slabs for both?

    #trans_list.append(["add_prefetch", ["vec", "j,e_inner_outer,e_inner_inner"],
    #    {"temporary_name":"vecf", "default_tag":"l.auto"}])
    #trans_list.append(["tag_array_axes", ["vecf", "f,f"]])
    if "inv_jac_t" in arg_names:
        trans_list.append(["add_prefetch", ["vec", "j,e_inner_outer,e_inner_inner"],
            {"temporary_name":"vecf", "default_tag":"l.auto"}])
        trans_list.append(["tag_array_axes", ["vecf", "N0,N1"]])
 
        trans_list.append(["add_prefetch", ["inv_jac_t", "r,j,e_inner_outer,e_inner_inner"],
            {"temporary_name":"inv_jac_tf", "default_tag":"l.auto"}])
        trans_list.append(["tag_array_axes", ["inv_jac_tf", "N2,N0,N1"]])
    elif "jac_surf" in arg_names:
        trans_list.append(["add_prefetch", ["vec", "f,j,e_inner_outer,e_inner_inner"],
            {"temporary_name":"vecf", "default_tag":"l.auto"}])
        trans_list.append(["tag_array_axes", ["vecf", "N1,N0,N2"]])
 
        trans_list.append(["add_prefetch", ["jac_surf", "f,j,e_inner_outer,e_inner_inner"],
            {"temporary_name":"inv_jac_tf", "default_tag":"l.auto"}])
        trans_list.append(["tag_array_axes", ["inv_jac_tf", "N1,N0,N2"]])
 
    trans_list.append(["split_iname", ["j", ji], {"outer_tag":"for", "inner_tag":"for"}])
    trans_list.append(["add_inames_for_unused_hw_axes"]) 
    return trans_list 

def einsum4to2_kernel_pspace_generator(queue, knl, start_param=None):

    local_mem_size = queue.device.local_mem_size
    max_work_group_size = queue.device.max_work_group_size    
    lmem_divisor = 0

    for arg in knl.default_entrypoint.args:
        if IsDOFArray() in arg.tags:
            n_elem, n_out = arg.shape
            fp_bytes = arg.dtype.dtype.itemsize
        elif IsVecDOFArray() in arg.tags:
            n_r, n_elem, n_out = arg.shape
            fp_bytes = arg.dtype.dtype.itemsize
        elif IsFaceDOFArray() in arg.tags:
            n_r, n_elem, n_in = arg.shape
            fp_bytes = arg.dtype.dtype.itemsize
        elif IsVecOpArray() in arg.tags:
            n_r, n_out, n_in = arg.shape
            lmem_divisor = n_r + 1
        elif IsFaceMassOpArray() in arg.tags:
            n_out, n_r, n_in = arg.shape
            lmem_divisor = 2*n_r

    if start_param is not None:
        kio_s, kii_s, iio_s, iii_s, ji_s = start_param
    else:
        kio_s, kii_s, iio_s, iii_s, ji_s = (None, None, None, None, None)

    # Iterate over five search dimensions
    parameter_list = []
    for kii in k_inner_inner_options(start_val=kii_s):
        # Both inv_jac_t and vec are prefetched so the amount of available local memory per array is reduced
        for kio in k_inner_outer_options(n_in, kii, local_mem_size // lmem_divisor, fp_bytes=fp_bytes,start_val=kio_s):
            kio_s = None # Set to None so will form the full set the next time around
            for iii in i_inner_inner_options(n_out, kii,
                    max_work_group_size=max_work_group_size, start_val=iii_s):
                iii_s = None
                for iio in i_inner_outer_options(n_out, iii, start_val=iio_s):
                    iio_s = None
                    for ji in j_inner_options(n_in, start_val=ji_s):
                        ji_s = None
                        choices = (kio, kii, iio, iii, ji)
                        parameter_list.append(choices)

    return parameter_list


def einsum5to3_kernel_tlist_generator(params, **kwargs):
    trans_list = []
    kio, kii, iio, iii, ji = params
    trans_list.append(["tag_inames", ["r: unr"]])
    trans_list.append(["tag_inames", ["x: ilp"]])
    trans_list.append(["split_iname", ["e", kio], {"outer_tag": "g.0", "slabs":(0,1)}])
    trans_list.append(["split_iname", ["e_inner", kii], 
        {"outer_tag": "ilp", "inner_tag":"l.0", "slabs":(0,1)}])
    trans_list.append(["split_iname", ["i", iio], {"outer_tag": "g.1", "slabs":(0,0)}])
    trans_list.append(["split_iname", ["i_inner", iii], 
        {"outer_tag": "ilp", "inner_tag":"l.1", "slabs":(0,1)}])
    # Should the i loop have (0,1) slabs for both?

    trans_list.append(["add_prefetch", ["vec", "j,e_inner_outer,e_inner_inner"],
        {"temporary_name":"vecf", "default_tag":"l.auto"}])
    trans_list.append(["tag_array_axes", ["vecf", "f,f"]])
    trans_list.append(["add_prefetch", ["inv_jac_t", "x,r,j,e_inner_outer,e_inner_inner"],
        {"temporary_name":"inv_jac_tf", "default_tag":"l.auto"}])
    trans_list.append(["tag_array_axes", ["inv_jac_tf", "N3,N2,N0,N1"]])

    trans_list.append(["split_iname", ["j", ji], {"outer_tag":"for", "inner_tag":"for"}])
    trans_list.append(["add_inames_for_unused_hw_axes"]) 
    return trans_list 

def einsum5to3_kernel_pspace_generator(queue, knl, start_param=None):

    local_mem_size = queue.device.local_mem_size
    max_work_group_size = queue.device.max_work_group_size    

    for arg in knl.default_entrypoint.args:
        if IsDOFArray() in arg.tags:
            n_elem, n_out = arg.shape
            fp_bytes = arg.dtype.dtype.itemsize
        elif IsFourAxisDOFArray() in arg.tags:
            n_r, n_x, n_elem, n_out = arg.shape
            fp_bytes = arg.dtype.dtype.itemsize
        elif IsVecOpArray() in arg.tags:
            n_r, n_out, n_in = arg.shape

    if start_param is not None:
        kio_s, kii_s, iio_s, iii_s, ji_s = start_param
    else:
        kio_s, kii_s, iio_s, iii_s, ji_s = (None, None, None, None, None)

    # Iterate over five search dimensions
    parameter_list = []
    for kii in k_inner_inner_options(start_val=kii_s):
        # Both inv_jac_t and vec are prefetched so the amount of available local memory per array is reduced
        for kio in k_inner_outer_options(n_in, kii, local_mem_size // (n_r*n_x + 1), fp_bytes=fp_bytes,start_val=kio_s):
            kio_s = None # Set to None so will form the full set the next time around
            for iii in i_inner_inner_options(n_out, kii,
                    max_work_group_size=max_work_group_size, start_val=iii_s):
                iii_s = None
                for iio in i_inner_outer_options(n_out, iii, start_val=iio_s):
                    iio_s = None
                    for ji in j_inner_options(n_in, start_val=ji_s):
                        ji_s = None
                        choices = (kio, kii, iio, iii, ji)
                        parameter_list.append(choices)

    return parameter_list
