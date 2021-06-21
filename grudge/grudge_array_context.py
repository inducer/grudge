from meshmode.array_context import PyOpenCLArrayContext
from pytools import memoize_method
import loopy as lp
import pyopencl.array as cla
import grudge.loopy_dg_kernels as dgk
from grudge.grudge_tags import (IsDOFArray, IsVecDOFArray, IsFaceDOFArray, 
    IsOpArray, IsVecOpArray, ParameterValue, IsFaceMassOpArray)
from numpy import prod
import hjson
import numpy as np

#from grudge.loopy_dg_kernels.run_tests import analyzeResult
import pyopencl as cl

try:
    import importlib.resources as pkg_resources
except ImportError:
    # Use backported version for python < 3.7
    import importlib_resources as pkg_resources

ctof_knl = lp.make_copy_kernel("f,f", old_dim_tags="c,c")
#ftoc_knl = lp.make_copy_kernel("c,c", old_dim_tags="f,f")

def get_transformation_id(device_id):
    hjson_file = pkg_resources.open_text(dgk, "device_mappings.hjson") 
    hjson_text = hjson_file.read()
    hjson_file.close()
    od = hjson.loads(hjson_text)
    return od[device_id]

def get_fp_string(dtype):
    return "FP64" if dtype == np.float64 else "FP32"

def get_order_from_dofs(dofs):
    dofs_to_order = {10: 2, 20: 3, 35: 4, 56: 5, 84: 6, 120: 7}
    return dofs_to_order[dofs]

def set_memory_layout(program):
    # This assumes arguments have only one tag
    for arg in program.args:
        if isinstance(arg.tags, IsDOFArray):
            program = lp.tag_array_axes(program, arg.name, "f,f")
        elif isinstance(arg.tags, IsVecDOFArray):
            program = lp.tag_array_axes(program, arg.name, "sep,f,f")
        elif isinstance(arg.tags, IsVecOpArray):
            program = lp.tag_array_axes(program, arg.name, "sep,c,c")
        elif isinstance(arg.tags, IsFaceDOFArray):
            program = lp.tag_array_axes(program, arg.name, "N1,N0,N2")
        elif isinstance(arg.tags, ParameterValue):
            program = lp.fix_parameters(program, **{arg.name: arg.tags.value})

        program = lp.set_options(program, lp.Options(no_numpy=True, return_dict=True))
    return program


class GrudgeArrayContext(PyOpenCLArrayContext):

    def empty(self, shape, dtype):
        return cla.empty(self.queue, shape=shape, dtype=dtype,
                allocator=self.allocator, order="F")

    def zeros(self, shape, dtype):
        return cla.zeros(self.queue, shape=shape, dtype=dtype,
                allocator=self.allocator, order="F")

    def thaw(self, array):
        thawed = super().thaw(array)
        if type(getattr(array, "tags", None)) == IsDOFArray:
            cq = thawed.queue
            # Should this be run through the array context
            #evt, out = self.call_loopy(ctof_knl, **{input: thawed})
            _, (out,) = ctof_knl(cq, input=thawed)
            thawed = out
        return thawed


    def from_numpy(self, np_array: np.ndarray):
        cl_a = super().from_numpy(np_array)
        tags = getattr(np_array, "tags", None)
        if tags is not None and IsDOFArray() in tags:
            # Should this call go through the array context?
            evt, (out,) = ctof_knl(self.queue, input=cl_a)
            cl_a = out
        return cl_a

    @memoize_method
    def transform_loopy_program(self, program):
        print(program.name)

        # Set no_numpy and return_dict options here?
        program = set_memory_layout(program)
        
        device_id = "NVIDIA Titan V"
        # This read could be slow
        transform_id = get_transformation_id(device_id)

        if "diff" in program.name: #and "diff_2" not in program.name:
            #program = lp.set_options(program, "write_cl")
            # TODO: Dynamically determine device id,
            # Rename this file
            fp_format = None
            print(program)
            for arg in program.args:
                if isinstance(arg.tags, IsOpArray):
                    dim = 1
                    ndofs = arg.shape[1]
                    fp_format = arg.dtype.numpy_dtype
                    break
                elif isinstance(arg.tags, IsVecOpArray):
                    dim = arg.shape[0]
                    ndofs = arg.shape[2]
                    fp_format = arg.dtype.numpy_dtype
                    break

            # FP format is very specific. Could have integer arrays?
            # What about mixed data types?
            fp_string = get_fp_string(fp_format)

            # Attempt to read from a transformation file in the current directory first,
            # then try to read from the package files
            #try:
            #hjson_file = open("test_write.hjson", "rt")
            #except FileNotFoundError:
            hjson_file = pkg_resources.open_text(dgk, "diff_{}d_transform.hjson".format(dim))

            # Probably need to generalize this
            indices = [transform_id, fp_string, str(ndofs)]
            transformations = dgk.load_transformations_from_file(hjson_file,
                indices)
            hjson_file.close()
            program = dgk.apply_transformation_list(program, transformations)

            # Print the Code
            """
            platform = cl.get_platforms()
            my_gpu_devices = platform[1].get_devices(device_type=cl.device_type.GPU)
            #ctx = cl.create_some_context(interactive=True)
            ctx = cl.Context(devices=my_gpu_devices)
            kern = program.copy(target=lp.PyOpenCLTarget(my_gpu_devices[0]))
            code = lp.generate_code_v2(kern).device_code()
            prog = cl.Program(ctx, code)
            prog = prog.build()
            ptx = prog.get_info(cl.program_info.BINARIES)[0]#.decode(
            #errors="ignore") #Breaks pocl
            from bs4 import UnicodeDammit
            dammit = UnicodeDammit(ptx)
            print(dammit.unicode_markup)
            print(program.options)
            exit()
            """

        elif "elwise_linear" in program.name:
            hjson_file = pkg_resources.open_text(dgk, "elwise_linear_transform.hjson")
            pn = -1
            fp_format = None
            print(program.args)
            for arg in program.args:
                if arg.name == "mat":
                    dofs = arg.shape[0]
                    pn = get_order_from_dofs(arg.shape[0])                    
                    fp_format = arg.dtype.numpy_dtype
                    break

            fp_string = get_fp_string(fp_format)
            indices = [transform_id, fp_string, str(dofs)]
            transformations = dgk.load_transformations_from_file(hjson_file,
                indices)
            hjson_file.close()
            print(transformations)
            program = dgk.apply_transformation_list(program, transformations)

        elif program.name == "face_mass":
            hjson_file = pkg_resources.open_text(dgk, "face_mass_transform.hjson")
            pn = -1
            fp_format = None
            for arg in program.args:
                if arg.name == "mat":
                    pn = get_order_from_dofs(arg.shape[0])                    
                    fp_format = arg.dtype.numpy_dtype
                    break

            fp_string = get_fp_string(fp_format)
            indices = [transform_id, fp_string, str(pn)]
            transformations = dgk.load_transformations_from_file(hjson_file,
                indices)
            hjson_file.close()
            program = dgk.apply_transformation_list(program, transformations)

        # These still depend on the polynomial order = 3
        elif program.name == "resample_by_mat":
            hjson_file = pkg_resources.open_text(dgk, "resample_by_mat.hjson")
    
            # Order 3: 10 x 10
            # Order 4: 15 x 35
            
            #print(program)
            #exit()
            pn = 3 # This needs to  be not fixed
            fp_string = "FP64"
            
            indices = [transform_id, fp_string, str(pn)]
            transformations = dgk.load_transformations_from_file(hjson_file,
                indices)
            hjson_file.close()
            print(transformations)
            program = dgk.apply_transformation_list(program, transformations)

        elif "actx_special" in program.name:
            program = lp.split_iname(program, "i0", 512, outer_tag="g.0",
                                        inner_tag="l.0", slabs=(0, 1))
            #program = lp.split_iname(program, "i0", 128, outer_tag="g.0",
            #                           slabs=(0,1))
            #program = lp.split_iname(program, "i0_inner", 32, outer_tag="ilp",
            #                           inner_tag="l.0")
            #program = lp.split_iname(program, "i1", 20, outer_tag="g.1",
            #                           inner_tag="l.1", slabs=(0,0))
            #program2 = lp.join_inames(program, ("i1", "i0"), "i")
            #from islpy import BasicMap
            #m = BasicMap("[x,y] -> {[n0,n1]->[i]:}")
            #program2 = lp.map_domain(program, m)
            #print(program2)
            #exit()

            #program = super().transform_loopy_program(program)
            #print(program)
            #print(lp.generate_code_v2(program).device_code())
 
        elif program.name == "nodes":
            program = lp.split_iname(program, "iel", 64, outer_tag="g.0", slabs=(0,1))
            program = lp.split_iname(program, "iel_inner", 16, outer_tag="ilp", inner_tag="l.0", slabs=(0,1))
            program = lp.split_iname(program, "idof", 20, outer_tag="g.1", slabs=(0,0))
            program = lp.split_iname(program, "idof_inner", 10, outer_tag="ilp", inner_tag="l.1", slabs=(0,0))
                      
        elif program.name == "resample_by_picking":
            program = lp.split_iname(program, "iel", 96, outer_tag="g.0",
                                        slabs=(0, 1))
            program = lp.split_iname(program, "iel_inner", 96, outer_tag="ilp",
                                        inner_tag="l.0")
            program = lp.split_iname(program, "idof", 10, outer_tag="g.1",
                                        inner_tag="l.1", slabs=(0, 0))
        elif "grudge_assign" in program.name or \
             "flatten" in program.name:
            # This is hardcoded. Need to move this to separate transformation file
            #program = lp.set_options(program, "write_cl")
            program = lp.split_iname(program, "iel", 128, outer_tag="g.0",
                                        slabs=(0, 1))
            program = lp.split_iname(program, "iel_inner", 32, outer_tag="ilp",
                                        inner_tag="l.0")
            program = lp.split_iname(program, "idof", 20, outer_tag="g.1",
                                        inner_tag="l.1", slabs=(0, 0))
        else:
            print("USING FALLBACK TRANSORMATIONS FOR " + program.name)
            program = super().transform_loopy_program(program)

        return program

    def call_loopy(self, program, **kwargs):

        evt, result = super().call_loopy(program, **kwargs)
        evt.wait()
        dt = evt.profile.end - evt.profile.start
        dt = dt / 1e9

        nbytes = 0
        # Could probably just use program.args but maybe all
        # parameters are not set

        #print("Input")

        if program.name == "resample_by_mat":
            n_to_nodes, n_from_nodes = kwargs["resample_mat"].shape
            nbytes = (kwargs["to_element_indices"].shape[0]*n_to_nodes +
                        n_to_nodes*n_from_nodes +
                        kwargs["from_element_indices"].shape[0]*n_from_nodes) * 8
        elif program.name == "resample_by_picking":
            # Double check this
            nbytes = kwargs["pick_list"].shape[0] * (kwargs["from_element_indices"].shape[0]
                        + kwargs["to_element_indices"].shape[0])*8
        else:
            print(program.name)
            #print(kwargs.keys())
            for key, val in kwargs.items():
                # output may be a list of pyopenclarrays or it could be a 
                # pyopenclarray. This prevents double counting (allowing
                # other for-loop to count the bytes in the former case)
                if key not in result.keys(): 
                    try: 
                        nbytes += prod(val.shape)*8
                        #print(val.shape)
                    except AttributeError:
                        nbytes += 0 # Or maybe 1*8 if this is a scalar
                #print(nbytes)
            #print("Output")
            #print(result.keys())
            for val in result.values():
                try:
                    nbytes += prod(val.shape)*8
                    #print(val.shape)
                except AttributeError:
                    nbytes += 0 # Or maybe this is a scalar?

        bw = nbytes / dt / 1e9

        print("Kernel {}, Time {}, Bytes {}, Bandwidth {}".format(program.name, dt, nbytes, bw))
       
        return evt, result

def convert(o):
    if isinstance(o, np.generic): return o.item()
    raise TypeError

class AutoTuningArrayContext(GrudgeArrayContext):

    @memoize_method
    def transform_loopy_program(self, program):
        print(program.name)

        device_id = "NVIDIA Titan V"
        # This read could be slow
        transform_id = get_transformation_id(device_id)

        if "diff" in program.name or \
           "elwise_linear" == program.name or \
           "face_mass" == program.name or \
           "nodes" == program.name:

            # Set no_numpy and return_dict options here?
            program = set_memory_layout(program)
 

            #program = lp.set_options(program, "write_cl")
            # TODO: Dynamically determine device id,
            # Rename this file
            fp_format = None
            #print(program)
            for arg in program.args:
                if isinstance(arg.tags, IsOpArray):
                    dim = 1
                    ndofs = arg.shape[0]
                    fp_format = arg.dtype.numpy_dtype
                    break
                elif isinstance(arg.tags, IsVecOpArray):
                    ndofs = arg.shape[1]
                    fp_format = arg.dtype.numpy_dtype
                    break
                elif isinstance(arg.tags, IsFaceMassOpArray):
                    ndofs = arg.shape[0]
                    fp_format = arg.dtype.numpy_dtype
                    break


            # FP format is very specific. Could have integer arrays?
            # What about mixed data types?
            fp_string = get_fp_string(fp_format)

            # TODO: Should search in current directory for transform file
            try:
                # Attempt to read from a transformation file in the current directory first,
                # then try to read from the package files
                #try:
                hjson_file = open(f"{program.name}.hjson", "rt")
                #except FileNotFoundError:

                # Probably need to generalize this
                indices = [transform_id, fp_string, str(ndofs)]
                transformations = dgk.load_transformations_from_file(hjson_file,
                    indices)
                hjson_file.close()
                program = dgk.apply_transformation_list(program, transformations)

            # File exists but a transformation is missing
            # How can the test times be excluded from the end-to-end time reports
            except KeyError:
                # There are other array sizes too. Need to handle those
                # What if already have a set of transformation parameters but want to
                # refine them more
                # If there is a key error, the autotuner should be called to figure
                # out a set of transformations and then write it back to the hjson file
                print("ARRAY SIZE NOT IN TRANSFORMATION FILE")

                from grudge.loopy_dg_kernels.run_tests import generic_test, random_search, exhaustive_search


                transformations = exhaustive_search(self.queue, program, generic_test, time_limit=np.inf)
                #parameters = random_search(self.queue, program, test_diff, time_limit=10)
                #transformations = dgk.generate_transformation_list(*parameters)
                program = dgk.apply_transformation_list(program, transformations)
                
                # How to save to file? It may need a new file or there may be an
                # existing file to add it into

                # Write the new transformations back to local file
                hjson_file = open(f"{program.name}.hjson", "rt")
                # Need to figure out how to copy existing transformations 
                import hjson
                od = hjson.load(hjson_file)
                hjson_file.close()
                od[transform_id][fp_string][ndofs] = transformations
                out_file = open(f"{program.name}.hjson", "wt")
                #from pprint import pprint
                #pprint(od)
                
                hjson.dump(od, out_file,default=convert)
                out_file.close()
                 
                #program = super().transform_loopy_program(program)
            # No transformation files exist
            except FileNotFoundError:
                from grudge.loopy_dg_kernels.run_tests import generic_test, random_search, exhaustive_search

                transformations = exhaustive_search(self.queue, program, generic_test, time_limit=np.inf)
                #parameters = random_search(self.queue, program, generic_test, time_limit=30)
                #transformations = dgk.generate_transformation_list(*parameters)
                #print(transformations)
                program = dgk.apply_transformation_list(program, transformations)
                
                # Write the new transformations to a file
                import hjson

                # Will need a new transform_id
                d = {transform_id: {fp_string: {str(ndofs): transformations} } }
                out_file = open(f"{program.name}.hjson", "wt")
                hjson.dump(d, out_file,default=convert)
                out_file.close()

        # Maybe this should have an autotuner
        elif program.name == "resample_by_picking":
            for arg in program.args:
                if arg.name == "n_to_nodes":
                    n_to_nodes = arg.tags.value

            l0 = ((1024 // n_to_nodes) // 32) * 32
            if l0 == 0:
                l0 = 16
            if n_to_nodes*16 > 1024:
                l0 = 8

            outer = max(l0, 32)

            program = set_memory_layout(program)
            program = lp.split_iname(program, "iel", outer, outer_tag="g.0",
                                        slabs=(0, 1))
            program = lp.split_iname(program, "iel_inner", l0, outer_tag="ilp",
                                        inner_tag="l.0")
            program = lp.split_iname(program, "idof", n_to_nodes, outer_tag="g.1",
                                        inner_tag="l.1", slabs=(0, 0))

        elif "actx_special" in program.name: # Fixed
            # Need to add autotuner support for this
            program = set_memory_layout(program)
            program = lp.split_iname(program, "i0", 512, outer_tag="g.0",
                                        inner_tag="l.0", slabs=(0, 1))
            #program = lp.split_iname(program, "i0", 128, outer_tag="g.0",
            #                           slabs=(0,1))
            #program = lp.split_iname(program, "i0_inner", 32, outer_tag="ilp",
            #                           inner_tag="l.0")
            #program = lp.split_iname(program, "i1", 20, outer_tag="g.1",
            #                           inner_tag="l.1", slabs=(0,0))
            #program2 = lp.join_inames(program, ("i1", "i0"), "i")
            #from islpy import BasicMap
            #m = BasicMap("[x,y] -> {[n0,n1]->[i]:}")
            #program2 = lp.map_domain(program, m)
            #print(program2)
            #exit()

            #program = super().transform_loopy_program(program)
            #print(program)
            #print(lp.generate_code_v2(program).device_code())

        # Not really certain how to do grudge_assign, done for flatten
        elif "flatten" in program.name: 

            program = set_memory_layout(program)
            # This is hardcoded. Need to move this to separate transformation file
            #program = lp.set_options(program, "write_cl")
            program = lp.split_iname(program, "iel", 128, outer_tag="g.0",
                                        slabs=(0, 1))
            program = lp.split_iname(program, "iel_inner", 32, outer_tag="ilp",
                                        inner_tag="l.0")
            program = lp.split_iname(program, "idof", 20, outer_tag="g.1",
                                        inner_tag="l.1", slabs=(0, 0))

        else:
            print(program)
            print("USING FALLBACK TRANSORMATIONS FOR " + program.name)
            program = super().transform_loopy_program(program)

        '''       
        # These still depend on the polynomial order = 3
        # Never called?
        # This is going away anyway probably
        elif program.name == "resample_by_mat":
            hjson_file = pkg_resources.open_text(dgk, "resample_by_mat.hjson")
    
            # Order 3: 10 x 10
            # Order 4: 15 x 35
            
            #print(program)
            #exit()
            pn = 3 # This needs to  be not fixed
            fp_string = "FP64"
            
            indices = [transform_id, fp_string, str(pn)]
            transformations = dgk.load_transformations_from_file(hjson_file,
                indices)
            hjson_file.close()
            print(transformations)
            program = dgk.apply_transformation_list(program, transformations)

        # Not really certain how to do grudge_assign, done for flatten
        elif "grudge_assign" in program.name or "flatten" in program.name: 
            # This is hardcoded. Need to move this to separate transformation file
            #program = lp.set_options(program, "write_cl")
            program = lp.split_iname(program, "iel", 128, outer_tag="g.0",
                                        slabs=(0, 1))
            program = lp.split_iname(program, "iel_inner", 32, outer_tag="ilp",
                                        inner_tag="l.0")
            program = lp.split_iname(program, "idof", 20, outer_tag="g.1",
                                        inner_tag="l.1", slabs=(0, 0))
        
        else:
            print("USING FALLBACK TRANSFORMATIONS FOR " + program.name)
            program = super().transform_loopy_program(program)
        '''

        return program
   

# vim: foldmethod=marker
