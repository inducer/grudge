from meshmode.array_context import PyOpenCLArrayContext
from pytools import memoize_method, memoize_in
import loopy as lp
import pyopencl as cl
import pyopencl.array as cla
import numpy as np

import grudge.loopy_dg_kernels as dgk
from grudge.grudge_tags import (IsDOFArray, IsSepVecDOFArray, IsFaceDOFArray, 
    IsOpArray, IsSepVecOpArray, ParameterValue, IsFaceMassOpArray, KernelDataTag,
    IsVecDOFArray, IsVecOpArray, IsFourAxisDOFArray)

from arraycontext.impl.pyopencl.fake_numpy import (PyOpenCLFakeNumpyNamespace)
from arraycontext.container.traversal import (rec_map_array_container,
    multimapped_over_array_containers)

from hashlib import md5
import hjson
from grudge.loopy_dg_kernels.run_tests import generic_test, random_search, exhaustive_search, exhaustive_search_v2
from arraycontext.container.traversal import rec_multimap_array_container

#from grudge.loopy_dg_kernels.run_tests import analyzeResult

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
    for arg in program.default_entrypoint.args:
        if IsDOFArray() in arg.tags:
            program = lp.tag_array_axes(program, arg.name, "f,f")
        elif IsSepVecDOFArray() in arg.tags:
            program = lp.tag_array_axes(program, arg.name, "sep,f,f")
        elif IsSepVecOpArray() in arg.tags:
            program = lp.tag_array_axes(program, arg.name, "sep,c,c")
        elif IsFaceDOFArray() in arg.tags:
            program = lp.tag_array_axes(program, arg.name, "N1,N0,N2")
        elif IsVecDOFArray() in arg.tags:
            program = lp.tag_array_axes(program, arg.name, "N2,N0,N1")
        elif IsVecOpArray() in arg.tags or IsFaceMassOpArray() in arg.tags:
            program = lp.tag_array_axes(program, arg.name, "c,c,c")
        elif IsFourAxisDOFArray() in arg.tags:
            program = lp.tag_array_axes(program, arg.name, "N3,N2,N0,N1")
        else:
            for tag in arg.tags:
                if isinstance(tag, ParameterValue):
                    program = lp.fix_parameters(program, **{arg.name: tag.value})

        program = lp.set_options(program, lp.Options(no_numpy=True, return_dict=True))
    return program


# {{{ _get_scalar_func_loopy_program

def _get_scalar_func_loopy_program(actx, c_name, nargs, naxes):
    @memoize_in(actx, _get_scalar_func_loopy_program)
    def get(c_name, nargs, naxes):
        from pymbolic import var

        var_names = ["i%d" % i for i in range(naxes)]
        size_names = ["n%d" % i for i in range(naxes)]
        subscript = tuple(var(vname) for vname in var_names)
        from islpy import make_zero_and_vars
        v = make_zero_and_vars(var_names, params=size_names)
        domain = v[0].domain()
        for vname, sname in zip(var_names, size_names):
            domain = domain & v[0].le_set(v[vname]) & v[vname].lt_set(v[sname])

        domain_bset, = domain.get_basic_sets()

        #import loopy as lp
        #from .loopy import make_loopy_program
        from arraycontext.loopy import make_loopy_program
        from arraycontext.transform_metadata import ElementwiseMapKernelTag
        prg = make_loopy_program(
                [domain_bset],
                [
                    lp.Assignment(
                        var("out")[subscript],
                        var(c_name)(*[
                            var("inp%d" % i)[subscript] for i in range(nargs)]))
                    ],
                name="actx_special_%s" % c_name,
                tags=(ElementwiseMapKernelTag(),))
        for arg in prg.default_entrypoint.args:
            if isinstance(arg, lp.ArrayArg) and len(arg.shape) > 1:
                arg.tags = [IsDOFArray()]

        return prg

    return get(c_name, nargs, naxes)

# }}}


class GrudgeFakeNumpyNamespace(PyOpenCLFakeNumpyNamespace):

    def ravel(self, a, order="C"):
        def _rec_ravel(a):
            if order == "C" and len(a.shape) == 2 and a.flags.f_contiguous:
                @memoize_in(self._array_context, (_rec_ravel, "flatten_grp_ary_prg"))
                def prg():
                    from arraycontext import make_loopy_program
                    t_unit = make_loopy_program(
                        [
                            "{[iel]: 0 <= iel < nelements}",
                            "{[idof]: 0 <= idof < ndofs_per_element}"
                        ],
                        """
                            result[iel * ndofs_per_element + idof] = grp_ary[iel, idof]
                        """,
                        [
                            lp.GlobalArg("result", None,
                                         shape="nelements * ndofs_per_element"),
                            lp.GlobalArg("grp_ary", None,
                                         shape=("nelements", "ndofs_per_element"), tags=[IsDOFArray()]),
                            lp.ValueArg("nelements", np.int32),
                            lp.ValueArg("ndofs_per_element", np.int32),
                            "..."
                        ],
                        name="flatten_grp_ary"
                    )
                    return t_unit
                    #return lp.tag_inames(t_unit, {
                    #    "iel": ConcurrentElementInameTag(),
                    #    "idof": ConcurrentDOFInameTag()})

                result = self._array_context.call_loopy(prg(), grp_ary=a)["result"]
                return result
            elif order in "FC":
                return a.reshape(-1, order=order)
            elif order == "A":
                # TODO: upstream this to pyopencl.array
                if a.flags.f_contiguous:
                    return a.reshape(-1, order="F")
                elif a.flags.c_contiguous:
                    return a.reshape(-1, order="C")
                else:
                    raise ValueError("For `order='A'`, array should be either"
                                     " F-contiguous or C-contiguous.")
            elif order == "K":
                raise NotImplementedError("PyOpenCLArrayContext.np.ravel not "
                                          "implemented for 'order=K'")
            else:
                raise ValueError("`order` can be one of 'F', 'C', 'A' or 'K'. "
                                 f"(got {order})")

        return rec_map_array_container(_rec_ravel, a)


    def stack(self,arrays, axis=0):
        from pytools.obj_array import make_obj_array

        if not axis == 0:
            raise NotImplementedError("Axes other than 0 are not currently supported")

        def _stack(arrays, queue):

            #print(len(arrays))
            #print(arrays[0].shape)
            #print(arrays[0].strides)

            # This sorts the strides from lowest to highest and then
            # uses their original indices to create a list of "N{i}"
            # strings.

            ndims = len(arrays[0].shape)
            lp_strides_ordered = np.array([f"N{i}" for i in range(ndims)])
            lp_strides = np.empty_like(lp_strides_ordered)
            sorted_estrides = np.array(sorted(list(enumerate(arrays[0].strides)), key=lambda tup : tup[1]))
            for i, j in enumerate(sorted_estrides[:,0]):
                lp_strides[j] = lp_strides_ordered[i]

            lp_strides_out = [f"N{ndims}"] + list(lp_strides)
            lp_strides_in = ["sep"] + list(lp_strides)

            # Loopy errors with this, constructing string instead
            #prg = lp.make_copy_kernel(lp_strides_out, old_dim_tags=lp_strides_in)

            # Loopy errors when try to use the lp_strides lists directly
            str_strides_in = ""
            str_strides_out = ""

            for s0, s1 in zip(lp_strides_out, lp_strides_in):
                str_strides_out += s0 + ","
                str_strides_in += s1 + ","
            str_strides_out = str_strides_out[:len(str_strides_out) - 1]
            str_strides_in = str_strides_in[:len(str_strides_in) - 1]
           
            #print(arrays[0].strides) 
            #print(str_strides_in)
            #print(str_strides_out)

            prg = lp.make_copy_kernel(str_strides_out, old_dim_tags=str_strides_in)

            # Fix the kernel parameters
            d = {"n{}".format(i+1): n for i,n in enumerate(arrays[0].shape)}
            d["n0"] = len(arrays)
            prg = lp.fix_parameters(prg,  **d)

            # Should call_loopy be used instead? Probably. No reason no to
            result = prg(queue, input=make_obj_array(arrays))[1][0]
            #print(result.shape)
            return result

        return rec_multimap_array_container(
                 lambda *args: _stack(args, self._array_context.queue),
                 *arrays)

        #return rec_multimap_array_container(
        #         lambda *args: cla.stack(arrays=args, axis=axis,
        #             queue=self._array_context.queue),
        #         *arrays)

    def __getattr__(self, name):
        def loopy_implemented_elwise_func(*args):
            actx = self._array_context
            prg = _get_scalar_func_loopy_program(actx,
                    c_name, nargs=len(args), naxes=len(args[0].shape))
            outputs = actx.call_loopy(prg,
                    **{"inp%d" % i: arg for i, arg in enumerate(args)})
            return outputs["out"]

        if name in self._c_to_numpy_arc_functions:
            from warnings import warn
            warn(f"'{name}' in ArrayContext.np is deprecated. "
                    "Use '{c_to_numpy_arc_functions[name]}' as in numpy. "
                    "The old name will stop working in 2021.",
                    DeprecationWarning, stacklevel=3)

        # normalize to C names anyway
        c_name = self._numpy_to_c_arc_functions.get(name, name)

        # limit which functions we try to hand off to loopy
        if name in self._numpy_math_functions:
            return multimapped_over_array_containers(loopy_implemented_elwise_func)
        else:
            raise AttributeError(name)

# The PyOpenCLArrayContext needs this since the array dimensions are
# Maybe the parameter fixing should be moved into the PyOpenCLArrayContext
class ParameterFixingPyOpenCLArrayContext(PyOpenCLArrayContext):

    @memoize_method
    def transform_loopy_program(self, program):

        # Set no_numpy and return_dict options here?
        for arg in program.default_entrypoint.args:
            for tag in arg.tags:
                if isinstance(tag, ParameterValue):
                    program = lp.fix_parameters(program, **{arg.name: tag.value})

        program = super().transform_loopy_program(program)
        return program


    def call_loopy(self, program, **kwargs):

        result = super().call_loopy(program, **kwargs)

        try: # Only if profiling is enabled
            evt = result["evt"]
            evt.wait()
            dt = evt.profile.end - evt.profile.start
            dt = dt / 1e9

            nbytes = 0
            # Could probably just use program.default_entrypoint.args but maybe all
            # parameters are not set
            if "resample_by_mat" in program.default_entrypoint.name:
                n_to_nodes, n_from_nodes = kwargs["resample_mat"].shape
                nbytes = (kwargs["to_element_indices"].shape[0]*n_to_nodes +
                            n_to_nodes*n_from_nodes +
                            kwargs["from_element_indices"].shape[0]*n_from_nodes) * 8
            elif "resample_by_picking" in program.default_entrypoint.name:
                # Double check this
                if "rhs" not in program.default_entrypoint.name:
                    nbytes = kwargs["pick_list"].shape[0] * (kwargs["from_element_indices"].shape[0]
                            + kwargs["to_element_indices"].shape[0])*8
                else:
                    nbytes = kwargs["pick_list"].shape[0] * (kwargs["from_element_indices"].shape[0])*8
            else:
                # This won't work because not all kernels have dimensions specified
                #for arg in program.default_entrypoint.args:
                #    nbytes += arg.dtype.dtype.itemsize*np.prod(arg.shape)
                for key, val in kwargs.items():
                    # output may be a list of pyopenclarrays or it could be a 
                    # pyopenclarray. This prevents double counting (allowing
                    # other for-loop to count the bytes in the former case)
                    if key not in result.keys(): 
                        try: 
                            nbytes += np.prod(val.shape)*8
                        except AttributeError:
                            nbytes += 0 # Or maybe 1*8 if this is a scalar
                for val in result.values():
                    try:
                        nbytes += np.prod(val.shape)*8
                    except AttributeError:
                        nbytes += 0 # Or maybe this is a scalar?
            bw = nbytes / dt / 1e9

            print("Kernel {}, Time {}, Bytes {}, Bandwidth {}".format(program.default_entrypoint.name, dt, nbytes, bw))

        except cl._cl.RuntimeError as e:
            pass 

        return result


class FortranOrderedArrayContext(ParameterFixingPyOpenCLArrayContext):

    def _get_fake_numpy_namespace(self):
        return GrudgeFakeNumpyNamespace(self)

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

    def einsum(self, spec, *args, arg_names=None, tagged=()):
        """Computes the result of Einstein summation following the
        convention in :func:`numpy.einsum`.

        :arg spec: a string denoting the subscripts for
            summation as a comma-separated list of subscript labels.
            This follows the usual :func:`numpy.einsum` convention.
            Note that the explicit indicator `->` for the precise output
            form is required.
        :arg args: a sequence of array-like operands, whose order matches
            the subscript labels provided by *spec*.
        :arg arg_names: an optional iterable of string types denoting
            the names of the *args*. If *None*, default names will be
            generated.
        :arg tagged: an optional sequence of :class:`pytools.tag.Tag`
            objects specifying the tags to be applied to the operation.

        :return: the output of the einsum :mod:`loopy` program
        """
        if arg_names is None:
            arg_names = tuple("arg%d" % i for i in range(len(args)))

        prg = self._get_einsum_prg(spec, arg_names, tagged)
        
        for tag in tagged:
            if isinstance(tag, KernelDataTag):
                ep = prg.default_entrypoint
                # Is there a better way to apply the kernel data besides making a new tunit object?
                prg = lp.make_kernel(ep.domains, ep.instructions, kernel_data=tag.kernel_data, name=ep.name)

        return self.call_loopy(
            prg, **{arg_names[i]: arg for i, arg in enumerate(args)}
        )["out"]

    def from_numpy(self, np_array: np.ndarray):
        cl_a = super().from_numpy(np_array)
        tags = getattr(np_array, "tags", None)
        if tags is not None and IsDOFArray() in tags:
            # Should this call go through the array context?
            evt, (out,) = ctof_knl(self.queue, input=cl_a)
            cl_a = out
        return cl_a


    def transform_loopy_program(self, program):
        program = lp.set_options(program, lp.Options(no_numpy=True, return_dict=True))
        program = set_memory_layout(program)

        # This should probably be a separate function
        for arg in program.default_entrypoint.args:
            for tag in arg.tags:
                if isinstance(tag, ParameterValue):
                    program = lp.fix_parameters(program, **{arg.name: tag.value})

        # PyOpenCLArrayContext default transformations can't handle fortran ordering
        #program = super().transform_loopy_program(program)
        return program


# This class could be used for some set of default transformations
class GrudgeArrayContext(FortranOrderedArrayContext):

    @memoize_method
    def transform_loopy_program(self, program):
        #print(program.default_entrypoint.name)

        #program = lp.set_options(program, lp.Options(no_numpy=True, return_dict=True))

        
        #device_id = "NVIDIA Titan V"
        #transform_id = get_transformation_id(device_id)

        # Static (non-autotuned) transformations for the GPU
        if "resample_by_picking" in program.default_entrypoint.name:
            for arg in program.default_entrypoint.args:
                print(arg.name, arg.tags)
                if arg.name == "n_to_nodes":
                    # Assumes this has has a single ParameterValue tag
                    n_to_nodes = arg.tags[0].value

            l0 = ((1024 // n_to_nodes) // 32) * 32
            if l0 == 0:
                l0 = 16
            if n_to_nodes*16 > 1024:
                l0 = 8

            outer = max(l0, 32)

            #program = set_memory_layout(program)
            program = lp.split_iname(program, "iel", outer, outer_tag="g.0",
                                        slabs=(0, 1))
            program = lp.split_iname(program, "iel_inner", l0, outer_tag="ilp",
                                        inner_tag="l.0")
            program = lp.split_iname(program, "idof", n_to_nodes, outer_tag="g.1",
                                        inner_tag="l.1", slabs=(0, 0))

        elif "actx_special" in program.default_entrypoint.name: # Fixed
            #program = set_memory_layout(program)
            # Sometimes sqrt is called on single values.
            if "i0" in program.default_entrypoint.inames:
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
        elif "flatten" in program.default_entrypoint.name: 

            #program = set_memory_layout(program)
            # This is hardcoded. Need to move this to separate transformation file
            #program = lp.set_options(program, "write_cl")
            program = lp.split_iname(program, "iel", 128, outer_tag="g.0",
                                        slabs=(0, 1))
            program = lp.split_iname(program, "iel_inner", 32, outer_tag="ilp",
                                        inner_tag="l.0")
            program = lp.split_iname(program, "idof", 20, outer_tag="g.1",
                                        inner_tag="l.1", slabs=(0, 0))

        #else:
            #print(program)
            #print("USING FALLBACK TRANSORMATIONS FOR " + program.default_entrypoint.name)
            #    The PyOpenCLArrayContext transformations can fail when inames are fixed.
        program = super().transform_loopy_program(program)

       
        '''
        if "diff_" in program.default_entrypoint.name: #and "diff_2" not in program.default_entrypoint.name:
            #program = lp.set_options(program, "write_cl")
            # TODO: Dynamically determine device id,
            # Rename this file
            fp_format = None
            print(program)
            for arg in program.default_entrypoint.args:
                if IsOpArray() in arg.tags:
                    dim = 1
                    ndofs = arg.shape[1]
                    fp_format = arg.dtype.numpy_dtype
                    break
                elif IsSepVecOpArray() in arg.tags or IsVecOpArray() in arg.tags:
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

        elif "elwise_linear" in program.default_entrypoint.name:
            hjson_file = pkg_resources.open_text(dgk, "elwise_linear_transform.hjson")
            pn = -1
            fp_format = None
            print(program.default_entrypoint.args)
            for arg in program.default_entrypoint.args:
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

        elif False:#program.default_entrypoint.name == "face_mass":
            hjson_file = pkg_resources.open_text(dgk, "face_mass_transform.hjson")
            pn = -1
            fp_format = None
            for arg in program.default_entrypoint.args:
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
        elif False:#"resample_by_mat" in program.default_entrypoint.name:
            print(program)
            program = lp.set_options(program, "write_cl")

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

        elif "actx_special" in program.default_entrypoint.name:
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
 
        elif program.default_entrypoint.name == "nodes":
            program = lp.split_iname(program, "iel", 64, outer_tag="g.0", slabs=(0,1))
            program = lp.split_iname(program, "iel_inner", 16, outer_tag="ilp", inner_tag="l.0", slabs=(0,1))
            program = lp.split_iname(program, "idof", 20, outer_tag="g.1", slabs=(0,0))
            program = lp.split_iname(program, "idof_inner", 10, outer_tag="ilp", inner_tag="l.1", slabs=(0,0))
                      
        elif "resample_by_picking" in program.default_entrypoint.name:
            program = lp.split_iname(program, "iel", 96, outer_tag="g.0",
                                        slabs=(0, 1))
            program = lp.split_iname(program, "iel_inner", 96, outer_tag="ilp",
                                        inner_tag="l.0")
            program = lp.split_iname(program, "idof", 10, outer_tag="g.1",
                                        inner_tag="l.1", slabs=(0, 0))
        elif "grudge_assign" in program.default_entrypoint.name or \
             "flatten" in program.default_entrypoint.name:
            # This is hardcoded. Need to move this to separate transformation file
            #program = lp.set_options(program, "write_cl")
            #program = lp.split_iname(program, "iel", 1024, outer_tag="g.0",
            #                            slabs=(0, 1))
            program = lp.split_iname(program, "iel", 128, outer_tag="g.0",
                                        slabs=(0, 1))
            program = lp.split_iname(program, "iel_inner", 32, outer_tag="ilp",
                                        inner_tag="l.0")
            program = lp.split_iname(program, "idof", 32, outer_tag="g.1",
                                        inner_tag="l.1", slabs=(0, 1))
        # No transformations for now
        elif "einsum" in program.default_entrypoint.name:
            pass
        else:
            print("USING FALLBACK TRANSORMATIONS FOR " + program.default_entrypoint.name)
            program = super().transform_loopy_program(program)
        '''

        return program


# Could be problematic if code generation is not deterministic.
def unique_program_id(program):
    code = lp.generate_code_v2(program).device_code()
    h = md5(code.encode())
    identifier = h.hexdigest()
    return identifier


def convert(o):
    if isinstance(o, np.generic): return o.item()
    raise TypeError

class AutotuningArrayContext(GrudgeArrayContext):

    @memoize_method
    def transform_loopy_program(self, program):

        # Really just need to add metadata to the hjson file
        # Could convert the kernel itself to base 64 and store it
        # in the hjson file
        # TODO: Dynamically determine device id,
        device_id = "NVIDIA Titan V"

        print(program.default_entrypoint.name)
        print(program)

        # These are the most compute intensive kernels
        to_optimize = {}
        if program.default_entrypoint.name in to_optimize:
            print(program)
            for arg in program.default_entrypoint.args:
                print(arg.tags)
            exit()

        # Meshmode and Grudge kernels to autotune
        autotuned_kernels = {"einsum3to2_kernel",
                             "einsum4to2_kernel", 
                             "einsum5to3_kernel", 
                             "einsum2to2_kernel",
                             "diff", 
                             "lp_nodes",
                             "grudge_elementwise_sum_knl"}

        if program.default_entrypoint.name in autotuned_kernels:
            # Set no_numpy and return_dict options here?
            program = lp.set_options(program, lp.Options(no_numpy=True, return_dict=True))
            program = set_memory_layout(program)
            pid = unique_program_id(program)
            print(pid)
            hjson_file_str = f"hjson/{program.default_entrypoint.name}_{pid}.hjson"

            try:
                # Attempt to read from a transformation file in the current directory first,
                # then try to read from the package files - this is not currently implemented
                # Maybe should have ability to search in arbitrary specified directories.

                hjson_file = open(hjson_file_str, "rt")

                try: # New hjson structure
                    transformations = dgk.load_transformations_from_file(hjson_file,
                        ["transformations"])
                except KeyError as e:
                    # This can eventually be removed since we're now using the hash of the program code to specify the file.
                    # Kernels with different dimensions will have different files.
                    hjson_file.seek(0,0) # Move read location back to beginning

                    fp_format = None
                    ndofs = None # The value doesn't matter now
                    transform_id = get_transformation_id(device_id)

                    for arg in program.default_entrypoint.args:
                        if IsOpArray() in arg.tags:
                            dim = 1
                            ndofs = arg.shape[0]
                            fp_format = arg.dtype.numpy_dtype
                            break
                        elif IsSepVecOpArray() in arg.tags or IsVecOpArray() in arg.tags:
                            ndofs = arg.shape[1]
                            fp_format = arg.dtype.numpy_dtype
                            break
                        elif IsFaceMassOpArray() in arg.tags:
                            ndofs = arg.shape[0]
                            fp_format = arg.dtype.numpy_dtype
                            break
                        elif IsDOFArray() in arg.tags:
                            ndofs = arg.shape[1]
                            fp_format = arg.dtype.numpy_dtype
                            break 

                    if fp_format is None:
                        print("Unknown fp_format")
                        exit()                
                    if ndofs is None:
                        print("Unknown ndofs")
                        exit()

                    fp_string = get_fp_string(fp_format)
                    indices = [transform_id, fp_string, str(ndofs)]
                    transformations = dgk.load_transformations_from_file(hjson_file,
                        indices)

                hjson_file.close()

            #except (KeyError, FileNotFoundError) as e:
            # There shouldn't be any more key errors now that PIDs are used
            except FileNotFoundError as e:
                
                search_fn = exhaustive_search_v2#random_search

                # Maybe the generators should be classes so we can use inheritance.
                if program.default_entrypoint.name == "einsum3to2_kernel":
                    from grudge.loopy_dg_kernels.generators import einsum3to2_kernel_tlist_generator as tlist_generator
                    from grudge.loopy_dg_kernels.generators import einsum3to2_kernel_pspace_generator as pspace_generator
                elif program.default_entrypoint.name == "einsum4to2_kernel":
                    from grudge.loopy_dg_kernels.generators import einsum4to2_kernel_tlist_generator as tlist_generator
                    from grudge.loopy_dg_kernels.generators import einsum4to2_kernel_pspace_generator as pspace_generator
                elif program.default_entrypoint.name == "einsum5to3_kernel":
                    from grudge.loopy_dg_kernels.generators import einsum5to3_kernel_tlist_generator as tlist_generator
                    from grudge.loopy_dg_kernels.generators import einsum5to3_kernel_pspace_generator as pspace_generator
                elif program.default_entrypoint.name == "einsum2to2_kernel":
                    from grudge.loopy_dg_kernels.generators import einsum2to2_kernel_tlist_generator as tlist_generator
                    from grudge.loopy_dg_kernels.generators import einsum2to2_kernel_pspace_generator as pspace_generator
                elif program.default_entrypoint.name == "grudge_elementwise_sum_knl":
                    from grudge.loopy_dg_kernels.generators import grudge_elementwise_sum_knl_tlist_generator as tlist_generator
                    from grudge.loopy_dg_kernels.generators import grudge_elementwise_sum_knl_pspace_generator as pspace_generator
                else:
                    from grudge.loopy_dg_kernels.generators import gen_autotune_list as pspace_generator
                    from grudge.loopy_dg_kernels.generators import mxm_trans_list_generator as tlist_generator

                try:
                    avg_time, transformations, data = search_fn(self.queue, program, generic_test, 
                                                pspace_generator, tlist_generator, time_limit=np.inf)
                except cl._cl.RuntimeError as e:
                    print(e)
                    print("Profiling is not enabled and the PID does not match any transformation file. Turn on profiling and run again.")

                #transformations = search_fn(self.queue, program, generic_test, time_limit=60)
                #transformations = dgk.generate_transformation_list(*transformations) 
                
                # Write the new transformations back to local file
                #if isinstance(e, KeyError):
                #    hjson_file = open(hjson_file_str, "rt")
                #    # Need to figure out how to copy existing transformations 
                #    od = hjson.load(hjson_file)
                #    hjson_file.close()
                #    od[transform_id][fp_string][ndofs] = transformations
                #else:
                #    od = {transform_id: {fp_string: {str(ndofs): transformations} } }
                od = {"transformations": transformations}
                
                out_file = open(hjson_file_str, "wt+")

                hjson.dump(od, out_file,default=convert)
                out_file.close()
                #from pprint import pprint
                #pprint(od)

            program = dgk.apply_transformation_list(program, transformations)

            """
            # Kernels to not autotune. Should probably still load the transformation from a
            # generator function. Should these be put in GrudgeArrayContext

            # Maybe this should have an autotuner
            # There isn't much room for optimization due to the indirection
            elif "resample_by_picking" in program.default_entrypoint.name:
                for arg in program.default_entrypoint.args:
                    if arg.name == "n_to_nodes":
                        # Assumes this has has a single ParameterValue tag
                        n_to_nodes = arg.tags[0].value

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

            elif "actx_special" in program.default_entrypoint.name: # Fixed
                program = set_memory_layout(program)
                # Sometimes sqrt is called on single values.
                if "i0" in program.default_entrypoint.inames:
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
            elif "flatten" in program.default_entrypoint.name: 

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
                #print(program)
                #print("USING FALLBACK TRANSORMATIONS FOR " + program.default_entrypoint.name)
                #    The PyOpenCLArrayContext transformations can fail when inames are fixed.
               program = super().transform_loopy_program(program)

            '''       
            # These still depend on the polynomial order = 3
            # Never called?
            # This is going away anyway probably
            elif "resample_by_mat" in program.default_entrypoint.name:
                hjson_file = pkg_resources.open_text(dgk, f"{program.default_entrypoint.name}.hjson")
        
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
            elif "grudge_assign" in program.default_entrypoint.name or "flatten" in program.default_entrypoint.name: 
                # This is hardcoded. Need to move this to separate transformation file
                #program = lp.set_options(program, "write_cl")
                program = lp.split_iname(program, "iel", 128, outer_tag="g.0",
                                            slabs=(0, 1))
                program = lp.split_iname(program, "iel_inner", 32, outer_tag="ilp",
                                            inner_tag="l.0")
                program = lp.split_iname(program, "idof", 20, outer_tag="g.1",
                                            inner_tag="l.1", slabs=(0, 0))
            

            '''
            """
        else:
            # print("USING FALLBACK TRANSFORMATIONS FOR " + program.default_entrypoint.name)
            program = super().transform_loopy_program(program)

        return program
   
# vim: foldmethod=marker
