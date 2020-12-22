from meshmode.array_context import PyOpenCLArrayContext
from meshmode.dof_array import DOFTag
#from grudge.execution import VecOpDOFTag
from grudge.execution import VecDOFTag, FaceDOFTag
import loopy as lp
#import pyopencl
import pyopencl.array as cla
import grudge.loopy_dg_kernels as dgk
#import numpy as np
try:
    import importlib.resources as pkg_resources
except ImportError:
    # Use backported version for python < 3.7
    import importlib_resources as pkg_resources

ctof_knl = lp.make_copy_kernel("f,f", old_dim_tags="c,c")
ftoc_knl = lp.make_copy_kernel("c,c", old_dim_tags="f,f")


class GrudgeArrayContext(PyOpenCLArrayContext):

    def empty(self, shape, dtype):
        return cla.empty(self.queue, shape=shape, dtype=dtype,
                allocator=self.allocator, order='F')

    def zeros(self, shape, dtype):
        return cla.zeros(self.queue, shape=shape, dtype=dtype,
                allocator=self.allocator, order='F')

    #@memoize_method
    def _get_scalar_func_loopy_program(self, name, nargs, naxes):
        prog = super()._get_scalar_func_loopy_program(name, nargs, naxes)
        for arg in prog.args:
            if type(arg) == lp.ArrayArg:
                arg.tags = DOFTag()
        return prog

    def thaw(self, array):
        thawed = super().thaw(array)
        if type(getattr(array, "tags", None)) == DOFTag:
            cq = thawed.queue
            _, (out,) = ctof_knl(cq, input=thawed)
            thawed = out
            # May or may not be needed
            #thawed.tags = "dof_array"
        return thawed

    #@memoize_method
    def transform_loopy_program(self, program):

        for arg in program.args:
            if isinstance(arg.tags, DOFTag):
                program = lp.tag_array_axes(program, arg.name, "f,f")

            elif isinstance(arg.tags, VecDOFTag):
                program = lp.tag_array_axes(program, arg.name, "sep,f,f")
            #elif isinstance(arg.tags, VecOpDOFTag):
            #    program = lp.tag_array_axes(program, arg.name, "sep,c,c")
            elif isinstance(arg.tags, FaceDOFTag):
                program = lp.tag_array_axes(program, arg.name, "N1,N0,N2")

        if program.name == "opt_diff":
            # TODO: Dynamically determine device id,
            # don't hardcode path to transform.hjson.
            # Also get pn from program
            hjson_file = pkg_resources.open_text(dgk, "transform.hjson")
            deviceID = "NVIDIA Titan V"

            pn = -1
            fp_format = None
            dofs_to_order = {10: 2, 20: 3, 35: 4, 56: 5, 84: 6, 120: 7}
            # Is this a list or a dictionary?
            for arg in program.args:
                if arg.name == "diff_mat":
                    pn = dofs_to_order[arg.shape[2]]
                    fp_format = arg.dtype.numpy_dtype
                    break

            #print(pn)
            #print(fp_format)
            #print(pn<=0)
            #exit()
            #print(type(fp_format) == None)
            #print(type(None) == None)
            # FP format is very specific. Could have integer arrays?
            # What about mixed data types?
            #if pn <= 0 or not isinstance(fp_format, :
                #print("Need to specify a polynomial order and data type")
                # Should throw an error
                #exit()

            transformations = dgk.loadTransformationsFromFile(hjson_file,
                deviceID, pn, fp_format=fp_format)
            program = dgk.applyTransformationList(program, transformations)
        else:
            program = super().transform_loopy_program(program)

        return program
