from meshmode.array_context import PyOpenCLArrayContext
import loopy as lp
import pyopencl
import pyopencl.array
import loopy_dg_kernels as dgk
#import numpy as np

class GrudgeArrayContext(PyOpenCLArrayContext):

    def __init__(self, queue, allocator=None):
        super().__init__(queue, allocator=allocator)

    def call_loopy(self, program, **kwargs):

        if program.name == "opt_diff":
            diff_mat = kwargs["diff_mat"]
            result = kwargs["result"]
            vec = kwargs["vec"]

            ctof_knl = lp.make_copy_kernel("f,f", old_dim_tags="c,c")
            ftoc_knl = lp.make_copy_kernel("c,c", old_dim_tags="f,f")

            # Create input array
            cq = vec.queue
            shape = vec.shape
            dtp = vec.dtype
            _,(inArg,) = ctof_knl(cq, input=vec)
            
            # Treat as c array, can do this to use c-format diff function
            #inArg.shape = (inArg.shape[1], inArg.shape[0])
            #inArg.strides = pyopencl.array._make_strides(vec.dtype.itemsize, inArg.shape, "c")

            argDict = { "result1": pyopencl.array.Array(cq, inArg.shape, dtp, order="f"),
                        "result2": pyopencl.array.Array(cq, inArg.shape, dtp, order="f"),
                        "result3": pyopencl.array.Array(cq, inArg.shape, dtp, order="f"),
                        "vec": inArg,
                        "mat1": diff_mat[0],
                        "mat2": diff_mat[1],
                        "mat3": diff_mat[2] }
            
            super().call_loopy(program, **argDict)

            result = kwargs["result"]

            # Change order of result back to c ordering
            for i, entry in enumerate(["result1", "result2", "result3"]):
                # Treat as fortran array
                #argDict[entry].shape = (argDict[entry].shape[1], argDict[entry].shape[0])
                #argDict[entry].strides = pyopencl.array._make_strides(argDict[entry].dtype.itemsize, argDict[entry].shape, "f")
                ftoc_knl(cq, input=argDict[entry], output=result[i])

            return result            

        else:
            result = super().call_loopy(program,**kwargs)

        return result

    #@memoize_method
    def transform_loopy_program(self, program):
        if program.name == "opt_diff":
            filename = "/home/njchris2/Workspace/nick/loopy_dg_kernels/transform.hjson"
            deviceID = "NVIDIA Titan V"
            pn = 3
            transformations = dgk.loadTransformationsFromFile(filename, deviceID, pn)            
            program = dgk.applyTransformationList(program, transformations)
        else:
            program = super().transform_loopy_program(program)

        '''
        # Broken currently
        ctof_knl = lp.make_copy_kernel("f,f", old_dim_tags="c,c")
        #ctof_knl = lp.rename_argument(ctof_knl, "input", "input_ctof")
        ctof_knl = lp.rename_argument(ctof_knl, "output", "vec")
        ctof_knl = lp.rename_iname(ctof_knl, "i0", "i0_ctof")
        ctof_knl = lp.rename_iname(ctof_knl, "i1", "i1_ctof")

        ftoc_knl = lp.make_copy_kernel("c,c", old_dim_tags="f,f")
        ftoc_knl = lp.rename_argument(ftoc_knl, "input", "result")
        ftoc_knl = lp.rename_argument(ftoc_knl, "n0", "nelements", existing_ok=True)
        ftoc_knl = lp.rename_argument(ftoc_knl, "n1", "ndiscr_nodes", existing_ok=True)
        #ftoc_knl = lp.rename_argument(ftoc_knl, "output", "output_ftoc")
        ftoc_knl = lp.rename_iname(ftoc_knl, "i0", "i0_ftoc")
        ftoc_knl = lp.rename_iname(ftoc_knl, "i1", "i1_ftoc")

        #dependencies = [("vec", ctof_knl, program), ("result", program, ctof_knl)] 
        dependencies =  [("vec", ctof_knl, program)]
        program = lp.fuse_kernels([ctof_knl, program], data_flow=dependencies)
        #program = lp.fuse_kernels([ctof_knl, program, ftoc_knl], data_flow=dependencies)
        '''

        #print(program)

        return program
