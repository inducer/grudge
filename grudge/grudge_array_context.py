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

        '''
        if program.name == "elwise_linear":
            #print("ELWISE LINEAR")
            mat = kwargs["mat"]
            result = kwargs["result"]
            vec = kwargs["vec"]

            ctof_knl = lp.make_copy_kernel("f,f", old_dim_tags="c,c")
            ftoc_knl = lp.make_copy_kernel("c,c", old_dim_tags="f,f")

            # Create input array
            cq = vec.queue
            shape = vec.shape
            dtp = vec.dtype
            inArg = pyopencl.array.Array(cq,shape,dtp, order="F")                   

            # Create output array
            shape = result.shape
            dtp = result.dtype
            outArg = pyopencl.array.Array(cq, shape, dtp, order="F")

            #inArg.set(np.asfortranarray(vec))
            ctof_knl(cq, input=vec, output=inArg)
            super().call_loopy(program,mat=mat,result=outArg,vec=inArg)
            #result.set(np.ascontiguousarray(outArt.get()))
            ftoc_knl(cq, input=outArg, output=result)
        '''
        if program.name == "diff" and len(kwargs["diff_mat"]) == 3:
            print(kwargs)
            diff_mat = kwargs["diff_mat"]
            result = kwargs["result"]
            vec = kwargs["vec"]
            print(vec)
            print(type(vec))
            print(vec.strides)
            print(vec.shape)

            ctof_knl = lp.make_copy_kernel("f,f", old_dim_tags="c,c")
            ftoc_knl = lp.make_copy_kernel("c,c", old_dim_tags="f,f")

            # Create input array
            cq = vec.queue
            shape = vec.shape
            dtp = vec.dtype
            #inArg = pyopencl.array.Array(cq,shape,dtp, order="f")                   
            print(vec.strides)
            print(vec.shape)

            _,(inArg,) = ctof_knl(cq, input=vec)

            print(inArg)

            print(inArg.strides)
            print(inArg.shape)

            # Treat as c array
            #inArg.shape = (inArg.shape[1], inArg.shape[0])
            #inArg.strides = pyopencl.array._make_strides(vec.dtype.itemsize, inArg.shape, "c")

            argDict = { "result1": pyopencl.array.Array(cq, inArg.shape, dtp, order="f"),
                        "result2": pyopencl.array.Array(cq, inArg.shape, dtp, order="f"),
                        "result3": pyopencl.array.Array(cq, inArg.shape, dtp, order="f"),
                        "vec": inArg,
                        "mat1": diff_mat[0],
                        "mat2": diff_mat[1],
                        "mat3": diff_mat[2] }
            
            result = super().call_loopy(program, **argDict)

            print(inArg)

            print(inArg.strides)
            print(inArg.shape)

            exit()

        else:
            result = super().call_loopy(program,**kwargs)

        return result

    #@memoize_method
    def transform_loopy_program(self, program):
        #print("TRANSFORMING LOOPY KERNEL")
        if program.name == "diff":
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
