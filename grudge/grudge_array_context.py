from meshmode.array_context import PyOpenCLArrayContext
import loopy as lp
import pyopencl
import pyopencl.array
import loopy_dg_kernels as dgk
import numpy as np

ctof_knl = lp.make_copy_kernel("f,f", old_dim_tags="c,c")
ftoc_knl = lp.make_copy_kernel("c,c", old_dim_tags="f,f")

class GrudgeArrayContext(PyOpenCLArrayContext):

    def __init__(self, queue, allocator=None):
        super().__init__(queue, allocator=allocator)

    def from_numpy(self, np_array: np.ndarray):
        # Should intercept this for the dof array
        # and make it return a fortran style array
        return super().from_numpy(np_array)
        

    def call_loopy(self, program, **kwargs):

        if program.name == "opt_diff":
            diff_mat = kwargs["diff_mat"]
            result = kwargs["result"]
            vec = kwargs["vec"]

            # Create input array
            cq = vec.queue
            dtp = vec.dtype

            # Esto no deberia hacerse aqui.
            _,(inArg,) = ctof_knl(cq, input=vec)
            
            # Treat as c array, can do this to use c-format diff function
            # np.array(A, format="F").flatten() == np.array(A.T, format="C").flatten()
            inArg.shape = (inArg.shape[1], inArg.shape[0])
            inArg.strides = pyopencl.array._make_strides(vec.dtype.itemsize, inArg.shape, "c")
            outShape = inArg.shape #(inArg.shape[1], inArg.shape[0])

            argDict = { "result1": pyopencl.array.Array(cq, outShape, dtp, order="c"),
                        "result2": pyopencl.array.Array(cq, outShape, dtp, order="c"),
                        "result3": pyopencl.array.Array(cq, outShape, dtp, order="c"),
                        "vec": inArg,
                        "mat1": diff_mat[0],
                        "mat2": diff_mat[1],
                        "mat3": diff_mat[2] }
              
            super().call_loopy(program, **argDict)

            result = kwargs["result"]

            # Treat as fortran style array again
            for i, entry in enumerate(["result1", "result2", "result3"]):
                argDict[entry].shape = (argDict[entry].shape[1], argDict[entry].shape[0])
                argDict[entry].strides = pyopencl.array._make_strides(argDict[entry].dtype.itemsize, argDict[entry].shape, "f")
                # This should be unnecessary
                # Il est necessaire pour le moment a cause du "ctof" d'ici. 
                ftoc_knl(cq, input=argDict[entry], output=result[i])

        else:
            result = super().call_loopy(program,**kwargs)

        return result

    #@memoize_method
    def transform_loopy_program(self, program):
        if program.name == "opt_diff":
            # TODO: Dynamically determine device id, don't hardcode path to transform.hjson.
            # Also get pn from program
            filename = "/home/njchris2/Workspace/nick/loopy_dg_kernels/transform.hjson"
            deviceID = "NVIDIA Titan V"
            pn = 3

            transformations = dgk.loadTransformationsFromFile(filename, deviceID, pn)            
            program = dgk.applyTransformationList(program, transformations)
        else:
            program = super().transform_loopy_program(program)

        return program
