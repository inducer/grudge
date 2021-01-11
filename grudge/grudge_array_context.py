from meshmode.array_context import PyOpenCLArrayContext
from meshmode.dof_array import IsDOFArray
#from grudge.execution import VecOpIsDOFArray
from grudge.execution import (VecIsDOFArray, FaceIsDOFArray,
    IsOpArray)
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
                allocator=self.allocator, order="F")

    def zeros(self, shape, dtype):
        return cla.zeros(self.queue, shape=shape, dtype=dtype,
                allocator=self.allocator, order="F")

    #@memoize_method
    def _get_scalar_func_loopy_program(self, name, nargs, naxes):
        prog = super()._get_scalar_func_loopy_program(name, nargs, naxes)
        for arg in prog.args:
            if type(arg) == lp.ArrayArg:
                arg.tags = IsDOFArray()
        return prog

    def thaw(self, array):
        thawed = super().thaw(array)
        if type(getattr(array, "tags", None)) == IsDOFArray:
            cq = thawed.queue
            _, (out,) = ctof_knl(cq, input=thawed)
            thawed = out
            # May or may not be needed
            #thawed.tags = "dof_array"
        return thawed

    #@memoize_method
    def transform_loopy_program(self, program):
        #print(program.name)

        for arg in program.args:
            if isinstance(arg.tags, IsDOFArray):
                program = lp.tag_array_axes(program, arg.name, "f,f")
            elif isinstance(arg.tags, IsOpArray):
                program = lp.tag_array_axes(program, arg.name, "f,f")
            elif isinstance(arg.tags, VecIsDOFArray):
                program = lp.tag_array_axes(program, arg.name, "sep,f,f")
            #elif isinstance(arg.tags, VecOpIsDOFArray):
            #    program = lp.tag_array_axes(program, arg.name, "sep,c,c")
            elif isinstance(arg.tags, FaceIsDOFArray):
                program = lp.tag_array_axes(program, arg.name, "N1,N0,N2")

        if program.name == "opt_diff":
            # TODO: Dynamically determine device id,
            # don't hardcode path to transform.hjson.
            # Also get pn from program
            hjson_file = pkg_resources.open_text(dgk, "transform.hjson")
            device_id = "NVIDIA Titan V"

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

            transformations = dgk.load_transformations_from_file(hjson_file,
                device_id, pn, fp_format=fp_format)
            program = dgk.apply_transformation_list(program, transformations)
        else:
            program = super().transform_loopy_program(program)

        return program


# {{{ pytest integration

# Should this method just be modified to accept a class _ContextFactory
# as an argument?
def pytest_generate_tests_for_grudge_array_context(metafunc):
    """Parametrize tests for pytest to use a :mod:`grudge` array context.

    Performs device enumeration analogously to
    :func:`pyopencl.tools.pytest_generate_tests_for_pyopencl`.

    Using the line:

    .. code-block:: python

       from grudge.grudge_array_context import pytest_generate_tests_for_pyopencl \
            as pytest_generate_tests

    in your pytest test scripts allows you to use the arguments ctx_factory,
    device, or platform in your test functions, and they will automatically be
    run for each OpenCL device/platform in the system, as appropriate.

    It also allows you to specify the ``PYOPENCL_TEST`` environment variable
    for device selection.
    """

    import pyopencl as cl
    from pyopencl.tools import _ContextFactory

    class ArrayContextFactory(_ContextFactory):
        def __call__(self):
            ctx = super().__call__()
            return GrudgeArrayContext(cl.CommandQueue(ctx))

        def __str__(self):
            return ("<array context factory for <pyopencl.Device '%s' on '%s'>" %
                    (self.device.name.strip(),
                     self.device.platform.name.strip()))

    import pyopencl.tools as cl_tools
    arg_names = cl_tools.get_pyopencl_fixture_arg_names(
            metafunc, extra_arg_names=["actx_factory"])

    if not arg_names:
        return

    arg_values, ids = cl_tools.get_pyopencl_fixture_arg_values()
    if "actx_factory" in arg_names:
        if "ctx_factory" in arg_names or "ctx_getter" in arg_names:
            raise RuntimeError("Cannot use both an 'actx_factory' and a "
                    "'ctx_factory' / 'ctx_getter' as arguments.")

        for arg_dict in arg_values:
            arg_dict["actx_factory"] = ArrayContextFactory(arg_dict["device"])

    arg_values = [
            tuple(arg_dict[name] for name in arg_names)
            for arg_dict in arg_values
            ]

    metafunc.parametrize(arg_names, arg_values, ids=ids)

# }}}


# vim: foldmethod=marker
