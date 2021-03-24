import numpy as np
from pytools import memoize_in

#import pyopencl as cl
#import pyopencl.array
#import pyopencl.clrandom

import loopy as lp
#from loopy.version import LOOPY_USE_LANGUAGE_VERSION_2018_2
#from loopy.kernel.data import AddressSpace

#import pycuda.gpuarray as cuarray
#import pycuda.driver as drv
#import pycuda.tools
#import pycuda.autoinit
#from pycuda.compiler import SourceModule
#from pycuda.curandom import rand as curand

#from modepy import equidistant_nodes

#from bs4 import UnicodeDammit
import hjson
#import time
#from math import ceil
#import sys

# setup
# -----
lp.set_caching_enabled(False)
import loopy.options
loopy.options.ALLOW_TERMINAL_COLORS = False

def gen_elwise_linear_knl(n_elem, n_in, n_out, fp_format):

    knl = lp.make_kernel(
        """{[iel, idof, j]:
            0<=iel<nelements and
            0<=idof<ndiscr_nodes_out and
            0<=j<ndiscr_nodes_in}""",
        "result[iel, idof] = sum(j, mat[idof, j] * vec[iel, j])",
        kernel_data=[
            lp.GlobalArg("result", fp_format, shape=(n_elem, n_out), order="F"),
            lp.GlobalArg("vec", fp_format, shape=(n_elem, n_in), order="F"),
            lp.GlobalArg("mat", fp_format, shape=(n_out, n_in), order="C")    
        ],
        name="elwise_linear")
    knl = lp.fix_parameters(knl, nelements=n_elem,
        ndiscr_nodes_in=n_in, ndiscr_nodes_out=n_out)


    #result = lp.tag_array_axes(result, "mat", "stride:auto,stride:auto")
    return knl

# Se podría usar el de Grudge.
#@memoize_method
def gen_diff_knl_fortran2(n_mat, n_elem, n_in, n_out, fp_format=np.float32,
        options=None):
    
    @memoize_in(gen_diff_knl_fortran2, "_gen_diff_knl")
    def _gen_diff_knl(n_mat, n_elem, n_in, n_out, fp_format):
        knl = lp.make_kernel(
        """{[imatrix,iel,idof,j]:
            0<=imatrix<nmatrices and
            0<=iel<nelements and
            0<=idof<ndiscr_nodes_out and
            0<=j<ndiscr_nodes_in}""",
        """
        result[imatrix,iel,idof] = simul_reduce(sum, j, diff_mat[imatrix, idof, j] * vec[iel, j])
        """,
        kernel_data=[
            lp.GlobalArg("result", fp_format, shape=(n_mat, n_elem, n_out),
                offset=lp.auto),
            lp.GlobalArg("diff_mat", fp_format, shape=(n_mat, n_out, n_in),
                order="C", offset=lp.auto),
            lp.GlobalArg("vec", fp_format, shape=(n_elem, n_in), order="F",
                offset=lp.auto)
        ],
        assumptions="nelements > 0 \
                     and ndiscr_nodes_out > 0 \
                     and ndiscr_nodes_in > 0 and nmatrices > 0",
        options=options,
        name="opt_diff_{}_axis".format(n_mat)
        )
        return knl

    knl = _gen_diff_knl(n_mat, n_elem, n_in, n_out, fp_format)

    # This should be in array context probably but need to avoid circular dependency
    knl = lp.tag_inames(knl, "imatrix: ilp")
    knl = lp.tag_array_axes(knl, "diff_mat", "sep,c,c")
    knl = lp.tag_array_axes(knl, "result", "sep,f,f")
    knl = lp.tag_array_axes(knl, "vec", "f,f")
    knl = lp.fix_parameters(knl, nmatrices=n_mat, nelements=n_elem,
        ndiscr_nodes_in=n_in, ndiscr_nodes_out=n_out)
    return knl


# Is k x i in F layout equivalent to i x k in C layout?
# If so, can we just call the gen_diff_knl?
# Pretty sure it is...
def gen_diff_knl_fortran(n_elem, n_in, n_out, fp_format=np.float32, options=None):
    knl = lp.make_kernel(
        """{[k,i,j]:
            0<=k<nelements and
            0<=i<ndiscr_nodes_out and
            0<=j<ndiscr_nodes_in}""",
        """
        result1[k,i] = simul_reduce(sum, j, mat1[i, j] * vec[k, j])
        result2[k,i] = simul_reduce(sum, j, mat2[i, j] * vec[k, j])
        result3[k,i] = simul_reduce(sum, j, mat3[i, j] * vec[k, j])
        """,
        kernel_data=[
            lp.GlobalArg("result1", fp_format, shape=(n_elem, n_out), order="F",
                offset=lp.auto),
            lp.GlobalArg("result2", fp_format, shape=(n_elem, n_out), order="F",
                offset=lp.auto),
            lp.GlobalArg("result3", fp_format, shape=(n_elem, n_out), order="F",
                offset=lp.auto),
            lp.GlobalArg("mat1", fp_format, shape=(n_out, n_in), order="C",
                offset=lp.auto),
            lp.GlobalArg("mat2", fp_format, shape=(n_out, n_in), order="C",
                offset=lp.auto),
            lp.GlobalArg("mat3", fp_format, shape=(n_out, n_in), order="C",
                offset=lp.auto),
            lp.GlobalArg("vec", fp_format, shape=(n_elem, n_in), order="F",
                offset=lp.auto)
        ],
        assumptions="nelements > 0 \
                     and ndiscr_nodes_out > 0 \
                     and ndiscr_nodes_in > 0",
        options=options,
        name="opt_diff"

    )

    knl = lp.fix_parameters(knl, nelements=n_elem, ndiscr_nodes_in=n_in,
        ndiscr_nodes_out=n_out)

    return knl

#@memoize_method
def gen_diff_knl(n_mat, n_elem, n_in, n_out, fp_format=np.float32, options=None):
    print(fp_format)
    knl = lp.make_kernel(
        """{[m,k,i,j]:
            0<=k<nelements and
            0<=i<ndiscr_nodes_out and
            0<=j<ndiscr_nodes_in and
            0<=m<nmatrices}""",
        """
        result[m, i ,k] = simul_reduce(sum, j, diff_mat[m, i, j] * vec[j, k])
        """,
        kernel_data=[
            lp.GlobalArg("result", fp_format, shape=(n_mat, n_out, n_elem),
                offset=lp.auto),
            lp.GlobalArg("diff_mat", fp_format, shape=(n_mat, n_out, n_in),
                order="C", offset=lp.auto),
            lp.GlobalArg("vec", fp_format, shape=(n_in, n_elem), order="C",
                offset=lp.auto)
        ],
        #kernel_data = [
        #    lp.GlobalArg("result1", fp_format, shape=None, strides=(n_elem,1),
        #       dim_tags=None, offset=lp.auto, order="C"),
        #    lp.GlobalArg("result2", fp_format, shape=None, strides=(n_elem,1),
        #       dim_tags=None, offset=lp.auto, order="C"),
        #    lp.GlobalArg("result3", fp_format, shape=None, strides=(n_elem,1),
        #       dim_tags=None, offset=lp.auto, order="C"),
        #    lp.GlobalArg("mat1", fp_format, shape=lp.auto, offset=lp.auto,
        #       order="C"),
        #    lp.GlobalArg("mat2", fp_format, shape=lp.auto, offset=lp.auto,
        #       order="C"),
        #    lp.GlobalArg("mat3", fp_format, shape=lp.auto, offset=lp.auto,
        #       order="C"),
        #    lp.GlobalArg("vec", fp_format, shape=None, strides=(1, n_elem),
        #       offset=lp.auto, order="C")
        #],
        assumptions="nelements > 0 \
                     and ndiscr_nodes_out > 0 \
                     and ndiscr_nodes_in > 0 \
                     and nmatrices > 0",
        options=options,
        name="opt_diff"
    )
    knl = lp.tag_array_axes(knl, "diff_mat", "sep,c,c")
    knl = lp.tag_array_axes(knl, "result", "sep,c,c")
    knl = lp.tag_array_axes(knl, "vec", "c,c")

    knl = lp.fix_parameters(knl, nmatrices=n_mat, nelements=n_elem,
        ndiscr_nodes_in=n_in, ndiscr_nodes_out=n_out)

    #mat_string = ["result1", "result2", "result3", "vec"]
    #for i in range(len(mat_string)):
    #   knl = lp.tag_array_axes(knl, mat_string, "stride:auto,stride:auto")
    #   knl = lp.tag_array_axes(knl, mat_string, "N1,N0")

    return knl


# This is redundant with the above but is more clear than the above
# so to keep it around may be worthwhile.
'''
def gen_diff_knl(n_elem, n_in, n_out, k_inner_outer,k_inner_inner,i_inner_outer,
                    i_inner_inner,j_inner, fp_format=np.float32):
    knl = lp.make_kernel(
        """{[k,i,j]:
            0<=k<nelements and
            0<=i<ndiscr_nodes_out and
            0<=j<ndiscr_nodes_in}""",
        """
        result1[i,k] = simul_reduce(sum, j, mat1[i, j] * vec[j, k])
        result2[i,k] = simul_reduce(sum, j, mat2[i, j] * vec[j, k])
        result3[i,k] = simul_reduce(sum, j, mat3[i, j] * vec[j, k])
        """,
        kernel_data = [
            lp.GlobalArg("result1", fp_format, shape=(n_out, n_elem), order="C"),
            lp.GlobalArg("result2", fp_format, shape=(n_out, n_elem), order="C"),
            lp.GlobalArg("result3", fp_format, shape=(n_out, n_elem), order="C"),
            lp.GlobalArg("mat1", fp_format, shape=(n_out, n_in), order="C"),
            lp.GlobalArg("mat2", fp_format, shape=(n_out, n_in), order="C"),
            lp.GlobalArg("mat3", fp_format, shape=(n_out, n_in), order="C"),
            lp.GlobalArg("vec", fp_format, shape=(n_in, n_elem), order="C")
        ],
        assumptions="nelements > 0 \
                     and ndiscr_nodes_out > 0 \
                     and ndiscr_nodes_in > 0",
        default_offset=None,
        name="opt_diff"
    )

    knl = lp.fix_parameters(knl, nelements=n_elem, ndiscr_nodes_in=n_in,
                                ndiscr_nodes_out=n_out)

    slabs0 = (0,0) if n_elem % k_inner_outer == 0 else (0,1)
    knl = lp.split_iname(knl, "k", k_inner_outer, outer_tag="g.0", slabs=slabs0)
    knl = lp.split_iname(knl, "k_inner", k_inner_inner, outer_tag="ilp",
                            inner_tag="l.0")
    knl = lp.split_iname(knl, "j", j_inner)
    knl = lp.split_iname(knl, "i", i_inner_outer, outer_tag="g.1")#slabs=(0,1))
    knl = lp.split_iname(knl, "i_inner", i_inner_inner, outer_tag="ilp",
                            inner_tag="l.1")

    #knl = lp.prioritize_loops(knl, "j_outer,j_inner,k_inner_outer")

    knl = lp.add_prefetch(knl, "vec", "j_outer,j_inner,k_inner_outer,k_inner_inner",
                            temporary_name="vecf", default_tag="l.auto")
    knl = lp.add_prefetch(knl, "mat1", "j_inner", temporary_name="mat1fp",
                            default_tag="unr")
    knl = lp.add_prefetch(knl, "mat2", "j_inner", temporary_name="mat2fp",
                            default_tag="unr")
    knl = lp.add_prefetch(knl, "mat3", "j_inner", temporary_name="mat3fp",
                            default_tag="unr")

    return knl
'''


def load_transformations_from_file(hjson_file, indices): 
    hjson_text = hjson_file.read()
    od = hjson.loads(hjson_text)
    for index in indices:
        od = od[index]
    return od

def generate_transformation_list_old(k_inner_outer, k_inner_inner, i_inner_outer,
                                    i_inner_inner, j_inner):
    transformations = []
    # transformation name, list of args, dict of keyward args
    transformations.append(("split_iname", ["k", k_inner_outer], {"outer_tag": "g.0",
                                "slabs": (0, 1)}))
    transformations.append(("split_iname", ["k_inner", k_inner_inner],
                            {"outer_tag": "ilp", "inner_tag": "l.0"}))
    transformations.append(("split_iname", ["j", j_inner]))
    transformations.append(("split_iname", ["i", i_inner_outer],
                            {"outer_tag": "g.1"}))
    transformations.append(("split_iname", ["i_inner", i_inner_inner],
                            {"outer_tag": "ilp", "inner_tag": "l.1"}))
    transformations.append(("add_prefetch", ["vec",
                            "j_outer,j_inner,k_inner_outer,k_inner_inner"],
                            {"temporary_name": "vecf", "default_tag": "l.auto"}))
    transformations.append(("add_prefetch", ["mat1", "j_inner"],
                            {"temporary_name": "mat1fp", "default_tag": "unr"}))
    transformations.append(("add_prefetch", ["mat2", "j_inner"],
                            {"temporary_name": "mat2fp", "default_tag": "unr"}))
    transformations.append(("add_prefetch", ["mat3", "j_inner"],
                            {"temporary_name": "mat3fp", "default_tag": "unr"}))
    return tuple(transformations)

def generate_transformation_list(k_inner_outer, k_inner_inner, i_inner_outer,
                                i_inner_inner, j_inner):
    transformations = []
    # transformation name, list of args, dict of keyward args
    transformations.append(("tag_array_axes", ["diff_mat", "sep,c,c"]))
    transformations.append(("tag_array_axes", ["result", "sep,f,f"]))
    transformations.append(("tag_inames", [[("m", "ilp")]]))
    transformations.append(("split_iname", ["k", k_inner_outer], {"outer_tag": "g.0",
                            "slabs": (0, 0)}))
    transformations.append(("split_iname", ["k_inner", k_inner_inner],
                            {"outer_tag": "ilp", "inner_tag": "l.0"}))
    transformations.append(("split_iname", ["j", j_inner]))
    transformations.append(("split_iname", ["i", i_inner_outer],
                            {"outer_tag": "g.1"}))
    transformations.append(("split_iname", ["i_inner", i_inner_inner],
                            {"outer_tag": "ilp", "inner_tag": "l.1"}))
    transformations.append(("add_prefetch", ["vec",
                            "j_outer,j_inner,k_inner_outer,k_inner_inner"],
                            {"temporary_name": "vecf", "default_tag": "l.auto"}))
    transformations.append(("add_prefetch", ["diff_mat", "j_inner"],
                            {"temporary_name": "matfp", "default_tag": "unr"}))
    return tuple(transformations)

#@memoize_method
def apply_transformation_list(knl, transformations):
    function_mapping = {"split_iname": lp.split_iname,
                        "add_prefetch": lp.add_prefetch,
                        "prioritize_loops": lp.prioritize_loops,
                        "rename_iname": lp.rename_iname,
                        "tag_array_axes": lp.tag_array_axes,
                        "tag_inames": lp.tag_inames}

    # Maybe add some logic to add slabs=(0,0) if n_elem % k_inner_outer == 0
    # Maybe can do this based on tranformation name, loop variable, and loop variable
    # bounds
    #print(knl)
    for t in transformations:
        #print(t)
        func = function_mapping[t[0]]
        args = [knl] + t[1]
        kwargs = t[2] if len(t) > 2 else {}
        #print(t)
        knl = func(*args, **kwargs)

    return knl

#knl = gen_diff_knl_fortran2(3, 128, 10, 10)
#trans = generate_transformation_list(64, 32, 10, 10, 10)
#knl = apply_transformation_list(knl, trans)
#code = lp.generate_code_v2(knl).device_code()
#print(code)
