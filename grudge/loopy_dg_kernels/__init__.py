import numpy as np

import pyopencl as cl
import pyopencl.array
import pyopencl.clrandom

import loopy as lp
from loopy.version import LOOPY_USE_LANGUAGE_VERSION_2018_2
from loopy.kernel.data import AddressSpace

import pycuda.gpuarray as cuarray
import pycuda.driver as drv
import pycuda.tools
import pycuda.autoinit
from pycuda.compiler import SourceModule
from pycuda.curandom import rand as curand

from modepy import equidistant_nodes

from bs4 import UnicodeDammit
import hjson
import time
from math import ceil
import sys

# setup
# -----
lp.set_caching_enabled(False)
import loopy.options
loopy.options.ALLOW_TERMINAL_COLORS = False

# Se podría usar el de Grudge. 
def gen_diff_knl_fortran2(n_mat, n_elem, n_in, n_out, fp_format=np.float32, options=None):
    knl = lp.make_kernel(
        """{[m,k,i,j]:
            0<=m<nmatrices and
            0<=k<nelements and
            0<=i<ndiscr_nodes_out and
            0<=j<ndiscr_nodes_in}""",
        """
        result[m,k,i] = simul_reduce(sum, j, diff_mat[m, i, j] * vec[k, j]) 
        """,
        kernel_data = [
            lp.GlobalArg("result", fp_format, shape=(n_mat, n_elem, n_out), offset=lp.auto),
            lp.GlobalArg("diff_mat", fp_format, shape=(n_mat, n_out, n_in), order="C", offset=lp.auto),
            lp.GlobalArg("vec", fp_format, shape=(n_elem, n_in), order="F", offset=lp.auto)
        ],
        assumptions = "nelements > 0 \
                     and ndiscr_nodes_out > 0 \
                     and ndiscr_nodes_in > 0 and nmatrices > 0",
        options=options,
        name="opt_diff"        
    )

    # This should be in array context probably but need to avoid circular dependency
    knl = lp.tag_array_axes(knl, "diff_mat", "sep,c,c")
    knl = lp.tag_array_axes(knl, "result", "sep,f,f") 
    knl = lp.tag_array_axes(knl, "vec", "f,f")
    knl = lp.fix_parameters(knl, nmatrices=n_mat, nelements=n_elem, ndiscr_nodes_in=n_in, ndiscr_nodes_out=n_out)

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
        kernel_data = [
            lp.GlobalArg("result1", fp_format, shape=(n_elem, n_out), order="F", offset=lp.auto),
            lp.GlobalArg("result2", fp_format, shape=(n_elem, n_out), order="F", offset=lp.auto),
            lp.GlobalArg("result3", fp_format, shape=(n_elem, n_out), order="F", offset=lp.auto),
            lp.GlobalArg("mat1", fp_format, shape=(n_out, n_in), order="C", offset=lp.auto),
            lp.GlobalArg("mat2", fp_format, shape=(n_out, n_in), order="C", offset=lp.auto),
            lp.GlobalArg("mat3", fp_format, shape=(n_out, n_in), order="C", offset=lp.auto),
            lp.GlobalArg("vec", fp_format, shape=(n_elem, n_in), order="F", offset=lp.auto)
        ],
        assumptions="nelements > 0 \
                     and ndiscr_nodes_out > 0 \
                     and ndiscr_nodes_in > 0",
        options=options,
        name="opt_diff"
        
    )

    knl = lp.fix_parameters(knl, nelements=n_elem, ndiscr_nodes_in=n_in, ndiscr_nodes_out=n_out)

    return knl

def gen_diff_knl(n_elem, n_in, n_out, fp_format=np.float32, options=None):
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
            lp.GlobalArg("result1", fp_format, shape=lp.auto, offset=lp.auto, order="C"),
            lp.GlobalArg("result2", fp_format, shape=lp.auto, offset=lp.auto, order="C"),
            lp.GlobalArg("result3", fp_format, shape=lp.auto, offset=lp.auto, order="C"),
            lp.GlobalArg("mat1", fp_format, shape=lp.auto, offset=lp.auto, order="C"),
            lp.GlobalArg("mat2", fp_format, shape=lp.auto, offset=lp.auto, order="C"),
            lp.GlobalArg("mat3", fp_format, shape=lp.auto, offset=lp.auto, order="C"),
            lp.GlobalArg("vec", fp_format, shape=lp.auto, offset=lp.auto, order="C")
        ],
#        kernel_data = [
#            lp.GlobalArg("result1", fp_format, shape=None, strides=(n_elem,1), dim_tags=None, offset=lp.auto, order="C"),
#            lp.GlobalArg("result2", fp_format, shape=None, strides=(n_elem,1), dim_tags=None, offset=lp.auto, order="C"),
#            lp.GlobalArg("result3", fp_format, shape=None, strides=(n_elem,1), dim_tags=None, offset=lp.auto, order="C"),
#            lp.GlobalArg("mat1", fp_format, shape=lp.auto, offset=lp.auto, order="C"),
#            lp.GlobalArg("mat2", fp_format, shape=lp.auto, offset=lp.auto, order="C"),
#            lp.GlobalArg("mat3", fp_format, shape=lp.auto, offset=lp.auto, order="C"),
#            lp.GlobalArg("vec", fp_format, shape=None, strides=(1, n_elem), offset=lp.auto, order="C")
#        ],
        assumptions="nelements > 0 \
                     and ndiscr_nodes_out > 0 \
                     and ndiscr_nodes_in > 0",
        options=options,
        name="opt_diff"
    )

    knl = lp.fix_parameters(knl, nelements=n_elem, ndiscr_nodes_in=n_in, ndiscr_nodes_out=n_out)
    mat_string = ["result1", "result2", "result3", "vec"]
    #for i in range(len(mat_string)):
        #knl = lp.tag_array_axes(knl, mat_string, "stride:auto,stride:auto")
        #knl = lp.tag_array_axes(knl, mat_string, "N1,N0")

    return knl

# This is redundant with the above but is more clear than the above 
# so to keep it around may be worthwhile.
'''
def gen_diff_knl(n_elem, n_in, n_out, k_inner_outer,k_inner_inner,i_inner_outer,i_inner_inner,j_inner, fp_format=np.float32):
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
        assumptions=d(N) ((int) get_local_id(N))
#define gid(N) ((int) get_group_id(N))

__kernel void __attribute__ ((reqd_work_group_size(32, 10, 1))) opt_diff(__global float *__restrict__ result_s0, __global float *__restrict__ result_s1, __global float *__restrict__ result_s2, __global float const *__restrict__ mat_s0, __global float const *__restrict__ mat_s1, __global float const *__restrict__ mat_s2, __global float const *__restrict__ vec)
{
  float acc_j_outer_j_inner[2 * 3];
  float matfp[10 * 3];
  __local float vecf[64 * 10];

  for (int vec_dim_0_outer = 0; vec_dim_0_outer <= 1; ++vec_dim_0_outer)
    vecf[10 * (32 * vec_dim_0_outer + lid(0)) + lid(1)] = vec[64 * gid(0) + 32 * vec_dim_0_outer + lid(0) + 128 * lid(1)];
  acc_j_outer_j_inner[0] = 0.0f;
  acc_j_outer_j_inner[1] = 0.0f;
  acc_j_outer_j_inner[2] = 0.0f;
  acc_j_outer_j_inner[3] = 0.0f;
  acc_j_outer_j_inner[4] = 0.0f;
  acc_j_outer_j_inner[5] = 0.0f;
  barrier(CLK_LOCAL_MEM_FENCE) /* for vecf (insn_j_outer_j_inner_update depends on vec_fetch_rule) */;
  {
    int const j_outer = 0;

    matfp[0] = mat_s0[10 * lid(1)];
    matfp[1] = mat_s1[10 * lid(1)];
    matfp[2] = mat_s2[10 * lid(1)];
    matfp[3] = mat_s0[10 * lid(1) + 1];
    matfp[4] = mat_s1[10 * lid(1) + 1];
    matfp[5] = mat_s2[10 * lid(1) + 1];
    matfp[6] = mat_s0[10 * lid(1) + 2];
    matfp[7] = mat_s1[10 * lid(1) + 2];
    matfp[8] = mat_s2[10 * lid(1) + 2];
    matfp[9] = mat_s0[10 * lid(1) + 3];
    matfp[10] = mat_s1[10 * lid(1) + 3];
    matfp[11] = mat_s2[10 * lid(1) + 3];
    matfp[12] = mat_s0[10 * lid(1) + 4];
    matfp[13] = mat_s1[10 * lid(1) + 4];
    matfp[14] = mat_s2[10 * lid(1) + 4];
    matfp[15] = mat_s0[10 * lid(1) + 5];
    matfp[16] = mat_s1[10 * lid(1) + 5];
    matfp[17] = mat_s2[10 * lid(1) + 5];
    matfp[18] = mat_s0[10 * lid(1) + 6];
    matfp[19] = mat_s1[10 * lid(1) + 6];
    matfp[20] = mat_s2[10 * lid(1) + 6];
    matfp[21] = mat_s0[10 * lid(1) + 7];
    matfp[22] = mat_s1[10 * lid(1) + 7];
    matfp[23] = mat_s2[10 * lid(1) + 7];
    matfp[24] = mat_s0[10 * lid(1) + 8];
    matfp[25] = mat_s1[10 * lid(1) + 8];
    matfp[26] = mat_s2[10 * lid(1) + 8];
    matfp[27] = mat_s0[10 * lid(1) + 9];
    matfp[28] = mat_s1[10 * lid(1) + 9];
    matfp[29] = mat_s2[10 * lid(1) + 9];
    for (int j_inner = 0; j_inner <= 9; ++j_inner)
    {
      acc_j_outer_j_inner[0] = acc_j_outer_j_inner[0] + matfp[3 * j_inner] * vecf[10 * lid(0) + j_inner];
      acc_j_outer_j_inner[1] = acc_j_outer_j_inner[1] + matfp[1 + 3 * j_inner] * vecf[10 * lid(0) + j_inner];
      acc_j_outer_j_inner[2] = acc_j_outer_j_inner[2] + matfp[2 + 3 * j_inner] * vecf[10 * lid(0) + j_inner];
      acc_j_outer_j_inner[3] = acc_j_outer_j_inner[3] + matfp[3 * j_inner] * vecf[10 * (32 + lid(0)) + j_inner];
      acc_j_outer_j_inner[4] = acc_j_outer_j_inner[4] + matfp[1 + 3 * j_inner] * vecf[10 * (32 + lid(0)) + j_inner];
      acc_j_outer_j_inner[5] = acc_j_outer_j_inner[5] + matfp[2 + 3 * j_inner] * vecf[10 * (32 + lid(0)) + j_inner];
    }
  }
  result_s0[64 * gid(0) + lid(0) + 128 * lid(1)] = acc_j_outer_j_inner[0];
  result_s1[64 * gid(0) + lid(0) + 128 * lid(1)] = acc_j_outer_j_inner[1];
  result_s2[64 * gid(0) + lid(0) + 128 * lid(1)] = acc_j_outer_j_inner[2];
  result_s0[64 * gid(0) + 32 + lid(0) + 128 * lid(1)] = acc_j_outer_j_inner[3];
  result_s1[64 * gid(0) + 32 + lid(0) + 128 * lid(1)] = acc_j_outer_j_inner[4];
  result_s2[64 * gid(0) + 32 + lid(0) + 128 * lid(1)] = acc_j_outer_j_inner[5];
}
"nelements > 0 \
                     and ndiscr_nodes_out > 0 \
                     and ndiscr_nodes_in > 0",
        default_offset=None, 
        name="diff"
    )

    knl = lp.fix_parameters(knl, nelements=n_elem, ndiscr_nodes_in=n_in, ndiscr_nodes_out=n_out)

    slabs0 = (0,0) if n_elem % k_inner_outer == 0 else (0,1)
    knl = lp.split_iname(knl, "k", k_inner_outer, outer_tag="g.0", slabs=slabs0)
    knl = lp.split_iname(knl, "k_inner", k_inner_inner, outer_tag="ilp", inner_tag="l.0")
    knl = lp.split_iname(knl, "j", j_inner)
    knl = lp.split_iname(knl, "i", i_inner_outer, outer_tag="g.1")#slabs=(0,1))
    knl = lp.split_iname(knl, "i_inner", i_inner_inner, outer_tag="ilp", inner_tag="l.1")

    #knl = lp.prioritize_loops(knl, "j_outer,j_inner,k_inner_outer") 
    
    knl = lp.add_prefetch(knl, "vec", "j_outer,j_inner,k_inner_outer,k_inner_inner", temporary_name="vecf", default_tag="l.auto")
    knl = lp.add_prefetch(knl, "mat1", "j_inner", temporary_name="mat1fp", default_tag="unr")
    knl = lp.add_prefetch(knl, "mat2", "j_inner", temporary_name="mat2fp", default_tag="unr")
    knl = lp.add_prefetch(knl, "mat3", "j_inner", temporary_name="mat3fp", default_tag="unr")

    return knl
'''

def loadTransformationsFromFile(hjson_file, device_id, pn, fp_format=np.float32):
    hjson_text = hjson_file.read()
    od = hjson.loads(hjson_text)
    transformID = od["devices"][device_id]
    fp_string = "FP64" if fp_format == np.float64 else "FP32"
    hjson_file.close()
    transformations = od["transformations"][transformID][fp_string][str(pn)] 
    return transformations

def generateTransformationListOld(k_inner_outer, k_inner_inner, i_inner_outer, i_inner_inner, j_inner):
    transformations = []
    # transformation name, list of args, dict of keyward args
    transformations.append(("split_iname", ["k", k_inner_outer], {"outer_tag": "g.0", "slabs": (0,1)}))
    transformations.append(("split_iname", ["k_inner", k_inner_inner], {"outer_tag": "ilp", "inner_tag": "l.0"}))
    transformations.append(("split_iname", ["j", j_inner]))
    transformations.append(("split_iname", ["i", i_inner_outer], {"outer_tag": "g.1"}))
    transformations.append(("split_iname", ["i_inner", i_inner_inner], {"outer_tag": "ilp", "inner_tag": "l.1"}))
    transformations.append(("add_prefetch", ["vec", "j_outer,j_inner,k_inner_outer,k_inner_inner"], \
      {"temporary_name": "vecf", "default_tag": "l.auto"}))
    transformations.append(("add_prefetch", ["mat1", "j_inner"], {"temporary_name": "mat1fp", "default_tag": "unr"}))
    transformations.append(("add_prefetch", ["mat2", "j_inner"], {"temporary_name": "mat2fp", "default_tag": "unr"}))
    transformations.append(("add_prefetch", ["mat3", "j_inner"], {"temporary_name": "mat3fp", "default_tag": "unr"}))
    return transformations

def generateTransformationList(k_inner_outer, k_inner_inner, i_inner_outer, i_inner_inner, j_inner):
    transformations = []
    # transformation name, list of args, dict of keyward args
    transformations.append(("tag_array_axes", ["diff_mat", "sep,c,c"]))
    transformations.append(("tag_array_axes", ["result", "sep,f,f"]))
    transformations.append(("tag_inames", [[("m", "ilp")]]))
    transformations.append(("split_iname", ["k", k_inner_outer], {"outer_tag": "g.0", "slabs": (0,0)}))
    transformations.append(("split_iname", ["k_inner", k_inner_inner], {"outer_tag": "ilp", "inner_tag": "l.0"}))
    transformations.append(("split_iname", ["j", j_inner]))
    transformations.append(("split_iname", ["i", i_inner_outer], {"outer_tag": "g.1"}))
    transformations.append(("split_iname", ["i_inner", i_inner_inner], {"outer_tag": "ilp", "inner_tag": "l.1"}))
    transformations.append(("add_prefetch", ["vec", "j_outer,j_inner,k_inner_outer,k_inner_inner"], \
      {"temporary_name": "vecf", "default_tag": "l.auto"}))
    transformations.append(("add_prefetch", ["diff_mat", "j_inner"], {"temporary_name": "matfp", "default_tag": "unr"}))
    return transformations


def applyTransformationList(knl, transformations):
    functionMapping = { "split_iname": lp.split_iname,
                        "add_prefetch": lp.add_prefetch,
                        "prioritize_loops": lp.prioritize_loops,
                        "rename_iname": lp.rename_iname,
                        "tag_array_axes": lp.tag_array_axes,
                        "tag_inames": lp.tag_inames }

    # Maybe add some logic to add slabs=(0,0) if n_elem % k_inner_outer == 0
    # Maybe can do this based on tranformation name, loop variable, and loop variable bounds
    for t in transformations:
      func = functionMapping[t[0]]
      args = [knl] + t[1]
      kwargs = t[2] if len(t) > 2 else {}
      #print(t)
      knl = func(*args, **kwargs)  

    return knl

#knl = gen_diff_knl_fortran2(3, 128, 10, 10)
#trans = generateTransformationList(64, 32, 10, 10, 10)
#knl = applyTransformationList(knl, trans)
#code = lp.generate_code_v2(knl).device_code()
#print(code)
