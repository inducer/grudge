"""
Core DG routines
^^^^^^^^^^^^^^^^

Elementwise differentiation
---------------------------

.. autofunction:: local_grad
.. autofunction:: local_d_dx
.. autofunction:: local_div

Weak derivative operators
-------------------------

.. autofunction:: weak_local_grad
.. autofunction:: weak_local_d_dx
.. autofunction:: weak_local_div

Mass, inverse mass, and face mass operators
-------------------------------------------

.. autofunction:: mass
.. autofunction:: inverse_mass
.. autofunction:: face_mass

Working around documentation tool awkwardness
---------------------------------------------

.. class:: TracePair

    See :class:`grudge.trace_pair.TracePair`.

Links to canonical locations of external symbols
------------------------------------------------

(This section only exists because Sphinx does not appear able to resolve
these symbols correctly.)

.. class:: ArrayOrContainer

    See :class:`arraycontext.ArrayOrContainer`.
"""

from __future__ import annotations

__copyright__ = """
Copyright (C) 2021 Andreas Kloeckner
Copyright (C) 2021 University of Illinois Board of Trustees
"""

__license__ = """
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""


from arraycontext import (ArrayContext, map_array_container, tag_axes,
        ArrayOrContainer)

from functools import partial

from meshmode.dof_array import DOFArray
from meshmode.transform_metadata import (FirstAxisIsElementsTag,
                                         DiscretizationDOFAxisTag,
                                         DiscretizationElementAxisTag,
                                         DiscretizationFaceAxisTag)
from meshmode.discretization.poly_element import TensorProductElementGroupBase

from modepy.tools import (
        reshape_array_for_tensor_product_space as fold,
        unreshape_array_for_tensor_product_space as unfold)

from grudge.discretization import DiscretizationCollection
from grudge.dof_desc import as_dofdesc
from grudge.array_context import (
        OutputIsTensorProductDOFArrayOrdered)

from pytools import keyed_memoize_in
from pytools.obj_array import make_obj_array

import numpy as np

import grudge.dof_desc as dof_desc
from grudge.dof_desc import (
    DD_VOLUME_ALL, FACE_RESTR_ALL, DISCR_TAG_BASE,
    DOFDesc, VolumeDomainTag
)

from grudge.interpolation import interp
from grudge.projection import project

from grudge.reductions import (
    norm,
    nodal_sum,
    nodal_min,
    nodal_max,
    nodal_sum_loc,
    nodal_min_loc,
    nodal_max_loc,
    integral,
    elementwise_sum,
    elementwise_max,
    elementwise_min,
    elementwise_integral,
)

from grudge.trace_pair import (
    project_tracepair,
    tracepair_with_discr_tag,
    interior_trace_pair,
    interior_trace_pairs,
    local_interior_trace_pair,
    connected_ranks,
    cross_rank_trace_pairs,
    bdry_trace_pair,
    bv_trace_pair
)


__all__ = (
    "project",
    "interp",

    "norm",
    "nodal_sum",
    "nodal_min",
    "nodal_max",
    "nodal_sum_loc",
    "nodal_min_loc",
    "nodal_max_loc",
    "integral",
    "elementwise_sum",
    "elementwise_max",
    "elementwise_min",
    "elementwise_integral",

    "project_tracepair",
    "tracepair_with_discr_tag",
    "interior_trace_pair",
    "interior_trace_pairs",
    "local_interior_trace_pair",
    "connected_ranks",
    "cross_rank_trace_pairs",
    "bdry_trace_pair",
    "bv_trace_pair",

    "local_grad",
    "local_d_dx",
    "local_div",

    "weak_local_grad",
    "weak_local_d_dx",
    "weak_local_div",

    "mass",
    "inverse_mass",
    "face_mass",
    )


# {{{ common derivative "kernels"

def _single_axis_derivative_kernel(
        actx, out_discr, in_discr, get_diff_mat, inv_jac_mat, xyz_axis, vec,
        *, metric_in_matvec):
    # This gets used from both the strong and the weak derivative. These differ
    # in three ways:
    # - which differentiation matrix gets used,
    # - whether inv_jac_mat is pre-multiplied by a factor that includes the
    #   area element, and
    # - whether the chain rule terms ("inv_jac_mat") sit outside (strong)
    #   or inside (weak) the matrix-vector product that carries out the
    #   derivative, cf. "metric_in_matvec".


    # {{{ tensor product single axis derivative

    # FIXME: actually implement single axis tensor product derivatives
    def compute_tensor_product_derivative(actx, grp, get_diff_mat, vec, ijm,
                                          xyz_axis, metric_in_matvec):


        return compute_simplicial_derivative(actx, grp, grp, get_diff_mat, vec,
                                             ijm, xyz_axis, metric_in_matvec)

    # }}}


    # {{{ simplicial single axis derivative

    def compute_simplicial_derivative(actx, in_grp, out_grp,
                                      get_diff_mat, vec_i, ijm_i,
                                      xyz_axis, metric_in_matvec):
        # r for rst axis
        return actx.einsum(
            "rej,rij,ej->ei" if metric_in_matvec else "rei,rij,ej->ei",
            ijm_i[xyz_axis],
            get_diff_mat(
                actx,
                out_element_group=out_grp,
                in_element_group=in_grp),
            vec_i,
            arg_names=("inv_jac_t", "ref_stiffT_mat", "vec", ),
            tagged=(FirstAxisIsElementsTag(),))

    # }}}


    return DOFArray(
        actx,
        data=tuple(
            compute_tensor_product_derivative(actx, in_grp, out_grp,
                                              get_diff_mat, vec_i, ijm_i,
                                              xyz_axis, metric_in_matvec)
            if isinstance(in_grp, TensorProductElementGroupBase)
            else compute_simplicial_derivative(actx, in_grp, out_grp,
                                               get_diff_mat, vec_i, ijm_i,
                                               xyz_axis, metric_in_matvec)
            for out_grp, in_grp, vec_i, ijm_i in zip(
                out_discr.groups, in_discr.groups, vec,
                inv_jac_mat)))


def _gradient_kernel(actx, out_discr, in_discr, get_diff_mat, inv_jac_mat, vec,
        *, metric_in_matvec):
    # See _single_axis_derivative_kernel for comments on the usage scenarios
    # (both strong and weak derivative) and their differences.


    # {{{ tensor product gradient

    def compute_tensor_product_grad(actx, grp, diff_mat, vec, ijm,
                                    metric_in_matvec):
        """
        Exploits tensor product structure to carry out differentiation with a
        differentiation operator containing only 1D information. For example,
        in the 2D strong form case, this computes partial derivatives in a
        similar manner to

            .. math::

                \partial_x \mathbf{f}_{ij} =
                \sum_{\ell,j}^n \mathbf{J}^e_{i\ell} \mathbf{D}_{i\ell}
                \mathbf{f}_{\ell j}

        where $\mathbf{D}$ is a 1D differentiation operator, $\mathbf{f}$ is a
        vector of function data, $\mathbf{J}^e$ is the element Jacobian matrix.
        The weak form uses a 1D mass operator and a 1D stiffness operator using
        the fact that

            .. math::

                \mathbf{M}^{2D}_{pq,rs} = \int_{\Omega} \phi_p(x) \phi_q(y)
                \phi_r(x) \phi_s(y) d\Omega = \int_{\Omega_x} \phi_p(x)
                \phi_r(x) dx \int_{\Omega_y}  \phi_q(y) \phi_s(y) dy =
                \mathbf{M}^{1D} \otimes \mathbf{M}^{1D}

        """

        # reshape u to expose tensor product structure
        vec = fold(grp.space, vec)
        inv_jac_mat_tp = fold(grp.space, ijm)

        # apply operators to function data
        dim = grp.dim
        diff_mat = get_diff_mat(actx, grp, grp)

        # weak form case:
        #   3D weak_x: einsum("estu,ps,qt,ru->epqr",
        #                      f, stiff_1D, mass_1D, mass_1D)
        if metric_in_matvec:
            stiff_1D, mass_1D = diff_mat
            if dim == 3:
                weak_x = actx.einsum(
                    "rejbd,ejbd,ij,ab,cd->eiac",
                    inv_jac_mat_tp[0],
                    vec,
                    stiff_1D,
                    mass_1D,
                    mass_1D,
                    arg_names=("inv_jac_mat", "vec", "stiff_1D_r", "mass_1D_s",
                               "mass_1D_t"),
                    tagged=(FirstAxisIsElementsTag(),
                            OutputIsTensorProductDOFArrayOrdered()))

                weak_y = actx.einsum(
                    "rebjd,ebjd,ij,ab,cd->eaic",
                    inv_jac_mat_tp[1],
                    vec,
                    stiff_1D,
                    mass_1D,
                    mass_1D,
                    arg_names=("inv_jac_mat", "vec", "mass_1D_r", "stiff_1D_s",
                               "mass_1D_t"),
                    tagged=(FirstAxisIsElementsTag(),
                            OutputIsTensorProductDOFArrayOrdered()))

                weak_z = actx.einsum(
                    "rebdj,ebdj,ij,ab,cd->eaci",
                    inv_jac_mat_tp[2],
                    vec,
                    stiff_1D,
                    mass_1D,
                    mass_1D,
                    arg_names=("inv_jac_mat", "vec", "mass_1D_r", "mass_1D_s",
                               "stiff_1D_t"),
                    tagged=(FirstAxisIsElementsTag(),
                            OutputIsTensorProductDOFArrayOrdered()))

                grad = make_obj_array([
                    weak_x,
                    weak_y,
                    weak_z
                ])

                # FIXME: causes an error: static maximum not found for PwAff ...
                # grad = make_obj_array([
                #     actx.einsum(
                #         f"re{'bd'[:i]}j{'bd'[i:]}," +
                #         f"e{'bd'[:i]}j{'bd'[i:]}," +
                #         "ij,ab,cd->" +
                #         f"e{'ac'[:i]}i{'ac'[i:]}",
                #         vec,
                #         stiff_1D,
                #         mass_1D,
                #         mass_1D,
                #         arg_names=("inv_jac_mat", "vec", "stiff_1D", "mass_1D", "mass_1D"),
                #         tagged=(FirstAxisIsElementsTag(),
                #                 OutputIsTensorProductDOFArrayOrdered()))
                #     for i in range(grp.dim)
                # ])

            elif dim == 2:
                weak_x = actx.einsum(
                    "rejb,ejb,ij,ab->eia",
                    inv_jac_mat_tp[0],
                    vec,
                    stiff_1D,
                    mass_1D,
                    arg_names=("inv_jac_mat", "vec", "stiff_1D_r", "mass_1D_s"),
                    tagged=(FirstAxisIsElementsTag(),
                            OutputIsTensorProductDOFArrayOrdered()))

                weak_y = actx.einsum(
                    "rebj,ebj,ij,ab->eai",
                    inv_jac_mat_tp[1],
                    vec,
                    stiff_1D,
                    mass_1D,
                    arg_names=("inv_jac_mat", "vec", "mass_1D_r", "stiff_1D_s"),
                    tagged=(FirstAxisIsElementsTag(),
                            OutputIsTensorProductDOFArrayOrdered()))

                grad = make_obj_array([
                    weak_x,
                    weak_y
                ])

        # Carries out, e.g., 3D strong form contraction
        #   x partial: einsum("il,eljk->eijk", D, f)
        else:
            grad = make_obj_array([
                actx.einsum(
                    f"re{'abcdfghijkl'[:i]}y{'mnopqstuvwx'[:dim-i-1]}," +
                    "yz," +
                    f"e{'abcdfghijkl'[:i]}z{'mnopqstuvwx'[:dim-i-1]}->" +
                    f"e{'abcdfghijkl'[:i]}y{'mnopqstuvwx'[:dim-i-1]}",
                    inv_jac_mat_tp[i],
                    diff_mat,
                    vec,
                    arg_names=("inv_jac_mat", "diff_mat", "vec"),
                    tagged=(FirstAxisIsElementsTag(),
                        OutputIsTensorProductDOFArrayOrdered()))
                for i in range(dim)
            ])

        # unreshape grad
        grad = make_obj_array([
            unfold(grp.space, grad[i])
            for i in range(grp.dim)
        ])

        return grad

    # }}}


    # {{{ simplicial grad

    def compute_simplicial_grad(actx, in_grp, out_grp, get_diff_mat, vec_i,
                                ijm_i, metric_in_matvec):
        return actx.einsum(
            "xrej,rij,ej->xei" if metric_in_matvec else "xrei,rij,ej->xei",
            ijm_i,
            get_diff_mat(
                actx,
                out_element_group=out_grp,
                in_element_group=in_grp
            ),
            vec_i,
            arg_names=("inv_jac_t", "ref_stiffT_mat", "vec"),
            tagged=(FirstAxisIsElementsTag(),))

    # }}}


    per_group_grads = [
        compute_tensor_product_grad(actx, in_grp, get_diff_mat, vec_i, ijm_i,
                                    metric_in_matvec)
        if isinstance(in_grp, TensorProductElementGroupBase)
        else compute_simplicial_grad(actx, in_grp, out_grp, get_diff_mat, vec_i,
                                     ijm_i, metric_in_matvec)

        for out_grp, in_grp, vec_i, ijm_i in zip(
            out_discr.groups, in_discr.groups, vec,
            inv_jac_mat)
    ]

    return make_obj_array([
            DOFArray(
                actx, data=tuple([pgg_i[xyz_axis] for pgg_i in per_group_grads]))
            for xyz_axis in range(out_discr.ambient_dim)])


def _divergence_kernel(actx, out_discr, in_discr, get_diff_mat, inv_jac_mat, vec,
        *, metric_in_matvec):
    # See _single_axis_derivative_kernel for comments on the usage scenarios
    # (both strong and weak derivative) and their differences.


    # {{{ tensor product div

    def compute_tensor_product_div(actx, grp, diff_mat, vec, ijm):
        """Exploits tensor product structure to differentiate each coordinate
        axis using a single differentiation matrix of shape (nnodes1d, nnodes1d)
        """

        # reshape u to expose tensor product structure
        vec = fold(grp.space, vec)
        inv_jac_mat_tp = fold(grp.space, ijm[0])

        dim = grp.dim
        diff_mat = get_diff_mat(actx, grp, grp)

        # weak form
        if metric_in_matvec:
            stiff_1D, mass_1D = diff_mat
            if dim == 3:
                weak_x = actx.einsum(
                    "rejbd,ejbd,ij,ab,cd->eiac",
                    inv_jac_mat_tp[0],
                    vec,
                    stiff_1D,
                    mass_1D,
                    mass_1D,
                    arg_names=("inv_jac_mat", "vec", "stiff_1D_r", "mass_1D_s",
                               "mass_1D_t"),
                    tagged=(FirstAxisIsElementsTag(),
                            OutputIsTensorProductDOFArrayOrdered()))

                weak_y = actx.einsum(
                    "rebjd,ebjd,ij,ab,cd->eaic",
                    inv_jac_mat_tp[1],
                    vec,
                    stiff_1D,
                    mass_1D,
                    mass_1D,
                    arg_names=("inv_jac_mat", "vec", "mass_1D_r", "stiff_1D_s",
                               "mass_1D_t"),
                    tagged=(FirstAxisIsElementsTag(),
                            OutputIsTensorProductDOFArrayOrdered()))

                weak_z = actx.einsum(
                    "rebdj,ebdj,ij,ab,cd->eaci",
                    inv_jac_mat_tp[2],
                    vec,
                    stiff_1D,
                    mass_1D,
                    mass_1D,
                    arg_names=("inv_jac_mat", "vec", "mass_1D_r", "mass_1D_s",
                               "stiff_1D_t"),
                    tagged=(FirstAxisIsElementsTag(),
                            OutputIsTensorProductDOFArrayOrdered()))

                partials = make_obj_array([
                    weak_x,
                    weak_y,
                    weak_z
                ])

                # FIXME: causes an error: static maximum not found for PwAff ...
                # grad = make_obj_array([
                #     actx.einsum(
                #         f"re{'bd'[:i]}j{'bd'[i:]}," +
                #         "ij,ab,cd->" +
                #         f"e{'ac'[:i]}i{'ac'[i:]}",
                #         vec,
                #         stiff_1D,
                #         mass_1D,
                #         mass_1D,
                #         arg_names=("inv_jac_mat", "vec", "stiff_1D", "mass_1D", "mass_1D"),
                #         tagged=(FirstAxisIsElementsTag(),
                #                 OutputIsTensorProductDOFArrayOrdered()))
                #     for i in range(grp.dim)
                # ])

            elif dim == 2:
                weak_x = actx.einsum(
                    "rejb,ejb,ij,ab->eia",
                    inv_jac_mat_tp[0],
                    vec,
                    stiff_1D,
                    mass_1D,
                    arg_names=("inv_jac_mat", "vec", "stiff_1D_r", "mass_1D_s"),
                    tagged=(FirstAxisIsElementsTag(),
                            OutputIsTensorProductDOFArrayOrdered()))

                weak_y = actx.einsum(
                    "rebj,ebj,ij,ab->eai",
                    inv_jac_mat_tp[1],
                    vec,
                    stiff_1D,
                    mass_1D,
                    arg_names=("inv_jac_mat", "vec", "mass_1D_r", "stiff_1D_s"),
                    tagged=(FirstAxisIsElementsTag(),
                            OutputIsTensorProductDOFArrayOrdered()))

                partials = make_obj_array([
                    weak_x,
                    weak_y
                ])

            else:
                raise Exception("Dimensions of 2 and 3 are supported by "
                                "tensor product elements. Found dim = {dim}")

        # strong form
        else:
            partials = make_obj_array([
                actx.einsum(
                    f"re{'abcdfghijkl'[:i]}y{'mnopqstuvwx'[:dim-i-1]}," +
                    "yz," +
                    f"e{'abcdfghijkl'[:i]}z{'mnopqstuvwx'[:dim-i-1]}->" +
                    f"e{'abcdfghijkl'[:i]}y{'mnopqstuvwx'[:dim-i-1]}",
                    inv_jac_mat_tp[i],
                    diff_mat,
                    vec,
                    arg_names=("inv_jac_mat", "diff_mat", "vec"),
                    tagged=(FirstAxisIsElementsTag(),
                        OutputIsTensorProductDOFArrayOrdered()))
                for i in range(dim)
            ])

        partials = make_obj_array([
            unfold(grp.space, partials[i])
            for i in range(partials.shape[0])
        ])

        partials = actx.np.stack([partials[i] for i in range(dim)])
        div = actx.einsum(
                "xei->ei",
                partials,
                arg_names=("partials",),
                tagged=(FirstAxisIsElementsTag(),))

        return div

    # }}}


    # {{{ simplicial div

    def compute_simplicial_div(actx, in_grp, out_grp, get_diff_mat, vec_i,
                               ijm_i, metric_in_matvec):
        return actx.einsum(
            "xrej,rij,xej->ei" if metric_in_matvec else "xrei,rij,xej->ei",
            ijm_i,
            get_diff_mat(
                actx,
                out_element_group=out_grp,
                in_element_group=in_grp
            ),
            vec_i,
            arg_names=("inv_jac_t", "ref_stiffT_mat", "vec"),
            tagged=(FirstAxisIsElementsTag(),))

    # }}}


    per_group_divs = [

        compute_tensor_product_div(actx, in_grp, get_diff_mat, vec_i, ijm_i)
        if isinstance(in_grp, TensorProductElementGroupBase)

        # r for rst axis
        # x for xyz axis
        else compute_simplicial_div(actx, in_grp, out_grp, get_diff_mat, vec_i,
                                    ijm_i, metric_in_matvec)

        for out_grp, in_grp, vec_i, ijm_i in zip(
            out_discr.groups, in_discr.groups, vec,
            inv_jac_mat)
    ]

    return DOFArray(actx, data=tuple(per_group_divs))

# }}}


# {{{ Derivative operators

def _reference_derivative_matrices(actx: ArrayContext,
        out_element_group, in_element_group):
    # We're accepting in_element_group for interface consistency with
    # _reference_stiffness_transpose_matrices.
    assert out_element_group is in_element_group

    @keyed_memoize_in(
        actx, _reference_derivative_matrices,
        lambda grp: grp.discretization_key())
    def get_ref_derivative_mats(grp):
        if isinstance(grp, TensorProductElementGroupBase):
            import modepy as mp
            import numpy.linalg as la

            #FIXME: Can be gotten rid of by updating meshmode
            nodes1d = grp.unit_nodes_1d
            bases_1d = grp.bases_1d()

            vdm_1d = mp.vandermonde(bases_1d.functions, nodes1d)
            vdm_p_1d = mp.vandermonde(bases_1d.gradients, nodes1d)[0]

            diff_mat = actx.from_numpy(vdm_p_1d @ la.inv(vdm_1d))

            from arraycontext.metadata import NameHint
            return actx.freeze(
                    actx.tag(NameHint("tp_diff_mat_1d"),
                             tag_axes(actx, {
                                1: DiscretizationDOFAxisTag()},
                                diff_mat)))

        else:
            from meshmode.discretization.poly_element import diff_matrices
            return actx.freeze(
                    actx.tag_axis(
                        1, DiscretizationDOFAxisTag(),
                        actx.from_numpy(
                            np.asarray(diff_matrices(grp)))))
    return get_ref_derivative_mats(out_element_group)


def _strong_scalar_grad(dcoll, dd_in, vec):
    assert isinstance(dd_in.domain_tag, VolumeDomainTag)

    from grudge.geometry import inverse_surface_metric_derivative_mat

    discr = dcoll.discr_from_dd(dd_in)
    actx = vec.array_context

    inverse_jac_mat = inverse_surface_metric_derivative_mat(actx, dcoll, dd=dd_in,
            _use_geoderiv_connection=actx.supports_nonscalar_broadcasting)
    return _gradient_kernel(actx, discr, discr,
            _reference_derivative_matrices, inverse_jac_mat, vec,
            metric_in_matvec=False)


def _strong_scalar_div(dcoll, dd, vecs):
    from grudge.geometry import inverse_surface_metric_derivative_mat
    from arraycontext import (get_container_context_recursively,
                              serialize_container)

    assert isinstance(vecs, np.ndarray)
    assert vecs.shape == (dcoll.ambient_dim,)

    discr = dcoll.discr_from_dd(dd)

    actx = get_container_context_recursively(vecs)
    vec = actx.np.stack([v for k, v in serialize_container(vecs)])

    inverse_jac_mat = inverse_surface_metric_derivative_mat(actx, dcoll, dd=dd,
            _use_geoderiv_connection=actx.supports_nonscalar_broadcasting)

    return _divergence_kernel(actx, discr, discr,
            _reference_derivative_matrices, inverse_jac_mat, vec,
            metric_in_matvec=False)


def local_grad(
        dcoll: DiscretizationCollection, *args, nested=False) -> ArrayOrContainer:
    r"""Return the element-local gradient of a function :math:`f` represented
    by *vec*:

    .. math::

        \nabla|_E f = \left(
            \partial_x|_E f, \partial_y|_E f, \partial_z|_E f \right)

    May be called with ``(vec)`` or ``(dd_in, vec)``.

    :arg vec: a :class:`~meshmode.dof_array.DOFArray` or an
        :class:`~arraycontext.ArrayContainer` of them.
    :arg dd_in: a :class:`~grudge.dof_desc.DOFDesc`, or a value convertible to one.
        Defaults to the base volume discretization if not provided.
    :arg nested: return nested object arrays instead of a single multidimensional
        array if *vec* is non-scalar.
    :returns: an object array (possibly nested) of
        :class:`~meshmode.dof_array.DOFArray`\ s or
        :class:`~arraycontext.ArrayContainer` of object arrays.
    """
    if len(args) == 1:
        vec, = args
        dd_in = DD_VOLUME_ALL
    elif len(args) == 2:
        dd_in, vec = args
    else:
        raise TypeError("invalid number of arguments")

    from grudge.tools import rec_map_subarrays
    return rec_map_subarrays(
        partial(_strong_scalar_grad, dcoll, dd_in),
        (), (dcoll.ambient_dim,),
        vec, scalar_cls=DOFArray, return_nested=nested,)


def local_d_dx(
        dcoll: DiscretizationCollection, xyz_axis, *args) -> ArrayOrContainer:
    r"""Return the element-local derivative along axis *xyz_axis* of a
    function :math:`f` represented by *vec*:

    .. math::

        \frac{\partial f}{\partial \lbrace x,y,z\rbrace}\Big|_E

    May be called with ``(vec)`` or ``(dd, vec)``.

    :arg xyz_axis: an integer indicating the axis along which the derivative
        is taken.
    :arg dd: a :class:`~grudge.dof_desc.DOFDesc`, or a value convertible to one.
        Defaults to the base volume discretization if not provided.
    :arg vec: a :class:`~meshmode.dof_array.DOFArray` or an
        :class:`~arraycontext.ArrayContainer` of them.
    :returns: a :class:`~meshmode.dof_array.DOFArray` or an
        :class:`~arraycontext.ArrayContainer` of them.
    """
    if len(args) == 1:
        vec, = args
        dd = DD_VOLUME_ALL
    elif len(args) == 2:
        dd, vec = args
    else:
        raise TypeError("invalid number of arguments")

    if not isinstance(vec, DOFArray):
        return map_array_container(partial(local_d_dx, dcoll, xyz_axis, dd), vec)

    discr = dcoll.discr_from_dd(dd)
    actx = vec.array_context

    from grudge.geometry import inverse_surface_metric_derivative_mat
    inverse_jac_mat = inverse_surface_metric_derivative_mat(actx, dcoll, dd=dd,
        _use_geoderiv_connection=actx.supports_nonscalar_broadcasting)

    return _single_axis_derivative_kernel(
        actx, discr, discr,
        _reference_derivative_matrices, inverse_jac_mat, xyz_axis, vec,
        metric_in_matvec=False)


def local_div(dcoll: DiscretizationCollection, *args) -> ArrayOrContainer:
    r"""Return the element-local divergence of the vector function
    :math:`\mathbf{f}` represented by *vecs*:

    .. math::

        \nabla|_E \cdot \mathbf{f} = \sum_{i=1}^d \partial_{x_i}|_E \mathbf{f}_i

    May be called with ``(vecs)`` or ``(dd, vecs)``.

    :arg dd: a :class:`~grudge.dof_desc.DOFDesc`, or a value convertible to one.
        Defaults to the base volume discretization if not provided.
    :arg vecs: an object array of
        :class:`~meshmode.dof_array.DOFArray`\s or an
        :class:`~arraycontext.ArrayContainer` object
        with object array entries. The last axis of the array
        must have length matching the volume dimension.
    :returns: a :class:`~meshmode.dof_array.DOFArray` or an
        :class:`~arraycontext.ArrayContainer` of them.
    """
    if len(args) == 1:
        vecs, = args
        dd = DD_VOLUME_ALL
    elif len(args) == 2:
        dd, vecs = args
    else:
        raise TypeError("invalid number of arguments")

    from grudge.tools import rec_map_subarrays
    return rec_map_subarrays(
        lambda vec: _strong_scalar_div(dcoll, dd, vec),
        (dcoll.ambient_dim,), (),
        vecs, scalar_cls=DOFArray)

# }}}


# {{{ Weak derivative operators

def _reference_stiffness_transpose_matrices(
        actx: ArrayContext, out_element_group, in_element_group):
    @keyed_memoize_in(
        actx, _reference_stiffness_transpose_matrices,
        lambda out_grp, in_grp: (out_grp.discretization_key(),
                                 in_grp.discretization_key()))
    def get_ref_stiffness_transpose_mat(out_grp, in_grp):
        if in_grp == out_grp:
            from meshmode.discretization.poly_element import \
                mass_matrix, diff_matrices

            # {{{ tensor product case

            if isinstance(out_grp, TensorProductElementGroupBase):
                import modepy as mp
                import numpy.linalg as la

                # FIXME: can be gotten rid of by updating meshmode operators
                basis_1d = out_grp.bases_1d()
                nodes_1d = out_grp.unit_nodes_1d

                vdm = mp.vandermonde(basis_1d.functions, nodes_1d)
                vdm_p = mp.vandermonde(basis_1d.gradients, nodes_1d)[0]

                mass_1D = la.inv(vdm @ vdm.T)
                diff_mat = la.solve(vdm.T, vdm_p.T).T

                stiff_1D = actx.freeze(
                        actx.tag_axis(1, DiscretizationDOFAxisTag(),
                                      actx.from_numpy(
                                      np.asarray(
                                          diff_mat.T @ mass_1D.T))))

                mass_1D = actx.freeze(
                        actx.tag_axis(1, DiscretizationDOFAxisTag(),
                                      actx.from_numpy(
                                          np.asarray(
                                              mass_1D))))

                return (stiff_1D, mass_1D)

            # }}}

            mmat = mass_matrix(out_grp)

            return actx.freeze(
                actx.tag_axis(1, DiscretizationDOFAxisTag(),
                    actx.from_numpy(
                        np.asarray(
                            [dmat.T @ mmat.T for dmat in diff_matrices(out_grp)]))))

        from modepy import vandermonde
        basis = out_grp.basis_obj()
        vand = vandermonde(basis.functions, out_grp.unit_nodes)
        grad_vand = vandermonde(basis.gradients, in_grp.unit_nodes)
        vand_inv_t = np.linalg.inv(vand).T

        if not isinstance(grad_vand, tuple):
            # NOTE: special case for 1d
            grad_vand = (grad_vand,)

        weights = in_grp.quadrature_rule().weights
        return actx.freeze(
            actx.from_numpy(
                np.einsum(
                    "c,bz,acz->abc",
                    weights,
                    vand_inv_t,
                    grad_vand
                ).copy()  # contigify the array
            )
        )
    return get_ref_stiffness_transpose_mat(out_element_group,
                                           in_element_group)


def _weak_scalar_grad(dcoll, dd_in, vec):
    from grudge.geometry import inverse_surface_metric_derivative_mat

    dd_in = as_dofdesc(dd_in)
    in_discr = dcoll.discr_from_dd(dd_in)
    out_discr = dcoll.discr_from_dd(dd_in.with_discr_tag(DISCR_TAG_BASE))

    actx = vec.array_context
    inverse_jac_mat = inverse_surface_metric_derivative_mat(actx, dcoll, dd=dd_in,
            times_area_element=True,
            _use_geoderiv_connection=actx.supports_nonscalar_broadcasting)

    return _gradient_kernel(actx, out_discr, in_discr,
            _reference_stiffness_transpose_matrices, inverse_jac_mat, vec,
            metric_in_matvec=True)


def _weak_scalar_div(dcoll, dd_in, vecs):
    from grudge.geometry import inverse_surface_metric_derivative_mat
    from arraycontext import (get_container_context_recursively,
                              serialize_container)

    assert isinstance(vecs, np.ndarray)
    assert vecs.shape == (dcoll.ambient_dim,)

    in_discr = dcoll.discr_from_dd(dd_in)
    out_discr = dcoll.discr_from_dd(dd_in.with_discr_tag(DISCR_TAG_BASE))

    actx = get_container_context_recursively(vecs)
    vec = actx.np.stack([v for k, v in serialize_container(vecs)])

    inverse_jac_mat = inverse_surface_metric_derivative_mat(actx, dcoll, dd=dd_in,
            times_area_element=True,
            _use_geoderiv_connection=actx.supports_nonscalar_broadcasting)

    return _divergence_kernel(actx, out_discr, in_discr,
            _reference_stiffness_transpose_matrices, inverse_jac_mat, vec,
            metric_in_matvec=True)


def weak_local_grad(
        dcoll: DiscretizationCollection, *args, nested=False) -> ArrayOrContainer:
    r"""Return the element-local weak gradient of the volume function
    represented by *vec*.

    May be called with ``(vec)`` or ``(dd_in, vec)``.

    Specifically, the function returns an object array where the :math:`i`-th
    component is the weak derivative with respect to the :math:`i`-th coordinate
    of a scalar function :math:`f`. See :func:`weak_local_d_dx` for further
    information. For non-scalar :math:`f`, the function will return a nested object
    array containing the component-wise weak derivatives.

    :arg dd_in: a :class:`~grudge.dof_desc.DOFDesc`, or a value convertible to one.
        Defaults to the base volume discretization if not provided.
    :arg vec: a :class:`~meshmode.dof_array.DOFArray` or an
        :class:`~arraycontext.ArrayContainer` of them.
    :arg nested: return nested object arrays instead of a single multidimensional
        array if *vec* is non-scalar
    :returns: an object array (possibly nested) of
        :class:`~meshmode.dof_array.DOFArray`\ s or
        :class:`~arraycontext.ArrayContainer` of object arrays.
    """
    if len(args) == 1:
        vecs, = args
        dd_in = DD_VOLUME_ALL
    elif len(args) == 2:
        dd_in, vecs = args
    else:
        raise TypeError("invalid number of arguments")

    from grudge.tools import rec_map_subarrays
    return rec_map_subarrays(
        partial(_weak_scalar_grad, dcoll, dd_in),
        (), (dcoll.ambient_dim,),
        vecs, scalar_cls=DOFArray, return_nested=nested)


def weak_local_d_dx(dcoll: DiscretizationCollection, *args) -> ArrayOrContainer:
    r"""Return the element-local weak derivative along axis *xyz_axis* of the
    volume function represented by *vec*.

    May be called with ``(xyz_axis, vec)`` or ``(dd_in, xyz_axis, vec)``.

    Specifically, this function computes the volume contribution of the
    weak derivative in the :math:`i`-th component (specified by *xyz_axis*)
    of a function :math:`f`, in each element :math:`E`, with respect to polynomial
    test functions :math:`\phi`:

    .. math::

        \int_E \partial_i\phi\,f\,\mathrm{d}x \sim
        \mathbf{D}_{E,i}^T \mathbf{M}_{E}^T\mathbf{f}|_E,

    where :math:`\mathbf{D}_{E,i}` is the polynomial differentiation matrix on
    an :math:`E` for the :math:`i`-th spatial coordinate, :math:`\mathbf{M}_E`
    is the elemental mass matrix (see :func:`mass` for more information), and
    :math:`\mathbf{f}|_E` is a vector of coefficients for :math:`f` on :math:`E`.

    :arg dd_in: a :class:`~grudge.dof_desc.DOFDesc`, or a value convertible to one.
        Defaults to the base volume discretization if not provided.
    :arg xyz_axis: an integer indicating the axis along which the derivative
        is taken.
    :arg vec: a :class:`~meshmode.dof_array.DOFArray` or an
        :class:`~arraycontext.ArrayContainer` of them.
    :returns: a :class:`~meshmode.dof_array.DOFArray` or an
        :class:`~arraycontext.ArrayContainer` of them.
    """
    if len(args) == 2:
        xyz_axis, vec = args
        dd_in = dof_desc.DD_VOLUME_ALL
    elif len(args) == 3:
        dd_in, xyz_axis, vec = args
    else:
        raise TypeError("invalid number of arguments")

    if not isinstance(vec, DOFArray):
        return map_array_container(
            partial(weak_local_d_dx, dcoll, dd_in, xyz_axis),
            vec
        )

    from grudge.geometry import inverse_surface_metric_derivative_mat

    dd_in = as_dofdesc(dd_in)
    in_discr = dcoll.discr_from_dd(dd_in)
    out_discr = dcoll.discr_from_dd(dd_in.with_discr_tag(DISCR_TAG_BASE))

    actx = vec.array_context
    inverse_jac_mat = inverse_surface_metric_derivative_mat(actx, dcoll, dd=dd_in,
            times_area_element=True,
            _use_geoderiv_connection=actx.supports_nonscalar_broadcasting)

    return _single_axis_derivative_kernel(
            actx, out_discr, in_discr, _reference_stiffness_transpose_matrices,
            inverse_jac_mat, xyz_axis, vec,
            metric_in_matvec=True)


def weak_local_div(dcoll: DiscretizationCollection, *args) -> ArrayOrContainer:
    r"""Return the element-local weak divergence of the vector volume function
    represented by *vecs*.

    May be called with ``(vecs)`` or ``(dd, vecs)``.

    Specifically, this function computes the volume contribution of the
    weak divergence of a vector function :math:`\mathbf{f}`, in each element
    :math:`E`, with respect to polynomial test functions :math:`\phi`:

    .. math::

        \int_E \nabla \phi \cdot \mathbf{f}\,\mathrm{d}x \sim
        \sum_{i=1}^d \mathbf{D}_{E,i}^T \mathbf{M}_{E}^T\mathbf{f}_i|_E,

    where :math:`\mathbf{D}_{E,i}` is the polynomial differentiation matrix on
    an :math:`E` for the :math:`i`-th spatial coordinate, and :math:`\mathbf{M}_E`
    is the elemental mass matrix (see :func:`mass` for more information).

    :arg dd: a :class:`~grudge.dof_desc.DOFDesc`, or a value convertible to one.
        Defaults to the base volume discretization if not provided.
    :arg vecs: an object array of
        :class:`~meshmode.dof_array.DOFArray`\s or an
        :class:`~arraycontext.ArrayContainer` object
        with object array entries. The last axis of the array
        must have length matching the volume dimension.
    :returns: a :class:`~meshmode.dof_array.DOFArray` or an
        :class:`~arraycontext.ArrayContainer` like *vec*.
    """
    if len(args) == 1:
        vecs, = args
        dd_in = DD_VOLUME_ALL
    elif len(args) == 2:
        dd_in, vecs = args
    else:
        raise TypeError("invalid number of arguments")

    from grudge.tools import rec_map_subarrays
    return rec_map_subarrays(
        lambda vec: _weak_scalar_div(dcoll, dd_in, vec),
        (dcoll.ambient_dim,), (),
        vecs, scalar_cls=DOFArray)

# }}}


# {{{ Mass operator

def reference_mass_matrix(actx: ArrayContext, out_element_group, in_element_group):
    @keyed_memoize_in(
        actx, reference_mass_matrix,
        lambda out_grp, in_grp: (out_grp.discretization_key(),
                                 in_grp.discretization_key()))
    def get_ref_mass_mat(out_grp, in_grp):
        if out_grp == in_grp:
            from meshmode.discretization.poly_element import mass_matrix

            return actx.freeze(
                actx.from_numpy(
                    np.asarray(
                        mass_matrix(out_grp),
                        order="C"
                    )
                )
            )

        from modepy import vandermonde
        basis = out_grp.basis_obj()
        vand = vandermonde(basis.functions, out_grp.unit_nodes)
        o_vand = vandermonde(basis.functions, in_grp.unit_nodes)
        vand_inv_t = np.linalg.inv(vand).T

        weights = in_grp.quadrature_rule().weights
        return actx.freeze(
                actx.tag_axis(0, DiscretizationDOFAxisTag(),
                    actx.from_numpy(
                        np.asarray(
                            np.einsum("j,ik,jk->ij", weights, vand_inv_t, o_vand),
                            order="C"))))

    return get_ref_mass_mat(out_element_group, in_element_group)


def _apply_mass_operator(
        dcoll: DiscretizationCollection, dd_out, dd_in, vec):
    if not isinstance(vec, DOFArray):
        return map_array_container(
            partial(_apply_mass_operator, dcoll, dd_out, dd_in), vec
        )

    from grudge.geometry import area_element

    in_discr = dcoll.discr_from_dd(dd_in)
    out_discr = dcoll.discr_from_dd(dd_out)

    actx = vec.array_context
    area_elements = area_element(actx, dcoll, dd=dd_in,
            _use_geoderiv_connection=actx.supports_nonscalar_broadcasting)
    return DOFArray(
        actx,
        data=tuple(
            actx.einsum("ij,ej,ej->ei",
                reference_mass_matrix(
                    actx,
                    out_element_group=out_grp,
                    in_element_group=in_grp
                    ),
                ae_i,
                vec_i,
                arg_names=("mass_mat", "jac", "vec"),
                tagged=(FirstAxisIsElementsTag(),))

            for in_grp, out_grp, ae_i, vec_i in zip(
                    in_discr.groups, out_discr.groups, area_elements, vec)
        )
    )


def mass(dcoll: DiscretizationCollection, *args) -> ArrayOrContainer:
    r"""Return the action of the DG mass matrix on a vector (or vectors)
    of :class:`~meshmode.dof_array.DOFArray`\ s, *vec*. In the case of
    *vec* being an :class:`~arraycontext.ArrayContainer`,
    the mass operator is applied component-wise.

    May be called with ``(vec)`` or ``(dd_in, vec)``.

    Specifically, this function applies the mass matrix elementwise on a
    vector of coefficients :math:`\mathbf{f}` via:
    :math:`\mathbf{M}_{E}\mathbf{f}|_E`, where

    .. math::

        \left(\mathbf{M}_{E}\right)_{ij} = \int_E \phi_i \cdot \phi_j\,\mathrm{d}x,

    where :math:`\phi_i` are local polynomial basis functions on :math:`E`.

    :arg dd_in: a :class:`~grudge.dof_desc.DOFDesc`, or a value convertible to one.
        Defaults to the base volume discretization if not provided.
    :arg vec: a :class:`~meshmode.dof_array.DOFArray` or an
        :class:`~arraycontext.ArrayContainer` of them.
    :returns: a :class:`~meshmode.dof_array.DOFArray` or an
        :class:`~arraycontext.ArrayContainer` like *vec*.
    """

    if len(args) == 1:
        vec, = args
        dd_in = dof_desc.DD_VOLUME_ALL
    elif len(args) == 2:
        dd_in, vec = args
    else:
        raise TypeError("invalid number of arguments")

    dd_out = dd_in.with_discr_tag(DISCR_TAG_BASE)

    return _apply_mass_operator(dcoll, dd_out, dd_in, vec)

# }}}


# {{{ Mass inverse operator

def reference_inverse_mass_matrix(actx: ArrayContext, element_group):
    @keyed_memoize_in(
        actx, reference_inverse_mass_matrix,
        lambda grp: grp.discretization_key())
    def get_ref_inv_mass_mat(grp):
        from modepy import inverse_mass_matrix
        basis = grp.basis_obj()

        return actx.freeze(
            actx.tag_axis(0, DiscretizationDOFAxisTag(),
                actx.from_numpy(
                    np.asarray(
                        inverse_mass_matrix(basis.functions, grp.unit_nodes),
                        order="C"))))

    return get_ref_inv_mass_mat(element_group)


def _apply_inverse_mass_operator(
        dcoll: DiscretizationCollection, dd_out, dd_in, vec):
    if not isinstance(vec, DOFArray):
        return map_array_container(
            partial(_apply_inverse_mass_operator, dcoll, dd_out, dd_in), vec
        )

    from grudge.geometry import area_element

    if dd_out != dd_in:
        raise ValueError(
            "Cannot compute inverse of a mass matrix mapping "
            "between different element groups; inverse is not "
            "guaranteed to be well-defined"
        )

    actx = vec.array_context
    discr = dcoll.discr_from_dd(dd_in)
    inv_area_elements = 1./area_element(actx, dcoll, dd=dd_in,
            _use_geoderiv_connection=actx.supports_nonscalar_broadcasting)
    group_data = [
            # Based on https://arxiv.org/pdf/1608.03836.pdf
            # true_Minv ~ ref_Minv * ref_M * (1/jac_det) * ref_Minv
            actx.einsum("ei,ij,ej->ei",
                        jac_inv,
                        reference_inverse_mass_matrix(actx, element_group=grp),
                        vec_i,
                        tagged=(FirstAxisIsElementsTag(),))
            for grp, jac_inv, vec_i in zip(discr.groups, inv_area_elements, vec)]

    return DOFArray(actx, data=tuple(group_data))


def inverse_mass(dcoll: DiscretizationCollection, *args) -> ArrayOrContainer:
    r"""Return the action of the DG mass matrix inverse on a vector
    (or vectors) of :class:`~meshmode.dof_array.DOFArray`\ s, *vec*.
    In the case of *vec* being an :class:`~arraycontext.ArrayContainer`,
    the inverse mass operator is applied component-wise.

    For affine elements :math:`E`, the element-wise mass inverse
    is computed directly as the inverse of the (physical) mass matrix:

    .. math::

        \left(\mathbf{M}_{J^e}\right)_{ij} =
            \int_{\widehat{E}} \widehat{\phi}_i\cdot\widehat{\phi}_j J^e
            \mathrm{d}\widehat{x},

    where :math:`\widehat{\phi}_i` are basis functions over the reference
    element :math:`\widehat{E}`, and :math:`J^e` is the (constant) Jacobian
    scaling factor (see :func:`grudge.geometry.area_element`).

    For non-affine :math:`E`, :math:`J^e` is not constant. In this case, a
    weight-adjusted approximation is used instead following [Chan_2016]_:

    .. math::

        \mathbf{M}_{J^e}^{-1} \approx
            \widehat{\mathbf{M}}^{-1}\mathbf{M}_{1/J^e}\widehat{\mathbf{M}}^{-1},

    where :math:`\widehat{\mathbf{M}}` is the reference mass matrix on
    :math:`\widehat{E}`.

    May be called with ``(vec)`` or ``(dd, vec)``.

    :arg vec: a :class:`~meshmode.dof_array.DOFArray` or an
        :class:`~arraycontext.ArrayContainer` of them.
    :arg dd: a :class:`~grudge.dof_desc.DOFDesc`, or a value convertible to one.
        Defaults to the base volume discretization if not provided.
    :returns: a :class:`~meshmode.dof_array.DOFArray` or an
        :class:`~arraycontext.ArrayContainer` like *vec*.
    """
    if len(args) == 1:
        vec, = args
        dd = DD_VOLUME_ALL
    elif len(args) == 2:
        dd, vec = args
    else:
        raise TypeError("invalid number of arguments")

    return _apply_inverse_mass_operator(dcoll, dd, dd, vec)

# }}}


# {{{ Face mass operator

def reference_face_mass_matrix(
        actx: ArrayContext, face_element_group, vol_element_group, dtype):
    @keyed_memoize_in(
        actx, reference_mass_matrix,
        lambda face_grp, vol_grp: (face_grp.discretization_key(),
                                   vol_grp.discretization_key()))
    def get_ref_face_mass_mat(face_grp, vol_grp):
        nfaces = vol_grp.mesh_el_group.nfaces
        assert face_grp.nelements == nfaces * vol_grp.nelements

        matrix = np.empty(
            (vol_grp.nunit_dofs,
            nfaces,
            face_grp.nunit_dofs),
            dtype=dtype
        )

        import modepy as mp
        from meshmode.discretization import ElementGroupWithBasis
        from meshmode.discretization.poly_element import \
            QuadratureSimplexElementGroup

        n = vol_grp.order
        m = face_grp.order
        vol_basis = vol_grp.basis_obj()
        faces = mp.faces_for_shape(vol_grp.shape)

        for iface, face in enumerate(faces):
            # If the face group is defined on a higher-order
            # quadrature grid, use the underlying quadrature rule
            if isinstance(face_grp, QuadratureSimplexElementGroup):
                face_quadrature = face_grp.quadrature_rule()
                if face_quadrature.exact_to < m:
                    raise ValueError(
                        "The face quadrature rule is only exact for polynomials "
                        f"of total degree {face_quadrature.exact_to}. Please "
                        "ensure a quadrature rule is used that is at least "
                        f"exact for degree {m}."
                    )
            else:
                # NOTE: This handles the general case where
                # volume and surface quadrature rules may have different
                # integration orders
                face_quadrature = mp.quadrature_for_space(
                    mp.space_for_shape(face, 2*max(n, m)),
                    face
                )

            # If the group has a nodal basis and is unisolvent,
            # we use the basis on the face to compute the face mass matrix
            if (isinstance(face_grp, ElementGroupWithBasis)
                    and face_grp.space.space_dim == face_grp.nunit_dofs):

                face_basis = face_grp.basis_obj()

                # Sanity check for face quadrature accuracy. Not integrating
                # degree N + M polynomials here is asking for a bad time.
                if face_quadrature.exact_to < m + n:
                    raise ValueError(
                        "The face quadrature rule is only exact for polynomials "
                        f"of total degree {face_quadrature.exact_to}. Please "
                        "ensure a quadrature rule is used that is at least "
                        f"exact for degree {n+m}."
                    )

                matrix[:, iface, :] = mp.nodal_mass_matrix_for_face(
                    face, face_quadrature,
                    face_basis.functions, vol_basis.functions,
                    vol_grp.unit_nodes,
                    face_grp.unit_nodes,
                )
            else:
                # Otherwise, we use a routine that is purely quadrature-based
                # (no need for explicit face basis functions)
                matrix[:, iface, :] = mp.nodal_quad_mass_matrix_for_face(
                    face,
                    face_quadrature,
                    vol_basis.functions,
                    vol_grp.unit_nodes,
                )

        return actx.freeze(
                tag_axes(actx, {
                    0: DiscretizationDOFAxisTag(),
                    2: DiscretizationDOFAxisTag()
                    },
                    actx.from_numpy(matrix)))

    return get_ref_face_mass_mat(face_element_group, vol_element_group)


def _apply_face_mass_operator(dcoll: DiscretizationCollection, dd_in, vec):
    if not isinstance(vec, DOFArray):
        return map_array_container(
            partial(_apply_face_mass_operator, dcoll, dd_in), vec
        )

    from grudge.geometry import area_element

    dd_out = DOFDesc(
        VolumeDomainTag(dd_in.domain_tag.volume_tag),
        DISCR_TAG_BASE)

    volm_discr = dcoll.discr_from_dd(dd_out)
    face_discr = dcoll.discr_from_dd(dd_in)
    dtype = vec.entry_dtype
    actx = vec.array_context

    assert len(face_discr.groups) == len(volm_discr.groups)
    surf_area_elements = area_element(actx, dcoll, dd=dd_in,
            _use_geoderiv_connection=actx.supports_nonscalar_broadcasting)

    return DOFArray(
        actx,
        data=tuple(
            actx.einsum("ifj,fej,fej->ei",
                        reference_face_mass_matrix(
                            actx,
                            face_element_group=afgrp,
                            vol_element_group=vgrp,
                            dtype=dtype),
                        actx.tag_axis(1, DiscretizationElementAxisTag(),
                            surf_ae_i.reshape(
                                vgrp.mesh_el_group.nfaces,
                                vgrp.nelements,
                                surf_ae_i.shape[-1])),
                        actx.tag_axis(0, DiscretizationFaceAxisTag(),
                            vec_i.reshape(
                                vgrp.mesh_el_group.nfaces,
                                vgrp.nelements,
                                afgrp.nunit_dofs)),
                        arg_names=("ref_face_mass_mat", "jac_surf", "vec"),
                        tagged=(FirstAxisIsElementsTag(),))

            for vgrp, afgrp, vec_i, surf_ae_i in zip(volm_discr.groups,
                                                     face_discr.groups,
                                                     vec,
                                                     surf_area_elements)))


def face_mass(dcoll: DiscretizationCollection, *args) -> ArrayOrContainer:
    r"""Return the action of the DG face mass matrix on a vector (or vectors)
    of :class:`~meshmode.dof_array.DOFArray`\ s, *vec*. In the case of
    *vec* being an arbitrary :class:`~arraycontext.ArrayContainer`,
    the face mass operator is applied component-wise.

    May be called with ``(vec)`` or ``(dd_in, vec)``.

    Specifically, this function applies the face mass matrix elementwise on a
    vector of coefficients :math:`\mathbf{f}` as the sum of contributions for
    each face :math:`f \subset \partial E`:

    .. math::

        \sum_{f=1}^{N_{\text{faces}} } \mathbf{M}_{f, E}\mathbf{f}|_f,

    where

    .. math::

        \left(\mathbf{M}_{f, E}\right)_{ij} =
            \int_{f \subset \partial E} \phi_i(s)\psi_j(s)\,\mathrm{d}s,

    where :math:`\phi_i` are (volume) polynomial basis functions on :math:`E`
    evaluated on the face :math:`f`, and :math:`\psi_j` are basis functions for
    a polynomial space defined on :math:`f`.

    :arg dd: a :class:`~grudge.dof_desc.DOFDesc`, or a value convertible to one.
        Defaults to the base ``"all_faces"`` discretization if not provided.
    :arg vec: a :class:`~meshmode.dof_array.DOFArray` or an
        :class:`~arraycontext.ArrayContainer` of them.
    :returns: a :class:`~meshmode.dof_array.DOFArray` or an
        :class:`~arraycontext.ArrayContainer` like *vec*.
    """

    if len(args) == 1:
        vec, = args
        dd_in = DD_VOLUME_ALL.trace(FACE_RESTR_ALL)
    elif len(args) == 2:
        dd_in, vec = args
    else:
        raise TypeError("invalid number of arguments")

    return _apply_face_mass_operator(dcoll, dd_in, vec)

# }}}


# vim: foldmethod=marker
