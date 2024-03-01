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

from meshmode.dof_array import DOFArray, warn
from meshmode.discretization.poly_element import (
    TensorProductElementGroupBase as TensorProductElementGroup,
    SimplexElementGroupBase as SimplexElementGroup)
from meshmode.transform_metadata import (FirstAxisIsElementsTag,
                                         DiscretizationDOFAxisTag,
                                         DiscretizationElementAxisTag,
                                         DiscretizationFaceAxisTag)

from modepy.tools import (
        reshape_array_for_tensor_product_space as fold,
        unreshape_array_for_tensor_product_space as unfold)

from grudge.discretization import DiscretizationCollection
from grudge.dof_desc import as_dofdesc
from grudge.array_context import (
    OutputIsTensorProductDOFArrayOrdered,
    TensorProductDOFAxis
)

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

    def compute_tensor_product_derivative(actx, grp, get_diff_mat, vec, ijm,
                                          xyz_axis, metric_in_matvec):

        vec = tag_axes(
            actx,
            { i: TensorProductDOFAxis() for i in range(1, grp.dim) },
            fold(grp.space, vec)
        )


        if metric_in_matvec:
            stiff_1d, mass_1d = get_diff_mat(actx, grp, grp)

            apply_mass_axes = set(range(grp.dim)) - {xyz_axis}

            for ax in apply_mass_axes:
                vec_mass_applied = single_axis_operator_application(
                    actx, grp.dim, mass_1d, ax, vec,
                    tagged=(FirstAxisIsElementsTag(),
                            OutputIsTensorProductDOFArrayOrdered(),),
                    arg_names=("mass_1d", "vec")
                )

            ref_weak_derivative = unfold(
                grp.space,
                single_axis_operator_application(
                    actx, grp.dim, stiff_1d, xyz_axis, vec_mass_applied,
                    tagged=(FirstAxisIsElementsTag(),
                            OutputIsTensorProductDOFArrayOrdered(),),
                    arg_names=("stiff_1d", "vec_with_mass_applied"))
            )

            derivative = actx.einsum(
                'rej,ej->ej',
                ijm[xyz_axis],
                ref_weak_derivative,
                tagged=(FirstAxisIsElementsTag(),),
                arg_names=("inv_jac_t", "ref_weak_derivative")
            )

        else:
            diff_mat = get_diff_mat(actx, grp, grp)

            ref_derivative = unfold(
                grp.space,
                single_axis_operator_application(
                    actx, grp.dim, diff_mat, xyz_axis, vec,
                    tagged=(FirstAxisIsElementsTag(),
                            OutputIsTensorProductDOFArrayOrdered(),),
                    arg_names=("diff_mat", "vec"))
            )

            derivative = actx.einsum(
                'rej,ej->ej',
                ijm[xyz_axis],
                ref_derivative,
                tagged=(FirstAxisIsElementsTag(),),
                arg_names=("inv_jac_t", "ref_derivs")
            )

        return derivative

    # }}}


    # {{{ simplicial single axis derivative

    def compute_simplicial_derivative(actx, in_grp, out_grp,
                                      get_diff_mat, vec, ijm,
                                      xyz_axis, metric_in_matvec):
        # r for rst axis
        return actx.einsum(
            "rej,rij,ej->ei" if metric_in_matvec else "rei,rij,ej->ei",
            ijm[xyz_axis],
            get_diff_mat(
                actx,
                out_element_group=out_grp,
                in_element_group=in_grp),
            vec,
            arg_names=("inv_jac_t", "ref_stiffT_mat", "vec", ),
            tagged=(FirstAxisIsElementsTag(),))

    # }}}


    return DOFArray(
        actx,
        data=tuple(
            compute_tensor_product_derivative(actx, in_grp, get_diff_mat, vec_i,
                                              ijm_i, xyz_axis, metric_in_matvec)
            if isinstance(in_grp, TensorProductElementGroup)
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
        Applies 1D operators one-axis-at-a-time to tensor-product discretized
        DOF data.
        """

        # reshape vector to expose tensor product structure
        vec = tag_axes(
            actx,
            { i: TensorProductDOFAxis() for i in range(1, grp.dim+1) },
            fold(grp.space, vec)
        )

        if metric_in_matvec:
            stiff_1d, mass_1d = get_diff_mat(actx, grp, grp)

            grad = []
            for xyz_axis in range(grp.dim):
                grad.append(vec)
                apply_mass_axes = set(range(grp.dim)) - {xyz_axis}

                # apply mass operators
                for ax in apply_mass_axes:
                    grad[xyz_axis] = single_axis_operator_application(
                        actx, grp.dim, mass_1d, ax, grad[xyz_axis],
                        tagged=(FirstAxisIsElementsTag(),
                                OutputIsTensorProductDOFArrayOrdered(),),
                        arg_names=("mass_1d", f"vec_{xyz_axis}")
                )

                # apply stiffness operator and unfold
                grad[xyz_axis] = unfold(
                    grp.space,
                    single_axis_operator_application(
                        actx, grp.dim, stiff_1d, xyz_axis, grad[xyz_axis],
                        tagged=(FirstAxisIsElementsTag(),
                                OutputIsTensorProductDOFArrayOrdered(),),
                        arg_names=("stiff_1d", f"vec_{xyz_axis}"))
                )

                # apply metric terms
                grad[xyz_axis] = actx.einsum(
                    'rej,ej->ej',
                    ijm[xyz_axis],
                    grad[xyz_axis],
                    tagged=(FirstAxisIsElementsTag(),),
                    arg_names=("inv_jac_t", f"vec_{xyz_axis}")
                )
        else:
            diff_mat = get_diff_mat(actx, grp, grp)

            grad = []
            for xyz_axis in range(grp.dim):
                grad.append(vec)
                grad[xyz_axis] = unfold(
                    grp.space,
                    single_axis_operator_application(
                        actx, grp.dim, diff_mat, xyz_axis, grad[xyz_axis],
                        tagged=(FirstAxisIsElementsTag(),
                                OutputIsTensorProductDOFArrayOrdered(),),
                        arg_names=("diff_mat", f"vec_{xyz_axis}")
                    )
                )

                grad[xyz_axis] = actx.einsum(
                    "rej,ej->ej",
                    ijm[xyz_axis],
                    grad[xyz_axis],
                    tagged=(FirstAxisIsElementsTag(),),
                    arg_names=("inv_jac_t", f"vec_{xyz_axis}")
                )

        return make_obj_array(grad)

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
        if isinstance(in_grp, TensorProductElementGroup)
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
        """
        Exploits tensor product structure to reduce complexity. See
        `_gradient_kernel.compute_tensor_product_grad` for more details.
        """

        vec = make_obj_array([
            tag_axes(
                actx,
                { i: TensorProductDOFAxis() for i in range(1,grp.dim+1) },
                fold(grp.space, vec[func_axis])
            )
            for func_axis in range(vec.shape[0])
        ])

        if metric_in_matvec:
            stiff_1d, mass_1d = get_diff_mat(actx, grp, grp)

            partials = []
            for func_axis in range(vec.shape[0]):
                ref = []
                for xyz_axis in range(grp.dim):
                    ref.append(vec[func_axis])

                    apply_mass_axes = set(range(grp.dim)) - {xyz_axis}
                    for ax in apply_mass_axes:
                        ref[xyz_axis] = single_axis_operator_application(
                            actx, grp.dim, mass_1d, ax, ref[xyz_axis],
                            tagged=(FirstAxisIsElementsTag(),
                                    OutputIsTensorProductDOFArrayOrdered(),),
                            arg_names=("mass_1d", f"vec_{func_axis}_{xyz_axis}")
                        )

                    ref[xyz_axis] = single_axis_operator_application(
                        actx, grp.dim, stiff_1d, xyz_axis, ref[xyz_axis],
                        tagged=(FirstAxisIsElementsTag(),
                                OutputIsTensorProductDOFArrayOrdered(),),
                        arg_names=("stiff_1d", f"vec_{func_axis}_{xyz_axis}")
                    )

                partials.append(ref)

        else:
            diff_mat = get_diff_mat(actx, grp, grp)

            partials = []
            for func_axis in range(vec.shape[0]):
                ref = []
                for xyz_axis in range(grp.dim):
                    ref.append(vec[func_axis])

                    ref[xyz_axis] = single_axis_operator_application(
                        actx, grp.dim, diff_mat, xyz_axis, ref[xyz_axis],
                        tagged=(FirstAxisIsElementsTag(),
                                OutputIsTensorProductDOFArrayOrdered(),),
                        arg_names=("diff_mat", f"vec_{func_axis}_{xyz_axis}")
                    )

                partials.append(ref)

        partials = actx.np.stack([
            unfold(grp.space, partials[func_axis][xyz_axis])
            for func_axis in range(grp.dim)
            for xyz_axis in range(grp.dim)
        ])
        partials = partials.reshape(grp.dim, grp.dim, *partials.shape[-2:])

        div = actx.einsum(
            'xrej,xrej->ej',
            ijm,
            partials,
            arg_names=("inv_jac_t", "partials"),
            tagged=(FirstAxisIsElementsTag(),)
        )

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
        if isinstance(in_grp, TensorProductElementGroup)

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
        if isinstance(grp, TensorProductElementGroup):
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

        elif isinstance(grp, SimplexElementGroup):
            from meshmode.discretization.poly_element import diff_matrices
            return actx.freeze(
                    actx.tag_axis(
                        1, DiscretizationDOFAxisTag(),
                        actx.from_numpy(
                            np.asarray(diff_matrices(grp)))))

        else:
            raise TypeError("grp must be either a TensorProductElementGroup or"
                            f" a SimplexElementGroup. Found {grp}")

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

            if isinstance(out_grp, TensorProductElementGroup):
                import modepy as mp
                import numpy.linalg as la

                # FIXME: can be gotten rid of by updating meshmode operators
                basis_1d = out_grp.bases_1d()
                nodes_1d = out_grp.unit_nodes_1d

                vdm = mp.vandermonde(basis_1d.functions, nodes_1d)
                vdm_p = mp.vandermonde(basis_1d.gradients, nodes_1d)[0]

                mass_1d = la.inv(vdm @ vdm.T)
                diff_mat = la.solve(vdm.T, vdm_p.T).T

                stiff_1d = actx.freeze(
                        actx.tag_axis(1, DiscretizationDOFAxisTag(),
                                      actx.from_numpy(
                                      np.asarray(
                                          diff_mat.T @ mass_1d.T))))

                from grudge.array_context import MassMatrix1d
                mass_1d = actx.freeze(
                    actx.tag_axis(
                        1, (DiscretizationDOFAxisTag(),),
                        actx.from_numpy(np.asarray(mass_1d)))
                )
                mass_1d = actx.tag(MassMatrix1d(), mass_1d)

                return (stiff_1d, mass_1d)

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

        if isinstance(grp, TensorProductElementGroup):
            basis_1d = grp.bases_1d()
            nodes_1d = grp.unit_nodes_1d
            inv_mass_1d = inverse_mass_matrix(basis_1d.functions, nodes_1d)

            from grudge.array_context import InverseMassMatrix1d
            inv_mass_1d = actx.tag_axis(0, DiscretizationDOFAxisTag(),
                                        actx.from_numpy(np.asarray(inv_mass_1d)))
            inv_mass_1d = actx.freeze(
                actx.tag(InverseMassMatrix1d(), inv_mass_1d))

            return inv_mass_1d
        elif isinstance(grp, SimplexElementGroup):
            basis = grp.basis_obj()

            return actx.freeze(
                actx.tag_axis(0, DiscretizationDOFAxisTag(),
                    actx.from_numpy(
                        np.asarray(
                            inverse_mass_matrix(basis.functions, grp.unit_nodes),
                            order="C"))))
        else:
            raise TypeError("grp must be either a TensorProductElementGroup or"
                            f" a SimplexElementGroup. Found {grp}")

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


    def apply_to_tensor_product_elements(grp, jac_inv, vec, ref_inv_mass):

        vec = fold(grp.space, vec)

        for xyz_axis in range(grp.dim):
            vec = single_axis_operator_application(
                actx, grp.dim, ref_inv_mass, xyz_axis, vec,
                tagged=(FirstAxisIsElementsTag(),
                        OutputIsTensorProductDOFArrayOrdered(),),
                arg_names=("ref_inv_mass_1d", "vec"))

        vec = unfold(grp.space, vec)

        return actx.einsum(
            "ei,ei->ei",
            jac_inv,
            vec,
            tagged=(FirstAxisIsElementsTag(),)
        )


    def apply_to_simplicial_elements(jac_inv, vec, ref_inv_mass):

        # Based on https://arxiv.org/pdf/1608.03836.pdf
        # true_Minv ~ ref_Minv * ref_M * (1/jac_det) * ref_Minv
        return actx.einsum(
            "ei,ij,ej->ei",
            jac_inv,
            ref_inv_mass,
            vec,
            tagged=(FirstAxisIsElementsTag(),))


    group_data = [
        apply_to_tensor_product_elements(
            grp, jac_inv, vec_i,
            reference_inverse_mass_matrix(actx, element_group=grp))
        if isinstance(grp, TensorProductElementGroup) else
        apply_to_simplicial_elements(jac_inv, vec_i,
            reference_inverse_mass_matrix(actx, element_group=grp))
        for grp, jac_inv, vec_i in zip(discr.groups, inv_area_elements, vec)
    ]

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


# {{{ general single axis operator application

def single_axis_operator_application(actx, dim, operator, axis, data,
                                     arg_names=None, tagged=None):
    """
    Used for applying 1D operators to a single axis of a tensor of DOF data.
    """

    if not isinstance(arg_names, tuple):
        raise TypeError("arg_names must be a tuple.")
    if not isinstance(tagged, tuple):
        raise TypeError("tagged must be a tuple.")

    operator_spec = 'ij'
    data_spec = f'e{"abcdefghklm"[:axis]}j{"nopqrstuvwxyz"[:dim-axis-1]}'
    out_spec = f'e{"abcdefghklm"[:axis]}i{"nopqrstuvwxyz"[:dim-axis-1]}'

    spec = operator_spec + ',' + data_spec + '->' + out_spec

    result = tag_axes(
        actx,
        { i: TensorProductDOFAxis() for i in range(1, dim+1) },
        actx.einsum(spec, operator, data,
                       arg_names=arg_names,
                       tagged=tagged)
    )

    return result

# }}}


# vim: foldmethod=marker
