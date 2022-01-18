"""
.. currentmodule:: grudge.op

Projections
-----------

.. autofunction:: project
"""

__copyright__ = """
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


from functools import partial

from arraycontext import ArrayContext, map_array_container
from arraycontext.container import ArrayOrContainerT

from grudge.discretization import DiscretizationCollection
from grudge.dof_desc import as_dofdesc

from meshmode.dof_array import DOFArray
from meshmode.transform_metadata import FirstAxisIsElementsTag

from pytools import keyed_memoize_in

from numbers import Number


def project(
        dcoll: DiscretizationCollection, src, tgt, vec) -> ArrayOrContainerT:
    """Project from one discretization to another, e.g. from the
    volume to the boundary, or from the base to the an overintegrated
    quadrature discretization.

    :arg src: a :class:`~grudge.dof_desc.DOFDesc`, or a value convertible to one.
    :arg tgt: a :class:`~grudge.dof_desc.DOFDesc`, or a value convertible to one.
    :arg vec: a :class:`~meshmode.dof_array.DOFArray` or an
        :class:`~arraycontext.container.ArrayContainer` of them.
    :returns: a :class:`~meshmode.dof_array.DOFArray` or an
        :class:`~arraycontext.container.ArrayContainer` like *vec*.
    """
    src = as_dofdesc(src)
    tgt = as_dofdesc(tgt)

    if isinstance(vec, Number) or src == tgt:
        return vec

    if not isinstance(vec, DOFArray):
        return map_array_container(
            partial(project, dcoll, src, tgt), vec
        )

    return dcoll.connection_from_dds(src, tgt)(vec)


# {{{ Projection matrices

def volume_quadrature_l2_projection_matrix(
        actx: ArrayContext, base_element_group, vol_quad_element_group):
    """todo.
    """
    @keyed_memoize_in(
        actx, volume_quadrature_l2_projection_matrix,
        lambda base_grp, vol_quad_grp: (base_grp.discretization_key(),
                                        vol_quad_grp.discretization_key()))
    def get_ref_l2_proj_mat(base_grp, vol_quad_grp):
        from grudge.interpolation import volume_quadrature_interpolation_matrix
        from grudge.op import reference_inverse_mass_matrix

        vdm_q = actx.to_numpy(
            volume_quadrature_interpolation_matrix(
                actx, base_grp, vol_quad_grp
            )
        )
        weights = vol_quad_grp.quadrature_rule().weights
        inv_mass_mat = actx.to_numpy(reference_inverse_mass_matrix(actx, base_grp))
        return actx.freeze(actx.from_numpy(inv_mass_mat @ (vdm_q.T * weights)))

    return get_ref_l2_proj_mat(base_element_group, vol_quad_element_group)

# }}}


def volume_quadrature_project(dcoll: DiscretizationCollection, dd_q, vec):
    """todo.
    """
    if not isinstance(vec, DOFArray):
        return map_array_container(
            partial(volume_quadrature_project, dcoll, dd_q), vec
        )

    actx = vec.array_context
    discr = dcoll.discr_from_dd("vol")
    quad_discr = dcoll.discr_from_dd(dd_q)

    return DOFArray(
        actx,
        data=tuple(
            actx.einsum("ij,ej->ei",
                        volume_quadrature_l2_projection_matrix(
                            actx,
                            base_element_group=bgrp,
                            vol_quad_element_group=qgrp
                        ),
                        vec_i,
                        arg_names=("Pq_mat", "vec"),
                        tagged=(FirstAxisIsElementsTag(),))

            for bgrp, qgrp, vec_i in zip(
                discr.groups,
                quad_discr.groups,
                vec
            )
        )
    )

# vim: foldmethod=marker
