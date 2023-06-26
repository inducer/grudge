"""
.. currentmodule:: grudge.op

Projections
-----------

.. autofunction:: project
.. autofunction:: volume_quadrature_project
"""

from __future__ import annotations

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
from numbers import Number

import numpy as np

from arraycontext import ArrayOrContainer, map_array_container

from grudge.discretization import DiscretizationCollection
from grudge.dof_desc import (
    as_dofdesc,
    VolumeDomainTag,
    BoundaryDomainTag,
    ConvertibleToDOFDesc)

from meshmode.dof_array import DOFArray
from meshmode.transform_metadata import FirstAxisIsElementsTag

from pytools import keyed_memoize_in


def project(
        dcoll: DiscretizationCollection,
        src: "ConvertibleToDOFDesc",
        tgt: "ConvertibleToDOFDesc", vec) -> ArrayOrContainer:
    """Project from one discretization to another, e.g. from the
    volume to the boundary, or from the base to the an overintegrated
    quadrature discretization.

    :arg src: a :class:`~grudge.dof_desc.DOFDesc`, or a value convertible to one.
    :arg tgt: a :class:`~grudge.dof_desc.DOFDesc`, or a value convertible to one.
    :arg vec: a :class:`~meshmode.dof_array.DOFArray` or an
        :class:`~arraycontext.ArrayContainer` of them.
    :returns: a :class:`~meshmode.dof_array.DOFArray` or an
        :class:`~arraycontext.ArrayContainer` like *vec*.
    """
    # {{{ process dofdesc arguments

    src_dofdesc = as_dofdesc(src)

    contextual_volume_tag = None
    if isinstance(src_dofdesc.domain_tag, VolumeDomainTag):
        contextual_volume_tag = src_dofdesc.domain_tag.tag
    elif isinstance(src_dofdesc.domain_tag, BoundaryDomainTag):
        contextual_volume_tag = src_dofdesc.domain_tag.volume_tag

    tgt_dofdesc = as_dofdesc(tgt, _contextual_volume_tag=contextual_volume_tag)

    del src
    del tgt

    # }}}

    if isinstance(vec, Number) or src_dofdesc == tgt_dofdesc:
        return vec

    return dcoll.connection_from_dds(src_dofdesc, tgt_dofdesc)(vec)


def volume_quadrature_project(
        dcoll: DiscretizationCollection, dd_q, vec) -> ArrayOrContainer:
    """Projects a field on the quadrature discreization, described by *dd_q*,
    into the polynomial space described by the volume discretization.

    :arg dd_q: a :class:`~grudge.dof_desc.DOFDesc`, or a value convertible to one.
    :arg vec: a :class:`~meshmode.dof_array.DOFArray` or an
        :class:`~arraycontext.container.ArrayContainer` of them.
    :returns: a :class:`~meshmode.dof_array.DOFArray` or an
        :class:`~arraycontext.container.ArrayContainer` like *vec*.
    """
    if not isinstance(vec, DOFArray):
        return map_array_container(
            partial(volume_quadrature_project, dcoll, dd_q), vec
        )

    from grudge.geometry import area_element
    from grudge.interpolation import volume_quadrature_interpolation_matrix
    from grudge.op import inverse_mass

    actx = vec.array_context
    discr = dcoll.discr_from_dd("vol")
    quad_discr = dcoll.discr_from_dd(dd_q)
    jacobians = area_element(
        actx, dcoll, dd=dd_q,
        _use_geoderiv_connection=actx.supports_nonscalar_broadcasting)

    @keyed_memoize_in(
        actx, volume_quadrature_project,
        lambda base_grp, vol_quad_grp: (base_grp.discretization_key(),
                                        vol_quad_grp.discretization_key()))
    def get_mat(base_grp, vol_quad_grp):
        vdm_q = actx.to_numpy(
            volume_quadrature_interpolation_matrix(
                actx, base_grp, vol_quad_grp
            )
        )
        weights = np.diag(vol_quad_grp.quadrature_rule().weights)
        return actx.freeze(actx.from_numpy(vdm_q.T @ weights))

    return inverse_mass(
        dcoll,
        DOFArray(
            actx,
            data=tuple(
                actx.einsum("ij,ej,ej->ei",
                            get_mat(bgrp, qgrp),
                            jac_i,
                            vec_i,
                            arg_names=("vqw_t", "jac", "vec"),
                            tagged=(FirstAxisIsElementsTag(),))
                for bgrp, qgrp, vec_i, jac_i in zip(
                    discr.groups, quad_discr.groups, vec, jacobians)
            )
        )
    )
