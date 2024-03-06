"""Helper functions for estimating stable time steps for RKDG methods.

Characteristic lengthscales
---------------------------

.. autofunction:: characteristic_lengthscales

Non-geometric quantities
------------------------

.. autofunction:: dt_non_geometric_factors

Mesh size utilities
-------------------

.. autofunction:: dt_geometric_factors
.. autofunction:: h_max_from_volume
.. autofunction:: h_min_from_volume
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


from typing import Optional, Sequence
import numpy as np

from arraycontext import ArrayContext, Scalar, tag_axes
from arraycontext.metadata import NameHint
from meshmode.transform_metadata import (FirstAxisIsElementsTag,
                                         DiscretizationDOFAxisTag,
                                         DiscretizationFaceAxisTag,
                                         DiscretizationElementAxisTag)

from grudge.dof_desc import (
        DD_VOLUME_ALL, DOFDesc, as_dofdesc, BoundaryDomainTag, FACE_RESTR_ALL)
from grudge.discretization import DiscretizationCollection

import grudge.op as op

from meshmode.dof_array import DOFArray

from pytools import memoize_on_first_arg, memoize_in


def characteristic_lengthscales(
        actx: ArrayContext, dcoll: DiscretizationCollection,
        dd: Optional[DOFDesc] = None) -> DOFArray:
    r"""Computes the characteristic length scale :math:`h_{\text{loc}}` at
    each node. The characteristic length scale is mainly useful for estimating
    the stable time step size. E.g. for a hyperbolic system, an estimate of the
    stable time step can be estimated as :math:`h_{\text{loc}} / c`, where
    :math:`c` is the characteristic wave speed. The estimate is obtained using
    the following formula:

    .. math::

        h_{\text{loc}} = \operatorname{min}\left(\Delta r_i\right) r_D

    where :math:`\operatorname{min}\left(\Delta r_i\right)` is the minimum
    node distance on the reference cell (see :func:`dt_non_geometric_factors`),
    and :math:`r_D` is the inradius of the cell (see :func:`dt_geometric_factors`).

    :returns: a :class:`~meshmode.dof_array.DOFArray` containing a characteristic
        lengthscale for each element, at each nodal location.

    .. note::

        While a prediction of stability is only meaningful in relation to a given
        time integrator with a known stability region, the lengthscale provided here
        is not intended to be specific to any one time integrator, though the
        stability region of standard four-stage, fourth-order Runge-Kutta
        methods has been used as a guide. Any concrete time integrator will
        likely require scaling of the values returned by this routine.
    """
    @memoize_in(dcoll, (characteristic_lengthscales, dd,
                        "compute_characteristic_lengthscales"))
    def _compute_characteristic_lengthscales():
        return actx.freeze(
                actx.tag(NameHint("char_lscales"),
                    DOFArray(
                        actx,
                        data=tuple(
                            # Scale each group array of geometric factors by the
                            # corresponding group non-geometric factor
                            cng * geo_facts
                            for cng, geo_facts in zip(
                                dt_non_geometric_factors(dcoll, dd),
                                actx.thaw(dt_geometric_factors(dcoll, dd)))))))

    return actx.thaw(_compute_characteristic_lengthscales())


@memoize_on_first_arg
def dt_non_geometric_factors(
        dcoll: DiscretizationCollection, dd: Optional[DOFDesc] = None
        ) -> Sequence[float]:
    r"""Computes the non-geometric scale factors following [Hesthaven_2008]_,
    section 6.4, for each element group in the *dd* discretization:

    .. math::

        c_{ng} = \operatorname{min}\left( \Delta r_i \right),

    where :math:`\Delta r_i` denotes the distance between two distinct
    nodal points on the reference element.

    :arg dd: a :class:`~grudge.dof_desc.DOFDesc`, or a value convertible to one.
        Defaults to the base volume discretization if not provided.
    :returns: a :class:`list` of :class:`float` values denoting the minimum
        node distance on the reference element for each group.
    """
    if dd is None:
        dd = DD_VOLUME_ALL

    discr = dcoll.discr_from_dd(dd)
    min_delta_rs = []
    for grp in discr.groups:
        nodes = np.asarray(list(zip(*grp.unit_nodes)))
        nnodes = grp.nunit_dofs

        # NOTE: order 0 elements have 1 node located at the centroid of
        # the reference element and is equidistant from each vertex
        if grp.order == 0:
            assert nnodes == 1
            min_delta_rs.append(
                np.linalg.norm(
                    nodes[0] - grp.mesh_el_group.vertex_unit_coordinates()[0]
                )
            )
        else:
            min_delta_rs.append(
                min(
                    np.linalg.norm(nodes[i] - nodes[j])
                    for i in range(nnodes) for j in range(nnodes) if i != j
                )
            )

    return min_delta_rs


# {{{ Mesh size functions

@memoize_on_first_arg
def h_max_from_volume(
        dcoll: DiscretizationCollection, dim=None,
        dd: Optional[DOFDesc] = None) -> Scalar:
    """Returns a (maximum) characteristic length based on the volume of the
    elements. This length may not be representative if the elements have very
    high aspect ratios.

    :arg dim: an integer denoting topological dimension. If *None*, the
        spatial dimension specified by
        :attr:`grudge.DiscretizationCollection.dim` is used.
    :arg dd: a :class:`~grudge.dof_desc.DOFDesc`, or a value convertible to one.
        Defaults to the base volume discretization if not provided.
    :returns: a scalar denoting the maximum characteristic length.
    """
    from grudge.reductions import nodal_max, elementwise_sum

    if dd is None:
        dd = DD_VOLUME_ALL
    dd = as_dofdesc(dd)

    if dim is None:
        dim = dcoll.dim

    ones = dcoll.discr_from_dd(dd).zeros(dcoll._setup_actx) + 1.0
    return nodal_max(
        dcoll,
        dd,
        elementwise_sum(dcoll, op.mass(dcoll, dd, ones))
    ) ** (1.0 / dim)


@memoize_on_first_arg
def h_min_from_volume(
        dcoll: DiscretizationCollection, dim=None,
        dd: Optional[DOFDesc] = None) -> Scalar:
    """Returns a (minimum) characteristic length based on the volume of the
    elements. This length may not be representative if the elements have very
    high aspect ratios.

    :arg dim: an integer denoting topological dimension. If *None*, the
        spatial dimension specified by
        :attr:`grudge.DiscretizationCollection.dim` is used.
    :arg dd: a :class:`~grudge.dof_desc.DOFDesc`, or a value convertible to one.
        Defaults to the base volume discretization if not provided.
    :returns: a scalar denoting the minimum characteristic length.
    """
    from grudge.reductions import nodal_min, elementwise_sum

    if dd is None:
        dd = DD_VOLUME_ALL
    dd = as_dofdesc(dd)

    if dim is None:
        dim = dcoll.dim

    ones = dcoll.discr_from_dd(dd).zeros(dcoll._setup_actx) + 1.0
    return nodal_min(
        dcoll,
        dd,
        elementwise_sum(dcoll, op.mass(dcoll, dd, ones))
    ) ** (1.0 / dim)


def dt_geometric_factors(
        dcoll: DiscretizationCollection, dd: Optional[DOFDesc] = None) -> DOFArray:
    r"""Computes a geometric scaling factor for each cell following
    [Hesthaven_2008]_, section 6.4, For simplicial elemenents, this factor is
    defined as the inradius (radius of an inscribed circle/sphere). For
    non-simplicial elements, a mean length measure is returned.

    Specifically, the inradius for each simplicial element is computed using the
    following formula from [Shewchuk_2002]_, Table 1 (triangles, tetrahedra):

    .. math::

        r_D = \frac{d~V}{\sum_{i=1}^{N_{faces}} F_i},

    where :math:`d` is the topological dimension, :math:`V` is the cell volume,
    and :math:`F_i` are the areas of each face of the cell.

    For non-simplicial elements, we use the following formula for a mean
    cell size measure:

    .. math::

        r_D = \frac{2~d~V}{\sum_{i=1}^{N_{faces}} F_i},

    where :math:`d` is the topological dimension, :math:`V` is the cell volume,
    and :math:`F_i` are the areas of each face of the cell. Other valid choices
    here include the shortest, longest, average of the cell diagonals, or edges.
    The value returned by this routine (i.e. the cell volume divided by the
    average cell face area) is bounded by the extrema of the cell edge lengths,
    is straightforward to calculate regardless of element shape, and jibes well
    with the foregoing calculation for simplicial elements.

    :arg dd: a :class:`~grudge.dof_desc.DOFDesc`, or a value convertible to one.
        Defaults to the base volume discretization if not provided.
    :returns: a frozen :class:`~meshmode.dof_array.DOFArray` containing the
        geometric factors for each cell and at each nodal location.
    """
    from meshmode.discretization.poly_element import SimplexElementGroupBase

    if dd is None:
        dd = DD_VOLUME_ALL

    actx = dcoll._setup_actx
    volm_discr = dcoll.discr_from_dd(dd)

    r_fac = dcoll.dim
    if any(not isinstance(grp, SimplexElementGroupBase)
           for grp in volm_discr.groups):
        r_fac = 2.0*r_fac

    if volm_discr.dim != volm_discr.ambient_dim:
        from warnings import warn
        warn("The geometric factor for the characteristic length scale in "
                "time step estimation is not necessarily valid for non-volume-"
                "filling discretizations. Continuing anyway.", stacklevel=3)

    cell_vols = abs(
        op.elementwise_integral(
            dcoll, dd, volm_discr.zeros(actx) + 1.0
        )
    )

    if dcoll.dim == 1:
        # Inscribed "circle" radius is half the cell size
        return actx.freeze(cell_vols/2)

    dd_face = dd.with_domain_tag(
            BoundaryDomainTag(FACE_RESTR_ALL, dd.domain_tag.tag))
    face_discr = dcoll.discr_from_dd(dd_face)

    # Compute areas of each face
    face_areas = abs(
        op.elementwise_integral(
            dcoll, dd_face, face_discr.zeros(actx) + 1.0
        )
    )

    if actx.supports_nonscalar_broadcasting:
        # Compute total surface area of an element by summing over the
        # individual face areas
        surface_areas = DOFArray(
            actx,
            data=tuple(
                actx.einsum(
                    "fej->e",
                    tag_axes(actx, {
                        0: DiscretizationFaceAxisTag(),
                        1: DiscretizationElementAxisTag(),
                        2: DiscretizationDOFAxisTag()
                        },
                        face_ae_i.reshape(
                            vgrp.mesh_el_group.nfaces,
                            vgrp.nelements,
                            face_ae_i.shape[-1])),
                    tagged=(FirstAxisIsElementsTag(),))

                for vgrp, face_ae_i in zip(volm_discr.groups, face_areas)))
    else:
        surface_areas = DOFArray(
            actx,
            data=tuple(
                # NOTE: Whenever the array context can't perform nonscalar
                # broadcasting, elementwise reductions
                # (like `elementwise_integral`) repeat the *same* scalar value of
                # the reduction at each degree of freedom. To get a single
                # value for the face area (per face),
                # we simply average over the nodes, which gives the desired result.
                actx.einsum(
                    "fej->e",
                    face_ae_i.reshape(
                        vgrp.mesh_el_group.nfaces,
                        vgrp.nelements,
                        face_ae_i.shape[-1]
                    ) / afgrp.nunit_dofs,
                    tagged=(FirstAxisIsElementsTag(),))

                for vgrp, afgrp, face_ae_i in zip(volm_discr.groups,
                                                  face_discr.groups,
                                                  face_areas)
            )
        )

    return actx.freeze(
            actx.tag(NameHint(f"dt_geometric_{dd.as_identifier()}"),
                DOFArray(actx,
                    data=tuple(
                        actx.einsum(
                            "e,ei->ei",
                            1/sae_i,
                            actx.tag_axis(1, DiscretizationDOFAxisTag(), cv_i),
                            tagged=(FirstAxisIsElementsTag(),)) * r_fac
                        for cv_i, sae_i in zip(cell_vols, surface_areas)))))

# }}}


# vim: foldmethod=marker
