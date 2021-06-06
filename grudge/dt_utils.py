"""Helper functions for estimating stable time steps for RKDG methods.

.. autofunction:: dt_non_geometric_factors
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


import numpy as np

from arraycontext import FirstAxisIsElementsTag

from grudge.dof_desc import DD_VOLUME, DOFDesc, as_dofdesc
from grudge.discretization import DiscretizationCollection

import grudge.op as op

from meshmode.dof_array import DOFArray

from pytools import memoize_on_first_arg


@memoize_on_first_arg
def dt_non_geometric_factors(
        dcoll: DiscretizationCollection, dd=None) -> list:
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
        dd = DD_VOLUME

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
        dcoll: DiscretizationCollection, dim=None, dd=None) -> float:
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
        dd = DD_VOLUME
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
        dcoll: DiscretizationCollection, dim=None, dd=None) -> float:
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
        dd = DD_VOLUME
    dd = as_dofdesc(dd)

    if dim is None:
        dim = dcoll.dim

    ones = dcoll.discr_from_dd(dd).zeros(dcoll._setup_actx) + 1.0
    return nodal_min(
        dcoll,
        dd,
        elementwise_sum(dcoll, op.mass(dcoll, dd, ones))
    ) ** (1.0 / dim)


@memoize_on_first_arg
def dt_geometric_factors(
        dcoll: DiscretizationCollection, dd=None) -> DOFArray:
    r"""Computes a geometric scaling factor for each cell following [Hesthaven_2008]_,
    section 6.4, defined as the inradius (radius of an inscribed circle/sphere).

    Specifically, the inradius for each element is computed using the following
    formula from [Shewchuk_2002]_, Table 1, for simplicial cells
    (triangles/tetrahedra):

    .. math::

        r_D = \frac{d V}{\sum_{i=1}^{N_{faces}} F_i},

    where :math:`d` is the topological dimension, :math:`V` is the cell volume,
    and :math:`F_i` are the areas of each face of the cell.

    :arg dd: a :class:`~grudge.dof_desc.DOFDesc`, or a value convertible to one.
        Defaults to the base volume discretization if not provided.
    :returns: a :class:`~meshmode.dof_array.DOFArray` containing the geometric
        factors for each cell and at each nodal location.
    """
    from meshmode.discretization.poly_element import SimplexElementGroupBase

    if dd is None:
        dd = DD_VOLUME

    actx = dcoll._setup_actx
    volm_discr = dcoll.discr_from_dd(dd)

    if any(not isinstance(grp, SimplexElementGroupBase)
           for grp in volm_discr.groups):
        raise NotImplementedError(
            "Geometric factors are only implemented for simplex element groups"
        )

    cell_vols = abs(
        op.elementwise_integral(
            dcoll, dd, volm_discr.zeros(actx) + 1.0
        )
    )

    if dcoll.dim == 1:
        return cell_vols

    dd_face = DOFDesc("all_faces", dd.discretization_tag)
    face_discr = dcoll.discr_from_dd(dd_face)

    # To get a single value for the total surface area of a cell, we
    # take the sum over the averaged face areas on each face.
    # NOTE: The face areas are the *same* at each face nodal location.
    # This assumes there are the *same* number of face nodes on each face.
    surface_areas = abs(
        op.elementwise_integral(
            dcoll, dd_face, face_discr.zeros(actx) + 1.0
        )
    )
    surface_areas = DOFArray(
        actx,
        data=tuple(
            actx.einsum("fej->e",
                        face_ae_i.reshape(
                            vgrp.mesh_el_group.nfaces,
                            vgrp.nelements,
                            afgrp.nunit_dofs
                        ),
                        tagged=(FirstAxisIsElementsTag(),)) / afgrp.nunit_dofs

            for vgrp, afgrp, face_ae_i in zip(volm_discr.groups,
                                              face_discr.groups,
                                              surface_areas)
        )
    )

    return DOFArray(
        actx,
        data=tuple(
            actx.einsum("e,ei->ei",
                        1/sae_i,
                        cv_i,
                        tagged=(FirstAxisIsElementsTag(),)) * dcoll.dim

            for cv_i, sae_i in zip(cell_vols, surface_areas)
        )
    )

# }}}


# vim: foldmethod=marker
