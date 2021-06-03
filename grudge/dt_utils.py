"""Helper functions for estimating stable time steps for RKDG methods.

.. autofunction:: dt_non_geometric_factor
.. autofunction:: dt_geometric_factor
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

from grudge.dof_desc import DD_VOLUME, DOFDesc
from grudge.discretization import DiscretizationCollection

import grudge.op as op

from meshmode.dof_array import DOFArray

from pytools import memoize_on_first_arg


@memoize_on_first_arg
def dt_non_geometric_factor(
        dcoll: DiscretizationCollection, scaling=None, dd=None) -> float:
    r"""Computes the non-geometric scale factor following [Hesthaven_2008]_,
    section 6.4:

    .. math::

        c_{ng} = \operatorname{min}\left( \Delta r_i \right),

    where :math:`\Delta r_i` denotes the distance between two distinct
    nodes on the reference element.

    :arg scaling: a :class:`float` denoting the scaling factor. By default,
        the constant is set to 2/3.
    :arg dd: a :class:`~grudge.dof_desc.DOFDesc`, or a value convertible to one.
        Defaults to the base volume discretization if not provided.
    :returns: a :class:`float` denoting the minimum node distance on the
        reference element.
    """
    if dd is None:
        dd = DD_VOLUME

    if scaling is None:
        scaling = 2/3

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

    # Return minimum over all element groups in the discretization
    return scaling * min(min_delta_rs)


@memoize_on_first_arg
def dt_geometric_factor(dcoll: DiscretizationCollection, dd=None) -> float:
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
    :returns: a :class:`float` denoting the geometric scaling factor.
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

    # NOTE: The cell volumes are the *same* at each nodal location.
    # Take the cell vols at each nodal location and average them to get
    # a single value per cell.
    cell_vols = op.elementwise_integral(
        dcoll, dd, volm_discr.zeros(actx) + 1.0
    )
    cell_vols = DOFArray(
        actx,
        data=tuple(
            actx.einsum("ei->e",
                        cv_i,
                        tagged=(FirstAxisIsElementsTag(),)) / vgrp.nunit_dofs

            for vgrp, cv_i in zip(volm_discr.groups,
                                  actx.np.fabs(cell_vols))
        )
    )

    if dcoll.dim == 1:
        return op.nodal_min(dcoll, dd, cell_vols)

    dd_face = DOFDesc("all_faces", dd.discretization_tag)
    face_discr = dcoll.discr_from_dd(dd_face)

    # To get a single value for the total surface area of a cell, we
    # take the sum over the averaged face areas on each face.
    # NOTE: The face areas are the *same* at each face nodal location.
    # This assumes there are the *same* number of face nodes on each face.
    face_areas = op.elementwise_integral(
        dcoll, dd_face, face_discr.zeros(actx) + 1.0
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
                                              actx.np.fabs(face_areas))
        )
    )

    return op.nodal_min(dcoll, dd, dcoll.dim * cell_vols / surface_areas)
