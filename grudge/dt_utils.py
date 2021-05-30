"""Helper functions for estimating stable time steps for RKDG methods.

.. autofunction:: dt_non_geometric_factor
.. autofunction:: symmetric_eigenvalues
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

from arraycontext import ArrayContext, rec_map_array_container

from functools import reduce

from grudge.dof_desc import DD_VOLUME
from grudge.geometry import first_fundamental_form
from grudge.discretization import DiscretizationCollection

from pytools import memoize_on_first_arg


@memoize_on_first_arg
def dt_non_geometric_factor(dcoll: DiscretizationCollection, dd=None) -> float:
    r"""Computes the non-geometric scale factor:

    .. math::

        \frac{2}{3}\operatorname{min}_i\left( \Delta r_i \right),

    where :math:`\Delta r_i` denotes the distance between two distinct
    nodes on the reference element.

    :arg dd: a :class:`~grudge.dof_desc.DOFDesc`, or a value convertible to one.
        Defaults to the base volume discretization if not provided.
    :returns: a :class:`float` denoting the minimum node distance on the
        reference element.
    """
    if dd is None:
        dd = DD_VOLUME

    discr = dcoll.discr_from_dd(dd)
    min_delta_rs = []
    for mgrp in discr.mesh.groups:
        nodes = np.asarray(list(zip(*mgrp.unit_nodes)))
        nnodes = mgrp.nunit_nodes

        # NOTE: order 0 elements have 1 node located at the centroid of
        # the reference element and is equidistant from each vertex
        if mgrp.order == 0:
            assert nnodes == 1
            min_delta_rs.append(
                2/3 * np.linalg.norm(nodes[0] - mgrp.vertex_unit_coordinates()[0])
            )
        else:
            min_delta_rs.append(
                2/3 * min(
                    np.linalg.norm(nodes[i] - nodes[j])
                    for i in range(nnodes) for j in range(nnodes) if i != j
                )
            )

    # Return minimum over all element groups in the discretization
    return min(min_delta_rs)


def symmetric_eigenvalues(actx: ArrayContext, amat):
    """Analytically computes the eigenvalues of a self-adjoint matrix, up
    to matrices of size 3 by 3.

    :arg amat: a square array-like object.
    :returns: a :class:`list` of the eigenvalues of *amat*.

    .. note::

        *amat* must be complex-valued, or ``actx.np.sqrt`` must automatically
        up-cast to complex data.
    """

    # https://gist.github.com/inducer/75ede170638c389c387e72e0ef1f0ef4
    sqrt = actx.np.sqrt

    if amat.shape == (1, 1):
        (a,), = amat
        return a
    elif amat.shape == (2, 2):
        (a, b), (_b, c) = amat
        x0 = sqrt(a**2 - 2*a*c + 4*b**2 + c**2)/2
        x1 = a/2 + c/2

        return [-x0 + x1,
                x0 + x1]
    elif amat.shape == (3, 3):
        # This is likely awful numerically, but *shrug*, we're only using
        # it for time step estimation.
        (a, b, c), (_b, d, e), (_c, _e, f) = amat
        x0 = a*d
        x1 = f*x0
        x2 = b*c*e
        x3 = e**2
        x4 = a*x3
        x5 = b**2
        x6 = f*x5
        x7 = c**2
        x8 = d*x7
        x9 = -a - d - f
        x10 = x9**3
        x11 = a*f
        x12 = d*f
        x13 = (-9*a - 9*d - 9*f)*(x0 + x11 + x12 - x3 - x5 - x7)
        x14 = -3*x0 - 3*x11 - 3*x12 + 3*x3 + 3*x5 + 3*x7 + x9**2
        x15_0 = (-4*x14**3
                 + (-27*x1 + 2*x10 - x13 - 54*x2 + 27*x4 + 27*x6 + 27*x8)**2)
        x15_1 = sqrt(x15_0)
        x15_2 = (-27*x1/2 + x10 - x13/2 - 27*x2 + 27*x4/2 + 27*x6/2 + 27*x8/2
                 + x15_1/2)
        x15 = x15_2**(1/3)
        x16 = x15/3
        x17 = x14/(3*x15)
        x18 = a/3 + d/3 + f/3
        x19 = 3**(1/2)*1j/2
        x20 = x19 - 1/2
        x21 = -x19 - 1/2

        return [-x16 - x17 + x18,
                -x16*x20 - x17/x20 + x18,
                -x16*x21 - x17/x21 + x18]
    else:
        raise NotImplementedError(
            "Unsupported shape ({amat.shape}) for analytically computing eigenvalues"
        )


@memoize_on_first_arg
def dt_geometric_factor(dcoll: DiscretizationCollection, dd=None) -> float:
    """Computes a geometric scaling factor, determined by taking the minimum
    singular value of the coordinate transformation from reference to physical
    cells.

    :arg dd: a :class:`~grudge.dof_desc.DOFDesc`, or a value convertible to one.
        Defaults to the base volume discretization if not provided.
    :returns: a :class:`float` denoting the geometric scaling factor.
    """
    if dd is None:
        dd = DD_VOLUME

    actx = dcoll._setup_actx
    ata = first_fundamental_form(actx, dcoll, dd=dd)

    complex_dtype = dcoll.discr_from_dd(dd).complex_dtype

    ata_complex = rec_map_array_container(
        lambda ary: ary.astype(complex_dtype), ata
    )

    sing_values = [actx.np.sqrt(abs(v))
                   for v in symmetric_eigenvalues(actx, ata_complex)]

    return reduce(actx.np.minimum, sing_values)
