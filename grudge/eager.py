"""
.. autoclass:: EagerDGDiscretization
"""
__copyright__ = "Copyright (C) 2020 Andreas Kloeckner"

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

import grudge.op as op

from arraycontext import ArrayContext
from grudge.discretization import DiscretizationCollection
from meshmode.mesh import Mesh


class EagerDGDiscretization(DiscretizationCollection):
    """
    This class is deprecated and only part of the documentation in order to
    avoid breaking depending documentation builds.
    Use :class:`~grudge.discretization.DiscretizationCollection` instead in
    new code.
    """

    def __init__(self, array_context: ArrayContext, mesh: Mesh,
                 order=None,
                 discr_tag_to_group_factory=None,
                 mpi_communicator=None,
                 # FIXME: `quad_tag_to_group_factory` is deprecated
                 quad_tag_to_group_factory=None):
        from warnings import warn
        warn("EagerDGDiscretization is deprecated and will go away in 2022. "
                "Use the base DiscretizationCollection with grudge.op "
                "instead.",
                DeprecationWarning, stacklevel=2)

        if (quad_tag_to_group_factory is not None
                and discr_tag_to_group_factory is not None):
            raise ValueError(
                "Both `quad_tag_to_group_factory` and `discr_tag_to_group_factory` "
                "are specified. Use `discr_tag_to_group_factory` instead."
            )

        # FIXME: `quad_tag_to_group_factory` is deprecated
        if (quad_tag_to_group_factory is not None
                and discr_tag_to_group_factory is None):
            warn("`quad_tag_to_group_factory` is a deprecated kwarg and will "
                 "be dropped in version 2022.x. Use `discr_tag_to_group_factory` "
                 "instead.",
                 DeprecationWarning, stacklevel=2)
            discr_tag_to_group_factory = quad_tag_to_group_factory

        from meshmode.discretization.poly_element import \
            PolynomialWarpAndBlendGroupFactory

        from grudge.dof_desc import DISCR_TAG_BASE, DISCR_TAG_MODAL

        if discr_tag_to_group_factory is None:
            if order is None:
                raise TypeError(
                    "one of 'order' and 'discr_tag_to_group_factory' must be given"
                )

            # Default choice: warp and blend simplex element group
            discr_tag_to_group_factory = {
                DISCR_TAG_BASE: PolynomialWarpAndBlendGroupFactory(order=order)
            }
        else:
            if order is not None:
                discr_tag_to_group_factory = discr_tag_to_group_factory.copy()
                if DISCR_TAG_BASE in discr_tag_to_group_factory:
                    raise ValueError(
                        "if 'order' is given, 'discr_tag_to_group_factory' must "
                        "not have a key of DISCR_TAG_BASE"
                    )

                discr_tag_to_group_factory[DISCR_TAG_BASE] = \
                    PolynomialWarpAndBlendGroupFactory(order=order)

        # Modal discr should always comes from the base discretization
        from grudge.discretization import _generate_modal_group_factory

        discr_tag_to_group_factory[DISCR_TAG_MODAL] = \
            _generate_modal_group_factory(
                discr_tag_to_group_factory[DISCR_TAG_BASE]
            )

        # Define the base discretization
        from meshmode.discretization import Discretization

        volume_discr = Discretization(
            array_context, mesh,
            discr_tag_to_group_factory[DISCR_TAG_BASE]
        )

        # Define boundary connections
        from grudge.discretization import set_up_distributed_communication

        dist_boundary_connections = set_up_distributed_communication(
            array_context, mesh,
            volume_discr,
            discr_tag_to_group_factory, comm=mpi_communicator
        )

        super().__init__(
            array_context=array_context,
            mesh=mesh,
            discr_tag_to_group_factory=discr_tag_to_group_factory,
            volume_discr=volume_discr,
            dist_boundary_connections=dist_boundary_connections,
            mpi_communicator=mpi_communicator
        )

    def project(self, src, tgt, vec):
        return op.project(self, src, tgt, vec)

    def grad(self, vec):
        return op.local_grad(self, vec)

    def d_dx(self, xyz_axis, vec):
        return op.local_d_dx(self, xyz_axis, vec)

    def div(self, vecs):
        return op.local_div(self, vecs)

    def weak_grad(self, *args):
        return op.weak_local_grad(self, *args)

    def weak_d_dx(self, *args):
        return op.weak_local_d_dx(self, *args)

    def weak_div(self, *args):
        return op.weak_local_div(self, *args)

    def mass(self, *args):
        return op.mass(self, *args)

    def inverse_mass(self, vec):
        return op.inverse_mass(self, vec)

    def face_mass(self, *args):
        return op.face_mass(self, *args)

    def norm(self, vec, p=2, dd=None):
        return op.norm(self, vec, p, dd)

    def nodal_sum(self, dd, vec):
        return op.nodal_sum(self, dd, vec)

    def nodal_min(self, dd, vec):
        return op.nodal_min(self, dd, vec)

    def nodal_max(self, dd, vec):
        return op.nodal_max(self, dd, vec)


connected_ranks = op.connected_ranks
interior_trace_pair = op.interior_trace_pair
cross_rank_trace_pairs = op.cross_rank_trace_pairs

# vim: foldmethod=marker
