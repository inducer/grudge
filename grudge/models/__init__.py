"""Base classes for operators."""

__copyright__ = """
Copyright (C) 2007 Andreas Kloeckner
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

from abc import ABCMeta, abstractmethod


class Operator(metaclass=ABCMeta):
    """A base class for Discontinuous Galerkin operators.

    You may derive your own operators from this class, but, at present
    this class provides no functionality. Its function is merely as
    documentation, to group related classes together in an inheritance
    tree.
    """


class HyperbolicOperator(Operator):
    """A base class for hyperbolic Discontinuous Galerkin operators."""

    @abstractmethod
    def max_characteristic_velocity(self, t, fields, dcoll):
        """Return an upper bound on the characteristic
        velocities of the operator.
        """

    def estimate_rk4_timestep(self, dcoll, t=None, fields=None):
        """Estimate the largest stable timestep for an RK4 method."""
        from grudge.dt_utils import (dt_non_geometric_factors,
                                     dt_geometric_factors)
        from meshmode.dof_array import DOFArray
        import grudge.op as op

        actx = dcoll._setup_actx
        max_lambda = self.max_characteristic_velocity(t, fields, dcoll)
        # Scale each group array of geometric factors by the corresponding
        # group non-geometric factor
        dt_factors = DOFArray(
            actx,
            data=tuple(
                cng * geo_facts
                for cng, geo_facts in zip(dt_non_geometric_factors(dcoll),
                                          dt_geometric_factors(dcoll))
            )
        )

        return op.nodal_min(dcoll, "vol", dt_factors * (1 / max_lambda))
