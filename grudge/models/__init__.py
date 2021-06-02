"""Base classes for operators."""

__copyright__ = "Copyright (C) 2007 Andreas Kloeckner"

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

    def estimate_rk4_timestep(self, dcoll, t=None, fields=None, dt_scaling=None):
        """Estimate the largest stable timestep for an RK4 method."""
        from mpi4py import MPI
        from grudge.dt_utils import (dt_non_geometric_factor,
                                     dt_geometric_factor)

        max_lambda = self.max_characteristic_velocity(t, fields, dcoll)
        dt_factor = \
            (dt_non_geometric_factor(dcoll, scaling=dt_scaling)
             * dt_geometric_factor(dcoll))

        mpi_comm = dcoll.mpi_communicator
        if mpi_comm is None:
            return dt_factor * (1 / max_lambda)

        return (1 / mpi_comm.allreduce(max_lambda, op=MPI.MAX)) \
            * mpi_comm.allreduce(dt_factor, op=MPI.MIN)
