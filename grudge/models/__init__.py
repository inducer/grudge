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


class Operator:
    """A base class for Discontinuous Galerkin operators.

    You may derive your own operators from this class, but, at present
    this class provides no functionality. Its function is merely as
    documentation, to group related classes together in an inheritance
    tree.
    """


class HyperbolicOperator(Operator):
    """A base class for hyperbolic Discontinuous Galerkin operators."""

    def max_eigenvalue(self, t, fields, dcoll):
        raise NotImplementedError

    def estimate_rk4_timestep(self, dcoll, t=None, fields=None):
        """Estimate the largest stable timestep for an RK4 method.
        """
        from grudge.dt_utils import h_min_vertex_distance

        N = max([grp.order for grp in dcoll.discr_from_dd("vol").groups])
        h_min_factor = h_min_vertex_distance(dcoll) / (N ** 2)

        return h_min_factor * 1 / self.max_eigenvalue(t, fields, dcoll)
