__copyright__ = "Copyright (C) 2024 Addison Alvey-Blanco"

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

from pytools.tag import IgnoredForEqualityTag, Tag, tag_dataclass
from meshmode.transform_metadata import DiscretizationDOFAxisTag


# {{{ tensor product specific metadata

class OutputIsTensorProductDOFArrayOrdered(Tag):
    # FIXME: REMOVE THIS
    # /!\ THIS IS TEMPORARY AND WILL GO AWAY /!\
    """
    Signify that the strides will not be of order "C" or "F".

    Used to specify strides for eager einsums.
    """
    pass


@tag_dataclass
class TensorProductDOFAxisTag(DiscretizationDOFAxisTag):
    """
    Tag an axis as being an axis containing the DOFs of a tensor-product
    discretization. Used to signify the relative update speed of an axis for
    transformation (i.e. loop nest ordering) purposes.
    """
    iaxis: int


@tag_dataclass
class TensorProductOperatorAxisTag(IgnoredForEqualityTag):
    """
    Signify that an axis is an operator of a tensor-product discretization.
    Since these operators are reused, it is important to not propagate axis tags
    along their axes.
    """
    pass


class MassMatrix1DTag(Tag):
    """Used in DAG transformation to realize algebraic simplification of 1D
    inverse mass operator times mass operator.
    """
    pass


class InverseMassMatrix1DTag(Tag):
    """See MassMatrix1d.
    """

# }}}
