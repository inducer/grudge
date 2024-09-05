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

from pytools.tag import Tag, tag_dataclass
from pytato.transform.metadata import IgnoredForPropagationTag
from meshmode.transform_metadata import DiscretizationDOFAxisTag


class OutputIsTensorProductDOFArrayOrdered(Tag):
    """
    Signal an eager `arraycontext` to adjust strides to be compatible with
    tensor product element DOFs when generating a `loopy` program.
    """
    pass


@tag_dataclass
class TensorProductDOFAxisTag(DiscretizationDOFAxisTag):
    """
    Signify an axis as containing the DOFs of a tensor product discretization.
    `iaxis` is later interpreted to determine the relative update speed (i.e.
    the stride) of each axis.
    """
    iaxis: int


class TensorProductOperatorAxisTag(IgnoredForPropagationTag):
    """
    Signify an axis is part of a 1D operator applied to a tensor product
    discretization. No tags will be propagated to or along axes containing this
    tag.
    """
    pass


class ReferenceTensorProductMassOperatorTag(Tag):
    """
    Tag an operator as being a reference mass operator. Used to realize an
    algebraic simplification of redundant mass-times-mass-inverse operations
    when using a tensor product discretization.
    """
    pass


class ReferenceTensorProductMassInverseOperatorTag(Tag):
    """
    See `ReferenceTensorProductMassOperatorTag`, but inverse.
    """
    pass
