"""
.. currentmodule:: grudge.op

Interpolation
-------------

.. autofunction:: interp
"""
from __future__ import annotations


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


from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from grudge.discretization import DiscretizationCollection


# FIXME: Should revamp interp and make clear distinctions
# between projection and interpolations.
# Related issue: https://github.com/inducer/grudge/issues/38
def interp(dcoll: DiscretizationCollection, src, tgt, vec):
    from warnings import warn
    warn("'interp' currently calls to 'project'",
         UserWarning, stacklevel=2)

    from grudge.projection import project

    return project(dcoll, src, tgt, vec)
