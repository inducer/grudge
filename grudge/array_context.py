"""
.. autoclass:: PyOpenCLArrayContext
.. autoclass:: PytatoPyOpenCLArrayContext
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


from meshmode.array_context import (
        PyOpenCLArrayContext as _PyOpenCLArrayContextBase,
        PytatoPyOpenCLArrayContext as _PytatoPyOpenCLArrayContextBase,
        )
from arraycontext.pytest import (
        _PytestPyOpenCLArrayContextFactoryWithClass,
        register_pytest_array_context_factory)


class PyOpenCLArrayContext(_PyOpenCLArrayContextBase):
    """Inherits from :class:`meshmode.array_context.PyOpenCLArrayContext`. Extends it
    to understand :mod:`grudge`-specific transform metadata. (Of which there isn't
    any, for now.)
    """


class PytatoPyOpenCLArrayContext(_PytatoPyOpenCLArrayContextBase):
    """Inherits from :class:`meshmode.array_context.PytatoPyOpenCLArrayContext`.
    Extends it to understand :mod:`grudge`-specific transform metadata. (Of
    which there isn't any, for now.)
    """

    def transform_dag(self, dag: "pytato.DictOfNamedArrays") -> "pytato.DictOfNamedArrays":
        print("transform_dag", dag._data)

        from pytato import find_partition_distributed, gather_distributed_comm_info, generate_code_for_partition
        parts = find_partition_distributed(dag)

        distributed_parts = gather_distributed_comm_info(parts)
        prg_per_partition = generate_code_for_partition(distributed_parts)

        from pytato import get_dot_graph_from_partition
        get_dot_graph_from_partition(distributed_parts)

        return super().transform_dag(dag)


# {{{ pytest actx factory

class PytestPyOpenCLArrayContextFactory(
        _PytestPyOpenCLArrayContextFactoryWithClass):
    actx_class = PyOpenCLArrayContext


# deprecated
class PytestPyOpenCLArrayContextFactoryWithHostScalars(
        _PytestPyOpenCLArrayContextFactoryWithClass):
    actx_class = PyOpenCLArrayContext
    force_device_scalars = False


register_pytest_array_context_factory("grudge.pyopencl",
        PytestPyOpenCLArrayContextFactory)

# }}}


# vim: foldmethod=marker
