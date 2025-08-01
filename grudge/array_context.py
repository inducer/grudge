"""
.. autoclass:: PyOpenCLArrayContext
.. autoclass:: PytatoPyOpenCLArrayContext
.. autoclass:: MPIBasedArrayContext
.. autoclass:: MPIPyOpenCLArrayContext
.. autoclass:: MPINumpyArrayContext
.. class:: MPIPytatoArrayContext
.. autofunction:: get_reasonable_array_context_class
"""

from __future__ import annotations


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

# {{{ imports

import logging
from abc import ABC
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar, Literal, overload
from warnings import warn

from typing_extensions import Self, override

from meshmode.array_context import (
    PyOpenCLArrayContext as _PyOpenCLArrayContextBase,
    PytatoPyOpenCLArrayContext as _PytatoPyOpenCLArrayContextBase,
)
from pytools import to_identifier


logger = logging.getLogger(__name__)


try:
    # FIXME: temporary workaround while FusionContractorArrayContext
    # is not available in meshmode's main branch
    from meshmode.array_context import FusionContractorArrayContext

    try:
        # Crude check if we have the correct loopy branch
        # (https://github.com/kaushikcfd/loopy/tree/pytato-array-context-transforms)
        from loopy.transform.loop_fusion import get_kennedy_unweighted_fusion_candidates  # noqa
    except ImportError:
        warn("Your loopy and meshmode branches are mismatched. "
             "Please make sure that you have the "
             "https://github.com/kaushikcfd/loopy/tree/pytato-array-context-transforms "
             "branch of loopy.", stacklevel=1)
        _HAVE_FUSION_ACTX = False
    else:
        _HAVE_FUSION_ACTX = True

except ImportError:
    _HAVE_FUSION_ACTX = False


from arraycontext import ArrayContext, NumpyArrayContext
from arraycontext.impl.pytato.compile import LazilyPyOpenCLCompilingFunctionCaller
from arraycontext.pytest import (
    _PytestNumpyArrayContextFactory,
    _PytestPyOpenCLArrayContextFactoryWithClass,
    _PytestPytatoPyOpenCLArrayContextFactory,
    register_pytest_array_context_factory,
)


if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    import pytato as pt
    from mpi4py import MPI
    from pytato import DictOfNamedArrays, DistributedGraphPartition
    from pytato.distributed.partition import PartId
    from pytato.target import BoundProgram

    import pyopencl
    import pyopencl.array as cl_array
    from arraycontext.container import ArrayContainer
    from pytools.tag import Tag


class PyOpenCLArrayContext(_PyOpenCLArrayContextBase):
    """Inherits from :class:`meshmode.array_context.PyOpenCLArrayContext`. Extends it
    to understand :mod:`grudge`-specific transform metadata. (Of which there isn't
    any, for now.)
    """
    def __init__(self, queue: pyopencl.CommandQueue,
            allocator: cl_array.Allocator | None = None,
            wait_event_queue_length: int | None = None,
            force_device_scalars: bool | None = None) -> None:

        if allocator is None:
            warn("No memory allocator specified, please pass one. "
                 "(Preferably a pyopencl.tools.MemoryPool in order "
                 "to reduce device allocations)", stacklevel=2)

        super().__init__(queue, allocator,
                         wait_event_queue_length, force_device_scalars)

# }}}


# {{{ pytato

class PytatoPyOpenCLArrayContext(_PytatoPyOpenCLArrayContextBase):
    """Inherits from :class:`meshmode.array_context.PytatoPyOpenCLArrayContext`.
    Extends it to understand :mod:`grudge`-specific transform metadata. (Of
    which there isn't any, for now.)
    """
    def __init__(self,
            queue: pyopencl.CommandQueue,
            allocator: cl_array.Allocator | None = None,
            *,
            compile_trace_callback: Callable[[Any, str, Any], None] | None
             = None) -> None:
        """
        :arg compile_trace_callback: A function of three arguments
            *(what, stage, ir)*, where *what* identifies the object
            being compiled, *stage* is a string describing the compilation
            pass, and *ir* is an object containing the intermediate
            representation. This interface should be considered
            unstable.
        """
        if allocator is None:
            warn("No memory allocator specified, please pass one. "
                 "(Preferably a pyopencl.tools.MemoryPool in order "
                 "to reduce device allocations)", stacklevel=2)

        super().__init__(queue, allocator,
                compile_trace_callback=compile_trace_callback)

# }}}


class MPIBasedArrayContext(ArrayContext, ABC):
    mpi_communicator: MPI.Intracomm


# {{{ distributed + pytato

@dataclass(frozen=True)
class _DistributedPartProgramID:
    f: Callable[..., Any]
    part_id: Any

    def __str__(self):
        name = getattr(self.f, "__name__", "anonymous")
        if not name.isidentifier():
            name = to_identifier(name)

        part = to_identifier(str(self.part_id))
        if part:
            return f"{name}_part{part}"
        else:
            return name


class _DistributedLazilyPyOpenCLCompilingFunctionCaller(
        LazilyPyOpenCLCompilingFunctionCaller):
    def _dag_to_compiled_func(self, dict_of_named_arrays: DictOfNamedArrays,
            input_id_to_name_in_program, output_id_to_name_in_program,
            output_template):

        import pytato as pt

        from pytools import ProcessLogger

        self.actx._compile_trace_callback(self.f, "pre_deduplicate",
                dict_of_named_arrays)

        with ProcessLogger(logger, "deduplicate"):
            dict_of_named_arrays = pt.transform.deduplicate(dict_of_named_arrays)

        self.actx._compile_trace_callback(self.f, "post_deduplicate",
                dict_of_named_arrays)

        self.actx._compile_trace_callback(self.f, "pre_deduplicate_data_wrappers",
                dict_of_named_arrays)

        with ProcessLogger(logger, "deduplicate_data_wrappers[pre-partition]"):
            dict_of_named_arrays = pt.transform.deduplicate_data_wrappers(
                dict_of_named_arrays)

        self.actx._compile_trace_callback(self.f, "post_deduplicate_data_wrappers",
                dict_of_named_arrays)

        self.actx._compile_trace_callback(self.f, "pre_materialize",
                dict_of_named_arrays)

        with ProcessLogger(logger, "materialize_with_mpms[pre-partition]"):
            dict_of_named_arrays = pt.transform.materialize_with_mpms(
                dict_of_named_arrays)

        self.actx._compile_trace_callback(self.f, "post_materialize",
                dict_of_named_arrays)

        # FIXME: Remove the import failure handling once this is in upstream grudge
        self.actx._compile_trace_callback(self.f, "pre_infer_axes_tags",
                dict_of_named_arrays)

        with ProcessLogger(logger,
                           "transform_dag.infer_axes_tags[pre-partition]"):
            from meshmode.transform_metadata import DiscretizationEntityAxisTag
            dict_of_named_arrays = pt.unify_axes_tags(
                dict_of_named_arrays,
                tag_t=DiscretizationEntityAxisTag,
            )

        self.actx._compile_trace_callback(self.f, "post_infer_axes_tags",
                dict_of_named_arrays)

        self.actx._compile_trace_callback(self.f, "pre_find_distributed_partition",
                dict_of_named_arrays)

        distributed_partition = pt.find_distributed_partition(
            # pylint-ignore-reason:
            # '_BasePytatoArrayContext' has no
            # 'mpi_communicator' member
            # pylint: disable=no-member
            self.actx.mpi_communicator, dict_of_named_arrays)

        if __debug__:
            # pylint-ignore-reason:
            # '_BasePytatoArrayContext' has no 'mpi_communicator' member
            pt.verify_distributed_partition(
                self.actx.mpi_communicator,  # pylint: disable=no-member
                distributed_partition)

        self.actx._compile_trace_callback(self.f, "post_find_distributed_partition",
                distributed_partition)

        # {{{ turn symbolic tags into globally agreed-upon integers

        self.actx._compile_trace_callback(self.f, "pre_number_distributed_tags",
                distributed_partition)

        from pytato import number_distributed_tags
        prev_mpi_base_tag = self.actx.mpi_base_tag

        # type-ignore-reason: 'PytatoPyOpenCLArrayContext' has no 'mpi_communicator'
        # pylint: disable=no-member
        distributed_partition, new_mpi_base_tag = number_distributed_tags(
                self.actx.mpi_communicator,
                distributed_partition,
                base_tag=prev_mpi_base_tag)

        assert prev_mpi_base_tag == self.actx.mpi_base_tag
        # FIXME: Updating stuff inside the array context from here is *cough*
        # not super pretty.
        self.actx.mpi_base_tag = new_mpi_base_tag

        self.actx._compile_trace_callback(self.f, "post_number_distributed_tags",
                distributed_partition)

        # }}}

        part_id_to_prg = {}
        name_in_program_to_tags = {}
        name_in_program_to_axes = {}

        from pytato import make_dict_of_named_arrays
        for part in distributed_partition.parts.values():
            d = make_dict_of_named_arrays(
                        {var_name: distributed_partition.name_to_output[var_name]
                            for var_name in part.output_names
                         })
            (
                part_id_to_prg[part.pid],
                part_prg_name_to_tags,
                part_prg_name_to_axes
            ) = self._dag_to_transformed_pytato_prg(
                    d, prg_id=_DistributedPartProgramID(self.f, part.pid))

            assert not (set(name_in_program_to_tags.keys())
                        & set(part_prg_name_to_tags.keys()))
            assert not (set(name_in_program_to_axes.keys())
                        & set(part_prg_name_to_axes.keys()))
            name_in_program_to_tags.update(part_prg_name_to_tags)
            name_in_program_to_axes.update(part_prg_name_to_axes)

        from constantdict import constantdict
        return _DistributedCompiledFunction(
                actx=self.actx,
                distributed_partition=distributed_partition,
                part_id_to_prg=part_id_to_prg,
                input_id_to_name_in_program=input_id_to_name_in_program,
                output_id_to_name_in_program=output_id_to_name_in_program,
                name_in_program_to_tags=constantdict(name_in_program_to_tags),
                name_in_program_to_axes=constantdict(name_in_program_to_axes),
                output_template=output_template)


@dataclass(frozen=True)
class _DistributedCompiledFunction:
    """
    A callable which captures the :class:`pytato.target.BoundProgram`  resulting
    from calling :attr:`~LazilyPyOpenCLCompilingFunctionCaller.f` with a given
    set of input types, and generating :mod:`loopy` IR from it.

    .. attribute:: pytato_program

    .. attribute:: input_id_to_name_in_program

        A mapping from input id to the placeholder name in
        :attr:`CompiledFunction.pytato_program`. Input id is represented as the
        position of :attr:`~LazilyPyOpenCLCompilingFunctionCaller.f`'s argument
        augmented with the leaf array's key if the argument is an array
        container.

    .. attribute:: output_id_to_name_in_program

        A mapping from output id to the name of
        :class:`pytato.array.NamedArray` in
        :attr:`CompiledFunction.pytato_program`. Output id is represented by
        the key of a leaf array in the array container
        :attr:`CompiledFunction.output_template`.

    .. attribute:: output_template

       An instance of :class:`arraycontext.ArrayContainer` that is the return
       type of the callable.
    """

    actx: MPIBasedArrayContext
    distributed_partition: DistributedGraphPartition
    part_id_to_prg: Mapping[PartId, BoundProgram]
    input_id_to_name_in_program: Mapping[tuple[Any, ...], str]
    output_id_to_name_in_program: Mapping[tuple[Any, ...], str]
    name_in_program_to_tags: Mapping[str, frozenset[Tag]]
    name_in_program_to_axes: Mapping[str, tuple[pt.Axis, ...]]
    output_template: ArrayContainer

    def __call__(self, arg_id_to_arg) -> ArrayContainer:
        """
        :arg arg_id_to_arg: Mapping from input id to the passed argument. See
            :attr:`CompiledFunction.input_id_to_name_in_program` for input id's
            representation.
        """

        from arraycontext.impl.pyopencl.taggable_cl_array import to_tagged_cl_array
        from arraycontext.impl.pytato.compile import _args_to_device_buffers
        from arraycontext.impl.pytato.utils import get_cl_axes_from_pt_axes
        input_args_for_prg = _args_to_device_buffers(
                self.actx, self.input_id_to_name_in_program, arg_id_to_arg)

        from pytato import execute_distributed_partition
        assert isinstance(self.actx, PytatoPyOpenCLArrayContext | PyOpenCLArrayContext)
        out_dict = execute_distributed_partition(
                self.distributed_partition, self.part_id_to_prg,
                self.actx.queue, self.actx.mpi_communicator,  # pylint: disable=no-member
                allocator=self.actx.allocator,  # pylint: disable=no-member
                input_args=input_args_for_prg)

        def to_output_template(keys, _):
            ary_name_in_prg = self.output_id_to_name_in_program[keys]
            return self.actx.thaw(to_tagged_cl_array(
                out_dict[ary_name_in_prg],
                axes=get_cl_axes_from_pt_axes(
                    self.name_in_program_to_axes[ary_name_in_prg]),
                tags=self.name_in_program_to_tags[ary_name_in_prg]))

        from arraycontext.container.traversal import rec_keyed_map_array_container
        return rec_keyed_map_array_container(to_output_template,
                                             self.output_template)


# }}}


# {{{ distributed + pyopencl

class MPIPyOpenCLArrayContext(PyOpenCLArrayContext, MPIBasedArrayContext):
    """An array context for using distributed computation with :mod:`pyopencl`
    eager evaluation.

    .. autofunction:: __init__
    """

    def __init__(self,
            mpi_communicator: MPI.Intracomm,
            queue: pyopencl.CommandQueue,
            *, allocator: cl_array.Allocator | None = None,
            wait_event_queue_length: int | None = None,
            force_device_scalars: bool | None = None) -> None:
        """
        See :class:`arraycontext.impl.pyopencl.PyOpenCLArrayContext` for most
        arguments.
        """
        super().__init__(queue, allocator=allocator,
                wait_event_queue_length=wait_event_queue_length,
                force_device_scalars=force_device_scalars)
        self.mpi_communicator = mpi_communicator

    @override
    def clone(self) -> Self:
        return type(self)(self.mpi_communicator, self.queue,
                allocator=self.allocator,
                wait_event_queue_length=self._wait_event_queue_length,
                force_device_scalars=self._force_device_scalars)

# }}}


# {{{ distributed + numpy

class MPINumpyArrayContext(NumpyArrayContext, MPIBasedArrayContext):
    """An array context for using distributed computation with :mod:`numpy`
    eager evaluation.

    .. autofunction:: __init__
    """

    def __init__(self, mpi_communicator: MPI.Intracomm) -> None:
        super().__init__()

        self.mpi_communicator = mpi_communicator

    @override
    def clone(self) -> Self:
        return type(self)(self.mpi_communicator)

# }}}


# {{{ distributed + pytato array context subclasses

class MPIBasePytatoPyOpenCLArrayContext(
        MPIBasedArrayContext, PytatoPyOpenCLArrayContext):
    """
    .. autofunction:: __init__
    """
    mpi_base_tag: int

    def __init__(
            self,
            mpi_communicator: MPI.Intracomm,
            queue: pyopencl.CommandQueue, *,
            mpi_base_tag: int,
            allocator: cl_array.Allocator | None = None,
            compile_trace_callback: Callable[[Any, str, Any], None] | None = None,
            ) -> None:
        """
        :arg compile_trace_callback: A function of three arguments
            *(what, stage, ir)*, where *what* identifies the object
            being compiled, *stage* is a string describing the compilation
            pass, and *ir* is an object containing the intermediate
            representation. This interface should be considered
            unstable.
        """
        if allocator is None:
            warn("No memory allocator specified, please pass one. "
                 "(Preferably a pyopencl.tools.MemoryPool in order "
                 "to reduce device allocations)", stacklevel=2)

        super().__init__(queue, allocator,
                compile_trace_callback=compile_trace_callback)

        self.mpi_communicator = mpi_communicator
        self.mpi_base_tag = mpi_base_tag

    # FIXME: implement distributed-aware freeze

    def compile(self, f: Callable[..., Any]) -> Callable[..., Any]:
        return _DistributedLazilyPyOpenCLCompilingFunctionCaller(self, f)

    def clone(self) -> Self:
        return type(self)(self.mpi_communicator, self.queue,
                mpi_base_tag=self.mpi_base_tag,
                allocator=self.allocator)


MPIPytatoArrayContext: type[MPIBasedArrayContext] = MPIBasePytatoPyOpenCLArrayContext


if _HAVE_FUSION_ACTX:
    class MPIFusionContractorArrayContext(
            MPIBasePytatoPyOpenCLArrayContext, FusionContractorArrayContext):
        """
        .. autofunction:: __init__
        """

    MPIPytatoArrayContext = MPIFusionContractorArrayContext
else:
    MPIPytatoArrayContext = MPIBasePytatoPyOpenCLArrayContext

# }}}


# {{{ pytest actx factory

class PytestPyOpenCLArrayContextFactory(
        _PytestPyOpenCLArrayContextFactoryWithClass):
    actx_class = PyOpenCLArrayContext

    def __call__(self):
        from pyopencl.tools import ImmediateAllocator, MemoryPool

        _ctx, queue = self.get_command_queue()
        alloc = MemoryPool(ImmediateAllocator(queue))

        return self.actx_class(queue, allocator=alloc)


class PytestPytatoPyOpenCLArrayContextFactory(
        _PytestPytatoPyOpenCLArrayContextFactory):
    actx_class = PytatoPyOpenCLArrayContext

    def __call__(self):
        _ctx, queue = self.get_command_queue()

        from pyopencl.tools import ImmediateAllocator, MemoryPool
        alloc = MemoryPool(ImmediateAllocator(queue))

        return self.actx_class(queue, allocator=alloc)


class PytestNumpyArrayContextFactory(_PytestNumpyArrayContextFactory):
    actx_class: ClassVar[type[NumpyArrayContext]] = NumpyArrayContext

    def __call__(self):
        return self.actx_class()


register_pytest_array_context_factory("grudge.pyopencl",
        PytestPyOpenCLArrayContextFactory)
register_pytest_array_context_factory("grudge.pytato-pyopencl",
        PytestPytatoPyOpenCLArrayContextFactory)
register_pytest_array_context_factory("grudge.numpy",
        PytestNumpyArrayContextFactory)

# }}}


# {{{ actx selection


def _get_single_grid_pytato_actx_class(distributed: bool) -> type[ArrayContext]:
    warn("No device-parallel actx available, execution will be slow.",
         stacklevel=1)
    # lazy, non-distributed
    if not distributed:
        return PytatoPyOpenCLArrayContext
    else:
        # distributed+lazy:
        return MPIBasePytatoPyOpenCLArrayContext


@overload
def get_reasonable_array_context_class(
        lazy: bool = True, distributed: Literal[True] = True,
        fusion: bool | None = None, numpy: bool = False,
        ) -> type[MPIBasedArrayContext]: ...
@overload
def get_reasonable_array_context_class(
        lazy: bool = True, distributed: bool = True,
        fusion: bool | None = None, numpy: bool = False,
        ) -> type[ArrayContext]: ...


def get_reasonable_array_context_class(
        lazy: bool = True, distributed: bool = True,
        fusion: bool | None = None, numpy: bool = False,
        ) -> type[ArrayContext]:
    """Returns a reasonable :class:`~arraycontext.ArrayContext` currently
    supported given the constraints of *lazy*, *distributed*, and *numpy*."""
    if fusion is None:
        fusion = lazy

    if numpy:
        assert not (lazy or fusion)
        if distributed:
            actx_class: type[ArrayContext] = MPINumpyArrayContext
        else:
            actx_class = NumpyArrayContext

        return actx_class

    if lazy:
        if fusion:
            if not _HAVE_FUSION_ACTX:
                warn("No device-parallel actx available, execution will be slow. "
                     "Please make sure you have the right branches for loopy "
                     "(https://github.com/kaushikcfd/loopy/tree/pytato-array-context-transforms) "  # noqa
                     "and meshmode "
                     "(https://github.com/kaushikcfd/meshmode/tree/pytato-array-context-transforms).",
                     stacklevel=1)
            # lazy+fusion, non-distributed

            if _HAVE_FUSION_ACTX:
                if distributed:
                    actx_class = MPIFusionContractorArrayContext
                else:
                    actx_class = FusionContractorArrayContext
            else:
                actx_class = _get_single_grid_pytato_actx_class(distributed)
        else:
            actx_class = _get_single_grid_pytato_actx_class(distributed)
    else:
        if fusion:
            raise ValueError("No eager actx's support op-fusion.")
        if distributed:
            actx_class = MPIPyOpenCLArrayContext
        else:
            actx_class = PyOpenCLArrayContext

    logger.info("get_reasonable_array_context_class: %s lazy=%r distributed=%r "
                "device-parallel=%r",
                actx_class.__name__, lazy, distributed,
                # eager is always device-parallel:
                (_HAVE_FUSION_ACTX or not lazy))
    return actx_class

# }}}


# vim: foldmethod=marker
