"""
.. autoclass:: PyOpenCLArrayContext
.. autoclass:: PytatoPyOpenCLArrayContext
.. autoclass:: MPIBasedArrayContext
.. autoclass:: MPIPyOpenCLArrayContext
.. class:: MPIPytatoArrayContext
.. autofunction:: get_reasonable_array_context_class
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

# {{{ imports

import logging
from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    FrozenSet,
    Mapping,
    Optional,
    Tuple,
    Type,
)
from warnings import warn

from meshmode.array_context import (
    PyOpenCLArrayContext as _PyOpenCLArrayContextBase,
    PytatoPyOpenCLArrayContext as _PytatoPyOpenCLArrayContextBase,
)
from pytools import to_identifier
from pytools.tag import Tag


logger = logging.getLogger(__name__)

try:
    # FIXME: temporary workaround while SingleGridWorkBalancingPytatoArrayContext
    # is not available in meshmode's main branch
    # (it currently needs
    # https://github.com/kaushikcfd/meshmode/tree/pytato-array-context-transforms)
    from meshmode.array_context import SingleGridWorkBalancingPytatoArrayContext

    try:
        # Crude check if we have the correct loopy branch
        # (https://github.com/kaushikcfd/loopy/tree/pytato-array-context-transforms)
        from loopy.codegen.result import get_idis_for_kernel  # noqa
    except ImportError:
        # warn("Your loopy and meshmode branches are mismatched. "
        #      "Please make sure that you have the "
        #      "https://github.com/kaushikcfd/loopy/tree/pytato-array-context-transforms "  # noqa
        #      "branch of loopy.")
        _HAVE_SINGLE_GRID_WORK_BALANCING = False
    else:
        _HAVE_SINGLE_GRID_WORK_BALANCING = True

except ImportError:
    _HAVE_SINGLE_GRID_WORK_BALANCING = False

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


from arraycontext import ArrayContext
from arraycontext.container import ArrayContainer
from arraycontext.impl.pytato.compile import LazilyPyOpenCLCompilingFunctionCaller
from arraycontext.pytest import (
    _PytestPyOpenCLArrayContextFactoryWithClass,
    _PytestPytatoPyOpenCLArrayContextFactory,
    register_pytest_array_context_factory,
)


if TYPE_CHECKING:
    import pytato as pt
    from mpi4py import MPI
    from pytato import DistributedGraphPartition
    from pytato.partition import PartId

    import pyopencl
    import pyopencl.tools


class PyOpenCLArrayContext(_PyOpenCLArrayContextBase):
    """Inherits from :class:`meshmode.array_context.PyOpenCLArrayContext`. Extends it
    to understand :mod:`grudge`-specific transform metadata. (Of which there isn't
    any, for now.)
    """
    def __init__(self, queue: "pyopencl.CommandQueue",
            allocator: Optional["pyopencl.tools.AllocatorBase"] = None,
            wait_event_queue_length: Optional[int] = None,
            force_device_scalars: bool = True) -> None:

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
    def __init__(self, queue, allocator=None,
            *,
            compile_trace_callback: Optional[Callable[[Any, str, Any], None]]
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


# {{{ Profiling

import pyopencl as cl
import loopy as lp
import pytools
from pytools.py_codegen import PythonFunctionGenerator
import numpy as np

class StatisticsAccumulator:
    """Class that provides statistical functions for multiple values.

    .. automethod:: __init__
    .. automethod:: add_value
    .. automethod:: sum
    .. automethod:: mean
    .. automethod:: max
    .. automethod:: min
    .. autoattribute:: num_values
    """

    def __init__(self, scale_factor: float = 1) -> None:
        """Initialize an empty StatisticsAccumulator object.

        Parameters
        ----------
        scale_factor
            Scale returned statistics by this factor.
        """
        # Number of values stored in the StatisticsAccumulator
        self.num_values: int = 0

        self._sum = 0
        self._min = None
        self._max = None
        self.scale_factor = scale_factor

    def add_value(self, v: float) -> None:
        """Add a new value to the statistics."""
        if v is None:
            return
        self.num_values += 1
        self._sum += v
        if self._min is None or v < self._min:
            self._min = v
        if self._max is None or v > self._max:
            self._max = v

    def sum(self) -> Optional[float]:
        """Return the sum of added values."""
        if self.num_values == 0:
            return None

        return self._sum * self.scale_factor

    def mean(self) -> Optional[float]:
        """Return the mean of added values."""
        if self.num_values == 0:
            return None

        return self._sum / self.num_values * self.scale_factor

    def max(self) -> Optional[float]:
        """Return the max of added values."""
        if self.num_values == 0:
            return None

        return self._max * self.scale_factor

    def min(self) -> Optional[float]:
        """Return the min of added values."""
        if self.num_values == 0:
            return None

        return self._min * self.scale_factor


@dataclass
class SingleCallKernelProfile:
    """Class to hold the results of a single kernel execution."""

    time: int
    flops: int
    bytes_accessed: int
    footprint_bytes: int


@dataclass
class MultiCallKernelProfile:
    """Class to hold the results of multiple kernel executions."""

    num_calls: int
    time: StatisticsAccumulator
    flops: StatisticsAccumulator
    bytes_accessed: StatisticsAccumulator
    footprint_bytes: StatisticsAccumulator


@dataclass
class ProfileEvent:
    """Holds a profile event that has not been collected by the profiler yet."""

    cl_event: cl._cl.Event
    translation_unit: lp.TranslationUnit
    args_tuple: tuple


class ProfilingActx:
    """An array context that profiles OpenCL kernel executions.

    .. automethod:: tabulate_profiling_data
    .. automethod:: call_loopy
    .. automethod:: get_profiling_data_for_kernel
    .. automethod:: reset_profiling_data_for_kernel

    Inherits from :class:`arraycontext.PyOpenCLArrayContext`.

    .. note::

       Profiling of :mod:`pyopencl` kernels (that is, kernels that do not get
       called through :meth:`call_loopy`) is restricted to a single instance of
       this class. If there are multiple instances, only the first one created
       will be able to profile these kernels.
    """

    def __init__(self, queue, allocator=None, logmgr: Any = None, **kwargs) -> None:
        super().__init__(queue, allocator)

        if not queue.properties & cl.command_queue_properties.PROFILING_ENABLE:
            raise RuntimeError("Profiling was not enabled in the command queue. "
                 "Please create the queue with "
                 "cl.command_queue_properties.PROFILING_ENABLE.")

        # list of ProfileEvents that haven't been transferred to profiled results yet
        self.profile_events = []

        # dict of kernel name -> SingleCallKernelProfile results
        self.profile_results = {}

        # dict of (Kernel, args_tuple) -> calculated number of flops, bytes
        self.kernel_stats = {}
        self.logmgr = logmgr

        # Only store the first kernel exec hook for elwise kernels
        # if cl.array.ARRAY_KERNEL_EXEC_HOOK is None:
        #     cl.array.ARRAY_KERNEL_EXEC_HOOK = self.array_kernel_exec_hook

    def clone(self):
        """Return a semantically equivalent but distinct version of *self*."""
        from warnings import warn
        warn("Cloned PyOpenCLProfilingArrayContexts can not "
             "profile elementwise PyOpenCL kernels.")
        return type(self)(self.queue, self.allocator, self.logmgr)

    def __del__(self):
        """Release resources and undo monkey patching."""
        del self.profile_events[:]
        self.profile_results.clear()
        self.kernel_stats.clear()

    def array_kernel_exec_hook(self, knl, queue, gs, ls, *actual_args, wait_for):
        """Extract data from the elementwise array kernel."""
        evt = knl(queue, gs, ls, *actual_args, wait_for=wait_for)

        name = knl.function_name

        args_tuple = tuple(
            (arg.size)
            for arg in actual_args if isinstance(arg, cl.array.Array))

        try:
            self.kernel_stats[knl][args_tuple]
        except KeyError:
            nbytes = 0
            nops = 0
            for arg in actual_args:
                if isinstance(arg, cl.array.Array):
                    nbytes += arg.size * arg.dtype.itemsize
                    nops += arg.size
            res = SingleCallKernelProfile(time=0, flops=nops, bytes_accessed=nbytes,
                                footprint_bytes=nbytes)
            self.kernel_stats.setdefault(knl, {})[args_tuple] = res

        if self.logmgr and f"{name}_time" not in self.logmgr.quantity_data:
            self.logmgr.add_quantity(KernelProfile(self, name))

        self.profile_events.append(ProfileEvent(evt, knl, args_tuple))

        return evt

    def _wait_and_transfer_profile_events(self) -> None:
        # First, wait for completion of all events
        if self.profile_events:
            cl.wait_for_events([pevt.cl_event for pevt in self.profile_events])

        # Then, collect all events and store them
        for t in self.profile_events:
            t_unit = t.translation_unit
            if isinstance(t_unit, lp.TranslationUnit):
                name = t_unit.default_entrypoint.name
            else:
                # It's actually a cl.Kernel
                name = t_unit.function_name

            r = self._get_kernel_stats(t_unit, t.args_tuple)
            time = t.cl_event.profile.end - t.cl_event.profile.start

            new = SingleCallKernelProfile(time, r.flops, r.bytes_accessed,
                                          r.footprint_bytes)

            self.profile_results.setdefault(name, []).append(new)

        self.profile_events = []

    def get_profiling_data_for_kernel(self, kernel_name: str) \
          -> MultiCallKernelProfile:
        """Return profiling data for kernel `kernel_name`."""
        self._wait_and_transfer_profile_events()

        time = StatisticsAccumulator(scale_factor=1e-9)
        gflops = StatisticsAccumulator(scale_factor=1e-9)
        gbytes_accessed = StatisticsAccumulator(scale_factor=1e-9)
        fprint_gbytes = StatisticsAccumulator(scale_factor=1e-9)
        num_calls = 0

        if kernel_name in self.profile_results:
            knl_results = self.profile_results[kernel_name]

            num_calls = len(knl_results)

            for r in knl_results:
                time.add_value(r.time)
                gflops.add_value(r.flops)
                gbytes_accessed.add_value(r.bytes_accessed)
                fprint_gbytes.add_value(r.footprint_bytes)

        return MultiCallKernelProfile(num_calls, time, gflops, gbytes_accessed,
                                      fprint_gbytes)

    def reset_profiling_data_for_kernel(self, kernel_name: str) -> None:
        """Reset profiling data for kernel `kernel_name`."""
        self.profile_results.pop(kernel_name, None)

    def tabulate_profiling_data(self) -> pytools.Table:
        """Return a :class:`pytools.Table` with the profiling results."""
        self._wait_and_transfer_profile_events()

        tbl = pytools.Table()

        # Table header
        tbl.add_row(["Function", "Calls",
            "Time_sum [s]", "Time_min [s]", "Time_avg [s]", "Time_max [s]",
            "GFlops/s_min", "GFlops/s_avg", "GFlops/s_max",
            "BWAcc_min [GByte/s]", "BWAcc_mean [GByte/s]", "BWAcc_max [GByte/s]",
            "BWFoot_min [GByte/s]", "BWFoot_mean [GByte/s]", "BWFoot_max [GByte/s]",
            "Intensity (flops/byte)"])

        # Precision of results
        g = ".4g"

        total_calls = 0
        total_time = 0

        for knl in self.profile_results.keys():
            r = self.get_profiling_data_for_kernel(knl)

            # Extra statistics that are derived from the main values returned by
            # self.get_profiling_data_for_kernel(). These are already GFlops/s and
            # GBytes/s respectively, so no need to scale them.
            flops_per_sec = StatisticsAccumulator()
            bandwidth_access = StatisticsAccumulator()

            knl_results = self.profile_results[knl]
            for knl_res in knl_results:
                flops_per_sec.add_value(knl_res.flops/knl_res.time)
                bandwidth_access.add_value(knl_res.bytes_accessed/knl_res.time)

            total_calls += r.num_calls

            total_time += r.time.sum()

            time_sum = f"{r.time.sum():{g}}"
            time_min = f"{r.time.min():{g}}"
            time_avg = f"{r.time.mean():{g}}"
            time_max = f"{r.time.max():{g}}"

            if r.footprint_bytes.sum() is not None:
                fprint_mean = f"{r.footprint_bytes.mean():{g}}"
                fprint_min = f"{r.footprint_bytes.min():{g}}"
                fprint_max = f"{r.footprint_bytes.max():{g}}"
            else:
                fprint_mean = "--"
                fprint_min = "--"
                fprint_max = "--"

            if r.flops.sum() > 0:
                bytes_per_flop_mean = f"{r.bytes_accessed.sum() / r.flops.sum():{g}}"
                flops_per_sec_min = f"{flops_per_sec.min():{g}}"
                flops_per_sec_mean = f"{flops_per_sec.mean():{g}}"
                flops_per_sec_max = f"{flops_per_sec.max():{g}}"
            else:
                bytes_per_flop_mean = "--"
                flops_per_sec_min = "--"
                flops_per_sec_mean = "--"
                flops_per_sec_max = "--"

            bandwidth_access_min = f"{bandwidth_access.min():{g}}"
            bandwidth_access_mean = f"{bandwidth_access.sum():{g}}"
            bandwidth_access_max = f"{bandwidth_access.max():{g}}"

            tbl.add_row([knl, r.num_calls, time_sum,
                time_min, time_avg, time_max,
                flops_per_sec_min, flops_per_sec_mean, flops_per_sec_max,
                bandwidth_access_min, bandwidth_access_mean, bandwidth_access_max,
                fprint_min, fprint_mean, fprint_max,
                bytes_per_flop_mean])

        tbl.add_row(["Total", total_calls, f"{total_time:{g}}"] + ["--"] * 13)

        return tbl

    def _get_kernel_stats(self, t_unit: lp.TranslationUnit, args_tuple: tuple) \
      -> SingleCallKernelProfile:
        return self.kernel_stats[t_unit][args_tuple]

    def _cache_kernel_stats(self, t_unit: lp.TranslationUnit, kwargs: dict) \
      -> tuple:
        """Generate the kernel stats for a program with its args."""
        args_tuple = tuple(
            (key, value.shape) if hasattr(value, "shape") else (key, value)
            for key, value in kwargs.items())

        # Are kernel stats already in the cache?
        try:
            self.kernel_stats[t_unit][args_tuple]
            return args_tuple
        except KeyError:
            # If not, calculate and cache the stats
            ep_name = t_unit.default_entrypoint.name
            executor = t_unit.target.get_kernel_executor(t_unit, self.queue,
                    entrypoint=ep_name)
            info = executor.translation_unit_info(
                ep_name, executor.arg_to_dtype_set(kwargs))

            typed_t_unit = executor.get_typed_and_scheduled_translation_unit(
                ep_name, executor.arg_to_dtype_set(kwargs))
            kernel = typed_t_unit[ep_name]

            idi = info.implemented_data_info

            param_dict = kwargs.copy()
            param_dict.update({k: None for k in kernel.arg_dict.keys()
                if k not in param_dict})

            param_dict.update(
                {d.name: None for d in idi if d.name not in param_dict})

            # Generate the wrapper code
            wrapper = executor.get_wrapper_generator()

            gen = PythonFunctionGenerator("_mcom_gen_args_profile", list(param_dict))

            wrapper.generate_integer_arg_finding_from_shapes(gen, kernel, idi)
            wrapper.generate_integer_arg_finding_from_offsets(gen, kernel, idi)
            wrapper.generate_integer_arg_finding_from_strides(gen, kernel, idi)

            param_names = kernel.all_params()
            gen("return {%s}" % ", ".join(
                f"{repr(name)}: {name}" for name in param_names))

            # Run the wrapper code, save argument values in domain_params
            domain_params = gen.get_picklable_function()(**param_dict)

            # Get flops/memory statistics
            op_map = lp.get_op_map(typed_t_unit, subgroup_size="guess")
            # bytes_accessed = lp.get_mem_access_map(
            #     typed_t_unit, subgroup_size="guess") \
            #                 .to_bytes().eval_and_sum(domain_params)

            bytes_accessed = 0

            flops = op_map.filter_by(dtype=[np.float32, np.float64]).eval_and_sum(
                domain_params)

            # Footprint gathering is not yet available in loopy with
            # kernel callables:
            # https://github.com/inducer/loopy/issues/399
            if 0:
                try:
                    footprint = lp.gather_access_footprint_bytes(typed_t_unit)
                    footprint_bytes = sum(footprint[k].eval_with_dict(domain_params)
                        for k in footprint)

                except lp.symbolic.UnableToDetermineAccessRange:
                    footprint_bytes = None
            else:
                footprint_bytes = None

            res = SingleCallKernelProfile(
                time=0, flops=flops, bytes_accessed=bytes_accessed,
                footprint_bytes=footprint_bytes)

            self.kernel_stats.setdefault(t_unit, {})[args_tuple] = res

            if self.logmgr:
                if f"{ep_name}_time" not in self.logmgr.quantity_data:
                    self.logmgr.add_quantity(KernelProfile(self, ep_name))

            return args_tuple

    def call_loopy(self, t_unit, **kwargs) -> dict:
        """Execute the loopy kernel and profile it."""
        try:
            t_unit = self._loopy_transform_cache[t_unit]
        except KeyError:
            orig_t_unit = t_unit
            t_unit = self.transform_loopy_program(t_unit)
            self._loopy_transform_cache[orig_t_unit] = t_unit
            del orig_t_unit

        evt, result = t_unit(self.queue, **kwargs, allocator=self.allocator)

        if self._wait_event_queue_length is not False:
            prg_name = t_unit.default_entrypoint.name
            wait_event_queue = self._kernel_name_to_wait_event_queue.setdefault(
                prg_name, [])

            wait_event_queue.append(evt)
            if len(wait_event_queue) > self._wait_event_queue_length:
                wait_event_queue.pop(0).wait()

        # Generate the stats here so we don't need to carry around the kwargs
        args_tuple = self._cache_kernel_stats(t_unit, kwargs)

        self.profile_events.append(ProfileEvent(evt, t_unit, args_tuple))

        return result



# }}}


class MPIBasedArrayContext:
    mpi_communicator: "MPI.Comm"


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
    def _dag_to_compiled_func(self, dict_of_named_arrays,
            input_id_to_name_in_program, output_id_to_name_in_program,
            output_template):

        import pytato as pt

        from pytools import ProcessLogger

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
        distributed_partition, _new_mpi_base_tag = number_distributed_tags(
                self.actx.mpi_communicator,
                distributed_partition,
                base_tag=prev_mpi_base_tag)

        assert prev_mpi_base_tag == self.actx.mpi_base_tag
        # FIXME: Updating stuff inside the array context from here is *cough*
        # not super pretty.
        self.actx.mpi_base_tag = _new_mpi_base_tag

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

        from immutabledict import immutabledict
        return _DistributedCompiledFunction(
                actx=self.actx,
                distributed_partition=distributed_partition,
                part_id_to_prg=part_id_to_prg,
                input_id_to_name_in_program=input_id_to_name_in_program,
                output_id_to_name_in_program=output_id_to_name_in_program,
                name_in_program_to_tags=immutabledict(name_in_program_to_tags),
                name_in_program_to_axes=immutabledict(name_in_program_to_axes),
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

    actx: "MPISingleGridWorkBalancingPytatoArrayContext"
    distributed_partition: "DistributedGraphPartition"
    part_id_to_prg: "Mapping[PartId, pt.target.BoundProgram]"
    input_id_to_name_in_program: Mapping[Tuple[Any, ...], str]
    output_id_to_name_in_program: Mapping[Tuple[Any, ...], str]
    name_in_program_to_tags: Mapping[str, FrozenSet[Tag]]
    name_in_program_to_axes: Mapping[str, Tuple["pt.Axis", ...]]
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
        out_dict = execute_distributed_partition(
                self.distributed_partition, self.part_id_to_prg,
                self.actx.queue, self.actx.mpi_communicator,
                allocator=self.actx.allocator,
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


class MPIPytatoArrayContextBase(MPIBasedArrayContext):
    def __init__(
            self, mpi_communicator, queue, *, mpi_base_tag, allocator=None,
            compile_trace_callback: Optional[Callable[[Any, str, Any], None]]
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

        self.mpi_communicator = mpi_communicator
        self.mpi_base_tag = mpi_base_tag

    # FIXME: implement distributed-aware freeze

    def compile(self, f: Callable[..., Any]) -> Callable[..., Any]:
        return _DistributedLazilyPyOpenCLCompilingFunctionCaller(self, f)

    def clone(self):
        # type-ignore-reason: 'DistributedLazyArrayContext' has no 'queue' member
        # pylint: disable=no-member
        return type(self)(self.mpi_communicator, self.queue,
                mpi_base_tag=self.mpi_base_tag,
                allocator=self.allocator)

# }}}


# {{{ distributed + pyopencl

class MPIPyOpenCLArrayContext(ProfilingActx, PyOpenCLArrayContext, MPIBasedArrayContext):
    """An array context for using distributed computation with :mod:`pyopencl`
    eager evaluation.

    .. autofunction:: __init__
    """

    def __init__(self,
            mpi_communicator,
            queue: "pyopencl.CommandQueue",
            *, allocator: Optional["pyopencl.tools.AllocatorBase"] = None,
            wait_event_queue_length: Optional[int] = None,
            force_device_scalars: bool = True) -> None:
        """
        See :class:`arraycontext.impl.pyopencl.PyOpenCLArrayContext` for most
        arguments.
        """
        super().__init__(queue, allocator=allocator,
                wait_event_queue_length=wait_event_queue_length,
                force_device_scalars=force_device_scalars)

        self.mpi_communicator = mpi_communicator

    def clone(self):
        # type-ignore-reason: 'DistributedLazyArrayContext' has no 'queue' member
        # pylint: disable=no-member
        return type(self)(self.mpi_communicator, self.queue,
                allocator=self.allocator,
                wait_event_queue_length=self._wait_event_queue_length,
                force_device_scalars=self._force_device_scalars)

# }}}


# {{{ distributed + pytato array context subclasses

class MPIBasePytatoPyOpenCLArrayContext(
        MPIPytatoArrayContextBase, PytatoPyOpenCLArrayContext):
    """
    .. autofunction:: __init__
    """
    pass


if _HAVE_SINGLE_GRID_WORK_BALANCING:
    class MPISingleGridWorkBalancingPytatoArrayContext(
            MPIPytatoArrayContextBase, SingleGridWorkBalancingPytatoArrayContext):
        """
        .. autofunction:: __init__
        """

    MPIPytatoArrayContext = MPISingleGridWorkBalancingPytatoArrayContext
else:
    MPIPytatoArrayContext = MPIBasePytatoPyOpenCLArrayContext


if _HAVE_FUSION_ACTX:
    class MPIFusionContractorArrayContext(
            MPIPytatoArrayContextBase, FusionContractorArrayContext):
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

        return self.actx_class(
                queue,
                allocator=alloc,
                force_device_scalars=self.force_device_scalars)


class PytestPytatoPyOpenCLArrayContextFactory(
        _PytestPytatoPyOpenCLArrayContextFactory):
    actx_class = PytatoPyOpenCLArrayContext

    def __call__(self):
        _ctx, queue = self.get_command_queue()

        from pyopencl.tools import ImmediateAllocator, MemoryPool
        alloc = MemoryPool(ImmediateAllocator(queue))

        return self.actx_class(queue, allocator=alloc)


register_pytest_array_context_factory("grudge.pyopencl",
        PytestPyOpenCLArrayContextFactory)
register_pytest_array_context_factory("grudge.pytato-pyopencl",
        PytestPytatoPyOpenCLArrayContextFactory)

# }}}


# {{{ actx selection


def _get_single_grid_pytato_actx_class(distributed: bool) -> Type[ArrayContext]:
    if not _HAVE_SINGLE_GRID_WORK_BALANCING:
        warn("No device-parallel actx available, execution will be slow. "
             "Please make sure you have the right branches for loopy "
             "(https://github.com/kaushikcfd/loopy/tree/pytato-array-context-transforms) "  # noqa
             "and meshmode "
             "(https://github.com/kaushikcfd/meshmode/tree/pytato-array-context-transforms).",
             stacklevel=1)
    # lazy, non-distributed
    if not distributed:
        if _HAVE_SINGLE_GRID_WORK_BALANCING:
            return SingleGridWorkBalancingPytatoArrayContext
        else:
            return PytatoPyOpenCLArrayContext
    else:
        # distributed+lazy:
        if _HAVE_SINGLE_GRID_WORK_BALANCING:
            return MPISingleGridWorkBalancingPytatoArrayContext
        else:
            return MPIBasePytatoPyOpenCLArrayContext


def get_reasonable_array_context_class(
        lazy: bool = True, distributed: bool = True,
        fusion: Optional[bool] = None,
        ) -> Type[ArrayContext]:
    """Returns a reasonable :class:`PyOpenCLArrayContext` currently
    supported given the constraints of *lazy* and *distributed*."""
    if fusion is None:
        fusion = lazy

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
                (_HAVE_SINGLE_GRID_WORK_BALANCING or _HAVE_FUSION_ACTX or not lazy))
    return actx_class

# }}}


# vim: foldmethod=marker
