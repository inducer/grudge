from pytools.tag import Tag, UniqueTag
from meshmode.transform_metadata import IsDOFArray, IsOpArray, ParameterValue, EinsumArgsTags

class KernelDataTag(Tag): # Delete this when no longer needed
    """A tag that applies to :class:`loopy.LoopKernel`. Kernel data provided
    with this tag can be later applied to the kernel. This is used, for
    instance, to specify kernel data in einsum kernels."""

    def __init__(self, kernel_data):
        self.kernel_data = kernel_data


class IsVecDOFArray(Tag):
    pass

class IsFaceDOFArray(Tag):
    pass

class IsVecOpArray(Tag):
    pass

class IsSepVecDOFArray(Tag):
    pass

class IsSepVecOpArray(Tag):
    pass

class IsFaceMassOpArray(Tag):
    pass

class IsFourAxisDOFArray(Tag):
    pass
