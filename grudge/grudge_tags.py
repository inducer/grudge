from pytools.tag import Tag, UniqueTag
from meshmode.transform_metadata import IsDOFArray, IsOpArray, ParameterValue, KernelDataTag

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

#class KernelDataTag(Tag):
#
#    def __init__(self, kernel_data):
#        self.kernel_data = kernel_data
