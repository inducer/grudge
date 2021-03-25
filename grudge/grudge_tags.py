from pytools.tag import Tag, UniqueTag
from meshmode.dof_array import IsDOFArray

class IsVecDOFArray(Tag):
    pass


class IsFaceDOFArray(Tag):
    pass


class IsVecOpDOFArray(Tag):
    pass


class ParameterValue(UniqueTag):
    
    def __init__(self, value):
        self.value = value

