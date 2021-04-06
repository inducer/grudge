from pytools.tag import Tag, UniqueTag
from meshmode.array_context import IsDOFArray, ParameterValue

class IsVecDOFArray(Tag):
    pass


class IsFaceDOFArray(Tag):
    pass


class IsVecOpArray(Tag):
    pass
