from pytools.tag import UniqueTag, tag_dataclass
from .dof_desc import DOFDesc


@tag_dataclass
class DiscretizationElementAxisTag(UniqueTag):
    """
    Tagged to an array's axis travesing elements in a discretization.

    :attr dd: One of the domain discretization tags defined in
        :mod:`grudge.dof_desc`.
    """
    dd: DOFDesc


@tag_dataclass
class DiscretizationDOFAxisTag(UniqueTag):
    """
    Tagged to an array's axis travesing an DOFs in a discretization.

    :attr dd: One of the domain discretization tags defined in
        :mod:`grudge.dof_desc`.
    """
    dd: DOFDesc


@tag_dataclass
class DicretizationElementProjector(UniqueTag):
    """
    Tagged to an array used an indirection map from one discretization's
    elements to another discretrization's elements.
    """
    from_dd: DOFDesc
    to_dd: DOFDesc


@tag_dataclass
class DicretizationDOFProjector(UniqueTag):
    """
    Tagged to an array used an indirection map from one discretization's
    elements to another discretrization's DOFs.
    """
    from_dd: DOFDesc
    to_dd: DOFDesc
