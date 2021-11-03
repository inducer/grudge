from pytools.tag import UniqueTag, tag_dataclass
from .dof_desc import DOFDesc


class DiscretizationEntityTag(UniqueTag):
    pass


@tag_dataclass
class DiscretizationElementAxisTag(DiscretizationEntityTag):
    """
    Tagged to an array's axis representing element indices in the
    discretization identifier by :attr:`dd`.

    :attr dd: One of the domain discretization tags defined in
        :mod:`grudge.dof_desc`.
    """
    dd: DOFDesc


@tag_dataclass
class DiscretizationDOFAxisTag(DiscretizationEntityTag):
    """
    Tagged to an array's axis representing DOF-indices in the
    discretization identifier by :attr:`dd`.

    :attr dd: One of the domain discretization tags defined in
        :mod:`grudge.dof_desc`.
    """
    dd: DOFDesc
