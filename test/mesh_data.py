from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, ClassVar

import numpy as np
from typing_extensions import override

import meshmode.mesh.generation as mgen
from meshmode.mesh.io import read_gmsh


if TYPE_CHECKING:
    from collections.abc import Hashable, Sequence

    from meshmode.mesh import Mesh


class MeshBuilder(ABC):
    resolutions: ClassVar[Sequence[Hashable]]
    ambient_dim: ClassVar[int]

    @abstractmethod
    def get_mesh(
             self,
             resolution: Hashable,
             mesh_order: int | None = None
         ) -> Mesh:
        ...


class _GmshMeshBuilder(MeshBuilder):
    resolutions: ClassVar[Sequence[Hashable]] = [None]

    def __init__(self, filename: str) -> None:
        self._mesh_fn: str = filename

    @override
    def get_mesh(self, resolution, mesh_order=None) -> Mesh:
        assert resolution is None
        assert mesh_order is None
        return read_gmsh(self._mesh_fn, force_ambient_dim=self.ambient_dim)


class GmshMeshBuilder2D(_GmshMeshBuilder):
    ambient_dim: ClassVar[int] = 2


class GmshMeshBuilder3D(_GmshMeshBuilder):
    ambient_dim: ClassVar[int] = 3


class Curve2DMeshBuilder(MeshBuilder):
    ambient_dim: ClassVar[int] = 2
    resolutions: ClassVar[Sequence[Hashable]] = [16, 32, 64, 128]

    @override
    def get_mesh(self, resolution, mesh_order=None):
        if mesh_order is None:
            mesh_order = 4
        return mgen.make_curve_mesh(
                self.curve_fn,      # pylint: disable=no-member
                np.linspace(0.0, 1.0, resolution + 1),
                mesh_order)


class EllipseMeshBuilder(Curve2DMeshBuilder):
    def __init__(self, radius=3.1, aspect_ratio: float = 2):
        self.radius: float = radius
        self.aspect_ratio: float = aspect_ratio

    @property
    def curve_fn(self):
        return lambda t: self.radius * mgen.ellipse(self.aspect_ratio, t)


class StarfishMeshBuilder(Curve2DMeshBuilder):
    narms: ClassVar[int] = 5
    amplitude: ClassVar[float] = 0.25

    @property
    def curve_fn(self):
        return mgen.NArmedStarfish(self.narms, self.amplitude)


class SphereMeshBuilder(MeshBuilder):
    ambient_dim: ClassVar[int] = 3

    resolutions: ClassVar[Sequence[Hashable]] = [0, 1, 2, 3]

    radius: float

    def __init__(self, radius: float = 1):
        self.radius = radius

    @override
    def get_mesh(self, resolution, mesh_order=4):
        from meshmode.mesh.generation import generate_sphere
        return generate_sphere(self.radius, order=mesh_order,
                uniform_refinement_rounds=resolution)


class SpheroidMeshBuilder(MeshBuilder):
    ambient_dim: ClassVar[int] = 3

    resolutions: ClassVar[Sequence[Hashable]] = [0, 1, 2, 3]

    radius: float
    aspect_ratio: float

    def __init__(self, radius: float = 1, aspect_ratio: float = 2):
        self.radius = radius
        self.aspect_ratio = aspect_ratio

    @override
    def get_mesh(self, resolution, mesh_order=4):
        from meshmode.mesh.generation import generate_sphere
        mesh = generate_sphere(self.radius, order=mesh_order,
                uniform_refinement_rounds=resolution)

        from meshmode.mesh.processing import affine_map
        return affine_map(mesh, A=np.diag([1.0, 1.0, self.aspect_ratio]))


class _BoxMeshBuilderBase(MeshBuilder):
    resolutions: ClassVar[Sequence[Hashable]] = [4, 8, 16]
    mesh_order: ClassVar[int] = 1

    a: ClassVar[tuple[float, ...]] = (-0.5, -0.5, -0.5)
    b: ClassVar[tuple[float, ...]] = (+0.5, +0.5, +0.5)

    @override
    def get_mesh(self, resolution, mesh_order=4):
        if not isinstance(resolution, list | tuple):
            resolution = (resolution,) * self.ambient_dim

        return mgen.generate_regular_rect_mesh(
                a=self.a, b=self.b,
                nelements_per_axis=resolution,
                order=mesh_order)


class BoxMeshBuilder1D(_BoxMeshBuilderBase):
    ambient_dim: ClassVar[int] = 1


class BoxMeshBuilder2D(_BoxMeshBuilderBase):
    ambient_dim: ClassVar[int] = 2


class BoxMeshBuilder3D(_BoxMeshBuilderBase):
    ambient_dim: ClassVar[int] = 2


class WarpedRectMeshBuilder(MeshBuilder):
    resolutions: ClassVar[Sequence[Hashable]] = [4, 6, 8]

    def __init__(self, dim):
        self.dim: int = dim

    @override
    def get_mesh(self, resolution, mesh_order=4):
        return mgen.generate_warped_rect_mesh(
                dim=self.dim, order=mesh_order, nelements_side=resolution)
