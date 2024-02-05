import numpy as np
import meshmode.mesh.generation as mgen


class MeshBuilder:
    order = 4
    mesh_order = None

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

        if self.mesh_order is None:
            self.mesh_order = self.order

    @property
    def ambient_dim(self):
        raise NotImplementedError

    @property
    def resolutions(self):
        raise NotImplementedError

    def get_mesh(self, resolution, mesh_order):
        raise NotImplementedError


class Curve2DMeshBuilder(MeshBuilder):
    ambient_dim = 2
    resolutions = [16, 32, 64, 128]

    def get_mesh(self, resolution, mesh_order):
        return mgen.make_curve_mesh(
                self.curve_fn,      # pylint: disable=no-member
                np.linspace(0.0, 1.0, resolution + 1),
                mesh_order)


class EllipseMeshBuilder(Curve2DMeshBuilder):
    radius = 3.1
    aspect_ratio = 2.0

    @property
    def curve_fn(self):
        return lambda t: self.radius * mgen.ellipse(self.aspect_ratio, t)


class StarfishMeshBuilder(Curve2DMeshBuilder):
    narms = 5
    amplitude = 0.25

    @property
    def curve_fn(self):
        return mgen.NArmedStarfish(self.narms, self.amplitude)


class SphereMeshBuilder(MeshBuilder):
    ambient_dim = 3

    resolutions = [0, 1, 2, 3]
    radius = 1.0

    def get_mesh(self, resolution, mesh_order):
        from meshmode.mesh.generation import generate_sphere
        return generate_sphere(self.radius, order=mesh_order,
                uniform_refinement_rounds=resolution)


class SpheroidMeshBuilder(MeshBuilder):
    ambient_dim = 3

    mesh_order = 4
    resolutions = [0, 1, 2, 3]

    radius = 1.0
    aspect_ratio = 2.0

    def get_mesh(self, resolution, mesh_order):
        from meshmode.mesh.generation import generate_sphere
        mesh = generate_sphere(self.radius, order=mesh_order,
                uniform_refinement_rounds=resolution)

        from meshmode.mesh.processing import affine_map
        return affine_map(mesh, A=np.diag([1.0, 1.0, self.aspect_ratio]))


class BoxMeshBuilder(MeshBuilder):
    ambient_dim = 2
    group_cls = None

    mesh_order = 1
    resolutions = [4, 8, 16]

    a = (-0.5, -0.5, -0.5)
    b = (+0.5, +0.5, +0.5)

    def get_mesh(self, resolution, mesh_order):
        if not isinstance(resolution, (list, tuple)):
            resolution = (resolution,) * self.ambient_dim

        return mgen.generate_regular_rect_mesh(
                a=self.a, b=self.b,
                nelements_per_axis=resolution,
                group_cls=self.group_cls,
                order=mesh_order)


class WarpedRectMeshBuilder(MeshBuilder):
    resolutions = [4, 6, 8]

    def __init__(self, dim):
        self.dim = dim

    def get_mesh(self, resolution, mesh_order):
        return mgen.generate_warped_rect_mesh(
                dim=self.dim, order=4, nelements_side=6)
