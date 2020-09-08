import numpy as np


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
        from meshmode.mesh.generation import make_curve_mesh
        return make_curve_mesh(
                self.curve_fn,
                np.linspace(0.0, 1.0, resolution + 1),
                mesh_order)


class EllipseMeshBuilder(Curve2DMeshBuilder):
    radius = 3.1
    aspect_ratio = 2.0

    @property
    def curve_fn(self):
        from meshmode.mesh.generation import ellipse
        return lambda t: self.radius * ellipse(self.aspect_ratio, t)


class StarfishMeshBuilder(Curve2DMeshBuilder):
    narms = 5
    amplitude = 0.25

    @property
    def curve_fn(self):
        from meshmode.mesh.generation import NArmedStarfish
        return NArmedStarfish(self.narms, self.amplitude)


class SphereMeshBuilder(MeshBuilder):
    ambient_dim = 3

    resolutions = [0, 1, 2, 3]
    radius = 1.0

    def get_mesh(self, resolution, mesh_order):
        from meshmode.mesh.generation import generate_icosphere
        return generate_icosphere(self.radius, order=mesh_order,
                uniform_refinement_rounds=resolution)


class SpheroidMeshBuilder(MeshBuilder):
    ambient_dim = 3

    mesh_order = 4
    resolutions = [1.0, 0.11, 0.05]
    # resolutions = [1.0, 0.11, 0.05, 0.03, 0.015]

    @property
    def radius(self):
        return 0.25

    @property
    def diameter(self):
        return 2.0 * self.radius

    @property
    def aspect_ratio(self):
        return 2.0

    def get_mesh(self, resolution, mesh_order):
        from meshmode.mesh.io import ScriptSource
        source = ScriptSource("""
            SetFactory("OpenCASCADE");
            Sphere(1) = {{0, 0, 0, {r}}};
            Dilate {{ {{0, 0, 0}}, {{ {r}, {r}, {rr} }} }} {{ Volume{{1}}; }}
            """.format(r=self.diameter, rr=self.aspect_ratio * self.diameter),
            "geo"
        )

        from meshmode.mesh.io import generate_gmsh
        mesh = generate_gmsh(source, 2, order=mesh_order,
                other_options=[
                    "-optimize_ho",
                    "-string", "Mesh.CharacteristicLengthMax = %g;" % resolution
                    ],
                target_unit="MM")

        from meshmode.mesh.processing import perform_flips
        return perform_flips(mesh, np.ones(mesh.nelements))


class BoxMeshBuilder(MeshBuilder):
    ambient_dim = 2

    mesh_order = 1
    resolutions = [8, 16, 32]

    a = (-0.5, -0.5, -0.5)
    b = (+0.5, +0.5, +0.5)

    def get_mesh(self, resolution, mesh_order):
        if not isinstance(resolution, (list, tuple)):
            resolution = (resolution,) * self.ambient_dim

        from meshmode.mesh.generation import generate_regular_rect_mesh
        mesh = generate_regular_rect_mesh(
                a=self.a, b=self.b,
                n=resolution,
                order=mesh_order)

        return mesh
