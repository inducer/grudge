__copyright__ = """
Copyright (C) 2021 University of Illinois Board of Trustees
"""

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

import numpy as np

from arraycontext import pytest_generate_tests_for_array_contexts

from grudge.array_context import (
    PytestPyOpenCLArrayContextFactory,
    PytestPytatoPyOpenCLArrayContextFactory,
)


pytest_generate_tests = pytest_generate_tests_for_array_contexts(
        [PytestPyOpenCLArrayContextFactory,
         PytestPytatoPyOpenCLArrayContextFactory])

import logging

import mesh_data
import pytest

import grudge.op as op
from grudge import DiscretizationCollection


logger = logging.getLogger(__name__)
from meshmode import _acf  # noqa: F401


@pytest.mark.parametrize("name", ["interval", "box2d", "box3d"])
def test_geometric_factors_regular_refinement(actx_factory, name):
    from grudge.dt_utils import dt_geometric_factors

    actx = actx_factory()

    # {{{ cases

    if name == "interval":
        builder = mesh_data.BoxMeshBuilder1D()
    elif name == "box2d":
        builder = mesh_data.BoxMeshBuilder2D()
    elif name == "box3d":
        builder = mesh_data.BoxMeshBuilder3D()
    else:
        raise ValueError(f"unknown geometry name: {name}")

    # }}}

    order = 4

    min_factors = []
    for resolution in builder.resolutions:
        mesh = builder.get_mesh(resolution, order)
        dcoll = DiscretizationCollection(actx, mesh, order=order)
        min_factors.append(
            actx.to_numpy(
                op.nodal_min(dcoll, "vol", actx.thaw(dt_geometric_factors(dcoll))))
        )

    # Resolution is doubled each refinement, so the ratio of consecutive
    # geometric factors should satisfy: gfi+1 / gfi = 2
    min_factors = np.asarray(min_factors)
    ratios = min_factors[:-1] / min_factors[1:]
    assert np.all(np.isclose(ratios, 2))

    # Make sure it works with empty meshes
    mesh = builder.get_mesh(0)
    dcoll = DiscretizationCollection(actx, mesh, order=order)
    factors = actx.thaw(dt_geometric_factors(dcoll))  # noqa: F841


@pytest.mark.parametrize("name", ["interval", "box2d", "box3d"])
def test_non_geometric_factors(actx_factory, name):
    from grudge.dt_utils import dt_non_geometric_factors

    actx = actx_factory()

    # {{{ cases

    if name == "interval":
        builder = mesh_data.BoxMeshBuilder1D()
    elif name == "box2d":
        builder = mesh_data.BoxMeshBuilder2D()
    elif name == "box3d":
        builder = mesh_data.BoxMeshBuilder3D()
    else:
        raise ValueError(f"unknown geometry name: {name}")

    # }}}

    factors = []
    degrees = list(range(1, 8))
    for degree in degrees:
        mesh = builder.get_mesh(1, degree)
        dcoll = DiscretizationCollection(actx, mesh, order=degree)
        factors.append(min(dt_non_geometric_factors(dcoll)))

    # Crude estimate, factors should behave like 1/N**2
    factors = np.asarray(factors)
    lower_bounds = 1/(np.asarray(degrees)**2)
    upper_bounds = 6.295*lower_bounds

    assert all(lower_bounds <= factors)
    assert all(factors <= upper_bounds)


def test_build_jacobian(actx_factory):
    actx = actx_factory()
    import meshmode.mesh.generation as mgen

    mesh = mgen.generate_regular_rect_mesh(a=[0], b=[1], nelements_per_axis=(3,))
    assert mesh.dim == 1

    dcoll = DiscretizationCollection(actx, mesh, order=1)

    def rhs(x):
        return 3*x**2 + 2*x + 5

    from pytools.obj_array import make_obj_array
    base_state = make_obj_array([dcoll.zeros(actx)+2])

    from grudge.tools import build_jacobian
    mat = build_jacobian(actx, rhs, base_state, 1e-5)

    assert np.array_equal(mat, np.diag(np.diag(mat)))
    assert np.allclose(np.diag(mat), 3*2*2 + 2)


@pytest.mark.parametrize("dim", [1, 2])
@pytest.mark.parametrize("degree", [2, 4])
def test_wave_dt_estimate(actx_factory, dim, degree, visualize=False):
    actx = actx_factory()

    import meshmode.mesh.generation as mgen

    a = [0, 0, 0]
    b = [1, 1, 1]
    mesh = mgen.generate_regular_rect_mesh(
            a=a[:dim], b=b[:dim],
            nelements_per_axis=(3,)*dim)
    assert mesh.dim == dim

    dcoll = DiscretizationCollection(actx, mesh, order=degree)

    from grudge.models.wave import WeakWaveOperator
    wave_op = WeakWaveOperator(dcoll, c=1)
    rhs = actx.compile(
            lambda w: wave_op.operator(t=0, w=w))

    from pytools.obj_array import make_obj_array
    fields = make_obj_array([dcoll.zeros(actx) for i in range(dim+1)])

    from grudge.tools import build_jacobian
    mat = build_jacobian(actx, rhs, fields, 1)

    import numpy.linalg as la
    eigvals = la.eigvals(mat)

    assert (eigvals.real <= 1e-12).all()

    import sympy as sp
    from leap.rk import RK4MethodBuilder, stability_function
    stab_func = sp.lambdify(*stability_function(
        RK4MethodBuilder.a_explicit,
        RK4MethodBuilder.output_coeffs))

    dt_est = actx.to_numpy(wave_op.estimate_rk4_timestep(actx, dcoll))

    if visualize:
        re, im = np.mgrid[-4:1:30j, -5:5:30j]
        sf_grid = np.abs(stab_func(re+1j*im))

        import matplotlib.pyplot as plt
        plt.contour(re, im, sf_grid, [0.25, 0.5, 0.75, 0.9, 1, 1.1])
        plt.colorbar()
        plt.plot(dt_est * eigvals.real, dt_est * eigvals.imag, "x")
        plt.grid()
        plt.show()

    thresh = 1+1e-8
    max_stab = np.max(np.abs(stab_func(dt_est*eigvals)))
    assert max_stab < thresh, max_stab

    dt_factors = 2**np.linspace(0, 4, 40)[1:]
    stable_dt_factors = [
            dt_factor
            for dt_factor in dt_factors
            if np.max(np.abs(stab_func(dt_factor*dt_est*eigvals))) < thresh]

    if stable_dt_factors:
        print(f"Stable timestep is {max(stable_dt_factors):.2f}x the estimate")
    else:
        print("Stable timestep estimate appears to be sharp")
    assert not stable_dt_factors or max(stable_dt_factors) < 1.5, stable_dt_factors


# You can test individual routines by typing
# $ python test_grudge.py 'test_routine()'

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        pytest.main([__file__])
