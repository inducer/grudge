__copyright__ = "Copyright (C) 2021 University of Illinois Board of Trustees"

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

from arraycontext import thaw

from grudge import DiscretizationCollection
from grudge.dof_desc import DISCR_TAG_BASE

import grudge.op as op

from pytools.obj_array import make_obj_array

import pytest

from grudge.array_context import PytestPyOpenCLArrayContextFactory
from arraycontext import pytest_generate_tests_for_array_contexts
pytest_generate_tests = pytest_generate_tests_for_array_contexts(
        [PytestPyOpenCLArrayContextFactory])

import logging

logger = logging.getLogger(__name__)


def test_variable_transformations(actx_factory):
    from grudge.models.euler import (
        primitive_to_conservative_vars,
        conservative_to_primitive_vars,
        entropy_to_conservative_vars,
        conservative_to_entropy_vars
    )

    actx = actx_factory()
    gamma = 1.4  # Adiabatic expansion factor for single-gas Euler model

    def vortex(x_vec, t=0):
        mach = 0.5    # Mach number
        _x0 = 5
        epsilon = 1   # vortex strength
        x, y = x_vec
        actx = x.array_context

        fxyt = 1 - (((x - _x0) - t)**2 + y**2)
        expterm = actx.np.exp(fxyt/2)

        u = 1 - (epsilon*y/(2*np.pi))*expterm
        v = ((epsilon*(x - _x0) - t)/(2*np.pi))*expterm

        velocity = make_obj_array([u, v])
        mass = (
            1 - ((epsilon**2*(gamma - 1)*mach**2)/(8*np.pi**2))*actx.np.exp(fxyt)
        ) ** (1 / (gamma - 1))
        p = (mass**gamma)/(gamma*mach**2)

        return primitive_to_conservative_vars((mass, velocity, p), gamma=gamma)

    from meshmode.mesh.generation import generate_regular_rect_mesh

    dim = 2
    res = 5
    mesh = generate_regular_rect_mesh(
        a=(0, -5),
        b=(20, 5),
        nelements_per_axis=(2*res, res),
        periodic=(True, True))

    from meshmode.discretization.poly_element import \
        default_simplex_group_factory

    order = 3
    dcoll = DiscretizationCollection(
        actx, mesh,
        discr_tag_to_group_factory={
            DISCR_TAG_BASE: default_simplex_group_factory(dim, order)
        }
    )

    # Fields in conserved variables
    fields = vortex(thaw(dcoll.nodes(), actx))

    # Map back and forth between primitive and conserved vars
    fields_prim = conservative_to_primitive_vars(fields, gamma=gamma)
    prim_fields_to_cons = primitive_to_conservative_vars(fields_prim, gamma=gamma)

    assert op.norm(
        dcoll, prim_fields_to_cons.mass - fields.mass, np.inf) < 1e-13
    assert op.norm(
        dcoll, prim_fields_to_cons.energy - fields.energy, np.inf) < 1e-13
    assert op.norm(
        dcoll, prim_fields_to_cons.momentum - fields.momentum, np.inf) < 1e-13

    # Map back and forth between entropy and conserved vars
    fields_ev = conservative_to_entropy_vars(fields, gamma=gamma)
    ev_fields_to_cons = entropy_to_conservative_vars(fields_ev, gamma=gamma)

    assert op.norm(
        dcoll, ev_fields_to_cons.mass - fields.mass, np.inf) < 1e-13
    assert op.norm(
        dcoll, ev_fields_to_cons.energy - fields.energy, np.inf) < 1e-13
    assert op.norm(
        dcoll, ev_fields_to_cons.momentum - fields.momentum, np.inf) < 1e-13


# You can test individual routines by typing
# $ python test_grudge.py 'test_routine()'

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        pytest.main([__file__])
