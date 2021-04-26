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
import numpy.linalg as la

from meshmode import _acf       # noqa: F401
from meshmode.dof_array import flatten, thaw
from meshmode.array_context import (  # noqa
        pytest_generate_tests_for_pyopencl_array_context
        as pytest_generate_tests)
import meshmode.mesh.generation as mgen

from pytools.obj_array import flat_obj_array, make_obj_array
import grudge.op as op

from grudge import DiscretizationCollection
import grudge.dof_desc as dof_desc

import pytest

import logging

logger = logging.getLogger(__name__)


# {{{ mass operator trig integration

@pytest.mark.parametrize("ambient_dim", [1, 2, 3])
@pytest.mark.parametrize("quad_tag", [dof_desc.QTAG_NONE, "OVSMP"])
def test_mass_mat_trig(actx_factory, ambient_dim, quad_tag):
    """Check the integral of some trig functions on an interval using the mass
    matrix.
    """
    actx = actx_factory()

    nelements = 17
    order = 4

    a = -4.0 * np.pi
    b = +9.0 * np.pi
    true_integral = 13*np.pi/2 * (b - a)**(ambient_dim - 1)

    from meshmode.discretization.poly_element import \
        QuadratureSimplexGroupFactory

    dd_quad = dof_desc.DOFDesc(dof_desc.DTAG_VOLUME_ALL, quad_tag)

    if quad_tag is dof_desc.QTAG_NONE:
        quad_tag_to_group_factory = {}
    else:
        quad_tag_to_group_factory = {
            quad_tag: QuadratureSimplexGroupFactory(order=2*order)
        }

    mesh = mgen.generate_regular_rect_mesh(
        a=(a,)*ambient_dim, b=(b,)*ambient_dim,
        n=(nelements,)*ambient_dim,
        order=1
    )
    dcoll = DiscretizationCollection(
        actx, mesh, order=order,
        quad_tag_to_group_factory=quad_tag_to_group_factory
    )

    def f(x):
        return actx.np.sin(x[0])**2

    volm_disc = dcoll.discr_from_dd(dof_desc.DD_VOLUME)
    x_volm = thaw(actx, volm_disc.nodes())
    f_volm = f(x_volm)
    ones_volm = volm_disc.zeros(actx) + 1

    quad_disc = dcoll.discr_from_dd(dd_quad)
    x_quad = thaw(actx, quad_disc.nodes())
    f_quad = f(x_quad)
    ones_quad = quad_disc.zeros(actx) + 1

    mop = op.mass_operator(dcoll, dd_quad, f_quad)
    num_integral_1 = np.dot(
            actx.to_numpy(flatten(ones_volm)),
            actx.to_numpy(flatten(mop)))
    err_1 = abs(num_integral_1 - true_integral)
    assert err_1 < 1e-9, err_1

    # num_integral_2 = np.dot(f_volm, actx.to_numpy(flatten(mass_op(f=ones_quad))))
    # err_2 = abs(num_integral_2 - true_integral)
    # assert err_2 < 1.0e-9, err_2

    # if quad_tag is dof_desc.QTAG_NONE:
    #     # NOTE: `integral` always makes a square mass matrix and
    #     # `QuadratureSimplexGroupFactory` does not have a `mass_matrix` method.
    #     num_integral_3 = bind(discr,
    #             dof_desc.integral(sym_f, dd=dd_quad))(f=f_quad)
    #     err_3 = abs(num_integral_3 - true_integral)
    #     assert err_3 < 5.0e-10, err_3

# }}}


# vim: foldmethod=marker
