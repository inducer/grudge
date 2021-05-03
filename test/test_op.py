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

import meshmode.mesh.generation as mgen
from meshmode.dof_array import thaw

from pytools.obj_array import make_obj_array

from grudge import sym, op, DiscretizationCollection
from grudge.dof_desc import DOFDesc

import pytest
from meshmode.array_context import (  # noqa
        pytest_generate_tests_for_pyopencl_array_context
        as pytest_generate_tests)

import logging

logger = logging.getLogger(__name__)


# {{{ gradient

@pytest.mark.parametrize("dim", [1, 2, 3])
@pytest.mark.parametrize("form", ["strong", "weak"])
@pytest.mark.parametrize("order", [2, 3])
@pytest.mark.parametrize(("vectorize", "stack"), [
    (False, False),
    (True, False),
    (True, True)
    ])
def test_gradient(actx_factory, dim, form, order, vectorize, stack):
    actx = actx_factory()

    from pytools.convergence import EOCRecorder
    eoc_rec = EOCRecorder()

    for n in [4, 6, 8]:
        mesh = mgen.generate_regular_rect_mesh(
                a=(-1,)*dim, b=(1,)*dim,
                nelements_per_axis=(n,)*dim)

        dcoll = DiscretizationCollection(actx, mesh, order=order)

        def f(x):
            result = dcoll.zeros(actx) + 1.
            for i in range(dim-1):
                result *= actx.np.sin(np.pi*x[i])
            result *= actx.np.cos(np.pi/2*x[dim-1])
            return result

        def grad_f(x):
            result = make_obj_array([dcoll.zeros(actx) + 1. for _ in range(dim)])
            for i in range(dim-1):
                for j in range(i):
                    result[i] *= actx.np.sin(np.pi*x[j])
                result[i] *= np.pi*actx.np.cos(np.pi*x[i])
                for j in range(i+1, dim-1):
                    result[i] *= actx.np.sin(np.pi*x[j])
                result[i] *= actx.np.cos(np.pi/2*x[dim-1])
            for j in range(dim-1):
                result[dim-1] *= actx.np.sin(np.pi*x[j])
            result[dim-1] *= -np.pi/2*actx.np.sin(np.pi/2*x[dim-1])
            return result

        x = thaw(actx, op.nodes(dcoll))

        if vectorize:
            u = make_obj_array([(idim+1)*f(x) for idim in range(dim)])
        else:
            u = f(x)

        def get_flux(u_tpair):
            dd = u_tpair.dd
            dd_allfaces = dd.with_dtag("all_faces")
            normal = thaw(actx, op.normal(dcoll, dd))
            u_avg = u_tpair.avg
            if vectorize:
                if stack:
                    flux = np.outer(u_avg, normal)
                else:
                    flux = make_obj_array([u_avg_i * normal for u_avg_i in u_avg])
            else:
                flux = u_avg * normal
            return op.project(dcoll, dd, dd_allfaces, flux)

        dd_intfaces = DOFDesc("int_faces")
        dd_allfaces = DOFDesc("all_faces")

        if form == "strong":
            # FIXME: this doesn't work
            u_intfaces = op.project(dcoll, "vol", dd_intfaces, u)
            grad_u = op.inverse_mass(dcoll,
                op.local_grad(dcoll, u, stack=stack)
                -  # noqa: W504
                op.face_mass(dcoll,
                    dd_allfaces,
                    # Note: no boundary flux terms here because u_ext == u_int == 0
                    get_flux(sym.TracePair(dd_intfaces,
                        interior=u_intfaces,
                        exterior=u_intfaces))
                    -  # noqa: W504
                    get_flux(op.interior_trace_pair(dcoll, u)))
                )
        elif form == "weak":
            grad_u = op.inverse_mass(dcoll,
                -op.weak_local_grad(dcoll, u, stack=stack)
                +  # noqa: W504
                op.face_mass(dcoll,
                    dd_allfaces,
                    # Note: no boundary flux terms here because u_ext == u_int == 0
                    get_flux(op.interior_trace_pair(dcoll, u)))
                )

        if vectorize:
            expected_grad_u = make_obj_array(
                [(idim+1)*grad_f(x) for idim in range(dim)])
            if stack:
                expected_grad_u = np.stack(expected_grad_u, axis=0)
        else:
            expected_grad_u = grad_f(x)

        rel_linf_err = (
            op.norm(dcoll, grad_u - expected_grad_u, np.inf)
            / op.norm(dcoll, expected_grad_u, np.inf))
        eoc_rec.add_data_point(1./n, rel_linf_err)

    print("L^inf error:")
    print(eoc_rec)
    assert(eoc_rec.order_estimate() >= order - 0.5
                or eoc_rec.max_error() < 1e-11)

# }}}


# You can test individual routines by typing
# $ python test_grudge.py 'test_routine()'

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        pytest.main([__file__])

# vim: fdm=marker
