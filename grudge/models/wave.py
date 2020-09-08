"""Wave equation operators."""

__copyright__ = "Copyright (C) 2009 Andreas Kloeckner"

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
from grudge.models import HyperbolicOperator
from meshmode.mesh import BTAG_ALL, BTAG_NONE
from grudge import sym
from pytools.obj_array import flat_obj_array


# {{{ constant-velocity

class WeakWaveOperator(HyperbolicOperator):
    r"""This operator discretizes the wave equation
    :math:`\partial_t^2 u = c^2 \Delta u`.

    To be precise, we discretize the hyperbolic system

    .. math::

        \partial_t u - c \\nabla \\cdot v = 0

        \partial_t v - c \\nabla u = 0

    The sign of :math:`v` determines whether we discretize the forward or the
    backward wave equation.

    :math:`c` is assumed to be constant across all space.
    """

    def __init__(self, c, ambient_dim, source_f=0,
            flux_type="upwind",
            dirichlet_tag=BTAG_ALL,
            dirichlet_bc_f=0,
            neumann_tag=BTAG_NONE,
            radiation_tag=BTAG_NONE):
        assert isinstance(ambient_dim, int)

        self.c = c
        self.ambient_dim = ambient_dim
        self.source_f = source_f

        if self.c > 0:
            self.sign = 1
        else:
            self.sign = -1

        self.dirichlet_tag = dirichlet_tag
        self.neumann_tag = neumann_tag
        self.radiation_tag = radiation_tag

        self.dirichlet_bc_f = dirichlet_bc_f

        self.flux_type = flux_type

    def flux(self, w):
        u = w[0]
        v = w[1:]
        normal = sym.normal(w.dd, self.ambient_dim)

        central_flux_weak = -self.c*flat_obj_array(
                np.dot(v.avg, normal),
                u.avg * normal)

        if self.flux_type == "central":
            return central_flux_weak
        elif self.flux_type == "upwind":
            return central_flux_weak - self.c*self.sign*flat_obj_array(
                    0.5*(u.ext-u.int),
                    0.5*(normal * np.dot(normal, v.ext-v.int)))
        else:
            raise ValueError("invalid flux type '%s'" % self.flux_type)

    def sym_operator(self):
        d = self.ambient_dim

        w = sym.make_sym_array("w", d+1)
        u = w[0]
        v = w[1:]

        # boundary conditions -------------------------------------------------

        # dirichlet BCs -------------------------------------------------------
        dir_u = sym.cse(sym.project("vol", self.dirichlet_tag)(u))
        dir_v = sym.cse(sym.project("vol", self.dirichlet_tag)(v))
        if self.dirichlet_bc_f:
            # FIXME
            from warnings import warn
            warn("Inhomogeneous Dirichlet conditions on the wave equation "
                    "are still having issues.")

            dir_g = sym.Field("dir_bc_u")
            dir_bc = flat_obj_array(2*dir_g - dir_u, dir_v)
        else:
            dir_bc = flat_obj_array(-dir_u, dir_v)

        dir_bc = sym.cse(dir_bc, "dir_bc")

        # neumann BCs ---------------------------------------------------------
        neu_u = sym.cse(sym.project("vol", self.neumann_tag)(u))
        neu_v = sym.cse(sym.project("vol", self.neumann_tag)(v))
        neu_bc = sym.cse(flat_obj_array(neu_u, -neu_v), "neu_bc")

        # radiation BCs -------------------------------------------------------
        rad_normal = sym.normal(self.radiation_tag, d)

        rad_u = sym.cse(sym.project("vol", self.radiation_tag)(u))
        rad_v = sym.cse(sym.project("vol", self.radiation_tag)(v))

        rad_bc = sym.cse(flat_obj_array(
                0.5*(rad_u - self.sign*np.dot(rad_normal, rad_v)),
                0.5*rad_normal*(np.dot(rad_normal, rad_v) - self.sign*rad_u)
                ), "rad_bc")

        # entire operator -----------------------------------------------------
        def flux(pair):
            return sym.project(pair.dd, "all_faces")(self.flux(pair))

        result = sym.InverseMassOperator()(
                flat_obj_array(
                    -self.c*np.dot(sym.stiffness_t(self.ambient_dim), v),
                    -self.c*(sym.stiffness_t(self.ambient_dim)*u)
                    )

                - sym.FaceMassOperator()(flux(sym.int_tpair(w))
                    + flux(sym.bv_tpair(self.dirichlet_tag, w, dir_bc))
                    + flux(sym.bv_tpair(self.neumann_tag, w, neu_bc))
                    + flux(sym.bv_tpair(self.radiation_tag, w, rad_bc))

                    ))

        result[0] += self.source_f

        return result

    def check_bc_coverage(self, mesh):
        from meshmode.mesh import check_bc_coverage
        check_bc_coverage(mesh, [
            self.dirichlet_tag,
            self.neumann_tag,
            self.radiation_tag])

    def max_eigenvalue(self, t, fields=None, discr=None):
        return abs(self.c)


# }}}


# {{{ variable-velocity

class VariableCoefficientWeakWaveOperator(HyperbolicOperator):
    r"""This operator discretizes the wave equation
    :math:`\partial_t^2 u = c^2 \Delta u`.

    To be precise, we discretize the hyperbolic system

    .. math::

        \partial_t u - c \\nabla \\cdot v = 0

        \partial_t v - c \\nabla u = 0

    The sign of :math:`v` determines whether we discretize the forward or the
    backward wave equation.

    :math:`c` is assumed to be constant across all space.
    """

    def __init__(self, c, ambient_dim, source_f=0,
            flux_type="upwind",
            dirichlet_tag=BTAG_ALL,
            dirichlet_bc_f=0,
            neumann_tag=BTAG_NONE,
            radiation_tag=BTAG_NONE):
        assert isinstance(ambient_dim, int)

        self.c = c
        self.ambient_dim = ambient_dim
        self.source_f = source_f

        self.sign = sym.If(sym.Comparison(
                            self.c, ">", 0),
                                            np.int32(1), np.int32(-1))

        self.dirichlet_tag = dirichlet_tag
        self.neumann_tag = neumann_tag
        self.radiation_tag = radiation_tag

        self.dirichlet_bc_f = dirichlet_bc_f

        self.flux_type = flux_type

    def flux(self, w):
        c = w[0]
        u = w[1]
        v = w[2:]
        normal = sym.normal(w.dd, self.ambient_dim)

        flux_central_weak = -0.5 * flat_obj_array(
            np.dot(v.int*c.int + v.ext*c.ext, normal),
            (u.int * c.int + u.ext*c.ext) * normal)

        if self.flux_type == "central":
            return flux_central_weak

        elif self.flux_type == "upwind":
            return flux_central_weak - 0.5 * flat_obj_array(
                    c.ext*u.ext - c.int * u.int,

                    normal * (np.dot(normal, c.ext * v.ext - c.int * v.int)))

        else:
            raise ValueError("invalid flux type '%s'" % self.flux_type)

    def sym_operator(self):
        d = self.ambient_dim

        w = sym.make_sym_array("w", d+1)
        u = w[0]
        v = w[1:]
        flux_w = flat_obj_array(self.c, w)

        # boundary conditions -------------------------------------------------

        # dirichlet BCs -------------------------------------------------------
        dir_c = sym.cse(sym.project("vol", self.dirichlet_tag)(self.c))
        dir_u = sym.cse(sym.project("vol", self.dirichlet_tag)(u))
        dir_v = sym.cse(sym.project("vol", self.dirichlet_tag)(v))
        if self.dirichlet_bc_f:
            # FIXME
            from warnings import warn
            warn("Inhomogeneous Dirichlet conditions on the wave equation "
                    "are still having issues.")

            dir_g = sym.Field("dir_bc_u")
            dir_bc = flat_obj_array(dir_c, 2*dir_g - dir_u, dir_v)
        else:
            dir_bc = flat_obj_array(dir_c, -dir_u, dir_v)

        dir_bc = sym.cse(dir_bc, "dir_bc")

        # neumann BCs ---------------------------------------------------------
        neu_c = sym.cse(sym.project("vol", self.neumann_tag)(self.c))
        neu_u = sym.cse(sym.project("vol", self.neumann_tag)(u))
        neu_v = sym.cse(sym.project("vol", self.neumann_tag)(v))
        neu_bc = sym.cse(flat_obj_array(neu_c, neu_u, -neu_v), "neu_bc")

        # radiation BCs -------------------------------------------------------
        rad_normal = sym.normal(self.radiation_tag, d)

        rad_c = sym.cse(sym.project("vol", self.radiation_tag)(self.c))
        rad_u = sym.cse(sym.project("vol", self.radiation_tag)(u))
        rad_v = sym.cse(sym.project("vol", self.radiation_tag)(v))

        rad_bc = sym.cse(flat_obj_array(rad_c,
                0.5*(rad_u - sym.project("vol", self.radiation_tag)(self.sign)
                    * np.dot(rad_normal, rad_v)),
                0.5*rad_normal*(np.dot(rad_normal, rad_v)
                    - sym.project("vol", self.radiation_tag)(self.sign)*rad_u)
                ), "rad_bc")

        # entire operator -----------------------------------------------------
        def flux(pair):
            return sym.project(pair.dd, "all_faces")(self.flux(pair))

        result = sym.InverseMassOperator()(
                flat_obj_array(
                    -self.c*np.dot(sym.stiffness_t(self.ambient_dim), v),
                    -self.c*(sym.stiffness_t(self.ambient_dim)*u)
                    )

                - sym.FaceMassOperator()(flux(sym.int_tpair(flux_w))
                    + flux(sym.bv_tpair(self.dirichlet_tag, flux_w, dir_bc))
                    + flux(sym.bv_tpair(self.neumann_tag, flux_w, neu_bc))
                    + flux(sym.bv_tpair(self.radiation_tag, flux_w, rad_bc))

                    ))

        result[0] += self.source_f

        return result

    def check_bc_coverage(self, mesh):
        from meshmode.mesh import check_bc_coverage
        check_bc_coverage(mesh, [
            self.dirichlet_tag,
            self.neumann_tag,
            self.radiation_tag])

    def max_eigenvalue(self, t, fields=None, discr=None):
        return sym.NodalMax()(sym.FunctionSymbol("fabs")(self.c))

# }}}


# vim: foldmethod=marker
