"""grudge operators modelling electromagnetic phenomena."""

__copyright__ = """
Copyright (C) 2007-2017 Andreas Kloeckner
Copyright (C) 2010 David Powell
Copyright (C) 2017 Bogdan Enache
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

from pytools import memoize_method

from grudge.models import HyperbolicOperator
from meshmode.mesh import BTAG_ALL, BTAG_NONE
from grudge import sym
from pytools.obj_array import flat_obj_array, make_obj_array


class MaxwellOperator(HyperbolicOperator):
    """A strong-form 3D Maxwell operator which supports fixed or variable
    isotropic, non-dispersive, positive epsilon and mu.

    Field order is [Ex Ey Ez Hx Hy Hz].
    """

    _default_dimensions = 3

    def __init__(self, epsilon, mu,
            flux_type,
            bdry_flux_type=None,
            pec_tag=BTAG_ALL,
            pmc_tag=BTAG_NONE,
            absorb_tag=BTAG_NONE,
            incident_tag=BTAG_NONE,
            incident_bc=lambda maxwell_op, e, h: 0, current=0, dimensions=None):
        """
        :arg flux_type: can be in [0,1] for anything between central and upwind,
          or "lf" for Lax-Friedrichs
        :arg epsilon: can be a number, for fixed material throughout the
            computation domain, or a TimeConstantGivenFunction for spatially
            variable material coefficients
        :arg mu: can be a number, for fixed material throughout the computation
            domain, or a TimeConstantGivenFunction for spatially variable material
            coefficients
        :arg incident_bc_getter: a function of signature *(maxwell_op, e, h)* that
            accepts *e* and *h* as a symbolic object arrays
            returns a symbolic expression for the incident
            boundary condition
        """

        self.dimensions = dimensions or self._default_dimensions

        space_subset = [True]*self.dimensions + [False]*(3-self.dimensions)

        e_subset = self.get_eh_subset()[0:3]
        h_subset = self.get_eh_subset()[3:6]

        from grudge.tools import SubsettableCrossProduct
        self.space_cross_e = SubsettableCrossProduct(
                op1_subset=space_subset,
                op2_subset=e_subset,
                result_subset=h_subset)
        self.space_cross_h = SubsettableCrossProduct(
                op1_subset=space_subset,
                op2_subset=h_subset,
                result_subset=e_subset)

        self.epsilon = epsilon
        self.mu = mu

        from pymbolic.primitives import is_constant
        self.fixed_material = is_constant(epsilon) and is_constant(mu)

        self.flux_type = flux_type
        if bdry_flux_type is None:
            self.bdry_flux_type = flux_type
        else:
            self.bdry_flux_type = bdry_flux_type

        self.pec_tag = pec_tag
        self.pmc_tag = pmc_tag
        self.absorb_tag = absorb_tag
        self.incident_tag = incident_tag

        self.current = current
        self.incident_bc_data = incident_bc

    def flux(self, w):
        """The numerical flux for variable coefficients.

        :param flux_type: can be in [0,1] for anything between central and upwind,
          or "lf" for Lax-Friedrichs.

        As per Hesthaven and Warburton page 433.
        """

        normal = sym.normal(w.dd, self.dimensions)

        if self.fixed_material:
            e, h = self.split_eh(w)
            epsilon = self.epsilon
            mu = self.mu

        Z_int = (mu/epsilon)**0.5  # noqa: N806
        Y_int = 1/Z_int  # noqa: N806
        Z_ext = (mu/epsilon)**0.5  # noqa: N806
        Y_ext = 1/Z_ext  # noqa: N806

        if self.flux_type == "lf":
            # if self.fixed_material:
            #     max_c = (self.epsilon*self.mu)**(-0.5)

            return flat_obj_array(
                    # flux e,
                    1/2*(
                        -self.space_cross_h(normal, h.ext-h.int)
                        # multiplication by epsilon undoes material divisor below
                        #-max_c*(epsilon*e.int - epsilon*e.ext)
                    ),
                    # flux h
                    1/2*(
                        self.space_cross_e(normal, e.ext-e.int)
                        # multiplication by mu undoes material divisor below
                        #-max_c*(mu*h.int - mu*h.ext)
                    ))
        elif isinstance(self.flux_type, (int, float)):
            # see doc/maxima/maxwell.mac
            return flat_obj_array(
                    # flux e,
                    (
                        -1/(Z_int+Z_ext)*self.space_cross_h(normal,
                            Z_ext*(h.ext-h.int)
                            - self.flux_type*self.space_cross_e(normal, e.ext-e.int))
                        ),
                    # flux h
                    (
                        1/(Y_int + Y_ext)*self.space_cross_e(normal,
                            Y_ext*(e.ext-e.int)
                            + self.flux_type*self.space_cross_h(normal, h.ext-h.int))
                        ),
                    )
        else:
            raise ValueError("maxwell: invalid flux_type (%s)"
                    % self.flux_type)

    def local_derivatives(self, w):
        """Template for the spatial derivatives of the relevant components of
        :math:`E` and :math:`H`
        """

        e, h = self.split_eh(w)

        nabla = sym.nabla(self.dimensions)

        def e_curl(field):
            return self.space_cross_e(nabla, field)

        def h_curl(field):
            return self.space_cross_h(nabla, field)

        # in conservation form: u_t + A u_x = 0
        return flat_obj_array(
                (self.current - h_curl(h)),
                e_curl(e)
                )

    def pec_bc(self, w):
        """Construct part of the flux operator template for PEC boundary conditions
        """
        e, h = self.split_eh(w)

        pec_e = sym.cse(sym.project("vol", self.pec_tag)(e))
        pec_h = sym.cse(sym.project("vol", self.pec_tag)(h))

        return flat_obj_array(-pec_e, pec_h)

    def pmc_bc(self, w):
        """Construct part of the flux operator template for PMC boundary conditions
        """
        e, h = self.split_eh(w)

        pmc_e = sym.cse(sym.project("vol", self.pmc_tag)(e))
        pmc_h = sym.cse(sym.project("vol", self.pmc_tag)(h))

        return flat_obj_array(pmc_e, -pmc_h)

    def absorbing_bc(self, w):
        """Construct part of the flux operator template for 1st order
        absorbing boundary conditions.
        """

        absorb_normal = sym.normal(self.absorb_tag, self.dimensions)

        e, h = self.split_eh(w)

        if self.fixed_material:
            epsilon = self.epsilon
            mu = self.mu

        absorb_Z = (mu/epsilon)**0.5  # noqa: N806
        absorb_Y = 1/absorb_Z  # noqa: N806

        absorb_e = sym.cse(sym.project("vol", self.absorb_tag)(e))
        absorb_h = sym.cse(sym.project("vol", self.absorb_tag)(h))

        bc = flat_obj_array(
                absorb_e + 1/2*(self.space_cross_h(absorb_normal, self.space_cross_e(
                    absorb_normal, absorb_e))
                    - absorb_Z*self.space_cross_h(absorb_normal, absorb_h)),
                absorb_h + 1/2*(
                    self.space_cross_e(absorb_normal, self.space_cross_h(
                        absorb_normal, absorb_h))
                    + absorb_Y*self.space_cross_e(absorb_normal, absorb_e)))

        return bc

    def incident_bc(self, w):
        """Flux terms for incident boundary conditions"""
        # NOTE: Untested for inhomogeneous materials, but would usually be
        # physically meaningless anyway (are there exceptions to this?)

        e, h = self.split_eh(w)

        from grudge.tools import count_subset
        fld_cnt = count_subset(self.get_eh_subset())

        from grudge.tools import is_zero
        incident_bc_data = self.incident_bc_data(self, e, h)
        if is_zero(incident_bc_data):
            return make_obj_array([0]*fld_cnt)
        else:
            return sym.cse(-incident_bc_data)

    def sym_operator(self, w=None):
        """The full operator template - the high level description of
        the Maxwell operator.

        Combines the relevant operator templates for spatial
        derivatives, flux, boundary conditions etc.
        """
        from grudge.tools import count_subset
        w = sym.make_sym_array("w", count_subset(self.get_eh_subset()))

        elec_components = count_subset(self.get_eh_subset()[0:3])
        mag_components = count_subset(self.get_eh_subset()[3:6])

        if self.fixed_material:
            # need to check this
            material_divisor = (
                    [self.epsilon]*elec_components+[self.mu]*mag_components)

        tags_and_bcs = [
                (self.pec_tag, self.pec_bc(w)),
                (self.pmc_tag, self.pmc_bc(w)),
                (self.absorb_tag, self.absorbing_bc(w)),
                (self.incident_tag, self.incident_bc(w)),
                ]

        def flux(pair):
            return sym.project(pair.dd, "all_faces")(self.flux(pair))

        return (
                - self.local_derivatives(w)
                - sym.InverseMassOperator()(sym.FaceMassOperator()(
                    flux(sym.int_tpair(w))
                    + sum(
                        flux(sym.bv_tpair(tag, w, bc))
                        for tag, bc in tags_and_bcs)
                    ))) / material_divisor

    @memoize_method
    def partial_to_eh_subsets(self):
        """Helps find the indices of the E and H components, which can vary
        depending on number of dimensions and whether we have a full/TE/TM
        operator.
        """

        e_subset = self.get_eh_subset()[0:3]
        h_subset = self.get_eh_subset()[3:6]

        from grudge.tools import partial_to_all_subset_indices
        return tuple(partial_to_all_subset_indices(
            [e_subset, h_subset]))

    def split_eh(self, w):
        """Splits an array into E and H components"""
        e_idx, h_idx = self.partial_to_eh_subsets()
        e, h = w[e_idx], w[h_idx]

        return e, h

    def get_eh_subset(self):
        """Return a 6-tuple of :class:`bool` objects indicating whether field
        components are to be computed. The fields are numbered in the order
        specified in the class documentation.
        """
        return 6*(True,)

    def max_eigenvalue_expr(self):
        """Return the largest eigenvalue of Maxwell's equations as a hyperbolic
        system.
        """
        from math import sqrt
        if self.fixed_material:
            return 1/sqrt(self.epsilon*self.mu)  # a number
        else:
            import grudge.symbolic as sym
            return sym.NodalMax()(1/sym.FunctionSymbol("sqrt")(self.epsilon*self.mu))

    def max_eigenvalue(self, t, fields=None, discr=None, context={}):
        if self.fixed_material:
            return self.max_eigenvalue_expr()
        else:
            raise ValueError("max_eigenvalue is no longer supported for "
                    "variable-coefficient problems--use max_eigenvalue_expr")

    def check_bc_coverage(self, mesh):
        from meshmode.mesh import check_bc_coverage
        check_bc_coverage(mesh, [
            self.pec_tag,
            self.pmc_tag,
            self.absorb_tag,
            self.incident_tag])


class TMMaxwellOperator(MaxwellOperator):
    """A 2D TM Maxwell operator with PEC boundaries.

    Field order is [Ez Hx Hy].
    """

    _default_dimensions = 2

    def get_eh_subset(self):
        return (
                (False, False, True)  # only ez
                + (True, True, False)  # hx and hy
                )


class TEMaxwellOperator(MaxwellOperator):
    """A 2D TE Maxwell operator.

    Field order is [Ex Ey Hz].
    """

    _default_dimensions = 2

    def get_eh_subset(self):
        return (
                (True, True, False)  # ex and ey
                + (False, False, True)  # only hz
                )


class TE1DMaxwellOperator(MaxwellOperator):
    """A 1D TE Maxwell operator.

    Field order is [Ex Ey Hz].
    """

    _default_dimensions = 1

    def get_eh_subset(self):
        return (
                (True, True, False)
                + (False, False, True)
                )


class SourceFree1DMaxwellOperator(MaxwellOperator):
    """A 1D TE Maxwell operator.

    Field order is [Ey Hz].
    """

    _default_dimensions = 1

    def get_eh_subset(self):
        return (
                (False, True, False)
                + (False, False, True)
                )


def get_rectangular_cavity_mode(E_0, mode_indices):  # noqa: N803
    """A rectangular TM cavity mode for a rectangle / cube
    with one corner at the origin and the other at (1,1[,1])."""
    dims = len(mode_indices)
    if dims != 2 and dims != 3:
        raise ValueError("Improper mode_indices dimensions")
    import numpy

    factors = [n*numpy.pi for n in mode_indices]

    kx, ky = factors[0:2]
    if dims == 3:
        kz = factors[2]

    omega = numpy.sqrt(sum(f**2 for f in factors))

    nodes = sym.nodes(dims)
    x = nodes[0]
    y = nodes[1]
    if dims == 3:
        z = nodes[2]

    sx = sym.sin(kx*x)
    cx = sym.cos(kx*x)
    sy = sym.sin(ky*y)
    cy = sym.cos(ky*y)
    if dims == 3:
        sz = sym.sin(kz*z)
        cz = sym.cos(kz*z)

    if dims == 2:
        tfac = sym.ScalarVariable("t") * omega

        result = flat_obj_array(
            0,
            0,
            sym.sin(kx * x) * sym.sin(ky * y) * sym.cos(tfac),  # ez
            -ky * sym.sin(kx * x) * sym.cos(ky * y) * sym.sin(tfac) / omega,  # hx
            kx * sym.cos(kx * x) * sym.sin(ky * y) * sym.sin(tfac) / omega,  # hy
            0,
            )
    else:
        tdep = sym.exp(-1j * omega * sym.ScalarVariable("t"))

        gamma_squared = ky**2 + kx**2
        result = flat_obj_array(
            -kx * kz * E_0*cx*sy*sz*tdep / gamma_squared,  # ex
            -ky * kz * E_0*sx*cy*sz*tdep / gamma_squared,  # ey
            E_0 * sx*sy*cz*tdep,  # ez

            -1j * omega * ky*E_0*sx*cy*cz*tdep / gamma_squared,  # hx
            1j * omega * kx*E_0*cx*sy*cz*tdep / gamma_squared,
            0,
            )

    return result
