"""grudge operators modelling electromagnetic phenomena."""

__copyright__ = """
Copyright (C) 2007-2017 Andreas Kloeckner
Copyright (C) 2010 David Powell
Copyright (C) 2017 Bogdan Enache
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

from arraycontext import get_container_context_recursively
from meshmode.mesh import BTAG_ALL, BTAG_NONE
from pytools import levi_civita, memoize_method
from pytools.obj_array import flat_obj_array, make_obj_array

import grudge.geometry as geo
import grudge.op as op
from grudge.models import HyperbolicOperator


# {{{ helpers

# NOTE: Hack for getting the derivative operators to play nice
# with grudge.tools.SubsettableCrossProduct
class _Dx:
    def __init__(self, dcoll, i):
        self.dcoll = dcoll
        self.i = i

    def __mul__(self, other):
        return op.local_d_dx(self.dcoll, self.i, other)


def is_zero(x):
    # DO NOT try to replace this with an attempted "== 0" comparison.
    # This will become an elementwise numpy operation and not do what
    # you want.

    if np.isscalar(x):
        return x == 0
    else:
        return False


def count_subset(subset):
    from pytools import len_iterable
    return len_iterable(uc for uc in subset if uc)


def partial_to_all_subset_indices(subsets, base=0):
    """Takes a sequence of bools and generates it into an array of indices
    to be used to insert the subset into the full set.
    Example:
    >>> list(partial_to_all_subset_indices([[False, True, True], [True,False,True]]))
    [array([0 1]), array([2 3]
    """

    idx = base
    for subset in subsets:
        result = []
        for is_in in subset:
            if is_in:
                result.append(idx)
                idx += 1

        yield np.array(result, dtype=np.intp)

# }}}


# {{{ SubsettableCrossProduct

class SubsettableCrossProduct:
    """A cross product that can operate on an arbitrary subsets of its
    two operands and return an arbitrary subset of its result.
    """

    full_subset = (True, True, True)

    def __init__(self, op1_subset=full_subset, op2_subset=full_subset,
            result_subset=full_subset):
        """Construct a subset-able cross product.
        :param op1_subset: The subset of indices of operand 1 to be taken into
            account.  Given as a 3-sequence of bools.
        :param op2_subset: The subset of indices of operand 2 to be taken into
            account.  Given as a 3-sequence of bools.
        :param result_subset: The subset of indices of the result that are
            calculated.  Given as a 3-sequence of bools.
        """
        def subset_indices(subset):
            return [i for i, use_component in enumerate(subset)
                    if use_component]

        self.op1_subset = op1_subset
        self.op2_subset = op2_subset
        self.result_subset = result_subset

        import pymbolic
        op1 = pymbolic.var("x")
        op2 = pymbolic.var("y")

        self.functions = []
        self.component_lcjk = []
        for i, use_component in enumerate(result_subset):
            if use_component:
                this_expr = 0
                this_component = []
                for j, j_real in enumerate(subset_indices(op1_subset)):
                    for k, k_real in enumerate(subset_indices(op2_subset)):
                        lc = levi_civita((i, j_real, k_real))
                        if lc != 0:
                            this_expr += lc*op1.index(j)*op2.index(k)
                            this_component.append((lc, j, k))
                self.functions.append(pymbolic.compile(this_expr,
                    variables=[op1, op2]))
                self.component_lcjk.append(this_component)

    def __call__(self, x, y, three_mult=None):
        """Compute the subsetted cross product on the indexables *x* and *y*.
        :param three_mult: a function of three arguments *sign, xj, yk*
          used in place of the product *sign*xj*yk*. Defaults to just this
          product if not given.
        """
        from pytools.obj_array import flat_obj_array
        if three_mult is None:
            return flat_obj_array(*[f(x, y) for f in self.functions])
        else:
            return flat_obj_array(
                    *[sum(three_mult(lc, x[j], y[k]) for lc, j, k in lcjk)
                    for lcjk in self.component_lcjk])


cross = SubsettableCrossProduct()

# }}}


# {{{ MaxwellOperator

class MaxwellOperator(HyperbolicOperator):
    """A strong-form 3D Maxwell operator which supports fixed or variable
    isotropic, non-dispersive, positive epsilon and mu.

    Field order is [Ex Ey Ez Hx Hy Hz].
    """

    _default_dimensions = 3

    def __init__(self, dcoll, epsilon, mu,
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

        self.dcoll = dcoll
        self.dimensions = dimensions or self._default_dimensions

        space_subset = [True]*self.dimensions + [False]*(3-self.dimensions)

        e_subset = self.get_eh_subset()[0:3]
        h_subset = self.get_eh_subset()[3:6]

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

    def flux(self, wtpair):
        """The numerical flux for variable coefficients.

        :param flux_type: can be in [0,1] for anything between central and upwind,
          or "lf" for Lax-Friedrichs.

        As per Hesthaven and Warburton page 433.
        """

        actx = get_container_context_recursively(wtpair)
        normal = geo.normal(actx, self.dcoll, wtpair.dd)

        if self.fixed_material:
            e, h = self.split_eh(wtpair)
            epsilon = self.epsilon
            mu = self.mu
        else:
            raise NotImplementedError("only fixed material spported for now")

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
                        # -max_c*(epsilon*e.int - epsilon*e.ext)
                    ),
                    # flux h
                    1/2*(
                        self.space_cross_e(normal, e.ext-e.int)
                        # multiplication by mu undoes material divisor below
                        # -max_c*(mu*h.int - mu*h.ext)
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
            raise ValueError(f"maxwell: invalid flux_type ({self.flux_type})")

    def local_derivatives(self, w):
        """Template for the spatial derivatives of the relevant components of
        :math:`E` and :math:`H`
        """

        e, h = self.split_eh(w)

        # Object array of derivative operators
        nabla = flat_obj_array(
            [_Dx(self.dcoll, i) for i in range(self.dimensions)]
        )

        def e_curl(field):
            return self.space_cross_e(nabla, field,
                                      three_mult=lambda lc, x, y: lc * (x * y))

        def h_curl(field):
            return self.space_cross_h(nabla, field,
                                      three_mult=lambda lc, x, y: lc * (x * y))

        # in conservation form: u_t + A u_x = 0
        return flat_obj_array(
                (self.current - h_curl(h)),
                e_curl(e)
                )

    def pec_bc(self, w):
        """Construct part of the flux operator template for PEC boundary conditions
        """
        e, h = self.split_eh(w)

        pec_e = op.project(self.dcoll, "vol", self.pec_tag, e)
        pec_h = op.project(self.dcoll, "vol", self.pec_tag, h)

        return flat_obj_array(-pec_e, pec_h)

    def pmc_bc(self, w):
        """Construct part of the flux operator template for PMC boundary conditions
        """
        e, h = self.split_eh(w)

        pmc_e = op.project(self.dcoll, "vol", self.pmc_tag, e)
        pmc_h = op.project(self.dcoll, "vol", self.pmc_tag, h)

        return flat_obj_array(pmc_e, -pmc_h)

    def absorbing_bc(self, w):
        """Construct part of the flux operator template for 1st order
        absorbing boundary conditions.
        """

        actx = get_container_context_recursively(w)
        absorb_normal = geo.normal(actx, self.dcoll, dd=self.absorb_tag)

        e, h = self.split_eh(w)

        if self.fixed_material:
            epsilon = self.epsilon
            mu = self.mu
        else:
            raise NotImplementedError("only fixed material supported for now")

        absorb_Z = (mu/epsilon)**0.5  # noqa: N806
        absorb_Y = 1/absorb_Z  # noqa: N806

        absorb_e = op.project(self.dcoll, "vol", self.absorb_tag, e)
        absorb_h = op.project(self.dcoll, "vol", self.absorb_tag, h)

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
        fld_cnt = count_subset(self.get_eh_subset())

        incident_bc_data = self.incident_bc_data(self, e, h)
        if is_zero(incident_bc_data):
            return make_obj_array([0]*fld_cnt)
        else:
            return -incident_bc_data

    def operator(self, t, w):
        """The full operator template - the high level description of
        the Maxwell operator.

        Combines the relevant operator templates for spatial
        derivatives, flux, boundary conditions etc.
        """
        elec_components = count_subset(self.get_eh_subset()[0:3])
        mag_components = count_subset(self.get_eh_subset()[3:6])

        if self.fixed_material:
            # need to check this
            material_divisor = (
                    [self.epsilon]*elec_components+[self.mu]*mag_components)
        else:
            raise NotImplementedError("only fixed material supported for now")

        tags_and_bcs = [
                (self.pec_tag, self.pec_bc(w)),
                (self.pmc_tag, self.pmc_bc(w)),
                (self.absorb_tag, self.absorbing_bc(w)),
                (self.incident_tag, self.incident_bc(w)),
                ]

        dcoll = self.dcoll

        def flux(pair):
            return op.project(dcoll, pair.dd, "all_faces", self.flux(pair))

        from grudge.dof_desc import as_dofdesc

        return (
            - self.local_derivatives(w)
            - op.inverse_mass(
                dcoll,
                op.face_mass(
                    dcoll,
                    sum(flux(tpair) for tpair in op.interior_trace_pairs(dcoll, w))
                    + sum(flux(op.bv_trace_pair(dcoll, as_dofdesc(tag), w, bc))
                          for tag, bc in tags_and_bcs)
                )
            )
        ) / material_divisor

    @memoize_method
    def partial_to_eh_subsets(self):
        """Helps find the indices of the E and H components, which can vary
        depending on number of dimensions and whether we have a full/TE/TM
        operator.
        """

        e_subset = self.get_eh_subset()[0:3]
        h_subset = self.get_eh_subset()[3:6]

        return tuple(partial_to_all_subset_indices([e_subset, h_subset]))

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

    def max_characteristic_velocity(self, actx, **kwargs):
        if self.fixed_material:
            return 1/np.sqrt(self.epsilon*self.mu)  # a number
        else:
            return op.nodal_max(self.dcoll, "vol",
                                1 / actx.np.sqrt(self.epsilon * self.mu))

    def check_bc_coverage(self, mesh):
        from meshmode.mesh import check_bc_coverage
        check_bc_coverage(mesh, [
            self.pec_tag,
            self.pmc_tag,
            self.absorb_tag,
            self.incident_tag,
            ])

# }}}


# {{{ TMMaxwellOperator

class TMMaxwellOperator(MaxwellOperator):
    """A 2D TM Maxwell operator with PEC boundaries.

    Field order is [Ez Hx Hy].
    """

    _default_dimensions = 2

    def get_eh_subset(self):
        return (
                (False, False, True, True, True, False)  # ez, hx and hy
                )

# }}}


# {{{ TEMaxwellOperator

class TEMaxwellOperator(MaxwellOperator):
    """A 2D TE Maxwell operator.

    Field order is [Ex Ey Hz].
    """

    _default_dimensions = 2

    def get_eh_subset(self):
        return (
                (True, True, False, False, False, True)  # ex and ey, only hz
                )

# }}}


# {{{ TE1DMaxwellOperator

class TE1DMaxwellOperator(MaxwellOperator):
    """A 1D TE Maxwell operator.

    Field order is [Ex Ey Hz].
    """

    _default_dimensions = 1

    def get_eh_subset(self):
        return (
                (True, True, False, False, False, True)
                )

# }}}


# {{{ SourceFree1DMaxwellOperator

class SourceFree1DMaxwellOperator(MaxwellOperator):
    """A 1D TE Maxwell operator.

    Field order is [Ey Hz].
    """

    _default_dimensions = 1

    def get_eh_subset(self):
        return (
                (False, True, False, False, False, True)
                )

# }}}


# {{{ get_rectangular_cavity_mode

def get_rectangular_cavity_mode(actx, nodes, t, E_0, mode_indices):  # noqa: N803
    """A rectangular TM cavity mode for a rectangle / cube
    with one corner at the origin and the other at (1,1[,1])."""
    dims = len(mode_indices)
    if dims != 2 and dims != 3:
        raise ValueError("Improper mode_indices dimensions")

    factors = [n*np.pi for n in mode_indices]

    kx, ky = factors[0:2]
    if dims == 3:
        kz = factors[2]

    omega = np.sqrt(sum(f**2 for f in factors))

    x = nodes[0]
    y = nodes[1]
    if dims == 3:
        z = nodes[2]

    zeros = 0*x
    sx = actx.np.sin(kx*x)
    cx = actx.np.cos(kx*x)
    sy = actx.np.sin(ky*y)
    cy = actx.np.cos(ky*y)

    if dims == 2:
        tfac = t * omega

        result = flat_obj_array(
            zeros,
            zeros,
            actx.np.sin(kx * x) * actx.np.sin(ky * y) * np.cos(tfac),  # ez
            (-ky * actx.np.sin(kx * x) * actx.np.cos(ky * y)
             * np.sin(tfac) / omega),  # hx
            (kx * actx.np.cos(kx * x) * actx.np.sin(ky * y)
             * np.sin(tfac) / omega),  # hy
            zeros,
        )
    elif dims == 3:
        sz = actx.np.sin(kz*z)
        cz = actx.np.cos(kz*z)

        tdep = np.exp(-1j * omega * t)

        gamma_squared = ky**2 + kx**2
        result = flat_obj_array(
            -kx * kz * E_0*cx*sy*sz*tdep / gamma_squared,  # ex
            -ky * kz * E_0*sx*cy*sz*tdep / gamma_squared,  # ey
            E_0 * sx*sy*cz*tdep,  # ez
            -1j * omega * ky*E_0*sx*cy*cz*tdep / gamma_squared,  # hx
            1j * omega * kx*E_0*cx*sy*cz*tdep / gamma_squared,
            zeros,
        )
    else:
        raise NotImplementedError("only 2D and 3D supported")

    return result

# }}}

# vim: foldmethod=marker
