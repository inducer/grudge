# Copyright (C) 2007 Andreas Kloeckner
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.


from __future__ import division
from __future__ import absolute_import
from __future__ import print_function
import numpy
from grudge import sym
from six.moves import zip


def get_rectangular_3D_cavity_mode(E_0, mode_indices, dimensions=(1, 1, 1)):
    """A rectangular TM cavity mode."""
    kx, ky, kz = factors = [n*numpy.pi/a for n,  a in zip(mode_indices, dimensions)]
    omega = numpy.sqrt(sum(f**2 for f in factors))

    gamma_squared = ky**2 + kx**2

    nodes = sym.nodes(3)
    x = nodes[0]
    y = nodes[1]
    z = nodes[2]

    sx = sym.sin(kx*x)
    sy = sym.sin(ky*y)
    sz = sym.sin(kz*z)
    cx = sym.cos(kx*x)
    cy = sym.cos(ky*y)
    cz = sym.cos(kz*z)

    tdep = sym.exp(-1j * omega * sym.ScalarVariable("t"))

    result = sym.join_fields(
            -kx * kz * E_0*cx*sy*sz*tdep / gamma_squared,  # ex
            -ky * kz * E_0*sx*cy*sz*tdep / gamma_squared,  # ey
            E_0 * sx*sy*cz*tdep,  # ez

            -1j * omega * ky*E_0*sx*cy*cz*tdep / gamma_squared,  # hx
            1j * omega * kx*E_0*cx*sy*cz*tdep / gamma_squared,
            0,
            )

    return result


def get_rectangular_2D_cavity_mode(E_0, mode_indices):
    """A TM cavity mode.

    Returns an expression depending on *epsilon*, *mu*, and *t*.
    """
    kx, ky = factors = [n*numpy.pi for n in mode_indices]
    omega = numpy.sqrt(sum(f**2 for f in factors))

    nodes = sym.nodes(2)
    x = nodes[0]
    y = nodes[1]

    tfac = sym.ScalarVariable("t") * omega

    result = sym.join_fields(
            0,
            0,
            sym.sin(kx * x) * sym.sin(ky * y) * sym.cos(tfac),  # ez
            -ky * sym.sin(kx * x) * sym.cos(ky * y) * sym.sin(tfac) / omega,  # hx
            kx * sym.cos(kx * x) * sym.sin(ky * y) * sym.sin(tfac) / omega,  # hy
            0,
            )

    return result
