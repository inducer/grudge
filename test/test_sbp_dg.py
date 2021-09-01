# Copyright (C) 2020 Cory Mikida
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


import numpy as np
import numpy.linalg as la

from grudge.array_context import PytestPyOpenCLArrayContextFactory
from arraycontext import pytest_generate_tests_for_array_contexts
pytest_generate_tests = pytest_generate_tests_for_array_contexts(
        [PytestPyOpenCLArrayContextFactory])

from arraycontext.container.traversal import thaw
import meshmode.mesh.generation as mgen

from grudge import DiscretizationCollection

import grudge.op as op
import grudge.dof_desc as dof_desc
from sbp_operators import (sbp21, sbp42, sbp63)
from projection import sbp_dg_projection


import pytest

import logging

logger = logging.getLogger(__name__)


# Domain: x = [-1,1], y = [-1,1]
# SBP Subdomain: x = [-1,0], y = [0,1] (structured mesh)
# DG Subdomain: x = [0,1], y = [0,1] (unstructured mesh)

def test_sbp_dg(actx_factory, write_output=True, order=4):
    actx = actx_factory()

    # DG Half.
    dim = 2

    nelem_x = 20
    nelem_y = 20
    mesh = mgen.generate_regular_rect_mesh(a=(0, -1), b=(1, 1),
                                           n=(nelem_x, nelem_y), order=order,
                                           boundary_tag_to_face={"btag_sbp": ["-x"],
                                              "btag_std": ["+y", "+x", "-y"]})

    # Check to make sure this actually worked.
    from meshmode.mesh import is_boundary_tag_empty
    assert not is_boundary_tag_empty(mesh, "btag_sbp")
    assert not is_boundary_tag_empty(mesh, "btag_std")

    # Check this new isolating discretization, as well as the inverse.
    dcoll = DiscretizationCollection(actx, mesh, order=order)
    sbp_bdry_discr = dcoll.discr_from_dd(dof_desc.DTAG_BOUNDARY("btag_sbp"))
    sbp_bdry_discr.cl_context = actx

    # Visualize our new boundary discretization
    # from grudge.shortcuts import make_visualizer
    # vis = make_visualizer(sbp_bdry_discr, vis_order=order)
    # vis.write_vtk_file("bdry_disc.vtu",
    #                            [], overwrite=True)

    std_bdry_discr = dcoll.discr_from_dd(dof_desc.DTAG_BOUNDARY("btag_std"))
    std_bdry_discr.cl_context = actx

    # Visualize our new boundary discretization
    # vis = make_visualizer(std_bdry_discr, vis_order=order)
    # vis.write_vtk_file("std_bdry_disc.vtu",
    #                            [], overwrite=True)

    dt_factor = 4
    h = 1/40

    c = np.array([0.1, 0.1])
    norm_c = la.norm(c)

    flux_type = "upwind"

    def f(x):
        return actx.np.sin(10*x)

    def u_analytic(x, t=0):
        return f(-c.dot(x)/norm_c+t*norm_c)

    from grudge.models.advection import WeakAdvectionSBPOperator

    std_tag = dof_desc.DTAG_BOUNDARY("btag_std")

    adv_operator = WeakAdvectionSBPOperator(dcoll, c,
                                            inflow_u=lambda t: u_analytic(
                                                thaw(dcoll.nodes(dd=std_tag),
                                                    actx),
                                                t=t
                                                ),
                                            flux_type=flux_type)

    nodes = thaw(dcoll.nodes(), actx)
    u = u_analytic(nodes, t=0)

    # Count the number of DG nodes - will need this later
    ngroups = len(mesh.groups)
    nnodes_grp = np.ones(ngroups)
    for i in range(0, dim):
        for j in range(0, ngroups):
            nnodes_grp[j] = nnodes_grp[j]*mesh.groups[j].nodes.shape[i*-1 - 1]

    nnodes = int(sum(nnodes_grp))

    final_time = 0.2

    dt = dt_factor * h/order**2
    nsteps = (final_time // dt) + 1
    dt = final_time/nsteps + 1e-15

    # SBP Half.

    # First, need to set up the structured mesh.
    n_sbp_x = 20
    n_sbp_y = 40
    x_sbp = np.linspace(-1, 0, n_sbp_x, endpoint=True)
    y_sbp = np.linspace(-1, 1, n_sbp_y, endpoint=True)
    dx = x_sbp[1] - x_sbp[0]
    dy = y_sbp[1] - y_sbp[0]

    # Set up solution vector:
    # For now, timestep is the same as DG.
    u_sbp = np.zeros(int(n_sbp_x*n_sbp_y))

    # Initial condition
    for j in range(0, n_sbp_y):
        for i in range(0, n_sbp_x):
            u_sbp[i + j*(n_sbp_x)] = np.sin(10*(-c.dot(
                                                [x_sbp[i], y_sbp[j]])/norm_c))

    # obtain P and Q
    order_sbp = 4
    if order_sbp == 2:
        [p_x, q_x] = sbp21(n_sbp_x)
        [p_y, q_y] = sbp21(n_sbp_y)
    elif order_sbp == 4:
        [p_x, q_x] = sbp42(n_sbp_x)
        [p_y, q_y] = sbp42(n_sbp_y)
    elif order_sbp == 6:
        [p_x, q_x] = sbp63(n_sbp_x)
        [p_y, q_y] = sbp63(n_sbp_y)

    tau_l = 1
    tau_r = 1

    # for the boundaries
    el_x = np.zeros(n_sbp_x)
    er_x = np.zeros(n_sbp_x)
    el_x[0] = 1
    er_x[n_sbp_x-1] = 1
    e_l_matx = np.zeros((n_sbp_x, n_sbp_x,))
    e_r_matx = np.zeros((n_sbp_x, n_sbp_x,))

    for i in range(0, n_sbp_x):
        for j in range(0, n_sbp_x):
            e_l_matx[i, j] = el_x[i]*el_x[j]
            e_r_matx[i, j] = er_x[i]*er_x[j]

    el_y = np.zeros(n_sbp_y)
    er_y = np.zeros(n_sbp_y)
    el_y[0] = 1
    er_y[n_sbp_y-1] = 1
    e_l_maty = np.zeros((n_sbp_y, n_sbp_y,))
    e_r_maty = np.zeros((n_sbp_y, n_sbp_y,))

    for i in range(0, n_sbp_y):
        for j in range(0, n_sbp_y):
            e_l_maty[i, j] = el_y[i]*el_y[j]
            e_r_maty[i, j] = er_y[i]*er_y[j]

    # construct the spatial operators
    d_x = np.linalg.inv(dx*p_x).dot(q_x - 0.5*e_l_matx + 0.5*e_r_matx)
    d_y = np.linalg.inv(dy*p_y).dot(q_y - 0.5*e_l_maty + 0.5*e_r_maty)

    # for the boundaries
    c_l_x = np.kron(tau_l, (np.linalg.inv(dx*p_x).dot(el_x)))
    c_r_x = np.kron(tau_r, (np.linalg.inv(dx*p_x).dot(er_x)))
    c_l_y = np.kron(tau_l, (np.linalg.inv(dy*p_y).dot(el_y)))
    c_r_y = np.kron(tau_r, (np.linalg.inv(dy*p_y).dot(er_y)))

    # For speed...
    dudx_mat = -np.kron(np.eye(n_sbp_y), d_x)
    dudy_mat = -np.kron(d_y, np.eye(n_sbp_x))

    # Number of nodes in our SBP-DG boundary discretization
    from meshmode.dof_array import flatten, unflatten
    sbp_nodes_y = flatten(thaw(sbp_bdry_discr.nodes(), actx)[1])
    # When projecting, we use nodes sorted in y, but we will have to unsort
    # afterwards to make sure projected solution is injected into DG BC
    # in the correct way.
    nodesort = np.argsort(sbp_nodes_y)
    nodesortlist = nodesort.tolist()
    rangex = np.array(range(sbp_nodes_y.shape[0]))
    unsort_args = [nodesortlist.index(x) for x in rangex]

    west_nodes = np.sort(np.array(sbp_nodes_y))

    # Make element-aligned glue grid.
    dg_side_gg = np.zeros(int(west_nodes.shape[0]/(order+1))+1)
    counter = 0
    for i in range(0, west_nodes.shape[0]):
        west_nodes[i] = west_nodes[i].get()
        if i % (order+1) == 0:
            dg_side_gg[counter] = west_nodes[i]
            counter += 1

    dg_side_gg[-1] = west_nodes[-1]
    n_west_elements = int(west_nodes.shape[0] / (order + 1))
    sbp2dg, dg2sbp = sbp_dg_projection(n_sbp_y-1, n_west_elements, order_sbp,
                                       order, dg_side_gg, west_nodes)

    def rhs(t, u):
        # Initialize the entire RHS to 0.
        rhs_out = np.zeros(int(n_sbp_x*n_sbp_y) + int(nnodes))

        # Fill the first part with the SBP half of the domain.

        # Pull the SBP vector out of device array for now.
        u_sbp_ts = u[0:int(n_sbp_x*n_sbp_y)]

        dudx = np.zeros((n_sbp_x*n_sbp_y))
        dudy = np.zeros((n_sbp_x*n_sbp_y))

        dudx = dudx_mat.dot(u_sbp_ts)
        dudy = dudy_mat.dot(u_sbp_ts)

        # Boundary condition
        dl_x = np.zeros(n_sbp_x*n_sbp_y)
        dr_x = np.zeros(n_sbp_x*n_sbp_y)
        dl_y = np.zeros(n_sbp_x*n_sbp_y)
        dr_y = np.zeros(n_sbp_x*n_sbp_y)

        # Need to fill this by looping through each segment.
        # X-boundary conditions:
        for j in range(0, n_sbp_y):
            u_bcx = u_sbp_ts[j*n_sbp_x:((j+1)*n_sbp_x)]
            v_l_x = np.transpose(el_x).dot(u_bcx)
            v_r_x = np.transpose(er_x).dot(u_bcx)
            left_bcx = np.sin(10*(-c.dot(
                                  [x_sbp[0], y_sbp[j]])/norm_c + norm_c*t))
            right_bcx = np.sin(10*(-c.dot(
                                   [x_sbp[n_sbp_x-1],
                                    y_sbp[j]])/norm_c + norm_c*t))
            dl_xbc = c_l_x*(v_l_x - left_bcx)
            dr_xbc = c_r_x*(v_r_x - right_bcx)
            dl_x[j*n_sbp_x:((j+1)*n_sbp_x)] = dl_xbc
            dr_x[j*n_sbp_x:((j+1)*n_sbp_x)] = dr_xbc
        # Y-boundary conditions:
        for i in range(0, n_sbp_x):
            u_bcy = u_sbp_ts[i::n_sbp_x]
            v_l_y = np.transpose(el_y).dot(u_bcy)
            v_r_y = np.transpose(er_y).dot(u_bcy)
            left_bcy = np.sin(10*(-c.dot(
                                [x_sbp[i], y_sbp[0]])/norm_c + norm_c*t))
            right_bcy = np.sin(10*(-c.dot(
                                [x_sbp[i],
                                 y_sbp[n_sbp_y-1]])/norm_c + norm_c*t))
            dl_ybc = c_l_y*(v_l_y - left_bcy)
            dr_ybc = c_r_y*(v_r_y - right_bcy)
            dl_y[i::n_sbp_x] = dl_ybc
            dr_y[i::n_sbp_x] = dr_ybc

        # Add these at each point on the SBP half to get the SBP RHS.
        rhs_sbp = c[0]*dudx + c[1]*dudy - dl_x - dr_x - dl_y - dr_y

        rhs_out[0:int(n_sbp_x*n_sbp_y)] = rhs_sbp

        sbp_east = np.zeros(n_sbp_y)
        # Pull SBP domain values off of east face.
        counter = 0
        for i in range(0, n_sbp_x*n_sbp_y):
            if i == n_sbp_x - 1:
                sbp_east[counter] = u_sbp_ts[i]
                counter += 1
            elif i % n_sbp_x == n_sbp_x - 1:
                sbp_east[counter] = u_sbp_ts[i]
                counter += 1

        # Projection from SBP to DG is now a two-step process.
        # First: SBP-to-DG.
        sbp_proj = sbp2dg.dot(sbp_east)
        # Second: Fix the ordering.
        sbp_proj = sbp_proj[unsort_args]
        sbp_tag = dof_desc.DTAG_BOUNDARY("btag_sbp")

        u_dg_in = unflatten(actx, dcoll.discr_from_dd("vol"),
                            actx.from_numpy(u[int(n_sbp_x*n_sbp_y):]))
        u_sbp_in = unflatten(actx, dcoll.discr_from_dd(sbp_tag),
                             actx.from_numpy(sbp_proj))
        # Grudge DG RHS.
        # Critical step - now need to apply projected SBP state to the
        # proper nodal locations in u_dg.
        dg_rhs = adv_operator.operator(
                t=t,
                u=u_dg_in,
                state_from_sbp=u_sbp_in,
                sbp_tag=sbp_tag, std_tag=std_tag)
        dg_rhs = flatten(dg_rhs)
        rhs_out[int(n_sbp_x*n_sbp_y):] = dg_rhs.get()

        return rhs_out

    # Timestepper.
    from grudge.shortcuts import set_up_rk4

    # Make a combined u with the SBP and the DG parts.
    u_comb = np.zeros(int(n_sbp_x*n_sbp_y) + nnodes)
    u_comb[0:int(n_sbp_x*n_sbp_y)] = u_sbp
    u_flat = flatten(u)
    for i in range(int(n_sbp_x*n_sbp_y), int(n_sbp_x*n_sbp_y) + nnodes):
        u_comb[i] = u_flat[i - int(n_sbp_x*n_sbp_y)].get()
    dt_stepper = set_up_rk4("u", dt, u_comb, rhs)

    from grudge.shortcuts import make_visualizer
    vis = make_visualizer(dcoll, vis_order=order)

    step = 0

    # Create mesh for structured grid output
    sbp_mesh = np.zeros((2, n_sbp_y, n_sbp_x))
    for j in range(0, n_sbp_y):
        sbp_mesh[0, j, :] = x_sbp
    for i in range(0, n_sbp_x):
        sbp_mesh[1, :, i] = y_sbp

    for event in dt_stepper.run(t_end=final_time):
        if isinstance(event, dt_stepper.StateComputed):

            step += 1

            last_t = event.t
            u_sbp = event.state_component[0:int(n_sbp_x*n_sbp_y)]
            u_dg = event.state_component[int(n_sbp_x*n_sbp_y):]

            error_l2_dg = op.norm(
                dcoll,
                u_dg - u_analytic(nodes, t=last_t),
                2
            )
            print('DG L2 Error after Step ', step, error_l2_dg)
            sbp_error = np.zeros((n_sbp_x*n_sbp_y))
            error_l2_sbp = 0
            for j in range(0, n_sbp_y):
                for i in range(0, n_sbp_x):
                    sbp_error[i + j*n_sbp_x] = u_sbp[i + j*n_sbp_x] - \
                        np.sin(10*(-c.dot([x_sbp[i], y_sbp[j]])/norm_c +
                                   - last_t*norm_c))
                    error_l2_sbp = error_l2_sbp + \
                        dx*dy*(sbp_error[i + j*n_sbp_x]) ** 2

            error_l2_sbp = np.sqrt(error_l2_sbp)
            print('SBP L2 Error after Step ', step, error_l2_sbp)

            # Write out the DG data only
            u_dg_plot = unflatten(actx, dcoll.discr_from_dd("vol"),
                                  actx.from_numpy(u_dg))
            vis.write_vtk_file("dg-%04d.vtu" % step,
                               [("u", u_dg_plot)], overwrite=True)

            # Try writing out a VTK file with the SBP data.
            from pyvisfile.vtk import write_structured_grid

            # Overwrite existing files - this is annoying when debugging.
            filename = "sbp_%04d.vts" % step
            import os
            if os.path.exists(filename):
                os.remove(filename)

            write_structured_grid(filename, sbp_mesh,
                                  point_data=[("u", u_sbp)])


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        pytest.main([__file__])
