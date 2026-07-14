import numpy as np
from sbp_operators import (sbp21, sbp42, sbp63)


def gaussian_quad(n):

    # Gaussian quadrature per the algorithm
    # of Hesthaven and Warburton.

    if n == 0:
        x = 0
        w = 2
    else:
        h1 = 2 * np.linspace(0, n, n+1, endpoint=True)
        je = 2/(h1[0:n] + 2)*np.linspace(1, n, n, endpoint=True) * \
            np.linspace(1, n, n, endpoint=True) / \
            np.sqrt((h1[0:n]+1)*(h1[0:n]+3))
        a = np.diag(je, 1)
        ap = np.diag(je, -1)
        a = a + ap
        [x, v] = np.linalg.eig(a)
        idx = x.argsort()[::1]
        x = x[idx]
        v = v[:, idx]
        w = 2 * (v[0, :]) ** 2

    return x, w


def legendre_vandermonde(x, n):

    v = np.zeros((x.shape[0], n+1))
    p_n_0 = x
    p_n_1 = np.ones(x.shape[0])
    if n >= 0:
        v[:, 0] = p_n_1 * np.sqrt(1.0/2.0)

    if n >= 1:
        v[:, 1] = p_n_0 * np.sqrt(3.0/2.0)

    for i in range(2, n+1):
        a = (2*i - 1)/i
        c = (i - 1)*(i - 1)*(2*i) / (i * i * (2*i - 2))
        p_n_2 = p_n_1
        p_n_1 = p_n_0
        p_n_0 = a*x*p_n_1 - c*p_n_2
        v[:, i] = p_n_0 * np.sqrt((2*i+1)/2)

    return v


def glue_pieces(n, vx, xg):

    # Remove leading dimensions of 1.
    if vx.shape[0] == 1:
        vx = np.squeeze(vx, axis=0)
    k = vx.shape[0] - 1

    [r, w_dump] = gaussian_quad(n)
    # Python uses zero-based indices.
    va = np.arange(0, k, 1)
    vb = np.arange(1, k+1, 1)

    # x = np.ones(n+1)*vx[va] + 0.5*(r+1)*(vx[vb] - vx[va])
    x = (np.ones(n+1).reshape(1, -1)).transpose()*vx[va] + \
        0.5*((r.reshape(1, -1)).transpose()+1)*(vx[vb] - vx[va])
    x = np.squeeze(x)

    m = np.diag(2/(2*np.arange(0, n+1, 1)+1))
    vr = legendre_vandermonde(r, n)
    u = np.zeros((n+1, k*(n+1)))

    xmin = vx[0]
    xmax = vx[-1]

    xbr = 2*np.divide((x-xmin), (xmax-xmin)) - 1
    vbr = (legendre_vandermonde((xbr.transpose()).flatten(),
                                n)).dot(np.sqrt(m))

    invsqm_invvr = np.sqrt(np.diag(
                           np.divide(1, np.diag(m)))).dot(np.linalg.inv(vr))

    for i_n in range(0, n+1):
        pbr = np.reshape(vbr[:, i_n], x.shape)
        tmp = invsqm_invvr.dot(pbr)
        u[i_n, :] = (tmp.transpose()).flatten()

    xgbr = 2*np.divide((xg-xmin), (xmax-xmin)) - 1
    v = (legendre_vandermonde(xgbr, n)).dot(np.sqrt(m))

    m = np.divide(m, 2)

    return u, v, m


def make_projection(n, order):

    p = order - 1

    # Diagonal SBP operators
    # Just orders 2 and 4 for now.
    if order == 2:
        # In K+W code, FD op size is n+1
        [p_sbp, q_sbp] = sbp21(n+1)
        m = 2
        # pb = 1
        # from opt files
        q = np.loadtxt("q_2.txt")
        r = 1
    elif order == 4:
        # In K+W code, FD op size is n+1
        [p_sbp, q_sbp] = sbp42(n+1)
        m = 4
        # pb = 2
        # from opt files
        q = np.loadtxt("q_4.txt")
        r = 5
    elif order == 6:
        # In K+W code, FD op size is n+1
        [p_sbp, q_sbp] = sbp63(n+1)
        m = 6
        # pb = 2
        # from opt files
        q = np.loadtxt("q_6.txt")
        r = 8

    h = p_sbp

    s = int(r - (m/2 - 1))

    # Make glue and finite difference grids.
    xf_b = np.linspace(0, s + m - 2, s + m - 1, endpoint=True)
    xg_b = np.linspace(0, s + m - 2, s + m - 1, endpoint=True)
    # Get the glue pieces.
    [u_dump, v_dump, mr_b] = glue_pieces(p, xf_b, xg_b)

    i = np.kron(np.linspace(s+1, n-s+1, n-2*s+1, endpoint=True),
                np.ones(m*(p+1)))
    j = np.kron(np.ones(n+1-2*s),
                np.linspace(1, m*(p+1), m*(p+1), endpoint=True)) +  \
        (p+1)*(i-1-m/2)
    # Interior solution.
    qi = np.kron(np.ones(n+1-2*s), q[0:(m*(p+1))])
    pg2f = np.zeros((n+1, n*(p+1)))
    for k in range(0, i.shape[0]):
        # Zero-based indices.
        pg2f[int(i[k])-1, int(j[k])-1] = qi[k]

    # Boundary solution.
    qb = np.reshape(q[m*(p+1):], (r*(p+1), s,), order="F")
    qb = np.transpose(qb)

    # Left block.
    pg2f[0:s, 0:r*(p+1)] = qb

    qb = np.reshape((np.diag(2*np.mod(np.linspace(1, p+1, p+1, endpoint=True),
                     2) - 1)).dot(np.flipud(np.reshape(np.rot90(qb,
                                  2).transpose(), (p+1, r*s,), order="F"))),
                    (r*(p+1), s,), order="F").transpose()

    # pg2f[np.arange(pg2f.shape[0]-s, pg2f.shape[0], 1),
    #     np.arange(pg2f.shape[1]+1-r*(p+1)-1, pg2f.shape[1], 1)] = qb

    # Right block.
    for ind_i in range(pg2f.shape[0]-s, pg2f.shape[0]):
        for ind_j in range(pg2f.shape[1]+1-r*(p+1)-1, pg2f.shape[1]):
            pg2f[ind_i, ind_j] = qb[ind_i-(pg2f.shape[0]-s),
                                    ind_j - (pg2f.shape[1]+1-r*(p+1)-1)]

    m = np.kron(np.eye(n), mr_b)

    # Pf2g comes directly from the compatibility equation.
    pf2g = (np.kron(np.eye(n),
            np.diag(1/np.diag(mr_b))).dot(np.transpose(pg2f))).dot(h)

    return pf2g, pg2f, m, h


def make_projection_g2g_hr(n, vxi_l, vxi_r):

    n_p = n + 1

    nv_l = vxi_l.shape[0]
    nv_r = vxi_r.shape[0]

    k_l = nv_l - 1
    k_r = nv_r - 1

    tol = 100.0*np.finfo(float).eps

    # Check to ensure that first and last grid points align.
    assert abs(vxi_l[-1] - vxi_r[-1]) <= \
        (tol*max(abs(vxi_l[-1]), abs(vxi_r[-1])) + np.finfo(float).eps)
    assert abs(vxi_l[-1] - vxi_r[-1]) <= \
        (tol*max(abs(vxi_l[-1]), abs(vxi_r[-1])) + np.finfo(float).eps)

    vxi_g = np.sort(np.concatenate([vxi_l, vxi_r]))
    vxi_g = np.unique(vxi_g)

    xi_g = np.vstack((vxi_g[0:-1], vxi_g[1:]))
    k_g = xi_g.shape[1]

    np_l = k_l * n_p
    np_r = k_r * n_p
    np_g = k_g * n_p

    pg2l = np.zeros((np_l, np_g,))
    pl2g = np.zeros((np_g, np_l,))

    for k in range(0, k_l):
        xia_l = vxi_l[k]
        xib_l = vxi_l[k+1]

        k_g = np.where(np.all(
                       np.vstack(([vxi_g < xib_l - tol],
                                 [vxi_g >= xia_l - tol])), axis=0))

        [pc2f, pf2c] = make_projection_g2g_h_gen(n, np.squeeze(xi_g[:, k_g]))

        idx_l = (k) * n_p
        idx_g = (k_g[0][0]) * n_p

        # pg2l[np.arange(idx_l, idx_l+n_p, 1),
        #       np.arange(idx_g, idx_g+n_p*k_g[0].shape[0], 1)] = pf2c
        # pl2g[np.arange(idx_g, idx_g+n_p*kg[0].shape[0], 1),
        #      np.arange(idx_l, idx_l+n_p, 1)] = pc2f

        for i in range(idx_l, idx_l+n_p):
            for j in range(idx_g, idx_g+n_p*k_g[0].shape[0]):
                pg2l[i, j] = pf2c[i-idx_l, j-idx_g]
                pl2g[j, i] = pc2f[j-idx_g, i-idx_l]

    pg2r = np.zeros((np_r, np_g))
    pr2g = np.zeros((np_g, np_r))

    for k in range(0, k_r):
        xia_r = vxi_r[k]
        xib_r = vxi_r[k+1]

        k_g = np.where(np.all(
                       np.vstack(([vxi_g < xib_r - tol],
                                 [vxi_g >= xia_r - tol])), axis=0))

        [pc2f, pf2c] = make_projection_g2g_h_gen(n, np.squeeze(xi_g[:, k_g]))

        idx_r = (k) * n_p
        idx_g = (k_g[0][0]) * n_p

        # pg2r[np.arange(idx_r, idx_r+n_p-1, 1),
        #      np.arange(idx_g, idx_g+n_p*k_g[0].shape[0]-1, 1)] = pf2c
        # pr2g[idx_g:idx_g+n_p*k_g[0].shape[0]-1,
        #      idx_r:idx_r+n_p-1] = pc2f

        for i in range(idx_r, idx_r+n_p):
            for j in range(idx_g, idx_g+n_p*k_g[0].shape[0]):
                pg2r[i, j] = pf2c[i-idx_r, j-idx_g]
                pr2g[j, i] = pc2f[j-idx_g, i-idx_r]

    return vxi_g, pl2g, pg2l, pr2g, pg2r


def make_projection_g2g_h_gen(n, xi_c):

    [r, w_dump] = gaussian_quad(n)

    if len(xi_c.shape) == 1:
        k = 1
        vx = xi_c
        fa = xi_c[0]
        fb = xi_c[-1]
    else:
        k = xi_c.shape[1]
        vx = np.append(xi_c[0, :], xi_c[1, -1])
        fa = xi_c[0, 0]
        fb = xi_c[1, -1]

    vx = 2/(fb - fa) * vx + 1 - 2*fb/(fb-fa)

    va = np.arange(0, k, 1)
    vb = np.arange(1, k+1, 1)

    x = (np.ones(n+1).reshape(1, -1)).transpose()*vx[va] + \
        0.5*((r.reshape(1, -1)).transpose()+1)*(vx[vb] - vx[va])

    jc = (vx[vb] - vx[va])/2
    jf = 1

    pc2f = np.zeros((x.size, r.shape[0]))

    m = (2.0/(2*np.linspace(0, n, n+1, endpoint=True) + 1))

    vr = legendre_vandermonde(r, n)

    sq_invm_invvr = np.diag(np.sqrt(np.divide(1, m))).dot(np.linalg.inv(vr))

    for k_i in range(0, k):
        pc2f[np.arange(k_i*(n+1), (k_i+1)*(n+1), 1), :] = \
                sq_invm_invvr.dot(legendre_vandermonde(x[:, k_i],
                                  n).dot(np.diag(np.sqrt(m))))

    pf2c = (1.0 / jf) * (np.diag(1.0 / m).dot(
        pc2f.transpose())).dot(np.kron(np.diag(jc), np.diag(m)))

    return pc2f, pf2c


def sbp_sbp_projection(na, qa, nb, qb):

    # Get the projection operators for each side.
    [paf2g, pag2f, ma_dump, ha_dump] = make_projection(na, qa)
    [pbf2g, pbg2f, mb_dump, hb_dump] = make_projection(nb, qb)

    # Modify the projection operators so they go to the
    # same order polynomials.

    pg = max(qa, qb)-1

    paf2g = np.kron(np.eye(na), np.eye(pg+1, qa)).dot(paf2g)
    pag2f = pag2f.dot(np.kron(np.eye(na), np.eye(qa, pg+1)))

    pbf2g = np.kron(np.eye(nb), np.eye(pg+1, qb)).dot(pbf2g)
    pbg2f = pbg2f.dot(np.kron(np.eye(nb), np.eye(qb, pg+1)))

    # Move to the glue space
    xa = np.linspace(-1, 1, na + 1, endpoint=True)
    xb = np.linspace(-1, 1, nb + 1, endpoint=True)

    # Glue-to-glue.
    [vxi_g_dump, pa2g, pg2a, pb2g, pg2b] = make_projection_g2g_hr(pg, xa, xb)

    # Stack the operators.
    pa2b = ((pbg2f.dot(pg2b)).dot(pa2g)).dot(paf2g)
    pb2a = ((pag2f.dot(pg2a)).dot(pb2g)).dot(pbf2g)

    return pa2b, pb2a


def sbp_sbp_test():

    na = 100
    nb = 230

    xa = np.linspace(-1, 1, na+1, endpoint=True)
    xb = np.linspace(-1, 1, nb+1, endpoint=True)

    eps = np.finfo(float).eps

    # FIXME: these test ranges are supposed to go to 10, but
    # we don't have those diagonal SBP operators coded yet.
    for qa in range(2, 6, 2):
        for qb in range(2, 6, 2):
            print("Creating projection for (qa, qb) ", qa, qb)
            [pa2b, pb2a] = sbp_sbp_projection(na, qa, nb, qb)

            # np.savetxt('pa2b_test.txt', pa2b)
            # np.savetxt('pb2a_test.txt', pb2a)

            print("Testing projection for (qa, qb) ", qa, qb)
            # Check the full operator
            for n in range(0, int(min(qa, qb)/2)-1):
                np.testing.assert_array_less(np.abs(pa2b.dot((xa ** n))
                                             - xb ** n), 1000*eps)
                np.testing.assert_array_less(np.abs(pb2a.dot((xb ** n))
                                             - xa ** n), 1000*eps)
            print("Test passed for (qa, qb) ", qa, qb)

            # Check the interior operator
            n_int = int(min(qa, qb))-1
            ta = pb2a.dot(xb ** n_int) - xa ** n_int
            tb = pa2b.dot(xa ** n_int) - xb ** n_int

            mb = np.argwhere(np.abs(tb) > 1000*eps).size / 2
            ma = np.argwhere(np.abs(ta) > 1000*eps).size / 2

            assert mb < nb / 2
            assert ma < na / 2

            # Look at interior part only - locate boundary portions ad-hoc.
            if mb > 0:
                assert np.max(abs(tb[int(mb):int(-mb)])) < 1000*eps
            if ma > 0:
                assert np.max(abs(ta[int(ma):int(-ma)])) < 1000*eps

    return


def sbp_dg_projection(na, nb, qa, qb, gg, dg_nodes):

    # Get the projection operators for SBP-side.
    [paf2g, pag2f, ma_dump, ha_dump] = make_projection(na, qa)
    [pbf2g, pbg2f] = glue_to_dg(qb, gg, dg_nodes)

    # Modify the projection operator so that it goes to the
    # same order polynomials as DG.

    pg = max(qa, qb)-1

    paf2g = np.kron(np.eye(na), np.eye(pg+1, qa)).dot(paf2g)
    pag2f = pag2f.dot(np.kron(np.eye(na), np.eye(qa, pg+1)))
    pbf2g = np.kron(np.eye(nb), np.eye(pg+1, qb)).dot(pbf2g)
    pbg2f = pbg2f.dot(np.kron(np.eye(nb), np.eye(qb, pg+1)))

    # Move to the glue space
    xa = np.linspace(-1, 1, na + 1, endpoint=True)
    xb = np.linspace(-1, 1, nb + 1, endpoint=True)

    # Glue-to-glue.
    [vxi_g_dump, pa2g, pg2a, pb2g, pg2b] = make_projection_g2g_hr(pg, xa, xb)

    # Stack the operators.
    pa2b = ((pbg2f.dot(pg2b)).dot(pa2g)).dot(paf2g)
    pb2a = ((pag2f.dot(pg2a)).dot(pb2g)).dot(pbf2g)

    return pa2b, pb2a


def glue_to_dg(order, gg, dg_nodes):

    from modepy.quadrature.jacobi_gauss import legendre_gauss_lobatto_nodes
    # Each element has order + 1 nodes.
    nelem = int(dg_nodes.shape[0] / (order + 1))

    # Create projection operators.
    pglue2dg = np.zeros((dg_nodes.shape[0], (gg.shape[0]-1)*order))
    pdg2glue = np.zeros(((gg.shape[0]-1)*order, dg_nodes.shape[0]))

    for i_elem in range(0, nelem):

        a_elem = dg_nodes[i_elem*(order+1)]
        b_elem = dg_nodes[i_elem*(order+1)+order]

        # Find the Legendre coefficients for coincident GG  interval, using
        # appropriate-order LGL quadrature.
        quad_nodes = legendre_gauss_lobatto_nodes(order)
        # Before transforming these, use them to get the weights.
        quad_weights = lgl_weights(order+1, quad_nodes)
        # Transform nodes to the appropriate interval.
        for i in range(0, order + 1):
            quad_nodes[i] = (quad_nodes[i] + 1) * (b_elem - a_elem) \
                            / 2.0 + a_elem
        # Transform the weights as well.
        for i in range(0, order + 1):
            quad_weights[i] = quad_weights[i] * (b_elem - a_elem) / 2.0
        # Verify that nodal points on the interface are already LGL points.
        for i in range(0, order+1):
            assert abs(quad_nodes[i] - dg_nodes[i_elem*(order+1)+i]) < 1e-12

        # Do the quadrature.
        # Coefficient loop.
        for i in range(0, order):
            # Quadrature point loop.
            for k in range(0, order+1):
                # Get Legendre polynomial of order i in
                # interval, evaluated at quad point k.
                poly_at_k = legendre_interval(i, a_elem,
                                              b_elem, quad_nodes[k])

                # Get Legendre coefficients.
                i_op = i_elem*order + i
                j_op = k + i_elem*(order+1)
                pdg2glue[i_op][j_op] = ((2*i+1)/(b_elem - a_elem))\
                    * quad_weights[k] * poly_at_k

    # Now that we have the projection from DG to glue, the projection
    # from glue to DG comes from the H-compatibility condition

    # In our case, H is a repeated diagonal of the quad weights.
    h = np.zeros((nelem*(order+1), nelem*(order+1)))
    for i_elem in range(0, nelem):
        for i in range(0, order+1):
            h[i_elem*(order+1)+i][i_elem*(order+1)+i] = quad_weights[i]

    # Will need to invert this.
    hinv = np.linalg.inv(h)

    # Now we need to also construct the per-interval mass matrix,
    # taking advantage of the fact that the Legendre polynomials
    # are orthogonal.

    m = np.zeros(((gg.shape[0]-1)*order, (gg.shape[0]-1)*order))
    m_small = np.zeros(order)
    # Only same-order Legendre polynomials in the same interval
    # will be nonzero, so the matrix is diagonal.
    for i in range(0, order):
        m_small[i] = (gg[1] - gg[0]) / (2.0 * i + 1)
    for i in range(0, (gg.shape[0]-1)*order):
        i_rem = i % order
        m[i][i] = m_small[i_rem]

    # Finally, we can use compatibility to get the reverse projection.
    pglue2dg = (m.dot(pdg2glue)).dot(hinv)
    pglue2dg = pglue2dg.transpose()

    return pdg2glue, pglue2dg


def legendre_interval(order, a_int, b_int, point_k):

    # First check if the query point
    # is within the interval. If it isn't,
    # return 0.
    if point_k >= a_int and point_k <= (b_int + 1e-12):
        pass
    else:
        return 0

    # With that out of the way, do some polynomial building
    # using Bonnet's recursion formula, remembering to
    # shift the Legendre polynomials to the interval
    x_trans = (point_k - a_int) / (b_int - a_int) \
        - (b_int - point_k) / (b_int - a_int)
    p_n_0 = 1
    p_n_1 = x_trans

    # If our input order is either of these, return them.
    if order == 0:
        return p_n_0
    elif order == 1:
        return p_n_1

    # Otherwise we enter recursion mode.
    p_n_m1 = p_n_0
    p_n = p_n_1
    for i in range(1, order):
        p_n_p1 = (1.0 / (i + 1)) * ((2*i + 1) * x_trans * p_n - i * p_n_m1)
        p_n_m1 = p_n
        p_n = p_n_p1

    return p_n_p1


def lgl_weights(n, nodes):

    weights = np.zeros(n)
    for i in range(0, n):
        weights[i] = (2.0 / (n*(n-1)
                             * (legendre_interval(n-1, -1, 1, nodes[i]))**2))

    return weights
