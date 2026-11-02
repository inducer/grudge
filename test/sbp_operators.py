import numpy as np


# Standard FD method: second-order
def sbp21(n):
    p = np.zeros((n, n, ))
    q = np.zeros((n, n, ))

    # norm matrix
    p[0, 0] = 0.5
    p[n-1, n-1] = 0.5
    for i in range(1,  n-1):
        p[i, i] = 1.0

    # now the q matrix
    q[0, 1] = 0.5
    q[n-1, n-2] = -0.5
    for i in range(1,  n-1):
        q[i, i-1] = -0.5
        q[i, i+1] = 0.5

    return p,  q


# Standard FD method: third-order
def sbp42(n):
    p = np.zeros((n, n, ))
    q = np.zeros((n, n, ))

    # norm matrix
    # upper subblock
    p[0, 0] = 17.0/48.0
    p[1, 1] = 59.0/48.0
    p[2, 2] = 43.0/48.0
    p[3, 3] = 49.0/48.0
    # lower subblock
    p[n-1, n-1] = 17.0/48.0
    p[n-2, n-2] = 59.0/48.0
    p[n-3, n-3] = 43.0/48.0
    p[n-4, n-4] = 49.0/48.0
    for i in range(4,  n-4):
        p[i, i] = 1.0

    # now the q matrix
    # upper subblock
    q[0, 1] = 59.0/96.0
    q[0, 2] = -1.0/12.0
    q[0, 3] = -1.0/32.0
    q[1, 2] = 59.0/96.0
    q[2, 3] = 59.0/96.0
    q[1, 0] = -q[0, 1]
    q[2, 0] = -q[0, 2]
    q[2, 1] = -q[1, 2]
    q[2, 4] = -1.0/12.0
    q[3, 0] = -q[0, 3]
    q[3, 2] = -q[2, 3]
    q[3, 4] = 2.0/3.0
    q[3, 5] = -1.0/12.0
    # lower subblock
    q[n-1, n-2] = -q[0, 1]
    q[n-1, n-3] = -q[0, 2]
    q[n-1, n-4] = -q[0, 3]
    q[n-2, n-1] = -q[1, 0]
    q[n-2, n-3] = -q[1, 2]
    q[n-3, n-1] = -q[2, 0]
    q[n-3, n-2] = -q[2, 1]
    q[n-3, n-4] = -q[2, 3]
    q[n-3, n-5] = 1.0/12.0
    q[n-4, n-1] = -q[3, 0]
    q[n-4, n-3] = -q[3, 2]
    q[n-4, n-5] = -2.0/3.0
    q[n-4, n-6] = 1.0/12.0
    for i in range(4,  n-4):
        q[i, i-2] = 1.0/12.0
        q[i, i-1] = -2.0/3.0
        q[i, i+1] = 2.0/3.0
        q[i, i+2] = -1.0/12.0

    return p,  q


# Standard FD method: fourth-order
def sbp63(n):
    p = np.zeros((n, n, ))
    q = np.zeros((n, n, ))

    # norm matrix
    # upper subblock
    p[0, 0] = 13649.0/43200.0
    p[1, 1] = 12013.0/8640.0
    p[2, 2] = 2711.0/4320.0
    p[3, 3] = 5359.0/4320.0
    p[4, 4] = 7877.0/8640.0
    p[5, 5] = 43801.0/43200.0
    # lower subblock
    p[n-1, n-1] = 13649.0/43200.0
    p[n-2, n-2] = 12013.0/8640.0
    p[n-3, n-3] = 2711.0/4320.0
    p[n-4, n-4] = 5359.0/4320.0
    p[n-5, n-5] = 7877.0/8640.0
    p[n-6, n-6] = 43801.0/43200.0
    for i in range(6,  n-6):
        p[i, i] = 1.0

    # now the q matrix
    # upper subblock
    q[0, 1] = 127/211
    q[0, 2] = 35/298
    q[0, 3] = -32/83
    q[0, 4] = 89/456
    q[0, 5] = -2/69

    q[1, 0] = -127/211
    q[1, 2] = -1/167
    q[1, 3] = 391/334
    q[1, 4] = -50/71
    q[1, 5] = 14/99

    q[2, 0] = -35/298
    q[2, 1] = 1/167
    q[2, 3] = -34/79
    q[2, 4] = 90/113
    q[2, 5] = -69/271

    q[3, 0] = 32/83
    q[3, 1] = -391/334
    q[3, 2] = 34/79
    q[3, 4] = 6/25
    q[3, 5] = 31/316
    q[3, 6] = 1/60

    q[4, 0] = -89/456
    q[4, 1] = 50/71
    q[4, 2] = -90/113
    q[4, 3] = -6/25
    q[4, 5] = 37/56
    q[4, 6] = -3/20
    q[4, 7] = 1/60

    q[5, 0] = 2/69
    q[5, 1] = -14/99
    q[5, 2] = 69/271
    q[5, 3] = -31/316
    q[5, 4] = -37/56
    q[5, 6] = 3/4
    q[5, 7] = -3/20
    q[5, 8] = 1/60

    # lower subblock
    for i in range(0, 6):
        for j in range(0, 9):
            q[n-1-i, n-1-j] = -q[i, j]

    # interior
    for i in range(6,  n-6):
        q[i, i-3] = -1.0/60.0
        q[i, i-2] = 3.0/20.0
        q[i, i-1] = -3.0/4.0
        q[i, i+1] = 3.0/4.0
        q[i, i+2] = -3.0/20.0
        q[i, i+3] = 1.0/60.0

    return p,  q
