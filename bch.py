import gf
import numpy as np
from scipy.linalg import hankel

# Support functions for coding:
def bool_polysum(p1, p2):
    # p1 and p2 are bool polynomials as bool numpy arrays
    size_diff = p1.size - p2.size
    if size_diff > 0:
        p2 = np.append(np.zeros(size_diff), p2)
    elif size_diff < 0:
        p1 = np.append(np.zeros(-size_diff), p1)
    poly_sum = np.logical_xor(p1, p2)
    nonzero_idx = np.nonzero(poly_sum)[0]
    if nonzero_idx.size == 0:
        return np.array([False])
    poly_sum = poly_sum[nonzero_idx[0]:]
    return poly_sum

def bool_polyprod(p1, p2):
    # p1 and p2 are bool polynomials as bool numpy arrays
    poly_prod = np.convolve(p1.astype(int), p2.astype(int))
    poly_prod = poly_prod % 2
    if np.all(poly_prod == 0):
        return np.array([0])
    poly_prod = poly_prod[np.nonzero(poly_prod)[0][0]:]
    return poly_prod

def bool_polydivmod(p1, p2):
    # p1 and p2 are bool polynomials as bool numpy arrays
    # p1 = q * p2 + r
    # returns tuple (q, r)
    p1 = p1.copy()
    p2 = p2.copy()
    q_deg = p1.size - p2.size
    if q_deg < 0:
        return (np.array([0]), p1)
    q = np.zeros(q_deg + 1, dtype=bool)
    while (p1.size >= p2.size):
        cur_q_pow = q_deg - (p1.size - p2.size)
        q[cur_q_pow] = 1
        sub_poly = bool_polyprod(q[cur_q_pow:], p2)
        p1 = bool_polysum(p1, sub_poly)
    r = p1
    return (q, r)

def coding(U, g):
    n_mes = U.shape[0]
    k = U.shape[1]
    m = g.size - 1
    x_pow_m = np.zeros((m + 1), dtype=int)
    x_pow_m[0] = 1
    V = np.zeros((n_mes, k + m), dtype=int)
    for i in range(n_mes):
        s = bool_polyprod(x_pow_m, U[i, :])
        r = bool_polydivmod(s, g)[1]
        v = bool_polysum(s, r)
        V[i, -v.size:] = v.copy()
    return V
    
def genpoly(n, t):
    # Very-very bad style:
    prim_poly_array = np.array([0, 0, 7, 11, 19, 37, 67, 131, 285, 529, 
                                1033, 2053, 4179, 8219, 16427, 32771, 65581])
    q = np.log2(n + 1).astype(int)
    if q < 2 or q > 16:
        raise ValueError("log2(n + 1) should be in [2, 16]")
    pm = gf.gen_pow_matrix(prim_poly_array[q])
    bch_zeros = pm[:(2 * t), 1]
    g = gf.minpoly(bch_zeros, pm)[0]
    return (g, bch_zeros, pm)

def dist(g, n):
    k = n - g.size + 1
    U = np.eye(k)
    V = coding(U, g)
    min_dist = n + 1
    for num in range(1, 2 ** k):
        num_list = list(bin(num)[2:])
        lin_comb_coefs = np.array(((k - len(num_list)) * [0] + num_list), dtype=int).reshape(k, 1)
        new_dist = np.sum(np.sum(V * lin_comb_coefs, axis=0) % 2)
        if new_dist < min_dist:
            min_dist = new_dist
    return min_dist

def decoding(W, R, pm, method='euclid'):
    n_mes = W.shape[0]
    n = W.shape[1]
    V = np.zeros((n_mes, n))
    t = int(R.size / 2)
    for mes_idx in range(n_mes):
        # Step 1. Computing syndrome polynomials:
        s = gf.polyval(W[mes_idx, :], R, pm)
        if np.count_nonzero(s) == 0:
            V[mes_idx, :] = W[mes_idx, :]
            continue
        # Step 2. Computing Lambda(z) coefficients:
        # PGZ decoder:
        if method == 'pgz':
            for nu in range(t, 0, -1):
                A = hankel(s[:nu], s[(nu-1) : (2*nu-1)])
                b = s[nu : (2*nu)]
                x = gf.linsolve(A, b, pm)
                if np.all(np.isnan(x) == False):
                    break
            if np.any(np.isnan(x) == True):
                W[mes_idx, :] = np.nan
                continue
            Lambda = np.append(x, [1])
        # Euclid decoder:
        elif method == 'euclid':
            S = np.append(s[::-1], [1])
            z_pow_d = np.zeros((2 * t + 2), dtype=int)
            z_pow_d[0] = 1
            Lambda = gf.euclid(z_pow_d, S, pm, max_deg=t)[2]
        else:
            raise ValueError("Unknown method name")
        # Step 3 and 4. Finding all roots of Lambda(z) and
        # Computing error positions:
        error_positions = np.where(gf.polyval(Lambda, pm[:, 1], pm) == 0)[0]
        # Step 5. Computing decoded message:
        v = W[mes_idx, :].copy()
        v[error_positions] = np.abs(v[error_positions] - 1)
        # Step 6. Checking decoded message:
        if np.count_nonzero(gf.polyval(v, R, pm)) == 0:
            V[mes_idx, :] = v.copy()
        else:
            V[mes_idx, :] = np.nan
    return V