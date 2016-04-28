import numpy as np

def gen_pow_matrix(primpoly):
    primpoly = int(primpoly)
    q = np.floor(np.log2(primpoly)).astype(int)
    pm = np.zeros((2 ** q - 1, 2), dtype=int)
    cur_elem = 2
    for i in range(1, pm.shape[0] + 1):
        pm[cur_elem - 1, 0] = i
        pm[i - 1, 1] = cur_elem
        new_elem = cur_elem << 1
        if new_elem >> q == 1:
            cur_elem = new_elem ^ primpoly
        else:
            cur_elem = new_elem
    return pm

def add(X, Y):
    return np.bitwise_xor(X.astype(int), Y.astype(int))

def sum(X, axis=0):
    if len(X.shape) != 2:
        raise ValueError("X is not 2-dimensional matrix")
    X = X.astype(int)
    if axis == 0:
        result = np.zeros((X.shape[1]), dtype=int)
        for i in range(X.shape[0]):
            result = add(result, X[i, :])
    elif axis == 1:
        result = np.zeros((X.shape[0]), dtype=int)
        for j in range(X.shape[1]):
            result = add(result, X[:, j])
    else:
        raise ValueError("Wrong axis value! It should be 0 or 1")
    return result

def prod(X, Y, pm):
    X = X.astype(int)
    Y = Y.astype(int)
    pow_X = pm[X - 1, 0]
    pow_Y = pm[Y - 1, 0]
    pow_hadamard_prod = (pow_X + pow_Y) % (pm.shape[0])
    hadamard_prod = pm[pow_hadamard_prod - 1, 1]
    hadamard_prod[np.logical_or(X == 0, Y == 0)] = 0
    return hadamard_prod.astype(int)

def divide(X, Y, pm):
    X = X.astype(int)
    Y = Y.astype(int)
    if np.any(Y == 0):
        raise ValueError("Division by zero")
    pow_X = pm[X - 1, 0]
    pow_Y = pm[Y - 1, 0]
    pow_hadamard_div = (pow_X - pow_Y) % (pm.shape[0])
    hadamard_div = pm[pow_hadamard_div - 1, 1]
    hadamard_div[X == 0] = 0
    return hadamard_div

def linsolve(A, b, pm):
    A = np.copy(A)
    b = np.copy(b)
    # Checking input data:
    if (A.shape[0] != A.shape[1]):
        raise ValueError("Matrix dimensions are not the same")
    if (b.size != A.shape[0]):
        raise ValueError("A and b dimensions are not equal")
    # Gauss-like method
    # Forward:
    for j in range(A.shape[1]):
        nz_col_idx = np.nonzero(A[j:, j])[0] + j
        if nz_col_idx.size == 0:
            return np.nan
        # Division:
        b[nz_col_idx] = divide(b[nz_col_idx], A[nz_col_idx, j], pm)
        A[nz_col_idx, :] = divide(A[nz_col_idx, :], 
                                  np.tile(A[nz_col_idx, j].reshape(-1, 1), A.shape[1]), pm)
        # Subtracting:
        for i in nz_col_idx[1:]:
            A[i, :] = add(A[i, :], A[nz_col_idx[0], :])
            b[i] = add(b[i], b[nz_col_idx[0]])
        # Swapping:
        if nz_col_idx[0] > j:
            tmp_row = A[j, :].copy()
            A[j, :] = A[nz_col_idx[0], :].copy()
            A[nz_col_idx[0], :] = tmp_row.copy()
            tmp = b[j].copy()
            b[j] = b[nz_col_idx[0]].copy()
            b[nz_col_idx[0]] = tmp.copy()
    # Backward:
    x = np.zeros(b.size, dtype=int)
    for i in range(b.size - 1, -1, -1):
        x[i] = add(b[i], sum(prod(x[(i + 1):], A[i, (i + 1):], pm).reshape(-1, 1))).copy()
    return x.astype(int)

def polyprod(p1, p2, pm):
    if p1.size >= p2.size:
        f1, f2 = p2, p1
    else:
        f1, f2 = p1, p2
    conv = np.zeros(f1.size + f2.size - 1)
    for i in range(f1.size):
        conv[i:(i + f2.size)] = add(conv[i:(i + f2.size)], prod(f1[i] * np.ones(f2.size), f2, pm))
    return conv.astype(int)

def minpoly(x, pm):
    # Suppose that x elements are not equal zero
    root_pows = np.zeros(shape=(pm.shape[0] + 1), dtype=bool)
    min_poly = np.array([1], dtype=int)
    # Generate cyclotomic cosets:
    pow_x = pm[x - 1, 0]
    root_pows[pow_x] = True
    prev_roots_number = np.unique(x).size
    while True:
        pow_x = 2 * pow_x % pm.shape[0]
        root_pows[pow_x] = True
        new_roots_number = np.sum(root_pows)
        if prev_roots_number == new_roots_number:
            break
        prev_roots_number = new_roots_number
    # Multiply polynomials:
    root_pows = np.nonzero(root_pows)[0]
    roots = pm[root_pows - 1, 1]
    for root in roots:
        min_poly = polyprod(np.array([1, root]), min_poly, pm)
    return (min_poly, root_pows)

def polyval(p, x, pm):
    result = np.zeros(x.size)
    # Horner's method:
    result = p[0]
    for i in range(1, p.size):
        result = add(prod(result, x, pm), np.array(p[i]))
    return result.astype(int)

def polysum(p1, p2):
    p1 = p1.copy()
    p2 = p2.copy()
    pow_diff = p1.size - p2.size
    if pow_diff > 0:
        p2 = np.append(np.zeros(pow_diff), p2)
    elif pow_diff < 0:
        p1 = np.append(np.zeros(-pow_diff), p1)
    poly_sum = add(p1, p2)
    nonzero_idx = np.nonzero(poly_sum)[0]
    if nonzero_idx.size == 0:
        return np.array([0])
    poly_sum = poly_sum[nonzero_idx[0]:]
    return poly_sum

def polydivmod(p1, p2, pm):
    if p2.size > p1.size:
        return (np.array([0], dtype=int), p1)
    p1 = p1.astype(int).copy()
    q = np.zeros(p1.size - p2.size + 1, dtype=int)
    for i in range(q.size):
        q[i] = divide(np.array([p1[0]]), np.array([p2[0]]), pm)[0]
        poly_prod = polyprod(q[i:], p2, pm)
        if poly_prod.size > p1.size:
            q[i] = 0
        else:
            p1 = polysum(p1, polyprod(q[i:], p2, pm))
    return (q, p1)

def euclid(p1, p2, pm, max_deg=0):
    p1 = p1[np.nonzero(p1)[0][0]:].copy()
    p2 = p2[np.nonzero(p2)[0][0]:].copy()
    if p2.size > p1.size:
        r_prev_prev, r_prev = p2, p1
    else:
        r_prev_prev, r_prev = p1, p2
    x_prev_prev, x_prev = np.array([1]), np.array([0])
    y_prev_prev, y_prev = np.array([0]), np.array([1])
    while True:
        (q, r) = polydivmod(r_prev_prev, r_prev, pm)
        x = polysum(x_prev_prev, polyprod(q, x_prev, pm))
        y = polysum(y_prev_prev, polyprod(q, y_prev, pm))
        r_prev_prev, r_prev = r_prev, r
        x_prev_prev, x_prev = x_prev, x
        y_prev_prev, y_prev = y_prev, y
        if (max_deg > 0 and r.size - 1 <= max_deg) or (max_deg == 0 and r.size == 1):
            break
    return (r_prev, x_prev, y_prev)
