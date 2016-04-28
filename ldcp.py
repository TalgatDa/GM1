import numpy as np
import sys
import matplotlib.pyplot as plt

''' Major functions: '''

def make_generator_matrix(H):
    m, n = H.shape
    Hcopy = H.copy()
    k = n - m
    not_ind = []
    cur_row = 0
    for col in range(n):
        # Searching for a pivot:
        nz = np.nonzero(Hcopy[cur_row:, col])[0] + cur_row
        if nz.size == 0:
            continue
        nonzero_row = nz[0]
        # Swap rows:
        Hcopy[nonzero_row, :], Hcopy[cur_row, :] = Hcopy[cur_row, :], Hcopy[nonzero_row, :].copy()
        # Subtract:
        if nz.size != 1:
            Hcopy[nz[1:], :] = np.logical_xor(Hcopy[cur_row, :][np.newaxis, :], Hcopy[nz[1:], :])
        cur_row += 1
        not_ind.append(col)
        if cur_row >= m:
            break
    if (len(not_ind) != m):
        return ()
    # Every leading coefficient is 1 and is the only nonzero entry in its column:
    for step, column in enumerate(not_ind):
        nz = np.where(Hcopy[:step, column])
        Hcopy[nz, :] = np.logical_xor(Hcopy[nz, :], Hcopy[step, :][np.newaxis, :])
    not_ind = np.array(not_ind)
    G = np.empty((n, k), dtype=H.dtype)
    ind = np.setdiff1d(np.arange(n), not_ind)
    G[ind, :] = np.eye(k, dtype=H.dtype)
    G[not_ind, :] = Hcopy[:, ind]
    return G, ind


def decode(s, H, q, schedule = 'parallel', damping = 1, max_iter = 300, 
           tol_beliefs = 1e-4, display = False, return_stab = False):
    zero_H_mask = H == 0
    m, n = H.shape
    vec_stab = np.full(max_iter, float(n))
    prior = np.array([1 - q, q])
    mu_he = np.zeros((m, n, 2))
    b = np.zeros((n, 2))
    eps = 1e-6
    status = 2
    lam = damping
    # Neighborhood precalculation for sequent schedule:
    Ni = []
    Nj = []
    for i in range(n):
        Ni.append(np.where(H[:, i])[0])
    for j in range(m):
        Nj.append(np.where(H[j, :])[0])
        
    # 1. Initialization:
    mu_eh = np.tile(prior, (m, n, 1))
    for n_iter in range(0, max_iter):
        if display:
            print("n_iter:\t", n_iter)
            sys.stdout.flush()
        mu_he_prev = mu_he.copy()
        mu_eh_prev = mu_eh.copy()
        b_prev = b.copy()
        if schedule == 'parallel':
            # 2. Parallel mu_h->e recalculation:
            delta_pk = mu_eh[:, :, 0] - mu_eh[:, :, 1]
            delta_pk[zero_H_mask] = 1.0
            num_zeros_in_row = np.sum(delta_pk == 0, axis=1)
            rows = np.where(num_zeros_in_row == 1)[0]
            cols = np.where(delta_pk[rows, :] == 0)[0]
            delta_pk[delta_pk == 0] = 1.0
            delta_pl = np.prod(delta_pk, axis=1)[:, np.newaxis] / delta_pk
            saved = delta_pl[rows, cols]
            delta_pl[num_zeros_in_row >= 1, :] = 0
            delta_pl[num_zeros_in_row == 1, cols] = saved
            pl = np.dstack(((1 + delta_pl) / 2, (1 - delta_pl) / 2))
            mu_he = pl.copy()
            idx = np.ix_(np.where(s)[0], np.arange(n))
            mu_he[:, :, 1][idx] = pl[:, :, 0][idx].copy()
            mu_he[:, :, 0][idx] = pl[:, :, 1][idx].copy()
            mu_he = lam * mu_he + (1 - lam) * mu_he_prev
            # 3.1 Parallel mu_e->h recalculation:
            mu_eh_prev = mu_he.copy()
            log_mu_he = mu_he.copy()
            log_mu_he[np.logical_or((H == 0)[:, :, np.newaxis], mu_he < eps)] = 1.0
            log_mu_he = np.log(log_mu_he)
            sum_log_mu_he = np.sum(log_mu_he, axis=0)
            mu_eh = prior.reshape(1, 1, 2) * np.exp(sum_log_mu_he[np.newaxis, :, :] - log_mu_he)
            mu_eh /= np.sum(mu_eh, axis=2)[:, :, np.newaxis]
            mu_eh = lam * mu_eh + (1 - lam) * mu_eh_prev
            # 3.2 Parallel beliefs recalculation:
            b = prior[np.newaxis, :] * np.exp(sum_log_mu_he)
            b /= b.sum(axis=1)[:, np.newaxis]
        elif schedule == 'sequent':
            # 2 & 3. Sequent mu_h->e, mu_e->h and belief recalculation:
            for j in range(m):
                # 2. mu_h[j]->e
                delta_pk = mu_eh[j, Nj[j], 0] - mu_eh[j, Nj[j], 1] # shape = n
                delta_pk = np.tile(delta_pk, (Nj[j].size, 1))
                delta_pk[np.arange(Nj[j].size), np.arange(Nj[j].size)] = 1.0
                delta_pl = np.prod(delta_pk, axis=1)
                pl = np.vstack((1.0 + delta_pl, 1.0 - delta_pl)).T / 2
                mu_he[j, :, :] = 0.0
                if s[j] == 0:
                    mu_he[j, Nj[j], :] = pl.copy()
                else:
                    mu_he[j, Nj[j], :] = pl[:, ::-1].copy()
                # 3.1 mu_e->h[j]
                mu_eh[j, :, :] = 0.0
                for i in Nj[j]:
                    k = np.setdiff1d(Ni[i], [j])
                    mu_eh[j, i, :] = prior * np.prod(mu_he[k, i, :], axis=0)
                    b[i, :] = mu_eh[j, i, :] * mu_he[j, i, :]
                # normalization
                sum_along_prob = mu_eh[j, Nj[j], :].sum(axis=1)
                mu_eh[j, Nj[j], :][sum_along_prob == 0, :] = prior[np.newaxis, :]
                sum_along_prob[sum_along_prob == 0] = 1.0
                mu_eh[j, Nj[j], :] /= sum_along_prob[:, np.newaxis]
                b_sum = b.sum(axis=1)
                b[b_sum == 0, :] = prior[np.newaxis, :]
                b_sum[b_sum == 0] = 1.0
                b /= b_sum[:, np.newaxis]
        else:
            raise ValueError("Unknown schedule!")
        # 4. Error estimation recalculation:
        e = np.argmax(b, axis=1)
        # 5. Termination criteria:
        if np.all(np.dot(H, e) % 2 == s):
            status = 0
            break
        vec_stab[n_iter] = np.sum(np.abs(b[:, 0] - b_prev[:, 0]) < tol_beliefs)
        if vec_stab[n_iter] == n:
            status = 1
            break
    if return_stab:
        return e, status, vec_stab.astype(float) / n
    return e, status


def estimate_errors(H, q, num_points = 300, schedule='parallel', damping=1.0, display=False, G=None):
    err_bit, err_block, diver = [0.0] * 3
    m, n = H.shape
    k = n - m
    if G is None:
        G, ind = make_generator_matrix(H)
    for n_iter in range(num_points):
        u = np.random.randint(0, 2, k)
        v = np.dot(G, u) % 2
        w = transfer(v, q)
        s = np.dot(H, w) % 2
        e, status = decode(s, H, q, max_iter=300, schedule=schedule, damping=damping)
        if status < 2:
            n_errors = np.sum(v != (w + e) % 2)
            if n_errors > 0:
                err_block += 1.0
                err_bit += float(n_errors) / n
        else:
            diver += 1.0
        if display and n_iter % 10 == 0:
            print(n_iter)
            sys.stdout.flush()
    err_bit /= (num_points - diver)
    err_block /= (num_points - diver)
    diver /= num_points
    return err_bit, err_block, diver

''' Additional functions: '''

def transfer(v, q):
    w = v.copy()
    idx = np.random.rand(v.size) < q
    w[idx] = np.logical_not(w[idx])
    return w

def generate_random_H(m, n, j):
    p = float(j) / m
    res = ()
    while len(res) == 0:
        H = np.random.binomial(1, p, (m, n))
        res = make_generator_matrix(H)
    return H, res[0], res[1]

def check_decoder(m, n, q, j, schedule='parallel'):
    H, G, ind = generate_random_H(m, n, j)
    u = np.random.randint(0, 2, n - m)
    v = np.dot(G, u) % 2
    w = transfer(v, q)
    s = np.dot(H, w) % 2
    true_e = (w - v) % 2
    e, status, part_stab = decode(s, H, q, max_iter=300, schedule=schedule, return_stab=True)
    print(status)
    if np.any(np.dot(H, e) % 2 != s):
        print("ERROR")
    else:
        print("YES")
    print("status:", status)
    print(true_e, e)
    # return H, v, w, s, ind

def compute_channel_capacity(q):
    return 1.0 + q * np.log2(q) + (1 - q) * np.log2(1 - q)

def estimate_bch_errors(H, q, g_poly, R, pm, num_points = 300, display=False):
    m, n = H.shape
    k = n - m
    U = np.random.randint(0, 2, (num_points, k))
    V = coding(U, g_poly)
    W = V.copy()
    idx = np.random.rand(W.shape[0], W.shape[1]) < q
    W[idx] = np.logical_not(W[idx])
    decoded = decoding(W, R, pm)
    idx = ~np.isnan(decoded).any(axis=1)
    decoded_without_nan = decoded[idx]
    if (decoded_without_nan.size == 0):
        return (1, 1, 1)
    diver = num_points - decoded_without_nan.shape[0]
    sum_err = np.sum(decoded_without_nan != V[idx, :], axis=1)
    err_bit = np.mean(sum_err) / n
    err_block = np.sum(sum_err != 0) / (num_points - diver)
    diver /= num_points
    return err_bit, err_block, diver
