import numpy as np
import jax.numpy as jnp
from jax import jit, value_and_grad
from scipy.optimize import minimize
import torch

from motion_code.sparse_gp import *
from motion_code.utils import *

def optimize_motion_codes(X_list, Y_list, labels, model_path, m=10, Q=8, latent_dim=3, sigma_y=0.1):
    '''
    Main algorithm to optimize all variables for the Motion Code model.
    '''
    num_motion = np.unique(labels).shape[0]
    dims = (num_motion, m, latent_dim, Q)

    # Initialize parameters
    X_m_start = np.repeat(sigmoid_inv(np.linspace(0.1, 0.9, m)).reshape(1, -1), latent_dim, axis=0).swapaxes(0, 1)
    Z_start = np.ones((num_motion, latent_dim))
    Sigma_start = softplus_inv(np.ones((num_motion, Q)))
    W_start = softplus_inv(np.ones((num_motion, Q)))

    # Optimize X_m, Z, and kernel parameters including Sigma, W
    res = minimize(fun=elbo_fn(X_list, Y_list, labels, sigma_y, dims),
        x0 = pack_params([X_m_start, Z_start, Sigma_start, W_start]),
        method='L-BFGS-B', jac=True)
    X_m, Z, Sigma, W = unpack_params(res.x, dims=dims)
    Sigma = softplus(Sigma)
    W = softplus(W)

    # We now optimize distribution params for each motion and store means in mu_ms, covariances in A_ms, and for convenient K_mm_invs
    mu_ms = []; A_ms = []; K_mm_invs = []

    # All timeseries of the same motion is put into a list, an element of X_motion_lists and Y_motion_lists
    X_motion_lists = []; Y_motion_lists = []
    for _ in range(num_motion):
        X_motion_lists.append([]); Y_motion_lists.append([])
    for i in range(len(Y_list)):
        X_motion_lists[labels[i]].append(X_list[i])
        Y_motion_lists[labels[i]].append(Y_list[i])

    # For each motion, using trained kernel parameter in "pair" form to obtain optimal distribution params for each motion.
    for k in range(num_motion):
        kernel_params = (Sigma[k], W[k])
        mu_m, A_m, K_mm_inv = phi_opt(sigmoid(X_m@Z[k]), X_motion_lists[k], Y_motion_lists[k], sigma_y, kernel_params) 
        mu_ms.append(mu_m); A_ms.append(A_m); K_mm_invs.append(K_mm_inv)
    
    # Save model to path.
    model = {'X_m': X_m, 'Z': Z, 'Sigma': Sigma, 'W': W, 
             'mu_ms': mu_ms, 'A_ms': A_ms, 'K_mm_invs': K_mm_invs}
    np.save(model_path, model)
    return

def optimize_motion_codes_gpu(X_list, Y_list, labels, model_path, m=10, Q=8, latent_dim=3, sigma_y=0.1, maxiter=500, tol=1e-5):
    '''
    GPU-accelerated version of optimize_motion_codes.
    Uses PyTorch's L-BFGS optimizer so the entire optimization loop
    (including L-BFGS state updates and gradient evaluations via autograd)
    stays on the GPU, avoiding per-iteration device round-trips.
    '''
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype = torch.float64

    num_motion = np.unique(labels).shape[0]
    dims = (num_motion, m, latent_dim, Q)

    # ---- PyTorch re-implementations of sparse_gp helpers ----
    def _spectral_kernel(X1, X2, sigma, alpha):
        X12 = (X1.unsqueeze(1) - X2.unsqueeze(0)).unsqueeze(-1)
        return torch.sum(
            alpha.reshape(1, 1, -1)
            * torch.exp(-0.5 * X12 * sigma.reshape(1, 1, -1) * X12),
            dim=-1,
        )

    jitter_val = torch.tensor(1e-6, device=device, dtype=dtype)

    def _elbo_from_kernel(K_mm, K_mn, y, trace_avg, sigma_y):
        n = y.shape[0]
        L = torch.linalg.cholesky(K_mm)
        A = torch.linalg.solve_triangular(L, K_mn, upper=False) / sigma_y
        AAT = A @ A.T
        B = torch.eye(K_mn.shape[0], device=device, dtype=dtype) + AAT
        LB = torch.linalg.cholesky(B)
        c = torch.linalg.solve_triangular(LB, A @ y, upper=False) / sigma_y

        lb = -n / 2 * np.log(2 * np.pi)
        lb = lb - torch.sum(torch.log(torch.diag(LB)))
        lb = lb - n / 2 * np.log(sigma_y ** 2)
        lb = lb - 0.5 / sigma_y ** 2 * (y.T @ y).squeeze()
        lb = lb + 0.5 * (c.T @ c).squeeze()
        lb = lb - 0.5 / sigma_y ** 2 * n * trace_avg
        lb = lb + 0.5 * torch.trace(AAT)
        return -lb

    # ---- Move data to device ----
    X_list_t = [torch.tensor(np.asarray(x), device=device, dtype=dtype) for x in X_list]
    Y_list_t = [torch.tensor(np.asarray(y), device=device, dtype=dtype) for y in Y_list]

    # ---- Initialize parameters ----
    X_m_start = np.repeat(
        sigmoid_inv(np.linspace(0.1, 0.9, m)).reshape(1, -1),
        latent_dim, axis=0,
    ).swapaxes(0, 1)
    Z_start = np.ones((num_motion, latent_dim))
    Sigma_start = softplus_inv(np.ones((num_motion, Q)))
    W_start = softplus_inv(np.ones((num_motion, Q)))

    x0 = pack_params([X_m_start, Z_start, Sigma_start, W_start])
    params = torch.tensor(x0, device=device, dtype=dtype, requires_grad=True)

    n_series = len(X_list_t)
    labels_arr = np.asarray(labels)

    def _unpack(p):
        idx = 0
        X_m = p[idx:idx + m * latent_dim].reshape(m, latent_dim); idx += m * latent_dim
        Z = p[idx:idx + num_motion * latent_dim].reshape(num_motion, latent_dim); idx += num_motion * latent_dim
        S = p[idx:idx + num_motion * Q].reshape(num_motion, Q); idx += num_motion * Q
        W = p[idx:idx + num_motion * Q].reshape(num_motion, Q)
        return X_m, Z, S, W

    def closure():
        optimizer.zero_grad()
        X_m, Z, S, W = _unpack(params)
        S = torch.nn.functional.softplus(S)
        W = torch.nn.functional.softplus(W)

        loss = torch.tensor(0.0, device=device, dtype=dtype)
        for i in range(n_series):
            k = labels_arr[i]
            X_m_k = torch.sigmoid(X_m @ Z[k])
            eye_m = torch.eye(X_m_k.shape[0], device=device, dtype=dtype) * jitter_val
            K_mm = _spectral_kernel(X_m_k, X_m_k, S[k], W[k]) + eye_m
            K_mn = _spectral_kernel(X_m_k, X_list_t[i], S[k], W[k])
            trace_avg = torch.sum(W[k] ** 2)
            y_n_k = Y_list_t[i].reshape(-1, 1)
            loss = loss + _elbo_from_kernel(K_mm, K_mn, y_n_k, trace_avg, sigma_y)

        loss = loss / n_series
        loss.backward()
        return loss

    optimizer = torch.optim.LBFGS(
        [params], max_iter=maxiter,
        tolerance_grad=tol, tolerance_change=tol,
        line_search_fn='strong_wolfe',
    )
    optimizer.step(closure)

    # ---- Convert optimized params back to numpy / JAX for phi_opt ----
    params_np = params.detach().cpu().numpy()
    X_m, Z, Sigma, W = unpack_params(params_np, dims=dims)
    Sigma = softplus(Sigma)
    W = softplus(W)

    mu_ms = []; A_ms = []; K_mm_invs = []

    X_motion_lists = [[] for _ in range(num_motion)]
    Y_motion_lists = [[] for _ in range(num_motion)]
    for i in range(len(Y_list)):
        X_motion_lists[labels[i]].append(X_list[i])
        Y_motion_lists[labels[i]].append(Y_list[i])

    for k in range(num_motion):
        kernel_params = (Sigma[k], W[k])
        mu_m, A_m, K_mm_inv = phi_opt(sigmoid(X_m @ Z[k]), X_motion_lists[k], Y_motion_lists[k], sigma_y, kernel_params)
        mu_ms.append(mu_m); A_ms.append(A_m); K_mm_invs.append(K_mm_inv)

    model = {'X_m': X_m, 'Z': Z, 'Sigma': Sigma, 'W': W,
             'mu_ms': mu_ms, 'A_ms': A_ms, 'K_mm_invs': K_mm_invs}
    np.save(model_path, model)
    return

def optimize_motion_codes_jax_gpu(X_list, Y_list, labels, model_path, m=10, Q=8, latent_dim=3, sigma_y=0.1, maxiter=500, tol=1e-5):
    '''
    GPU-accelerated version of optimize_motion_codes using JAX.
    Uses jaxopt's L-BFGS optimizer so the entire optimization loop
    (including L-BFGS state updates and gradient evaluations via autodiff)
    stays on the GPU, avoiding per-iteration device round-trips.
    '''
    import jaxopt

    num_motion = np.unique(labels).shape[0]
    dims = (num_motion, m, latent_dim, Q)

    X_list_jax = [jnp.array(x) for x in X_list]
    Y_list_jax = [jnp.array(y) for y in Y_list]
    n_series = len(X_list_jax)
    labels_arr = np.asarray(labels)

    # Initialize parameters
    X_m_start = np.repeat(sigmoid_inv(np.linspace(0.1, 0.9, m)).reshape(1, -1), latent_dim, axis=0).swapaxes(0, 1)
    Z_start = np.ones((num_motion, latent_dim))
    Sigma_start = softplus_inv(np.ones((num_motion, Q)))
    W_start = softplus_inv(np.ones((num_motion, Q)))

    x0 = jnp.array(pack_params([X_m_start, Z_start, Sigma_start, W_start]))

    def elbo(params):
        X_m, Z, Sigma, W = unpack_params(params, dims)
        Sigma = softplus(Sigma)
        W = softplus(W)

        loss = 0.0
        for i in range(n_series):
            k = labels_arr[i]
            X_m_k = sigmoid(X_m @ Z[k])
            K_mm = spectral_kernel(X_m_k, X_m_k, Sigma[k], W[k]) + jitter(X_m_k.shape[0])
            K_mn = spectral_kernel(X_m_k, X_list_jax[i], Sigma[k], W[k])
            trace_avg_all_comps = jnp.sum(W[k] ** 2)
            y_n_k = Y_list_jax[i].reshape(-1, 1)
            loss += elbo_fn_from_kernel(K_mm, K_mn, y_n_k, trace_avg_all_comps, sigma_y)

        return loss / n_series

    solver = jaxopt.LBFGS(fun=elbo, maxiter=maxiter, tol=tol, jit=True)
    result = solver.run(x0)
    params_opt = result.params

    params_np = np.array(params_opt)
    X_m, Z, Sigma, W = unpack_params(params_np, dims=dims)
    Sigma = softplus(Sigma)
    W = softplus(W)

    mu_ms = []; A_ms = []; K_mm_invs = []

    X_motion_lists = [[] for _ in range(num_motion)]
    Y_motion_lists = [[] for _ in range(num_motion)]
    for i in range(len(Y_list)):
        X_motion_lists[labels[i]].append(X_list[i])
        Y_motion_lists[labels[i]].append(Y_list[i])

    for k in range(num_motion):
        kernel_params = (Sigma[k], W[k])
        mu_m, A_m, K_mm_inv = phi_opt(sigmoid(X_m @ Z[k]), X_motion_lists[k], Y_motion_lists[k], sigma_y, kernel_params)
        mu_ms.append(mu_m); A_ms.append(A_m); K_mm_invs.append(K_mm_inv)

    model = {'X_m': X_m, 'Z': Z, 'Sigma': Sigma, 'W': W,
             'mu_ms': mu_ms, 'A_ms': A_ms, 'K_mm_invs': K_mm_invs}
    np.save(model_path, model)
    return

def classify_predict_helper(X_test, Y_test, kernel_params_all_motions, X_m, Z, mu_ms, A_ms, K_mm_invs, mode='dt'):
    """
    Classify by calculate distance between inducing (mean) values and interpolated test values at inducing pts.
    """
    num_motion = len(kernel_params_all_motions)
    ind = -1; min_ll = 1e9
    for k in range(num_motion):
        X_m_k = sigmoid(X_m @ Z[k])
        if mode == 'simple':
            Y = np.interp(X_m_k, X_test, Y_test)
            ll = ((mu_ms[k]-Y).T)@(mu_ms[k]-Y)
        elif mode == 'variational':
            Sigma, W = kernel_params_all_motions[k]
            K_mm = spectral_kernel(X_m_k, X_m_k, Sigma, W) + jitter(X_m_k.shape[0])
            K_mn = spectral_kernel(X_m_k, X_test, Sigma, W)
            trace_avg_all_comps = jnp.sum(W**2)
            y_n_k = Y_test.reshape(-1, 1) # shape (n, 1)
            ll = elbo_fn_from_kernel(K_mm, K_mn, y_n_k, trace_avg_all_comps, sigma_y=0.1)
        elif mode == 'dt':
            mean, _ = q(X_test, X_m_k, kernel_params_all_motions[k], mu_ms[k], A_ms[k], K_mm_invs[k])
            # ll = jnp.log(jnp.linalg.det(covar)) + ((Y_test-mean).T)@jnp.linalg.inv(covar)@(Y_test-mean)
            ll = ((mean-Y_test).T)@(mean-Y_test) 
        if ind == -1:
            ind = k; min_ll = ll
        elif min_ll > ll: 
            ind = k; min_ll = ll
    
    return ind
