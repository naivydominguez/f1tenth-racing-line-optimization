"""Minimum-curvature racing line (Heilmeier et al., Vehicle System Dynamics 2020), own implementation.

The path is p_i = r_i + alpha_i * n_i (reference point + lateral offset along the left normal).
A closed cubic spline through p_i gives node second derivatives M = T p with
T = 6 A^-1 D (A = cyclic tridiag(1,4,1), D = cyclic 2nd difference). Freezing the first
derivatives (and the speed term) at the previous iterate makes curvature linear in alpha:
    kappa = q + Q alpha
and   min ||kappa||^2  s.t.  -(w_r - c) <= alpha <= (w_l - c),  |kappa| <= kappa_max
is a QP (solved with CasADi's qpOASES / OSQP). Repeating the linearisation a few times
('iterative QP') removes most of the linearisation error.
"""
import time
import numpy as np
import casadi as ca

from .geometry import resample_closed, heading_curvature, normals_left
from .params import kappa_max as car_kappa_max, half_car_clearance
from .utils import quiet_stdout


def _spline_matrices(N):
    A = np.zeros((N, N))
    D = np.zeros((N, N))
    for i in range(N):
        A[i, i] = 4.0
        A[i, (i - 1) % N] = 1.0
        A[i, (i + 1) % N] = 1.0
        D[i, i] = -2.0
        D[i, (i - 1) % N] = 1.0
        D[i, (i + 1) % N] = 1.0
    T = 6.0 * np.linalg.solve(A, D)          # M = T p
    E = np.zeros((N, N))                      # first derivative at node: p_{i+1}-p_i-(2M_i+M_{i+1})/6
    for i in range(N):
        E[i, i] = -1.0
        E[i, (i + 1) % N] = 1.0
    F = E - (2.0 * np.eye(N) + np.roll(np.eye(N), 1, axis=1)) @ T / 6.0
    return T, F


def optimize_mincurv(track, p, ds=None, iterations=None, verbose=False):
    """Returns dict(alpha (offset at reference nodes), ref (resampled reference Track), xy (path at
    nodes), kappa, solve_time, status)."""
    ds = ds or p.mincurv.ds
    iterations = iterations or p.mincurv.iterations
    ref = track.resampled(ds=ds, smoothing=p.track.ref_smoothing)
    N = ref.n_points
    r = ref.xy
    nrm = ref.normal
    c = half_car_clearance(p)
    lb = -(ref.w_right - c)
    ub = ref.w_left - c
    if np.any(lb > ub):
        mask = lb > ub
        bad = int(np.sum(mask))
        mid = 0.5 * (lb + ub)
        lb[mask] = mid[mask]
        ub[mask] = mid[mask]
        if verbose:
            print(f'[mincurv] warning: {bad} points narrower than the car + margins; pinned to middle')
    T, F = _spline_matrices(N)
    kmax = car_kappa_max(p) if p.mincurv.limit_curvature else 1e3

    alpha = np.zeros(N)
    t0 = time.time()
    status = 'ok'
    solver_name = 'qpoases' if N <= 900 else 'osqp'
    for it in range(iterations):
        pts = r + nrm * alpha[:, None]
        d1 = F @ pts                                # first derivatives (spline param) at nodes
        den = np.power(np.sum(d1 * d1, axis=1), 1.5)
        ax_ = d1[:, 0] / den
        ay_ = d1[:, 1] / den
        Q = (ax_[:, None] * T) * nrm[:, 1][None, :] - (ay_[:, None] * T) * nrm[:, 0][None, :]
        q = ax_ * (T @ r[:, 1]) - ay_ * (T @ r[:, 0])
        H = 2.0 * Q.T @ Q
        H = 0.5 * (H + H.T) + 1e-9 * np.eye(N)
        g = 2.0 * Q.T @ q
        opts = {'print_problem': False, 'print_time': False}
        if solver_name == 'qpoases':
            opts.update({'printLevel': 'none', 'sparse': False, 'enableRegularisation': True})
        else:
            opts.update({'osqp': {'verbose': False, 'eps_abs': 1e-7, 'eps_rel': 1e-7, 'max_iter': 20000,
                                  'polish': True}})
        with quiet_stdout():
            S = ca.conic('mc', solver_name, {'h': ca.DM(H).sparsity(), 'a': ca.DM(Q).sparsity()}, opts)
            sol = S(h=ca.DM(H), g=ca.DM(g), a=ca.DM(Q), lba=-kmax - q, uba=kmax - q, lbx=lb, ubx=ub, x0=alpha)
        st = S.stats()
        if not st.get('success', True):
            status = f'qp_iter{it}_' + str(st.get('return_status', 'fail'))
        alpha = np.array(sol['x']).ravel()
        if verbose:
            kap = q + Q @ alpha
            print(f'[mincurv] iter {it}: sum k^2 = {np.sum(kap ** 2):.4f}, max|k| = {np.max(np.abs(kap)):.3f}')
    xy = r + nrm * alpha[:, None]
    _, kap = heading_curvature(xy)
    return dict(alpha=alpha, ref=ref, xy=xy, kappa=kap, solve_time=time.time() - t0, status=status,
                lb=lb, ub=ub)
