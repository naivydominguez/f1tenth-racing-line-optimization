"""Time-optimal racing line: optimal control problem in curvilinear coordinates, solved with
CasADi + IPOPT by direct multiple shooting (cf. Christ et al., Vehicle System Dynamics 2021,
who use a double-track model; we use a dynamic single-track model with a friction limit).

Independent variable: arc length s along the reference line (track centre line).
States  x = [v, beta, omega, delta, n, xi]
        speed, side-slip at CG, yaw rate, steering angle, lateral offset (left +), heading error
Controls u = [delta_dot, a]   steering rate, longitudinal acceleration
Vehicle dynamics: exactly the CommonRoad / f1tenth_gym single-track model (linear tyres whose
stiffness is scaled by the gym 'mu'), plus what the gym does NOT model: a friction limit per axle
    (a/g)^2 + (F_y,i / F_z,i)^2 <= (friction_usage * mu)^2        (load-proportional drive/brake)
Objective: lap time = sum over intervals of integral dt/ds ds  (+ tiny control-rate regularisation)
Every shooting interval is integrated with RK4 and continuity is imposed between intervals;
the last interval wraps onto the first (closed lap, periodic states).
"""
import time
import numpy as np
import casadi as ca

from .geometry import heading_curvature, resample_closed, wrap_angle
from .params import G, half_car_clearance
from .speed_profile import speed_profile
from .min_curv import optimize_mincurv


def _model(p):
    v_ = p.vehicle
    lf, lr, h, m, Iz = v_.lf, v_.lr, v_.h_cg, v_.mass, v_.I_z
    CSf, CSr, mus = v_.C_Sf, v_.C_Sr, v_.mu_stiffness
    L = lf + lr
    x = ca.SX.sym('x', 6)
    u = ca.SX.sym('u', 2)
    kap = ca.SX.sym('kap')
    v, beta, om, de, n, xi = [x[i] for i in range(6)]
    ddel, a = u[0], u[1]
    gf = G * lr - a * h          # proportional to front load
    gr = G * lf + a * h          # proportional to rear load
    om_dot = (-mus * m / (v * Iz * L) * (lf ** 2 * CSf * gf + lr ** 2 * CSr * gr) * om
              + mus * m / (Iz * L) * (lr * CSr * gr - lf * CSf * gf) * beta
              + mus * m / (Iz * L) * lf * CSf * gf * de)
    beta_dot = ((mus / (v ** 2 * L) * (CSr * gr * lr - CSf * gf * lf) - 1) * om
                - mus / (v * L) * (CSr * gr + CSf * gf) * beta
                + mus / (v * L) * CSf * gf * de)
    s_dot = v * ca.cos(xi + beta) / (1 - n * kap)
    xdot = ca.vertcat(a, beta_dot, om_dot, ddel, v * ca.sin(xi + beta), om - kap * s_dot)
    f_s = ca.Function('f_s', [x, u, kap], [xdot / s_dot, 1.0 / s_dot])
    # friction usage per axle (normalised by vertical load)
    alpha_f = de - beta - lf * om / v
    alpha_r = -beta + lr * om / v
    fy_f = mus * CSf * alpha_f
    fy_r = mus * CSr * alpha_r
    fric = ca.Function('fric', [x, u], [ca.vertcat((a / G) ** 2 + fy_f ** 2, (a / G) ** 2 + fy_r ** 2)])
    return f_s, fric


def _rk4_interval(f_s, steps):
    x = ca.SX.sym('x', 6)
    u = ca.SX.sym('u', 2)
    kap = ca.SX.sym('kap')
    hs = ca.SX.sym('h')
    dt = 0
    xc = x
    hh = hs / steps
    for _ in range(steps):
        k1, t1 = f_s(xc, u, kap)
        k2, t2 = f_s(xc + hh / 2 * k1, u, kap)
        k3, t3 = f_s(xc + hh / 2 * k2, u, kap)
        k4, t4 = f_s(xc + hh * k3, u, kap)
        xc = xc + hh / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        dt = dt + hh / 6 * (t1 + 2 * t2 + 2 * t3 + t4)
    return ca.Function('F', [x, u, kap, hs], [xc, dt])


def optimize_mintime(track, p, ds=None, warm=None, verbose=False):
    """Solve the minimum-lap-time OCP. Returns a dict with the racing line, speed, inputs, lap time."""
    ds = ds or p.mintime.ds
    lim = p.limits
    if warm is None:
        warm = optimize_mincurv(track, p, ds=ds)
    ref = warm['ref']
    N = ref.n_points
    seg = np.hypot(*(np.roll(ref.xy, -1, axis=0) - ref.xy).T)
    kap_n = ref.kappa
    kap_i = 0.5 * (kap_n + np.roll(kap_n, -1))          # curvature per interval
    c = half_car_clearance(p)
    lb_n, ub_n = warm['lb'], warm['ub']

    f_s, fric = _model(p)
    F = _rk4_interval(f_s, int(p.mintime.rk4_steps))
    Fm = F.map(N)
    Fr = fric.map(N)

    X = ca.MX.sym('X', 6, N)
    U = ca.MX.sym('U', 2, N)
    Xn, DT = Fm(X, U, ca.DM(kap_i).T, ca.DM(seg).T)
    Xnext = ca.horzcat(X[:, 1:], X[:, 0])
    g_cont = ca.vec(Xn - Xnext)
    g_fric = ca.vec(Fr(X, U))
    g_pow = ca.vec(U[1, :] * X[0, :])                  # a * v <= a_max * v_switch
    T = ca.sum2(DT)
    dU = ca.horzcat(U[:, 1:], U[:, :1]) - U
    J = (T + p.mintime.w_steer_rate * ca.sumsqr(U[0, :] / lim.delta_rate_max) / N
         + p.mintime.w_accel_rate * ca.sumsqr(dU[1, :] / lim.a_accel_max) / N)

    w = ca.vertcat(ca.vec(X), ca.vec(U))
    g = ca.vertcat(g_cont, g_fric, g_pow)
    mu_lim = (lim.friction_usage * lim.mu) ** 2
    lbg = np.concatenate([np.zeros(6 * N), np.full(2 * N, -np.inf), np.full(N, -np.inf)])
    ubg = np.concatenate([np.zeros(6 * N), np.full(2 * N, mu_lim), np.full(N, lim.a_accel_max * lim.v_switch)])

    vmin = p.mintime.v_min
    xi_max = float(p.mintime.get('xi_max', 0.8))
    xlb = np.array([vmin, -0.6, -20.0, -lim.delta_max, 0.0, -xi_max])
    xub = np.array([lim.v_max, 0.6, 20.0, lim.delta_max, 0.0, xi_max])
    Xlb = np.tile(xlb[:, None], (1, N))
    Xub = np.tile(xub[:, None], (1, N))
    Xlb[4, :] = lb_n
    Xub[4, :] = ub_n
    # keep 1 - n*kappa > 0 with margin (only matters on very tight references)
    with np.errstate(divide='ignore'):
        nmax_k = np.where(kap_n > 1e-6, 0.9 / np.abs(kap_n), np.inf)
    Xub[4, :] = np.where(kap_n > 0, np.minimum(Xub[4, :], nmax_k), Xub[4, :])
    Xlb[4, :] = np.where(kap_n < 0, np.maximum(Xlb[4, :], -nmax_k), Xlb[4, :])
    Ulb = np.tile(np.array([[-lim.delta_rate_max], [-lim.a_brake_max]]), (1, N))
    Uub = np.tile(np.array([[lim.delta_rate_max], [lim.a_accel_max]]), (1, N))
    lbw = np.concatenate([Xlb.ravel(order='F'), Ulb.ravel(order='F')])
    ubw = np.concatenate([Xub.ravel(order='F'), Uub.ravel(order='F')])

    # ---------------- warm start from the min-curvature line + its speed profile
    path = warm['xy']
    psi_p, kap_p = heading_curvature(path)
    sp = speed_profile(path, p, kappa=kap_p)
    Lwb = p.vehicle.lf + p.vehicle.lr
    v0 = np.clip(sp['v'], vmin, lim.v_max)
    de0 = np.clip(np.arctan(Lwb * kap_p), -lim.delta_max, lim.delta_max)
    beta0 = np.arctan(p.vehicle.lr / Lwb * np.tan(de0))
    xi0 = wrap_angle(psi_p - beta0 - ref.psi)
    X0 = np.vstack([v0, beta0, v0 * kap_p, de0, np.clip(warm['alpha'], lb_n, ub_n), np.clip(xi0, -xi_max, xi_max)])
    U0 = np.vstack([np.zeros(N), np.clip(sp['ax'], -lim.a_brake_max, lim.a_accel_max)])
    w0 = np.concatenate([X0.ravel(order='F'), U0.ravel(order='F')])

    opts = {'ipopt.max_iter': int(p.mintime.max_iter), 'ipopt.print_level': int(p.mintime.print_level),
            'print_time': False, 'ipopt.tol': 1e-6, 'ipopt.acceptable_tol': 1e-4,
            'ipopt.mu_strategy': 'adaptive', 'ipopt.sb': 'yes'}
    solver = ca.nlpsol('mintime', 'ipopt', {'x': w, 'f': J, 'g': g}, opts)
    t0 = time.time()
    sol = solver(x0=w0, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg)
    solve_time = time.time() - t0
    stats = solver.stats()
    wopt = np.array(sol['x']).ravel()
    Xo = wopt[:6 * N].reshape((6, N), order='F')
    Uo = wopt[6 * N:].reshape((2, N), order='F')
    dt_int = np.array(ca.Function('dtf', [X, U], [DT])(Xo, Uo)).ravel()
    fr = np.array(ca.Function('frf', [X, U], [Fr(X, U)])(Xo, Uo))
    v, beta, om, de, n, xi = Xo
    xy = ref.xy + ref.normal * n[:, None]
    t = np.concatenate([[0.0], np.cumsum(dt_int[:-1])])
    # sanity check: IPOPT can 'succeed' on a solution that exploits the discretisation (heading
    # swings within one interval -> kinked path). Reject kinks the car cannot steer and lines slower
    # than the min-curvature warm start; the robust wrapper then retries / falls back.
    from .params import kappa_max as _kmax
    try:
        xy_chk = resample_closed(xy, 0.1)
        kap_chk = heading_curvature(xy_chk)[1]
        kink = float(np.max(np.abs(kap_chk))) / _kmax(p)
    except Exception:
        kink = np.inf
    lap = float(np.sum(dt_int))
    valid = bool(kink <= 1.3 and lap <= 1.05 * sp['lap_time'])
    return dict(xy=xy, v=v, beta=beta, omega=om, delta=de, n=n, xi=xi, ddelta=Uo[0], ax=Uo[1],
                psi=ref.psi + xi, t=t, lap_time=float(np.sum(dt_int)), ref=ref,
                solve_time=solve_time, status=stats['return_status'], success=bool(stats['success']) and valid,
                valid=valid, kink_ratio=kink, warm_lap_time=float(sp['lap_time']),
                iterations=int(stats['iter_count']), friction_use=np.sqrt(np.maximum(fr, 0)) / np.sqrt(mu_lim),
                N=N)


def optimize_mintime_robust(track, p, warm=None, verbose=False):
    """Solve the OCP; if IPOPT fails, retry with slightly different discretisation / settings.
    Returns (result, attempts). If every attempt fails, result is None -> caller falls back to the
    minimum-curvature line (the plan in our proposal)."""
    from .params import with_overrides
    tries = [{}, {'mintime.rk4_steps': 2}, {'mintime.v_min': 0.5}, {'mintime.ds': round(0.8 * p.mintime.ds, 3)},
             {'track.ref_smoothing': max(0.1, 2 * p.track.ref_smoothing)},
             {'mintime.ds': round(1.2 * p.mintime.ds, 3), 'mintime.v_min': 0.5}]
    attempts = []
    for ov in tries:
        pp = with_overrides(p, ov) if ov else p
        w = warm if (not ov and warm is not None) else None
        try:
            res = optimize_mintime(track, pp, warm=w)
        except Exception as e:  # pragma: no cover
            attempts.append((ov, f'error: {e}'))
            continue
        st = res['status'] if res['valid'] else f"{res['status']}_REJECTED(kink {res['kink_ratio']:.1f}x)"
        attempts.append((ov, st))
        if verbose:
            print(f'[mintime] {ov or "default"}: {st} lap {res["lap_time"]:.3f}s '
                  f'({res["iterations"]} it, {res["solve_time"]:.1f}s)')
        if res['success']:
            res['attempts'] = attempts
            res['settings'] = ov
            return res, attempts
    return None, attempts
