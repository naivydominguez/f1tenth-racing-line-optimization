"""Quasi-steady-state speed profile on a fixed path (forward/backward pass with a friction ellipse).

This is what the car can follow with pure pursuit + a speed setpoint, and it is used for the
centre-line baseline and for the minimum-curvature line. Tunable knobs (config/params.yaml):
mu, friction_usage, v_max, a_accel_max / v_switch (motor), a_brake_max, ggv_exponent.
"""
import numpy as np

from .geometry import heading_curvature, arc_length, _open
from .params import G, a_lat_max


def _ax_accel_max(v, p):
    lim = p.limits
    motor = lim.a_accel_max * np.minimum(1.0, lim.v_switch / max(v, 1e-3))
    return min(motor, lim.friction_usage * lim.mu * G)


def _ax_brake_max(p):
    lim = p.limits
    return min(lim.a_brake_max, lim.friction_usage * lim.mu * G)


def speed_profile(xy, p, kappa=None):
    """Closed path xy (N,2), roughly uniform spacing. Returns dict with v, ax, ay, t, lap_time, s, kappa."""
    xy = _open(xy)
    N = len(xy)
    if kappa is None:
        _, kappa = heading_curvature(xy)
    seg = np.hypot(*(np.roll(xy, -1, axis=0) - xy).T)
    s, L = arc_length(xy)
    ay_max = a_lat_max(p)
    e = p.limits.ggv_exponent
    ak = np.abs(kappa)
    v_lim = np.minimum(p.limits.v_max, np.sqrt(ay_max / np.maximum(ak, 1e-9)))

    def remain(v, k):
        r = 1.0 - (v * v * k / ay_max) ** e
        return max(r, 0.0) ** (1.0 / e)

    i0 = int(np.argmin(v_lim))
    # forward (acceleration-limited) pass, two wraps starting at the slowest point
    vf = v_lim.copy()
    for step in range(2 * N):
        i = (i0 + step) % N
        j = (i + 1) % N
        a = _ax_accel_max(vf[i], p) * remain(vf[i], ak[i])
        vf[j] = min(vf[j], np.sqrt(vf[i] ** 2 + 2.0 * a * seg[i]))
    # backward (braking-limited) pass
    vb = vf.copy()
    abr = _ax_brake_max(p)
    for step in range(2 * N):
        j = (i0 - step) % N
        i = (j - 1) % N
        a = abr * remain(vb[j], ak[j])
        vb[i] = min(vb[i], np.sqrt(vb[j] ** 2 + 2.0 * a * seg[i]))
    v = vb
    vn = np.roll(v, -1)
    dt = 2.0 * seg / np.maximum(v + vn, 1e-6)
    ax = (vn ** 2 - v ** 2) / (2.0 * np.maximum(seg, 1e-9))
    t = np.concatenate([[0.0], np.cumsum(dt[:-1])])
    return dict(v=v, ax=ax, ay=v * v * kappa, t=t, lap_time=float(np.sum(dt)), s=s, kappa=kappa,
                length=L)
