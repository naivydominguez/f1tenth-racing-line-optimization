"""Friction (mu) calibration from a constant-steer circle test at slowly increasing speed.

Physics: in a steady circle a_y = v * yaw_rate = v^2 / R. While the tyres grip, the measured path
curvature (yaw_rate / v) stays close to the commanded curvature tan(delta)/L (slowly dropping
because of understeer). At the grip limit the car can no longer raise a_y: either the front
washes out (curvature collapses) or the rear lets go (yaw rate spikes, car spins). We take
    mu_est = max sustained a_y / g          (sustained = median over a 0.5 s window)
and report the slip-onset speed where measured/commanded curvature departs from its low-speed
trend. This is the proposal's mu = v^2/(r g) test, made automatic.

IMPORTANT on the real car: /odom from vesc_to_odom computes yaw rate FROM the steering command,
so it can never show slip. Use an independent yaw rate (IMU gyro z, or the particle-filter pose
differentiated) and a speed estimate (VESC wheel speed is fine for the lateral test).
"""
import numpy as np
from scipy.ndimage import median_filter

G = 9.81


def estimate_mu(t, v, yaw_rate, steer=None, wheelbase=0.33, window_s=0.5, v_min=0.8):
    t = np.asarray(t, float)
    v = np.asarray(v, float)
    w = np.asarray(yaw_rate, float)
    dt = float(np.median(np.diff(t)))
    k = max(int(round(window_s / dt)) | 1, 3)
    vs = median_filter(v, k, mode='nearest')
    ws = median_filter(w, k, mode='nearest')
    ay = np.abs(vs * ws)
    valid = vs > v_min
    ay_sust = median_filter(np.where(valid, ay, 0.0), k, mode='nearest')
    i_peak = int(np.argmax(ay_sust))
    out = dict(mu_est=float(ay_sust[i_peak] / G), ay_max=float(ay_sust[i_peak]), v_at_peak=float(vs[i_peak]),
               t_at_peak=float(t[i_peak]), ay=ay, v=vs, t=t)
    if steer is not None:
        st = median_filter(np.asarray(steer, float), k, mode='nearest')
        k_cmd = np.tan(np.abs(st)) / wheelbase
        k_meas = np.abs(ws) / np.maximum(vs, 1e-3)
        ratio = np.where(valid & (k_cmd > 1e-3), k_meas / np.maximum(k_cmd, 1e-6), np.nan)
        # linear (understeer) trend of the ratio vs a_y over the first half of the grip range
        sel = valid & np.isfinite(ratio) & (ay < 0.5 * out['ay_max'])
        if np.sum(sel) > 10:
            c = np.polyfit(ay[sel], ratio[sel], 1)
            trend = np.polyval(c, ay)
            dev = np.abs(ratio - trend)
            after = np.where(valid & (ay > 0.5 * out['ay_max']) & (dev > 0.10))[0]
            if len(after):
                j = int(after[0])
                out.update(slip_onset_v=float(vs[j]), slip_onset_ay=float(ay[j]), slip_onset_t=float(t[j]),
                           mu_onset=float(ay[j] / G))
        out.update(ratio=ratio)
    return out


def drift_speed_table(mu_eff, radii=(0.75, 1.0, 1.5, 2.0, 3.0, 5.0)):
    """Max steady speed before sliding in a corner of radius R: sqrt(mu_eff * g * R)."""
    return {float(r): float(np.sqrt(mu_eff * G * r)) for r in radii}
