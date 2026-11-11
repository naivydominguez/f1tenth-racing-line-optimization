#!/usr/bin/env python3
"""Synthetic circle test on the saturating-tyre simulator (known mu) -> CSV, to validate
tools/calibrate_mu.py before we trust it on the car.

    python tools/sim_circle_test.py --mu-true 0.7 --steer 0.25 --out results/circle_mu0.7.csv
"""
import argparse
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from raceline.params import load_params  # noqa: E402
from sim.vehicle import Car  # noqa: E402


def simulate(mu_true, steer=0.25, v0=1.0, ramp=0.15, t_end=60.0, p=None, seed=0, noise=True):
    p = p or load_params()
    rng = np.random.default_rng(seed)
    car = Car(p, tire_model='pacejka', mu_true=mu_true, dt=0.01, steer_delay=2)
    car.reset(0, 0, 0, v0)
    rows = []
    t = 0.0
    while t < t_end:
        v_cmd = v0 + ramp * t
        car.step(v_cmd, steer)
        t += 0.01
        s = car.state
        v = math.hypot(s[3], s[6])
        yaw_rate = s[5]
        if noise:   # IMU gyro + wheel-speed noise
            v += rng.normal(0, 0.03)
            yaw_rate += rng.normal(0, 0.02)
        if int(round(t / 0.01)) % 2 == 0:
            rows.append((t, v, yaw_rate, steer))
        if abs(s[6]) > 3.0 or abs(s[5]) > 15:   # spun out
            break
    return np.array(rows)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--mu-true', type=float, default=0.7)
    ap.add_argument('--steer', type=float, default=0.25)
    ap.add_argument('--out', default='results/circle_test.csv')
    a = ap.parse_args()
    d = simulate(a.mu_true, a.steer)
    os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)
    np.savetxt(a.out, d, delimiter=',', header='t,v,yaw_rate,steer', comments='', fmt='%.5f')
    print(f'wrote {len(d)} rows to {a.out}')
