#!/usr/bin/env python3
"""Estimate tyre-floor friction mu from a circle test log.

CSV columns (header names): t, v, yaw_rate, steer   (steer optional; SI units, rad)
    python tools/calibrate_mu.py results/circle_test.csv --plot results/circle_test.png

Then put the result into config/params.yaml -> limits.mu, and keep limits.friction_usage
at 0.8-0.9 for the first fast runs.
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from raceline.calibration import estimate_mu, drift_speed_table  # noqa: E402

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('csv')
    ap.add_argument('--wheelbase', type=float, default=0.33)
    ap.add_argument('--plot', default=None)
    a = ap.parse_args()
    d = np.genfromtxt(a.csv, delimiter=',', names=True)
    steer = d['steer'] if 'steer' in d.dtype.names else None
    r = estimate_mu(d['t'], d['v'], d['yaw_rate'], steer, wheelbase=a.wheelbase)
    print(f"mu estimate (max sustained a_y / g): {r['mu_est']:.3f}   (a_y max {r['ay_max']:.2f} m/s^2 at v = {r['v_at_peak']:.2f} m/s)")
    if 'mu_onset' in r:
        print(f"slip onset: v = {r['slip_onset_v']:.2f} m/s, a_y = {r['slip_onset_ay']:.2f} m/s^2 (mu_onset {r['mu_onset']:.3f})")
    print('max speed before drift at friction_usage 0.7 (car start):')
    for R, v in drift_speed_table(0.7 * r['mu_est']).items():
        print(f'   R = {R:4.2f} m  ->  {v:4.2f} m/s')
    if a.plot:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        ax[0].plot(r['v'], r['ay'], lw=1)
        ax[0].axhline(r['ay_max'], ls='--', c='k', lw=0.8)
        ax[0].set_xlabel('speed [m/s]')
        ax[0].set_ylabel('lateral accel v*yaw_rate [m/s^2]')
        if 'ratio' in r:
            ax[1].plot(r['ay'], r['ratio'], '.', ms=2)
            ax[1].set_xlabel('a_y [m/s^2]')
            ax[1].set_ylabel('measured / commanded curvature')
            ax[1].set_ylim(0, 1.5)
        fig.tight_layout()
        fig.savefig(a.plot, dpi=120)
        print('plot:', a.plot)
