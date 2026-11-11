#!/usr/bin/env python3
"""Plan racing lines for one track and export waypoint CSVs for the car / gym.

  python scripts/optimize.py tracks/technical_s1/technical_s1_centerline.csv
  python scripts/optimize.py tracks/siebel2/siebel2_centerline.csv --set limits.mu=0.7 limits.v_max=4 \
         limits.delta_max=0.36 track.wall_margin=0.25 --out results/lines/siebel2

Methods: centerline (baseline), mincurv (warm start / fallback), mintime (CasADi OCP).
Any parameter in config/params.yaml can be overridden with --set section.key=value.
Writes <out>/<method>_waypoints.csv (x,y,v,yaw,kappa,s,ax), summary.json and lines.png.
"""
import argparse
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from raceline.params import load_params, drift_speed, kappa_max  # noqa: E402
from raceline.track import Track  # noqa: E402
from raceline.pipeline import plan_lines, export_line, summary_json, LABELS  # noqa: E402

if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('centerline_csv')
    ap.add_argument('--params', default=None)
    ap.add_argument('--set', nargs='*', default=[], help='overrides like limits.mu=0.7')
    ap.add_argument('--methods', nargs='*', default=['centerline', 'mincurv', 'mintime'])
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    p = load_params(a.params, a.set)
    tr = Track.from_csv(a.centerline_csv)
    out = a.out or os.path.join(ROOT, 'results', 'lines', tr.name.replace('_centerline', ''))
    print(f'track {tr.summary()}')
    print(f"limits: mu={p.limits.mu} usage={p.limits.friction_usage} v_max={p.limits.v_max} "
          f"delta_max={p.limits.delta_max} (min turn radius {1 / kappa_max(p):.2f} m) margin={p.track.wall_margin}")
    print('max speed before drift: ' + ', '.join(f'R={r} m: {drift_speed(p, r):.2f} m/s' for r in (1, 2, 4)))
    lines = plan_lines(tr, p, methods=a.methods, verbose=True)
    for m, l in lines.items():
        path = export_line(l, out, m)
        print(f'{LABELS[m]:20s} planned lap {l["lap_time"]:7.3f} s   ({l["status"]}, {l["solve_time"]:.1f} s)  -> {path}')
    summary_json(lines, os.path.join(out, 'summary.json'), extra={'params_overrides': a.set})
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 8))
    for e in (tr.left_edge, tr.right_edge):
        ax.plot(*np.vstack([e, e[:1]]).T, 'k', lw=0.8)
    for m, c in zip(lines, ['0.6', 'tab:blue', 'tab:orange']):
        ax.plot(*np.vstack([lines[m]['xy'], lines[m]['xy'][:1]]).T, c=c, lw=1.2,
                label=f"{LABELS[m]}: {lines[m]['lap_time']:.2f} s")
    ax.set_aspect('equal')
    ax.legend(loc='best', fontsize=8)
    fig.savefig(os.path.join(out, 'lines.png'), dpi=120)
    print('plot ->', os.path.join(out, 'lines.png'))
