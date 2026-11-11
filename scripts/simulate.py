#!/usr/bin/env python3
"""Closed-loop laps (our pure pursuit on the f1tenth_gym dynamics, no ROS needed).

  python scripts/simulate.py tracks/technical_s1/technical_s1_centerline.csv \
         results/lines/technical_s1/mintime_waypoints.csv --map tracks/technical_s1/technical_s1_map.yaml
  # what if the floor only has mu = 0.6 (saturating tyres)?
  python scripts/simulate.py ... --tire pacejka --mu-true 0.6
"""
import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from raceline.params import load_params  # noqa: E402
from raceline.track import Track  # noqa: E402
from raceline.pipeline import load_line  # noqa: E402
from sim.closed_loop import run_laps  # noqa: E402

if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('centerline_csv')
    ap.add_argument('waypoints_csv')
    ap.add_argument('--map', default=None, help='map YAML for wall collisions (else centre-line widths)')
    ap.add_argument('--laps', type=int, default=10)
    ap.add_argument('--tire', default='gym', choices=['gym', 'pacejka'])
    ap.add_argument('--mu-true', type=float, default=1.0)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--set', nargs='*', default=[])
    a = ap.parse_args()
    p = load_params(None, a.set)
    tr = Track.from_csv(a.centerline_csv)
    xy, v = load_line(a.waypoints_csv)
    r = run_laps(tr, xy, v, p, laps=a.laps, tire_model=a.tire, mu_true=a.mu_true, seed=a.seed, collision_map=a.map)
    print(f"laps completed {r['laps_completed']}/{a.laps}  crashed={r['crashed']}  "
          f"mean {r['mean_lap']:.3f} s  std {r['std_lap']:.3f} s  min wall clearance {r['min_clearance']:.3f} m")
    print('lap times:', ' '.join(f'{t:.3f}' for t in r['lap_times']))
