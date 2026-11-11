#!/usr/bin/env python3
"""SLAM / gym map -> centre line + widths CSV (x_m, y_m, w_tr_right_m, w_tr_left_m).

  python scripts/extract_centerline.py tracks/siebel2/siebel2.yaml --start 0 0 --yaw 3.10 \
         --max-half-width 1.5 --smoothing 0.1 --out tracks/siebel2/siebel2_centerline.csv

--start : a point on the loop where the lap should start (slam_toolbox: (0,0) = where the car was)
--yaw   : driving direction at the start [rad] (otherwise counter-clockwise)
"""
import argparse
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from raceline.map_extract import extract_centerline  # noqa: E402

if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('map_yaml')
    ap.add_argument('--start', nargs=2, type=float, default=[0.0, 0.0])
    ap.add_argument('--yaw', type=float, default=None)
    ap.add_argument('--direction', default='ccw', choices=['ccw', 'cw'])
    ap.add_argument('--ds', type=float, default=0.1)
    ap.add_argument('--smoothing', type=float, default=0.05, help='RMS smoothing of the centre line [m]')
    ap.add_argument('--max-half-width', type=float, default=2.0)
    ap.add_argument('--open-radius', type=float, default=0.15)
    ap.add_argument('--hole-area-max', type=float, default=1.0)
    ap.add_argument('--clearance-ref', type=float, default=0.3)
    ap.add_argument('--out', default=None)
    ap.add_argument('--plot', default=None)
    a = ap.parse_args()
    tr, dbg = extract_centerline(a.map_yaml, start_xy=tuple(a.start), start_yaw=a.yaw, ds=a.ds,
                                 smoothing=a.smoothing, max_half_width=a.max_half_width,
                                 open_radius=a.open_radius, hole_area_max=a.hole_area_max,
                                 clearance_ref=a.clearance_ref, direction=a.direction, return_debug=True,
                                 name=os.path.basename(a.map_yaml).split('.')[0])
    out = a.out or a.map_yaml.rsplit('.', 1)[0] + '_centerline.csv'
    tr.to_csv(out)
    print(tr.summary(), '->', out)
    if a.plot:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        meta = dbg['meta']
        img = meta['img']
        H, W = img.shape
        r = meta['resolution']
        ox, oy = meta['origin'][:2]
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(img, cmap='gray', extent=[ox, ox + W * r, oy, oy + H * r])
        ax.plot(*np.vstack([tr.xy, tr.xy[:1]]).T, 'r', lw=1)
        for e in (tr.left_edge, tr.right_edge):
            ax.plot(*np.vstack([e, e[:1]]).T, 'c', lw=0.7)
        ax.plot(*tr.xy[0], 'go')
        fig.savefig(a.plot, dpi=120)
