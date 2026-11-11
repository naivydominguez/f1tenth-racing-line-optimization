#!/usr/bin/env python3
"""Generate random F1TENTH test tracks (map PNG/YAML for the gym + centre-line CSV).

Examples
  python scripts/generate_tracks.py --preset technical --seed 3
  python scripts/generate_tracks.py --preset all --seed 1
  python scripts/generate_tracks.py --random 10 --seed 0               # 10 random layouts
  python scripts/generate_tracks.py --length 80 --corners 12 --hairpins 2 --chicanes 1 --width 1.5
  python scripts/generate_tracks.py --preset narrow --style-map slam    # grey-unknown SLAM look

Output: tracks/<name>/<name>_map.png, _map.yaml (origin = start pose (0,0,0), heading +x),
        <name>_centerline.csv (x_m, y_m, w_tr_right_m, w_tr_left_m), <name>_preview.png
Gym / f1tenth_gym_ros: point sim.yaml map_path at tracks/<name>/<name>_map (no extension),
sx = sy = stheta = 0.
"""
import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from raceline.track_gen import PRESETS, generate_preset, generate_track, random_params  # noqa: E402
from raceline.map_io import render_map  # noqa: E402


def preview(tr, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 6))
    for e in (tr.left_edge, tr.right_edge):
        ax.plot(*np.vstack([e, e[:1]]).T, 'k', lw=1)
    ax.plot(*tr.xy.T, '--', c='gray', lw=0.6)
    ax.annotate('', xy=tr.xy[8], xytext=tr.xy[0], arrowprops=dict(arrowstyle='->', color='g'))
    ax.set_aspect('equal')
    s = tr.summary()
    ax.set_title(f"{tr.name}: {s['length_m']} m, width {s['width_min_m']}-{s['width_max_m']} m, min R {s['min_radius_m']} m",
                 fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def save(tr, out, style):
    d = os.path.join(out, tr.name)
    os.makedirs(d, exist_ok=True)
    tr.to_csv(os.path.join(d, f'{tr.name}_centerline.csv'))
    render_map(tr, d, tr.name, style=style)
    preview(tr, os.path.join(d, f'{tr.name}_preview.png'))
    with open(os.path.join(d, f'{tr.name}_info.json'), 'w') as f:
        json.dump(dict(summary=tr.summary(), generator=tr.gen_params, start_pose=[0.0, 0.0, 0.0]), f, indent=1,
                  default=float)
    print(f'{tr.name:22s} {tr.summary()}  -> {d}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--preset', default=None, help='one of %s or "all"' % ', '.join(PRESETS))
    ap.add_argument('--random', type=int, default=0, help='number of random layouts')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', default=os.path.join(ROOT, 'tracks'))
    ap.add_argument('--style-map', default='gym', choices=['gym', 'slam'])
    ap.add_argument('--name', default=None)
    g = ap.add_argument_group('custom track (used when no --preset/--random)')
    g.add_argument('--style', default='primitives', choices=['primitives', 'blob'])
    g.add_argument('--length', type=float, default=60.0)
    g.add_argument('--width', type=float, default=1.6)
    g.add_argument('--width-var', type=float, default=0.0)
    g.add_argument('--corners', type=int, default=10)
    g.add_argument('--hairpins', type=int, default=1)
    g.add_argument('--chicanes', type=int, default=1)
    g.add_argument('--p-right', type=float, default=0.3)
    g.add_argument('--r-min', type=float, default=None, help='min corner radius [m]')
    g.add_argument('--straight-max', type=float, default=8.0)
    a = ap.parse_args()

    if a.preset:
        names = list(PRESETS) if a.preset == 'all' else [a.preset]
        for n in names:
            save(generate_preset(n, seed=a.seed), a.out, a.style_map)
    elif a.random:
        rng = np.random.default_rng(a.seed)
        for i in range(a.random):
            gp = random_params(rng)
            save(generate_track(seed=a.seed * 1000 + i, name=f'random_{a.seed}_{i}', **gp), a.out, a.style_map)
    else:
        tr = generate_track(style=a.style, length=a.length, width=a.width, width_var=a.width_var,
                            n_corners=a.corners, hairpins=a.hairpins, chicanes=a.chicanes, p_right=a.p_right,
                            r_min=a.r_min, straight_range=(1.0, a.straight_max), seed=a.seed,
                            name=a.name or f'custom_s{a.seed}')
        save(tr, a.out, a.style_map)
