"""Procedural closed-track generator for F1TENTH-scale test tracks.

Two styles:

* 'primitives' (default): a random sequence of corners (turn angle + radius, incl. hairpins and
  chicanes) joined by straights. Straight lengths are solved by a small constrained least-squares
  problem so the loop closes exactly. Gives race-track-like layouts with real straights.
* 'blob': random star-shaped control polygon -> periodic spline (smooth, flowing layouts).

Every candidate is smoothed (corner entries get a short curvature ramp), checked for
self-intersection, minimum corner radius, and wall clearance between neighbouring sections,
then translated/rotated so it starts at (0, 0) heading +x on a straight -- like a slam_toolbox
map whose origin is where the car was switched on.
"""
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import lsq_linear
from scipy.spatial import cKDTree

from .geometry import (resample_closed, heading_curvature, arc_length, signed_area,
                       segments_intersect_count)
from .track import Track

PRESETS = {
    'oval':      dict(style='primitives', length=40.0, n_corners=2, width=1.6, hairpins=0, chicanes=0,
                      p_right=0.0, r_range=(3.0, 3.0), straight_range=(8.0, 8.0)),
    'flowing':   dict(style='blob', length=70.0, n_ctrl=8, irregularity=0.30, aspect=1.2, width=1.8),
    'technical': dict(style='primitives', length=60.0, n_corners=11, width=1.4, hairpins=1, chicanes=1,
                      p_right=0.35, r_range=(1.3, 2.6), straight_range=(0.8, 4.0)),
    'hairpins':  dict(style='primitives', length=70.0, n_corners=6, width=1.5, hairpins=2, chicanes=0,
                      p_right=0.2, r_range=(1.3, 3.0), straight_range=(4.0, 12.0)),
    'chicanes':  dict(style='primitives', length=65.0, n_corners=10, width=1.6, hairpins=0, chicanes=3,
                      p_right=0.1, r_range=(1.5, 3.0), straight_range=(2.0, 8.0)),
    'long':      dict(style='primitives', length=120.0, n_corners=12, width=2.0, hairpins=1, chicanes=1,
                      p_right=0.3, r_range=(1.5, 4.0), straight_range=(2.0, 14.0)),
    'narrow':    dict(style='primitives', length=50.0, n_corners=9, width=1.1, hairpins=1, chicanes=1,
                      p_right=0.3, r_range=(1.2, 2.5), straight_range=(1.0, 5.0)),
    'varwidth':  dict(style='primitives', length=70.0, n_corners=9, width=1.7, width_var=0.35, hairpins=1,
                      chicanes=1, p_right=0.3, r_range=(1.5, 3.0), straight_range=(1.5, 8.0)),
}


# ----------------------------------------------------------------------------- helpers
def _smooth_noise(n, rng, modes=4):
    s = np.linspace(0, 2 * np.pi, n, endpoint=False)
    f = np.zeros(n)
    for k in range(1, modes + 1):
        f += rng.normal() / k * np.cos(k * s + rng.uniform(0, 2 * np.pi))
    f -= f.mean()
    return f / max(np.max(np.abs(f)), 1e-9)


def _check_clearance(xy, s, L, d_req):
    """True if no two points that are far apart along the track come closer than d_req."""
    tree = cKDTree(xy)
    pairs = tree.query_pairs(d_req, output_type='ndarray')
    if len(pairs) == 0:
        return True
    ds = np.abs(s[pairs[:, 0]] - s[pairs[:, 1]])
    ds = np.minimum(ds, L - ds)
    return not np.any(ds > 0.5 * np.pi * d_req + 1e-6)


def _turn_sequence(rng, n_corners, hairpins, chicanes, p_right, r_range, r_min):
    """Random list of (angle, radius) with total turning +2*pi (counter-clockwise loop)."""
    if n_corners <= 2 and hairpins == 0 and chicanes == 0:   # oval
        r = rng.uniform(*r_range)
        return np.array([np.pi, np.pi]), np.array([r, r]), ['corner', 'corner']
    for _ in range(200):
        groups = []
        for _ in range(hairpins):
            groups.append([[np.deg2rad(rng.uniform(165, 185)), rng.uniform(r_min, r_min * 1.25), 'hairpin']])
        for _ in range(chicanes):  # S-bend: right then left (or reverse) back-to-back
            a = np.deg2rad(rng.uniform(35, 70))
            r = rng.uniform(r_range[0], r_range[1])
            sg = 1.0 if rng.random() < 0.5 else -1.0
            groups.append([[-sg * a, r, 'chicane_a'], [sg * a, r, 'chicane_b']])
        for _ in range(max(n_corners - hairpins - 2 * chicanes, 1)):
            sgn = -1.0 if rng.random() < p_right else 1.0
            groups.append([[sgn * np.deg2rad(rng.uniform(30, 130)), rng.uniform(*r_range), 'corner']])
        order = rng.permutation(len(groups))
        flat = [c for gi in order for c in groups[gi]]
        ang = np.array([c[0] for c in flat])
        free = np.array([c[2] == 'corner' and c[0] > 0 for c in flat])
        fixed_sum = np.sum(ang[~free])
        if np.sum(ang[free]) <= 0:
            continue
        scale = (2 * np.pi - fixed_sum) / np.sum(ang[free])
        if scale <= 0:
            continue
        ang[free] *= scale
        if np.any(np.abs(ang) > np.deg2rad(190)):
            continue
        rad = np.array([c[1] for c in flat])
        kinds = [c[2] for c in flat]
        return ang, rad, kinds
    raise RuntimeError('could not sample a turn sequence')


def _primitive_path(ang, rad, straights, ds):
    """Polyline from alternating straight / arc primitives starting at origin heading +x."""
    pts = [np.zeros(2)]
    pos = np.zeros(2)
    hd = 0.0
    for a, r, L in zip(ang, rad, straights):
        n = max(int(np.ceil(L / ds)), 1)
        for _ in range(n):
            pos = pos + (L / n) * np.array([np.cos(hd), np.sin(hd)])
            pts.append(pos.copy())
        arc = abs(a) * r
        n = max(int(np.ceil(arc / ds)), 2)
        sgn = np.sign(a)
        c = pos + sgn * r * np.array([-np.sin(hd), np.cos(hd)])
        for j in range(1, n + 1):
            h = hd + a * j / n
            pts.append(c - sgn * r * np.array([-np.sin(h), np.cos(h)]))
        pos = pts[-1].copy()
        hd = hd + a
    return np.array(pts[:-1])


def _close_straights(ang, rad, target, Lmin):
    """Straight lengths L >= Lmin (one before every corner) that close the loop, closest to target."""
    hd = np.concatenate([[0.0], np.cumsum(ang)[:-1]])
    U = np.stack([np.cos(hd), np.sin(hd)])            # straight k runs along heading hd[k]
    d = np.zeros(2)                                     # displacement contributed by the arcs
    for a, r, h in zip(ang, rad, hd):
        sgn = np.sign(a)
        d += sgn * r * (np.array([-np.sin(h), np.cos(h)]) - np.array([-np.sin(h + a), np.cos(h + a)]))
    # equality as a heavily weighted least-squares row (robust when U is rank deficient)
    w = 1e4
    A = np.vstack([w * U, np.eye(len(target))])
    b = np.concatenate([-w * d, target])
    res = lsq_linear(A, b, bounds=(Lmin, np.inf), lsmr_tol='auto', method='bvls')
    if np.max(np.abs(U @ res.x + d)) > 1e-3:
        return None
    return res.x


def _candidate_primitives(rng, length, n_corners, hairpins, chicanes, p_right, r_range,
                          straight_range, r_min, ds):
    ang, rad, kinds = _turn_sequence(rng, n_corners, hairpins, chicanes, p_right, r_range, r_min)
    target = rng.uniform(*straight_range, size=len(ang))
    for j, k in enumerate(kinds):          # short link inside a chicane
        if k == 'chicane_b':
            target[j] = rng.uniform(0.3, 1.0)
    arc_total = np.sum(np.abs(ang) * rad)
    budget = length - arc_total
    if budget > 0:
        target *= budget / max(np.sum(target), 1e-6)
    Lmin = 0.3
    L = _close_straights(ang, rad, target, Lmin)
    if L is None:
        return None
    return _primitive_path(ang, rad, L, ds / 4)


def _candidate_blob(rng, n_ctrl, irregularity, aspect):
    k = np.arange(n_ctrl)
    th = (k + rng.uniform(-0.35, 0.35, n_ctrl)) * 2 * np.pi / n_ctrl
    r = 1.0 + irregularity * rng.uniform(-1, 1, n_ctrl)
    ctrl = np.column_stack([aspect * r * np.cos(th), r * np.sin(th)])
    return resample_closed(ctrl, None, n=4000)


# ----------------------------------------------------------------------------- main API
def generate_track(style='primitives', length=60.0, width=1.6, width_var=0.0, r_min=None,
                   wall_gap=0.35, car_r_min=0.74, seed=0, ds=0.1, max_tries=400, name=None,
                   direction='ccw', n_corners=10, hairpins=1, chicanes=1, p_right=0.3,
                   r_range=(1.3, 3.0), straight_range=(1.0, 8.0),
                   n_ctrl=10, irregularity=0.35, aspect=1.2):
    """Generate a random valid closed track (see module doc). Lengths in metres.

    width / width_var : wall-to-wall width and +/- fractional variation along the lap
    r_min             : minimum centre-line corner radius; default max(1.5 * car turn radius,
                        (max width + wall_gap) / 2)
    car_r_min         : car's minimum turning radius = wheelbase / tan(max steer) (0.33/tan(0.4189))
    primitives style  : n_corners, hairpins, chicanes, p_right (share of right turns),
                        r_range (corner radii), straight_range (straight lengths)
    blob style        : n_ctrl, irregularity, aspect
    """
    rng = np.random.default_rng(seed)
    w_max = width * (1 + width_var)
    if r_min is None:
        r_min = max(1.5 * car_r_min, 0.5 * (w_max + wall_gap))
    r_range = (max(r_range[0], r_min), max(r_range[1], r_min * 1.05))
    d_req = w_max + wall_gap
    for attempt in range(max_tries):
        try:
            if style == 'blob':
                raw = _candidate_blob(rng, n_ctrl, irregularity, aspect)
            else:
                raw = _candidate_primitives(rng, length, n_corners, hairpins, chicanes, p_right,
                                            r_range, straight_range, r_min, ds)
            if raw is None:
                continue
            raw = resample_closed(raw, None, n=4000)
        except Exception:
            continue
        ok = False
        sig0 = 6 if style == 'primitives' else 0
        for sigma in (sig0, 15, 30, 50, 80):
            cur = raw if sigma == 0 else np.column_stack(
                [gaussian_filter1d(raw[:, 0], sigma, mode='wrap'),
                 gaussian_filter1d(raw[:, 1], sigma, mode='wrap')])
            _, L_ = arc_length(cur)
            cur = cur * (length / L_)
            try:
                cur = resample_closed(cur, ds)
            except Exception:
                break
            _, kap = heading_curvature(cur)
            if np.max(np.abs(kap)) <= 1.0 / r_min:
                ok = True
                break
        if not ok:
            continue
        s, L = arc_length(cur)
        if segments_intersect_count(cur[::3]) > 0:
            continue
        if not _check_clearance(cur, s, L, d_req):
            continue
        if (signed_area(cur) > 0) != (direction == 'ccw'):
            cur = cur[::-1]
        psi, kap = heading_curvature(cur)
        win = max(int(2.0 / ds), 1)
        kmax = np.max(np.abs(np.stack([np.roll(kap, j) for j in range(-win, win + 1)])), axis=0)
        i0 = int(np.argmin(kmax))
        cur = np.roll(cur, -i0, axis=0)
        psi = np.roll(psi, -i0)
        c, s0 = np.cos(-psi[0]), np.sin(-psi[0])
        R = np.array([[c, -s0], [s0, c]])
        cur = (cur - cur[0]) @ R.T
        n = len(cur)
        w = width * (1.0 + width_var * _smooth_noise(n, rng)) if width_var > 0 else np.full(n, width)
        tr = Track(cur, w / 2, w / 2, name=name or f'gen_{seed}')
        tr.gen_params = dict(style=style, length=length, width=width, width_var=width_var,
                             r_min=round(float(r_min), 3), seed=int(seed), attempts=attempt + 1)
        return tr
    raise RuntimeError('could not generate a valid track; relax r_min / corners / width')


def generate_preset(preset, seed=0, **overrides):
    p = dict(PRESETS[preset])
    p.update(overrides)
    return generate_track(seed=seed, name=f'{preset}_s{seed}', **p)


def random_params(rng):
    """Random generator settings for bulk testing (sweeps over track 'character')."""
    return dict(style='primitives', length=float(rng.uniform(35, 130)),
                n_corners=int(rng.integers(4, 14)), width=float(rng.uniform(1.2, 2.2)),
                width_var=float(rng.choice([0.0, 0.0, 0.25])), hairpins=int(rng.integers(0, 3)),
                chicanes=int(rng.integers(0, 3)), p_right=float(rng.uniform(0.0, 0.4)),
                r_range=(1.3, float(rng.uniform(2.0, 4.0))), straight_range=(1.0, float(rng.uniform(4, 14))))
