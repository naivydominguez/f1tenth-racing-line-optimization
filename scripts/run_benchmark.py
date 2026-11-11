#!/usr/bin/env python3
"""Full benchmark used for the standup slides. Writes results/benchmark.json and results/*.npz.

Sections (run all by default, or pick with --sections A B ...):
  A  generated tracks, end-to-end: generate -> render map -> extract centre line FROM THE MAP ->
     plan 3 lines -> 10 closed-loop laps each (f1tenth_gym dynamics) checked against the true walls
  B  f1tenth_racetracks maps (Spielberg, Oschersleben) with their centre-line files
  C  our real Siebel SLAM map (siebel2) with car-like limits (mu assumed until the circle test)
  D  20 random tracks: planned lap-time gain distribution + OCP convergence statistics
  E  friction mismatch: saturating-tyre sim with mu_true = 0.7, planner mu swept
  F  wall-margin sweep under localisation noise
  G  circle-test mu calibration on the saturating-tyre sim
  H  grid: 22 random tracks end-to-end, 10 closed-loop laps per line (used for the track grid figure)
"""
import argparse
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'tools'))

from raceline.params import load_params, with_overrides  # noqa: E402
from raceline.track import Track  # noqa: E402
from raceline.track_gen import generate_preset, generate_track, random_params  # noqa: E402
from raceline.map_io import render_map  # noqa: E402
from raceline.map_extract import extract_centerline  # noqa: E402
from raceline.pipeline import plan_lines, export_line, METHODS  # noqa: E402
from sim.closed_loop import run_laps  # noqa: E402

RES = os.path.join(ROOT, 'results')
TRACKS = os.path.join(ROOT, 'tracks')
EXT = os.environ.get('F1TENTH_RACETRACKS', os.path.join(os.path.dirname(ROOT), 'f1tenth_racetracks'))
SIEBEL = os.path.join(TRACKS, 'siebel2', 'siebel2.yaml')
GEN_PRESETS = ['technical', 'hairpins', 'chicanes', 'long', 'narrow', 'varwidth']


def log(*a):
    print(time.strftime('%H:%M:%S'), *a, flush=True)


def _eval_lines(track_eval, lines, p, laps, seed=0, tire_model=None, mu_true=None, cmap=None):
    from sim.closed_loop import GridCollision
    gc = GridCollision(cmap) if cmap else None
    out = {}
    for m, l in lines.items():
        r = run_laps(track_eval, l['xy'], l['v'], p, laps=laps, seed=seed, tire_model=tire_model, mu_true=mu_true,
                     collision_map=gc)
        out[m] = dict(plan=float(l['lap_time']), sim_mean=r['mean_lap'], sim_std=r['std_lap'],
                      laps=r['laps_completed'], crashed=r['crashed'], min_clear=r['min_clearance'],
                      status=str(l['status']), solve_time=float(l['solve_time']))
    return out


def _save_lines(path, track, lines, extra=None):
    d = {'track_xy': track.xy, 'track_wl': track.w_left, 'track_wr': track.w_right}
    for m, l in lines.items():
        d[f'{m}_xy'] = l['xy']
        d[f'{m}_v'] = l['v']
    if extra:
        d.update(extra)
    np.savez_compressed(path, **d)


# ----------------------------------------------------------------------------------- A
def job_generated(name, laps):
    p = load_params()
    tr = generate_preset(name, seed=1)
    out_dir = os.path.join(TRACKS, tr.name)
    os.makedirs(out_dir, exist_ok=True)
    tr.to_csv(os.path.join(out_dir, f'{tr.name}_centerline_groundtruth.csv'))
    y = render_map(tr, out_dir, tr.name, style='gym')
    render_map(tr, out_dir, tr.name + '_slamstyle', style='slam')
    t0 = time.time()
    ex = extract_centerline(y, start_xy=(0, 0), start_yaw=0.0, name=tr.name)
    t_ex = time.time() - t0
    ex.to_csv(os.path.join(out_dir, f'{tr.name}_centerline.csv'))
    s, n, _ = tr.project(ex.xy)
    lines = plan_lines(ex, p)
    for m, l in lines.items():
        export_line(l, os.path.join(RES, 'lines', tr.name), m)
    ev = _eval_lines(tr, lines, p, laps, cmap=y)          # walls = the rendered map (as in the gym)
    _save_lines(os.path.join(RES, f'lines_{tr.name}.npz'), tr, lines,
                extra={'extracted_xy': ex.xy, 'extracted_wl': ex.w_left, 'extracted_wr': ex.w_right})
    return tr.name, dict(track=tr.summary(), extraction=dict(center_err_mean_cm=float(100 * np.mean(np.abs(n))),
                                                            center_err_max_cm=float(100 * np.max(np.abs(n))),
                                                            time_s=t_ex),
                         results=ev, gen=tr.gen_params)


# ----------------------------------------------------------------------------------- B
def job_racetrack(name, laps):
    p = load_params(overrides={'mintime.ds': 0.5, 'mincurv.ds': 0.5, 'track.ref_smoothing': 0.1})
    from raceline.map_extract import widths_from_map
    tr_csv = Track.from_csv(os.path.join(EXT, name, f'{name}_centerline.csv'), name=name)
    tr = widths_from_map(tr_csv, os.path.join(EXT, name, f'{name}_map.yaml'))   # CSV widths overstate the map
    lines = plan_lines(tr, p)
    for m, l in lines.items():
        export_line(l, os.path.join(RES, 'lines', name), m)
    ev = _eval_lines(tr, lines, p, laps, cmap=os.path.join(EXT, name, f'{name}_map.yaml'))
    _save_lines(os.path.join(RES, f'lines_{name}.npz'), tr, lines)
    return name, dict(track=tr.summary(), results=ev,
                      csv_vs_map_width_m=float(np.mean(tr_csv.w_left + tr_csv.w_right - tr.w_left - tr.w_right)))


# ----------------------------------------------------------------------------------- C
CAR_LIKE = {'limits.delta_max': 0.36, 'limits.v_max': 5.0, 'limits.mu': 0.7, 'limits.friction_usage': 0.8,
            'limits.a_brake_max': 4.0, 'limits.a_accel_max': 4.0, 'track.wall_margin': 0.25,
            'track.ref_smoothing': 0.1}


def job_siebel(laps):
    p = load_params(overrides=CAR_LIKE)
    wp = np.loadtxt(os.path.join(TRACKS, 'siebel2', 'siebel_waypoints_lab5.csv'), delimiter=',', skiprows=1)
    tr = extract_centerline(SIEBEL, start_xy=(0, 0), start_yaw=float(wp[0, 2]), max_half_width=1.5,
                            smoothing=0.1, name='siebel2')
    tr.to_csv(os.path.join(TRACKS, 'siebel2', 'siebel2_centerline.csv'))
    lines = plan_lines(tr, p)
    for m, l in lines.items():
        export_line(l, os.path.join(RES, 'lines', 'siebel2'), m)
    ev = _eval_lines(tr, lines, p, laps, cmap=SIEBEL)
    _save_lines(os.path.join(RES, 'lines_siebel2.npz'), tr, lines)
    s, n, _ = tr.project(wp[:, :2])
    return 'siebel2', dict(track=tr.summary(), results=ev, params=CAR_LIKE,
                           lab5_waypoints_vs_centerline_cm=float(100 * np.mean(np.abs(n))))


# ----------------------------------------------------------------------------------- D
def job_random(seed):
    p = load_params()
    rng = np.random.default_rng(1000 + seed)
    gp = random_params(rng)
    try:
        tr = generate_track(seed=1000 + seed, name=f'random_{seed}', **gp)
    except RuntimeError:
        return seed, None
    t0 = time.time()
    lines = plan_lines(tr, p)
    mt = lines['mintime']
    return seed, dict(track=tr.summary(), gen=gp,
                      plan={m: float(l['lap_time']) for m, l in lines.items()},
                      mintime_status=str(mt['status']), mintime_attempts=len(mt.get('attempts', [])),
                      mintime_solve_s=float(mt['solve_time']), total_s=time.time() - t0)


# ----------------------------------------------------------------------------------- H
def job_grid(seed, laps=10):
    """Random track, end-to-end like A (map -> extracted centre line -> 3 lines -> closed loop)."""
    p = load_params()
    rng = np.random.default_rng(2000 + seed)
    gp = random_params(rng)
    try:
        tr = generate_track(seed=2000 + seed, name=f'grid_{seed:02d}', **gp)
    except RuntimeError:
        return seed, None
    out_dir = os.path.join(TRACKS, 'grid', tr.name)
    os.makedirs(out_dir, exist_ok=True)
    tr.to_csv(os.path.join(out_dir, f'{tr.name}_centerline_groundtruth.csv'))
    y = render_map(tr, out_dir, tr.name, style='gym')
    ex = extract_centerline(y, start_xy=(0, 0), start_yaw=0.0, name=tr.name)
    ex.to_csv(os.path.join(out_dir, f'{tr.name}_centerline.csv'))
    lines = plan_lines(ex, p)
    for m, l in lines.items():
        export_line(l, os.path.join(RES, 'lines', 'grid', tr.name), m)
    ev = _eval_lines(tr, lines, p, laps, cmap=y)
    _save_lines(os.path.join(RES, 'grid', f'lines_{tr.name}.npz'), tr, lines)
    return seed, dict(track=tr.summary(), gen=gp, results=ev)


# ----------------------------------------------------------------------------------- E
def job_mu(args):
    name, mu_plan, laps = args
    p = load_params(overrides={'limits.mu': mu_plan, 'limits.friction_usage': 1.0})
    tr = Track.from_csv(os.path.join(TRACKS, f'{name}_s1', f'{name}_s1_centerline.csv'), name=name)
    gt = Track.from_csv(os.path.join(TRACKS, f'{name}_s1', f'{name}_s1_centerline_groundtruth.csv'))
    lines = plan_lines(tr, p, methods=('mincurv', 'mintime'))
    ev = _eval_lines(gt, lines, p, laps, tire_model='pacejka', mu_true=0.7,
                     cmap=os.path.join(TRACKS, f'{name}_s1', f'{name}_s1_map.yaml'))
    return (name, mu_plan), ev


# ----------------------------------------------------------------------------------- F
def job_margin(args):
    margin, noise, seed, laps = args
    p = load_params(overrides={'track.wall_margin': margin, 'sim.pose_noise_xy': noise,
                               'sim.pose_noise_yaw': noise / 2})
    name = 'technical'
    tr = Track.from_csv(os.path.join(TRACKS, f'{name}_s1', f'{name}_s1_centerline.csv'), name=name)
    gt = Track.from_csv(os.path.join(TRACKS, f'{name}_s1', f'{name}_s1_centerline_groundtruth.csv'))
    lines = plan_lines(tr, p, methods=('mintime',))
    ev = _eval_lines(gt, lines, p, laps, seed=seed, cmap=os.path.join(TRACKS, f'{name}_s1', f'{name}_s1_map.yaml'))
    return (margin, noise, seed), ev['mintime']


# ----------------------------------------------------------------------------------- G
def job_circle(mu_true):
    from sim_circle_test import simulate
    from raceline.calibration import estimate_mu
    out = []
    for steer in (0.2, 0.3):
        for seed in range(3):
            d = simulate(mu_true, steer, seed=seed)
            r = estimate_mu(d[:, 0], d[:, 1], d[:, 2], d[:, 3])
            out.append(dict(steer=steer, seed=seed, mu_est=r['mu_est'], v_peak=r['v_at_peak']))
            if steer == 0.3 and seed == 0:
                np.savez_compressed(os.path.join(RES, f'circle_mu{mu_true}.npz'), t=d[:, 0], v=r['v'], ay=r['ay'])
    return mu_true, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sections', nargs='*', default=list('ABCDEFG'))
    ap.add_argument('--laps', type=int, default=10)
    ap.add_argument('--procs', type=int, default=2)
    a = ap.parse_args()
    os.makedirs(RES, exist_ok=True)
    jpath = os.path.join(RES, 'benchmark.json')
    bench = json.load(open(jpath)) if os.path.exists(jpath) else {}
    pool = Pool(a.procs)

    def save():
        with open(jpath, 'w') as f:
            json.dump(bench, f, indent=1, default=float)

    if 'A' in a.sections:
        log('A: generated tracks (end-to-end)')
        bench['A'] = dict(pool.starmap(job_generated, [(n, a.laps) for n in GEN_PRESETS]))
        save()
    if 'B' in a.sections:
        log('B: f1tenth_racetracks')
        bench['B'] = dict(pool.starmap(job_racetrack, [(n, a.laps) for n in ('Spielberg', 'Oschersleben')]))
        save()
    if 'C' in a.sections:
        log('C: Siebel SLAM map')
        bench['C'] = dict([job_siebel(a.laps)])
        save()
    if 'D' in a.sections:
        log('D: random tracks')
        bench['D'] = {str(k): v for k, v in pool.map(job_random, range(20)) if v is not None}
        save()
    if 'E' in a.sections:
        log('E: friction mismatch')
        jobs = [(n, mu, 3) for n in ('technical', 'chicanes', 'hairpins') for mu in (0.5, 0.6, 0.7, 0.8, 0.9, 1.0)]
        bench['E'] = {f'{k[0]}|{k[1]}': v for k, v in pool.map(job_mu, jobs)}
        save()
    if 'F' in a.sections:
        log('F: wall margin sweep')
        jobs = [(m, nz, sd, 3) for m in (0.05, 0.1, 0.15, 0.2, 0.3) for nz in (0.02, 0.05) for sd in range(3)]
        bench['F'] = {f'{k[0]}|{k[1]}|{k[2]}': v for k, v in pool.map(job_margin, jobs)}
        save()
    if 'H' in a.sections:
        log('H: grid of random tracks')
        os.makedirs(os.path.join(RES, 'grid'), exist_ok=True)
        bench['H'] = {str(k): v for k, v in pool.map(job_grid, range(22)) if v is not None}
        save()
    if 'G' in a.sections:
        log('G: circle-test calibration')
        bench['G'] = {str(k): v for k, v in pool.map(job_circle, (0.5, 0.7, 0.9))}
        save()
    pool.close()
    log('done ->', jpath)


if __name__ == '__main__':
    main()
