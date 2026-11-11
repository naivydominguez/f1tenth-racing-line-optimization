"""End-to-end smoke test: generate -> render map -> extract -> plan -> 2 closed-loop laps.
    python tests/test_smoke.py
"""
import os, sys, tempfile, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from raceline.params import load_params
from raceline.track_gen import generate_preset
from raceline.map_io import render_map
from raceline.map_extract import extract_centerline
from raceline.pipeline import plan_lines, export_line, load_line
from sim.closed_loop import run_laps

t0 = time.time()
p = load_params()
tr = generate_preset('oval', seed=2)
with tempfile.TemporaryDirectory() as d:
    y = render_map(tr, d, 'oval')
    ex = extract_centerline(y, start_xy=(0, 0), start_yaw=0.0)
    s, n, _ = tr.project(ex.xy)
    assert np.mean(np.abs(n)) < 0.08, 'centre-line extraction drifted'
    lines = plan_lines(ex, p)
    assert lines['mintime']['status'] == 'Solve_Succeeded', lines['mintime']['status']
    assert lines['mintime']['lap_time'] < lines['centerline']['lap_time']
    path = export_line(lines['mintime'], d, 'mintime')
    xy, v = load_line(path)
    r = run_laps(tr, xy, v, p, laps=2, collision_map=y)
    assert r['laps_completed'] == 2 and not r['crashed'], r
print(f"PASS  centre {lines['centerline']['lap_time']:.2f}s  min-curv {lines['mincurv']['lap_time']:.2f}s  "
      f"min-time {lines['mintime']['lap_time']:.2f}s  sim {r['mean_lap']:.2f}s  ({time.time() - t0:.0f}s)")
