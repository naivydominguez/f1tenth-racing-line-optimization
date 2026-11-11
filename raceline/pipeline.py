"""High-level helpers: plan centre-line / min-curvature / min-time lines and export them."""
import json
import os
import time
import numpy as np

from .geometry import resample_closed, arc_length, heading_curvature
from .speed_profile import speed_profile
from .min_curv import optimize_mincurv
from .min_time import optimize_mintime_robust

METHODS = ('centerline', 'mincurv', 'mintime')
LABELS = {'centerline': 'Centre line', 'mincurv': 'Min curvature', 'mintime': 'Min time (CasADi)'}


def _resample_line(xy, v=None, ds=0.1):
    """Uniform 0.1 m line; if v given (defined on the original nodes) it is interpolated by arc length."""
    s_o, L_o = arc_length(xy)
    new = resample_closed(xy, ds)
    if v is None:
        return new, None
    s_n, L_n = arc_length(new)
    vv = np.interp(s_n * L_o / L_n, np.r_[s_o, L_o], np.r_[v, v[0]])
    return new, vv


def plan_lines(track, p, methods=METHODS, verbose=False):
    """Returns {method: dict(xy, v, ax, kappa, lap_time, solve_time, ...)} on a 0.1 m grid."""
    out = {}
    ref = track.resampled(ds=p.track.ds, smoothing=p.track.ref_smoothing)
    if 'centerline' in methods:
        sp = speed_profile(ref.xy, p)
        out['centerline'] = dict(xy=ref.xy, v=sp['v'], ax=sp['ax'], kappa=sp['kappa'],
                                 lap_time=sp['lap_time'], solve_time=0.0, status='ok')
    mc = None
    if 'mincurv' in methods or 'mintime' in methods:
        t0 = time.time()
        mc = optimize_mincurv(track, p, ds=p.mintime.ds if 'mintime' in methods else p.mincurv.ds)
        xy = resample_closed(mc['xy'], p.track.ds)
        sp = speed_profile(xy, p)
        out['mincurv'] = dict(xy=xy, v=sp['v'], ax=sp['ax'], kappa=sp['kappa'], lap_time=sp['lap_time'],
                              solve_time=time.time() - t0, status=mc['status'],
                              n_offset=mc['alpha'], ref=mc['ref'])
    if 'mintime' in methods:
        res, attempts = optimize_mintime_robust(track, p, warm=mc, verbose=verbose)
        if res is None:   # fallback (proposal): use the min-curvature line
            d = dict(out['mincurv'])
            d.update(status='FALLBACK_mincurv', fallback=True, attempts=attempts)
            out['mintime'] = d
        else:
            xy, v = _resample_line(res['xy'], res['v'], p.track.ds)
            _, ax = _resample_line(res['xy'], res['ax'], p.track.ds)
            _, kap = heading_curvature(xy)
            out['mintime'] = dict(xy=xy, v=v, ax=ax, kappa=kap, lap_time=res['lap_time'],
                                  solve_time=res['solve_time'], status=res['status'], fallback=False,
                                  iterations=res['iterations'], ocp=res, attempts=attempts)
    if 'mincurv' not in methods and 'mincurv' in out:
        out.pop('mincurv')
    return out


def export_line(line, out_dir, stem):
    """Writes <stem>_waypoints.csv with header x,y,v,yaw,kappa,s,ax (x,y first so our Lab-5
    pure pursuit loader reads it unchanged; v is the planned speed)."""
    os.makedirs(out_dir, exist_ok=True)
    xy, v = line['xy'], line['v']
    psi, kap = heading_curvature(xy)
    s, _ = arc_length(xy)
    ax = line.get('ax', np.zeros(len(xy)))
    path = os.path.join(out_dir, f'{stem}_waypoints.csv')
    np.savetxt(path, np.column_stack([xy, v, psi, kap, s, ax]), delimiter=',', fmt='%.5f',
               header='x,y,v,yaw,kappa,s,ax', comments='')
    return path


def load_line(path):
    data = np.genfromtxt(path, delimiter=',', names=True)
    xy = np.column_stack([data['x'], data['y']])
    v = data['v'] if 'v' in data.dtype.names else None
    return xy, v


def summary_json(lines, path=None, extra=None):
    d = {m: dict(lap_time_plan_s=round(float(l['lap_time']), 4), solve_time_s=round(float(l['solve_time']), 3),
                 status=str(l['status']), length_m=round(float(arc_length(l['xy'])[1]), 3),
                 v_max_mps=round(float(np.max(l['v'])), 3), v_min_mps=round(float(np.min(l['v'])), 3))
         for m, l in lines.items()}
    if extra:
        d.update(extra)
    if path:
        with open(path, 'w') as f:
            json.dump(d, f, indent=2)
    return d
