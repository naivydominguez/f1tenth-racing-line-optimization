#!/usr/bin/env python3
"""Figures for the standup slides (dark theme to match our deck). Reads results/benchmark.json + npz."""
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from raceline.map_io import read_map  # noqa: E402

RES = os.path.join(ROOT, 'results')
OUT = os.path.join(RES, 'figures')
os.makedirs(OUT, exist_ok=True)

BG = '#1b1b1b'
INK = '#ecebe6'
INK2 = '#b9b8b0'
MUTED = '#8c8b85'
GRID = '#333331'
WALL = '#d8d7d0'
C_CENTRE = '#9a9a94'      # baseline: neutral grey
C_MINCURV = '#3987e5'     # categorical slot 1 (dark)
C_MINTIME = '#199e70'     # categorical slot 3 (dark) - validated pair, fits the deck's green
BLUE_RAMP = ['#104281', '#184f95', '#1c5cab', '#256abf', '#2a78d6', '#3987e5', '#5598e7', '#6da7ec',
             '#86b6ef', '#9ec5f4', '#b7d3f6', '#cde2fb']
SPEED_CMAP = LinearSegmentedColormap.from_list('speed', BLUE_RAMP)
GRID_WALL = '#9a9993'
GRID_LINE = '#58b956'
SUFFIX = ''

plt.rcParams.update({
    'figure.facecolor': BG, 'axes.facecolor': BG, 'savefig.facecolor': BG,
    'text.color': INK, 'axes.labelcolor': INK2, 'xtick.color': INK2, 'ytick.color': INK2,
    'axes.edgecolor': GRID, 'font.size': 13, 'axes.titlesize': 15, 'axes.titleweight': 'normal',
    'font.family': ['Red Hat Display', 'DejaVu Sans'], 'axes.spines.top': False, 'axes.spines.right': False,
})

B = json.load(open(os.path.join(RES, 'benchmark.json')))
NICE = {'technical_s1': 'Technical', 'hairpins_s1': 'Hairpins', 'chicanes_s1': 'Chicanes', 'long_s1': 'Long',
        'narrow_s1': 'Narrow', 'varwidth_s1': 'Var. width', 'Spielberg': 'Spielberg (F1/10)', 'Oschersleben': 'Oschersleben (F1/10)',
        'siebel2': 'Siebel (our SLAM map)'}
NICE_GEN = {k: v + ' (gen.)' for k, v in NICE.items() if k.endswith('_s1')}


def _closed(a):
    return np.vstack([a, a[:1]])


def _edges(d):
    from raceline.track import Track
    t = Track(d['track_xy'], d['track_wr'], d['track_wl'])
    return t.left_edge, t.right_edge


def speed_line(ax, xy, v, vmin, vmax, lw=3.2):
    pts = _closed(xy)
    seg = np.stack([pts[:-1], pts[1:]], axis=1)
    lc = LineCollection(seg, cmap=SPEED_CMAP, norm=plt.Normalize(vmin, vmax), linewidths=lw, capstyle='round')
    lc.set_array(v)
    ax.add_collection(lc)
    return lc


def fig_hero(name='chicanes_s1', fname='fig_hero.png'):
    d = np.load(os.path.join(RES, f'lines_{name}.npz'))
    r = B['A'][name]['results'] if name in B.get('A', {}) else None
    fig, ax = plt.subplots(figsize=(9.6, 7.0))
    le, re = _edges(d)
    for e in (le, re):
        ax.plot(*_closed(e).T, c=WALL, lw=1.6, solid_joinstyle='round', solid_capstyle='round')
    ax.plot(*_closed(d['centerline_xy']).T, c=C_CENTRE, lw=1.2, ls=(0, (4, 4)), label='centre line (baseline)')
    v = d['mintime_v']
    lc = speed_line(ax, d['mintime_xy'], v, v.min(), v.max())
    ax.plot(*d['mintime_xy'][0], 'o', ms=9, c=INK, mec=BG, mew=2)
    ax.set_aspect('equal')
    ax.axis('off')
    cb = fig.colorbar(lc, ax=ax, shrink=0.55, pad=0.01)
    cb.set_label('planned speed [m/s]', color=INK2)
    cb.outline.set_visible(False)
    cb.ax.tick_params(colors=INK2)
    ax.legend(loc='upper left', frameon=False, fontsize=12, labelcolor=INK2, bbox_to_anchor=(0.0, 1.04))
    fig.savefig(os.path.join(OUT, fname), dpi=200, bbox_inches='tight', pad_inches=0.15)
    plt.close(fig)


def fig_gallery(names=('technical_s1', 'hairpins_s1', 'chicanes_s1', 'long_s1', 'narrow_s1', 'varwidth_s1')):
    fig, axes = plt.subplots(2, 3, figsize=(10.5, 6.2))
    for ax, n in zip(axes.flat, names):
        d = np.load(os.path.join(RES, f'lines_{n}.npz'))
        le, re = _edges(d)
        for e in (le, re):
            ax.plot(*_closed(e).T, c=WALL, lw=1.0)
        ax.plot(*_closed(d['mintime_xy']).T, c=C_MINTIME, lw=1.6)
        s = B['A'][n]['track']
        ax.set_title(f"{NICE[n]}  ·  {s['length_m']:.0f} m  ·  {s['width_min_m']:.1f}"
                     + (f"-{s['width_max_m']:.1f}" if s['width_max_m'] != s['width_min_m'] else '') + ' m wide',
                     fontsize=11, color=INK2)
        ax.set_aspect('equal')
        ax.axis('off')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'fig_gallery.png'), dpi=200)
    plt.close(fig)


def fig_gain():
    rows = []
    for sec in ('A', 'B', 'C'):
        for n, v in B.get(sec, {}).items():
            r = v['results']
            c = r['centerline']['sim_mean']
            rows.append((NICE_GEN.get(n, NICE.get(n, n)), 100 * (1 - r['mincurv']['sim_mean'] / c), 100 * (1 - r['mintime']['sim_mean'] / c),
                         r['mincurv']['crashed'] or r['mintime']['crashed'] or r['centerline']['crashed'],
                         r['mintime']['sim_mean'], c))
    rows = rows[::-1]
    y = np.arange(len(rows))
    h = 0.36
    fig, ax = plt.subplots(figsize=(8.2, 6.0))
    ax.barh(y + h / 2 + 0.02, [r[1] for r in rows], height=h, color=C_MINCURV, label='Min curvature', zorder=3)
    ax.barh(y - h / 2 - 0.02, [r[2] for r in rows], height=h, color=C_MINTIME, label='Min time (CasADi)', zorder=3)
    for yi, r in zip(y, rows):
        ax.text(max(r[1], r[2]) + 0.4, yi - h / 2, f'{r[2]:.1f}%', va='center', fontsize=15, color=INK)
    ax.axvline(10, color=MUTED, lw=1.2, ls=(0, (3, 3)), zorder=2)
    ax.text(10.2, -0.78, 'proposal target 10%', color=MUTED, fontsize=14)
    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows], fontsize=16)
    ax.tick_params(axis='x', labelsize=15)
    ax.xaxis.label.set_size(16)
    ax.set_xlabel('lap time saved vs centre line [%]')
    ax.set_ylim(-0.95, len(rows) - 0.4)
    ax.grid(axis='x', color=GRID, lw=0.8, zorder=0)
    ax.tick_params(axis='y', length=0)
    ax.set_xlim(0, max(max(r[1], r[2]) for r in rows) * 1.15)
    ax.legend(loc='lower left', bbox_to_anchor=(-0.02, 1.0), ncol=2, frameon=False, labelcolor=INK2, fontsize=16)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'fig_gain.png'), dpi=200)
    plt.close(fig)
    return rows


def fig_mu():
    E = B.get('E', {})
    if not E:
        return
    tracks = sorted({k.split('|')[0] for k in E})
    mus = sorted({float(k.split('|')[1]) for k in E})
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for meth, col, off in (('mincurv', C_MINCURV, -0.008), ('mintime', C_MINTIME, 0.008)):
        ok_x, ok_y, bad_x = [], [], []
        for t in tracks:
            base = E[f'{t}|0.7'][meth]['sim_mean']
            for mu in mus:
                r = E[f'{t}|{mu}'][meth]
                if r['crashed'] or not np.isfinite(r['sim_mean']):
                    bad_x.append(mu + off)
                else:
                    ok_x.append(mu + off)
                    ok_y.append(100 * (r['sim_mean'] / base - 1))
        ax.plot(ok_x, ok_y, 'o', ms=8, c=col, mec=BG, mew=1.5,
                label=('Min curvature' if meth == 'mincurv' else 'Min time') + ': completed laps')
        ax.plot(bad_x, [14] * len(bad_x), 'x', ms=10, mew=2.5, c=col)
    ax.text(mus[-1] + 0.03, 14, 'slid into\nthe wall', va='center', fontsize=10, color=INK2)
    ax.axvline(0.7, color=MUTED, lw=1.2, ls=(0, (3, 3)))
    ax.text(0.705, -9.5, 'true floor μ = 0.7', color=MUTED, fontsize=10)
    ax.set_xlabel('μ assumed by the planner (friction usage 100%)')
    ax.set_ylabel('lap time vs planning with true μ [%]')
    ax.set_xlim(mus[0] - 0.05, mus[-1] + 0.15)
    ax.grid(color=GRID, lw=0.8)
    ax.legend(loc='upper left', frameon=False, fontsize=10, labelcolor=INK2)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'fig_mu.png'), dpi=200)
    plt.close(fig)


def fig_margin():
    F = B.get('F', {})
    if not F:
        return
    agg = {}
    for k, v in F.items():
        m, nz, sd = k.split('|')
        agg.setdefault((float(m), float(nz)), []).append(v)
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for nz, col, lab in ((0.02, C_MINCURV, 'pose noise 2 cm'), (0.05, C_MINTIME, 'pose noise 5 cm')):
        ms = sorted(m for (m, n) in agg if n == nz)
        plan = [agg[(m, nz)][0]['plan'] for m in ms]
        crash = [sum(x['crashed'] for x in agg[(m, nz)]) / len(agg[(m, nz)]) for m in ms]
        ax.plot(ms, [100 * c for c in crash], '-o', c=col, lw=2, ms=8, mec=BG, mew=1.5, label=lab)
    ax2_vals = sorted({m for (m, n) in agg})
    ax.set_xlabel('wall margin beyond half car width [m]')
    ax.set_ylabel('runs that hit a wall [%]')
    ax.set_xticks(ax2_vals)
    ax.grid(color=GRID, lw=0.8)
    ax.legend(frameon=False, labelcolor=INK2)
    plans = {m: agg[(m, 0.02)][0]['plan'] for m in ax2_vals}
    for m in ax2_vals:
        ax.text(m, 104, f'{plans[m]:.2f}s', ha='center', fontsize=9, color=MUTED)
    ax.text(ax2_vals[0] - 0.02, 112, 'planned lap:', fontsize=9, color=MUTED, ha='left')
    ax.set_ylim(-5, 118)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'fig_margin.png'), dpi=200)
    plt.close(fig)


def fig_siebel():
    d = np.load(os.path.join(RES, 'lines_siebel2.npz'))
    meta = read_map(os.path.join(ROOT, 'tracks', 'siebel2', 'siebel2.yaml'))
    img = meta['img'].astype(float)
    H, W = img.shape
    r = meta['resolution']
    ox, oy = meta['origin'][:2]
    # dark rendering of the SLAM map: free = dark grey, unknown = background, walls = light
    rgb = np.zeros((H, W, 3))
    bg = np.array([0x1b, 0x1b, 0x1b]) / 255
    rgb[:] = bg
    rgb[img > 250] = np.array([0x2c, 0x2c, 0x2a]) / 255
    rgb[img < 100] = np.array([0xd8, 0xd7, 0xd0]) / 255
    fig, ax = plt.subplots(figsize=(4.6, 7.4))
    ax.imshow(rgb, extent=[ox, ox + W * r, oy, oy + H * r], interpolation='nearest')
    ax.plot(*_closed(d['centerline_xy']).T, c=C_CENTRE, lw=1.0, ls=(0, (4, 4)))
    v = d['mintime_v']
    speed_line(ax, d['mintime_xy'], v, v.min(), v.max(), lw=2.4)
    ax.set_xlim(-4.5, 12.5)
    ax.set_ylim(-3.5, 27.5)
    ax.axis('off')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'fig_siebel.png'), dpi=200)
    plt.close(fig)


def fig_circle():
    G = B.get('G', {})
    if not G:
        return
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    cols = {'0.5': C_CENTRE, '0.7': C_MINCURV, '0.9': C_MINTIME}
    for mu, col in cols.items():
        f = os.path.join(RES, f'circle_mu{mu}.npz')
        if not os.path.exists(f):
            continue
        d = np.load(f)
        ax.plot(d['v'], d['ay'] / 9.81, c=col, lw=1.6, label=f'true μ = {mu}')
        est = np.mean([x['mu_est'] for x in G[mu]])
        ax.axhline(est, c=col, lw=1, ls=(0, (3, 3)))
        ax.text(0.2, est + 0.015, f'estimate {est:.2f} (true {mu})', color=INK2, fontsize=10)
    ax.set_xlabel('speed in the circle [m/s]')
    ax.set_ylabel('lateral acceleration [g]')
    ax.grid(color=GRID, lw=0.8)
    ax.legend(frameon=False, labelcolor=INK2, loc='lower right')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'fig_circle.png'), dpi=200)
    plt.close(fig)


if __name__ == '__main__' and not (len(sys.argv) > 1 and sys.argv[1] in ('deck', 'light')):
    for f in (fig_hero, fig_gallery, fig_gain, fig_mu, fig_margin, fig_siebel, fig_circle):
        try:
            f()
            print('ok', f.__name__)
        except Exception as e:  # keep going so partial results still render
            print('FAILED', f.__name__, repr(e))


# ----------------------------------------------------------------------------------- simplified deck figures
def _gain(r):
    c = r['centerline']
    t = r['mintime']
    if c['crashed'] or t['crashed'] or not np.isfinite(t['sim_mean']):
        return None
    return 100 * (1 - t['sim_mean'] / c['sim_mean'])


def fig_grid(ncols=6, nrows=3, fname='fig_grid.png'):
    """Grid of generated tracks with their time-optimal line; last tile = our Siebel SLAM map."""
    H = B.get('H', {})
    keys = sorted(H, key=int)[:ncols * nrows - 1]
    tiles = [(os.path.join(RES, 'grid', f"lines_{H[k]['track']['name']}.npz"), H[k]) for k in keys]
    tiles.append((os.path.join(RES, 'lines_siebel2.npz'), B['C']['siebel2']))
    fig, axes = plt.subplots(nrows, ncols, figsize=(16.6, 5.6))
    for ax, (f, info) in zip(axes.flat, tiles):
        d = np.load(f)
        le, re = _edges(d)
        for e in (le, re):
            ax.plot(*_closed(e).T, c=GRID_WALL, lw=0.9, solid_joinstyle='round')
        ax.plot(*_closed(d['mintime_xy']).T, c=GRID_LINE, lw=1.8, solid_joinstyle='round')
        g = _gain(info['results'])
        name = 'Siebel (ours)' if 'siebel' in f else f"{info['track']['length_m']:.0f} m"
        lab = f'{name}  ·  −{g:.0f}%' if g is not None else f'{name}  ·  hit wall'
        ax.set_title(lab, fontsize=18, color=(C_MINTIME if 'siebel' in f else INK2), pad=4)
        ax.set_aspect('equal', adjustable='datalim')
        ax.axis('off')
    for ax in list(axes.flat)[len(tiles):]:
        ax.axis('off')
    fig.subplots_adjust(left=0.005, right=0.995, top=0.93, bottom=0.01, wspace=0.08, hspace=0.32)
    fig.savefig(os.path.join(OUT, fname.replace('.png', SUFFIX + '.png')), dpi=160, transparent=bool(SUFFIX))
    plt.close(fig)


def all_closed_loop():
    rows = []
    for sec in ('A', 'B', 'C', 'H'):
        for n, v in B.get(sec, {}).items():
            r = v['results']
            c = r['centerline']['sim_mean']
            rows.append(dict(sec=sec, name=v['track']['name'] if sec == 'H' else n,
                             mc=100 * (1 - r['mincurv']['sim_mean'] / c), mt=100 * (1 - r['mintime']['sim_mean'] / c),
                             laps=sum(x['laps'] for x in r.values()),
                             crashes=sum(int(x['crashed']) for x in r.values()),
                             mt_status=r['mintime']['status']))
    return rows


def fig_strip(fname='fig_strip.png'):
    rows = all_closed_loop()
    fig, ax = plt.subplots(figsize=(9.0, 4.6))
    rng = np.random.default_rng(3)
    for yi, (key, col, lab) in enumerate((('mt', C_MINTIME, 'Min time'), ('mc', C_MINCURV, 'Min curvature'))):
        vals = np.array([r[key] for r in rows if np.isfinite(r[key])])
        jit = rng.uniform(-0.17, 0.17, len(vals))
        ax.plot(vals, yi + jit, 'o', ms=11, c=col, mec=BG, mew=2, zorder=3)
        m = vals.mean()
        ax.plot([m, m], [yi - 0.3, yi + 0.3], c=INK, lw=3, zorder=4, solid_capstyle='round')
        ax.text(m, yi + 0.36, f'avg {m:.0f}%', ha='center', va='bottom', fontsize=17, color=INK)
    ax.axvline(10, color=MUTED, lw=1.4, ls=(0, (3, 3)), zorder=1)
    ax.text(10.3, -0.62, '10% target', color=MUTED, fontsize=15, va='center')
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['Min time', 'Min curvature'], fontsize=18)
    ax.tick_params(axis='y', length=0)
    ax.tick_params(axis='x', labelsize=16)
    ax.set_xlabel('lap time saved vs centre line [%]   (one dot = one track, 10 sim laps)', fontsize=15)
    ax.set_ylim(-0.8, 1.75)
    ax.set_xlim(0, max(r['mt'] for r in rows) * 1.08)
    ax.grid(axis='x', color=GRID, lw=0.8, zorder=0)
    ax.spines['left'].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, fname.replace('.png', SUFFIX + '.png')), dpi=200, transparent=bool(SUFFIX))
    plt.close(fig)


def set_light_theme():
    """White background in Illinois colours (for the Illinois PowerPoint template)."""
    g = globals()
    g.update(BG='#ffffff', INK='#13294b', INK2='#3d4a5c', MUTED='#6b7280', GRID='#e3e5e8',
             C_MINCURV='#2870cd', C_MINTIME='#e8590c', GRID_WALL='#8c8b85', GRID_LINE='#e8590c', SUFFIX='_light')
    plt.rcParams.update({'figure.facecolor': BG, 'axes.facecolor': BG, 'savefig.facecolor': BG,
                         'text.color': INK, 'axes.labelcolor': INK2, 'xtick.color': INK2, 'ytick.color': INK2,
                         'axes.edgecolor': '#c8c6c7', 'font.family': ['Liberation Sans', 'Arial', 'DejaVu Sans']})


if __name__ == '__main__' and len(sys.argv) > 1 and sys.argv[1] == 'light':
    set_light_theme()
    fig_grid()
    fig_strip()
    print('ok light figures')
    sys.exit(0)

if __name__ == '__main__':      # figures used by the simplified deck (need section H)
    for f in (fig_grid, fig_strip):
        try:
            f()
            print('ok', f.__name__)
        except Exception as e:
            print('FAILED', f.__name__, repr(e))
