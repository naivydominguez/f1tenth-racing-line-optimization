"""Extract a closed centre line + track widths from an occupancy-grid map (e.g. slam_toolbox).

Method (robust to dead-end alcoves and laser 'leaks' that break skeleton methods):
  1. free space (map's own thresholds) -> morphological opening (drops slivers the car can't fit
     through) -> keep the connected component containing the start point
  2. fill small holes (furniture, noise); remaining non-free space splits into the OUTSIDE
     (touches the image border) and the INFIELD (largest enclosed blob)
  3. centre line = zero level set of  d_infield - d_outside  (marching squares -> ordered loop)
  4. smooth + resample; left/right widths by ray-casting along the normals; widths capped and
     made conservative with a running minimum
  5. start at the point closest to start_xy (slam_toolbox map origin = where the car started),
     direction CCW unless start_yaw / a reference waypoint file says otherwise
"""
import numpy as np
import cv2
from scipy import ndimage
from scipy.ndimage import minimum_filter1d
from skimage.measure import find_contours

from .map_io import read_map, free_mask, px_to_world, world_to_px
from .geometry import resample_closed, heading_curvature, normals_left, signed_area, polyline_length
from .track import Track


def _disk(r_px):
    r = max(int(round(r_px)), 0)
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))


def extract_centerline(yaml_path, start_xy=(0.0, 0.0), start_yaw=None, ds=0.1, smoothing=0.05,
                       open_radius=0.15, hole_area_max=1.0, max_half_width=2.0, wall_shrink=0.0,
                       width_min_window=0.6, clearance_ref=0.3, direction='ccw', name=None,
                       return_debug=False):
    """Returns a Track. All lengths in metres.

    open_radius    : morphological opening radius; free gaps narrower than 2*r are ignored
    hole_area_max  : enclosed obstacles smaller than this [m^2] are filled (treated as free)
    max_half_width : cap on each side's width (stops the planner using open rooms / SLAM leaks)
    wall_shrink    : subtract this from both widths (extra conservatism against map error)
    width_min_window: widths = running minimum over this arc length (conservative)
    clearance_ref  : widths are 'Euclidean-clearance' widths: w = n_max + clearance_ref, where n_max
                     is the furthest offset along the normal whose distance to ANY obstacle is still
                     >= clearance_ref. Unlike a plain normal ray this sees wall corners diagonally.
                     Exact when half car width + wall_margin == clearance_ref (0.155 + 0.15 ~ 0.3).
    """
    meta = read_map(yaml_path)
    res = meta['resolution']
    free = free_mask(meta).astype(np.uint8)
    free = cv2.morphologyEx(free, cv2.MORPH_OPEN, _disk(open_radius / res))
    lab, nlab = ndimage.label(free, structure=np.ones((3, 3)))
    if nlab == 0:
        raise RuntimeError('no free space in map')
    r0, c0 = world_to_px(start_xy[0], start_xy[1], meta)
    r0, c0 = int(round(float(r0))), int(round(float(c0)))
    H, W = free.shape
    comp = 0
    if 0 <= r0 < H and 0 <= c0 < W and lab[r0, c0] > 0:
        comp = lab[r0, c0]
    else:  # nearest free component to the start point
        dist, (ri, ci) = ndimage.distance_transform_edt(lab == 0, return_indices=True)
        rr, cc = np.clip(r0, 0, H - 1), np.clip(c0, 0, W - 1)
        comp = lab[ri[rr, cc], ci[rr, cc]]
    corridor = lab == comp

    # non-free components: outside (touches border) vs enclosed holes
    nonfree = ~corridor
    hl, nh = ndimage.label(nonfree, structure=np.ones((3, 3)))
    border = np.unique(np.concatenate([hl[0, :], hl[-1, :], hl[:, 0], hl[:, -1]]))
    border = border[border > 0]
    sizes = ndimage.sum(np.ones_like(hl), hl, index=np.arange(1, nh + 1)) * res * res
    holes = [i + 1 for i in range(nh) if (i + 1) not in border]
    if not holes:
        raise RuntimeError('free space is not a loop (no enclosed infield found)')
    big = [h for h in holes if sizes[h - 1] > hole_area_max]
    for h in holes:
        if h not in big:
            corridor |= hl == h  # fill small obstacles
    if not big:
        raise RuntimeError('no infield larger than hole_area_max')
    infield_id = max(big, key=lambda h: sizes[h - 1])
    infield = hl == infield_id
    outside = ~corridor & ~infield

    d_in = cv2.distanceTransform((~infield).astype(np.uint8), cv2.DIST_L2, 5)
    d_out = cv2.distanceTransform((~outside).astype(np.uint8), cv2.DIST_L2, 5)
    f = (d_in - d_out).astype(float)
    contours = find_contours(f, 0.0, mask=corridor)
    closed = [c for c in contours if np.linalg.norm(c[0] - c[-1]) < 2.0 and len(c) > 20]
    if not closed:
        closed = contours
    cpx = max(closed, key=len)
    x, y = px_to_world(cpx[:, 0], cpx[:, 1], meta)
    xy = np.column_stack([x, y])
    xy = resample_closed(xy, ds, smoothing=smoothing)

    # orientation
    want_ccw = direction == 'ccw'
    if start_yaw is not None:
        i = int(np.argmin(np.hypot(*(xy - np.asarray(start_xy)).T)))
        t = xy[(i + 1) % len(xy)] - xy[i]
        if np.dot(t, [np.cos(start_yaw), np.sin(start_yaw)]) < 0:
            xy = xy[::-1]
    elif (signed_area(xy) > 0) != want_ccw:
        xy = xy[::-1]
    i = int(np.argmin(np.hypot(*(xy - np.asarray(start_xy)).T)))
    xy = np.roll(xy, -i, axis=0)

    # widths by ray casting along the normals inside the corridor
    psi, _ = heading_curvature(xy)
    nrm = normals_left(psi)
    steps = np.arange(0, max_half_width + 0.5 * res, 0.5 * res)
    dist = cv2.distanceTransform(corridor.astype(np.uint8), cv2.DIST_L2, 5) * res   # m to nearest obstacle
    wl = _clearance_width(xy, nrm, steps, dist, meta, clearance_ref)
    wr = _clearance_width(xy, -nrm, steps, dist, meta, clearance_ref)
    win = max(int(round(width_min_window / ds)), 1)
    wl = minimum_filter1d(wl, win, mode='wrap') - wall_shrink
    wr = minimum_filter1d(wr, win, mode='wrap') - wall_shrink
    tr = Track(xy, np.maximum(wr, 0.05), np.maximum(wl, 0.05), name=name or 'extracted')
    if return_debug:
        return tr, dict(corridor=corridor, infield=infield, outside=outside, f=f, meta=meta)
    return tr


def _clearance_width(xy, dirs, steps, dist, meta, c_ref):
    """w = (furthest contiguous offset with clearance >= c_ref) + c_ref, capped by steps[-1]."""
    H, W = dist.shape
    nmax = np.zeros(len(xy))
    alive = np.ones(len(xy), bool)
    for d in steps:
        p = xy + dirs * d
        r, c = world_to_px(p[:, 0], p[:, 1], meta)
        r = np.clip(np.round(r).astype(int), 0, H - 1)
        c = np.clip(np.round(c).astype(int), 0, W - 1)
        ok = dist[r, c] >= c_ref
        alive &= ok
        nmax[alive] = d
        if not alive.any():
            break
    d0 = _sample(dist, xy, meta)
    w = np.where(d0 >= c_ref, nmax + c_ref, d0)   # narrower than 2*c_ref: fall back to plain clearance
    return np.minimum(w, steps[-1])


def _sample(img, xy, meta):
    H, W = img.shape
    r, c = world_to_px(xy[:, 0], xy[:, 1], meta)
    r = np.clip(np.round(r).astype(int), 0, H - 1)
    c = np.clip(np.round(c).astype(int), 0, W - 1)
    return img[r, c]


def _raycast(xy, dirs, steps, corridor, meta):
    H, W = corridor.shape
    out = np.full(len(xy), steps[-1])
    hit = np.zeros(len(xy), bool)
    for d in steps[1:]:
        p = xy + dirs * d
        r, c = world_to_px(p[:, 0], p[:, 1], meta)
        r = np.clip(np.round(r).astype(int), 0, H - 1)
        c = np.clip(np.round(c).astype(int), 0, W - 1)
        blocked = ~corridor[r, c] & ~hit
        out[blocked] = d - 0.5 * (steps[1] - steps[0])
        hit |= blocked
        if hit.all():
            break
    return out


def widths_from_map(track, yaml_path, clearance_ref=0.3, max_half_width=3.0, width_min_window=0.6,
                    open_radius=0.0):
    """Replace a centre line's widths with what the occupancy map actually allows (same
    Euclidean-clearance definition as extract_centerline). Use it when a width file and the map
    disagree (f1tenth_racetracks CSVs say 1.1 m; the drawn wall faces are ~1.0 m away)."""
    meta = read_map(yaml_path)
    res = meta['resolution']
    free = free_mask(meta).astype(np.uint8)
    if open_radius > 0:
        free = cv2.morphologyEx(free, cv2.MORPH_OPEN, _disk(open_radius / res))
    dist = cv2.distanceTransform(free, cv2.DIST_L2, 5) * res
    steps = np.arange(0, max_half_width + 0.5 * res, 0.5 * res)
    wl = _clearance_width(track.xy, track.normal, steps, dist, meta, clearance_ref)
    wr = _clearance_width(track.xy, -track.normal, steps, dist, meta, clearance_ref)
    ds = track.length / track.n_points
    win = max(int(round(width_min_window / ds)), 1)
    wl = minimum_filter1d(wl, win, mode='wrap')
    wr = minimum_filter1d(wr, win, mode='wrap')
    return Track(track.xy, np.maximum(wr, 0.05), np.maximum(wl, 0.05), name=track.name)
