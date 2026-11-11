"""Closed-curve geometry helpers (periodic splines, resampling, heading, curvature, normals)."""
import numpy as np
from scipy.interpolate import splprep, splev


def _open(xy):
    """Drop a duplicated closing point if the polyline repeats its first point."""
    xy = np.asarray(xy, dtype=float)
    if len(xy) > 2 and np.linalg.norm(xy[0] - xy[-1]) < 1e-9:
        xy = xy[:-1]
    return xy


def polyline_length(xy, closed=True):
    xy = _open(xy)
    d = np.diff(np.vstack([xy, xy[:1]]) if closed else xy, axis=0)
    return float(np.sum(np.hypot(d[:, 0], d[:, 1])))


def closed_spline(xy, smoothing=0.0, extra=None):
    """Periodic cubic smoothing spline through a closed polyline.

    smoothing: RMS deviation [m] allowed per point (0 = interpolate exactly).
    extra: optional (N, k) array of per-point values (e.g. track widths) splined alongside.
    Returns (tck, u_nodes, length) with u parameterised by approximate arc length in [0, 1).
    """
    xy = _open(xy)
    pts = xy if extra is None else np.hstack([xy, np.asarray(extra, float).reshape(len(xy), -1)])
    pts = np.vstack([pts, pts[:1]])  # splprep(per=1) needs the closing point
    seg = np.hypot(np.diff(pts[:, 0]), np.diff(pts[:, 1]))
    seg = np.maximum(seg, 1e-9)
    u = np.concatenate([[0.0], np.cumsum(seg)])
    L = u[-1]
    u = u / L
    s = len(pts) * smoothing ** 2
    tck, _ = splprep(pts.T, u=u, s=s, per=1, k=3)
    return tck, u[:-1], L


def resample_closed(xy, ds, smoothing=0.0, extra=None, n=None):
    """Resample a closed curve at (approximately) uniform arc-length spacing ds.

    Returns xy (N,2) and, if given, extra (N,k) interpolated on the same samples.
    The spline is evaluated densely first so that the final spacing is uniform in true arc length.
    """
    tck, _, _ = closed_spline(xy, smoothing, extra)
    uu = np.linspace(0, 1, 20000, endpoint=False)
    vals = np.array(splev(uu, tck)).T
    seg = np.hypot(*np.diff(np.vstack([vals[:, :2], vals[:1, :2]]), axis=0).T)
    s = np.concatenate([[0], np.cumsum(seg)])
    L = s[-1]
    if n is None:
        n = max(int(round(L / ds)), 8)
    s_new = np.linspace(0, L, n, endpoint=False)
    u_new = np.interp(s_new, s, np.concatenate([uu, [1.0]]))
    out = np.array(splev(u_new, tck)).T
    if extra is None:
        return out[:, :2]
    return out[:, :2], out[:, 2:]


def heading_curvature(xy):
    """Heading psi (rad, 0 = +x, CCW positive) and signed curvature (1/m, left turn > 0)
    of a closed curve, from an interpolating periodic spline (smooth derivatives)."""
    xy = _open(xy)
    tck, u, L = closed_spline(xy, 0.0)
    dx, dy = splev(u, tck, der=1)
    ddx, ddy = splev(u, tck, der=2)
    psi = np.arctan2(dy, dx)
    kappa = (dx * ddy - dy * ddx) / np.power(dx * dx + dy * dy, 1.5)
    return psi, kappa


def arc_length(xy, closed=True):
    """Cumulative arc length at each node (starts at 0) and total length."""
    xy = _open(xy)
    nxt = np.roll(xy, -1, axis=0)
    seg = np.hypot(*(nxt - xy).T)
    if not closed:
        seg[-1] = 0.0
    s = np.concatenate([[0.0], np.cumsum(seg[:-1])])
    return s, float(np.sum(seg))


def normals_left(psi):
    """Unit normal pointing to the left of the direction of travel."""
    return np.column_stack([-np.sin(psi), np.cos(psi)])


def signed_area(xy):
    xy = _open(xy)
    x, y = xy[:, 0], xy[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def wrap_angle(a):
    return (np.asarray(a) + np.pi) % (2 * np.pi) - np.pi


def segments_intersect_count(xy):
    """Number of proper self-intersections of a closed polyline (O(N^2) but vectorised;
    fine for a few thousand points)."""
    xy = _open(xy)
    a = xy
    b = np.roll(xy, -1, axis=0)
    n = len(a)
    count = 0
    for i in range(n):
        p, r = a[i], b[i] - a[i]
        q = a
        s = b - a
        rxs = r[0] * s[:, 1] - r[1] * s[:, 0]
        qp = q - p
        with np.errstate(divide='ignore', invalid='ignore'):
            t = (qp[:, 0] * s[:, 1] - qp[:, 1] * s[:, 0]) / rxs
            uu = (qp[:, 0] * r[1] - qp[:, 1] * r[0]) / rxs
        idx = np.arange(n)
        ok = (np.abs(rxs) > 1e-12) & (t > 1e-9) & (t < 1 - 1e-9) & (uu > 1e-9) & (uu < 1 - 1e-9)
        ok &= (idx != i) & (idx != (i + 1) % n) & (idx != (i - 1) % n)
        count += int(np.sum(ok))
    return count // 2
