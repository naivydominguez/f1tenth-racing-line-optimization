"""Track = closed reference line + left/right widths.

File format is the one used by f1tenth_racetracks / TUM global_racetrajectory_optimization:
    # x_m, y_m, w_tr_right_m, w_tr_left_m
Widths are measured from the reference line to the track edge (wall), right / left of the
direction of travel.
"""
import numpy as np
from scipy.spatial import cKDTree

from .geometry import (resample_closed, heading_curvature, arc_length, normals_left,
                       signed_area, _open)


class Track:
    def __init__(self, xy, w_right, w_left, name='track'):
        self.xy = _open(np.asarray(xy, float))
        n = len(self.xy)
        self.w_right = np.broadcast_to(np.asarray(w_right, float), (n,)).copy()
        self.w_left = np.broadcast_to(np.asarray(w_left, float), (n,)).copy()
        self.name = name
        self._update()

    # ------------------------------------------------------------------ derived quantities
    def _update(self):
        self.psi, self.kappa = heading_curvature(self.xy)
        self.s, self.length = arc_length(self.xy)
        self.normal = normals_left(self.psi)
        self._tree = cKDTree(self.xy)

    @property
    def n_points(self):
        return len(self.xy)

    @property
    def left_edge(self):
        return self.xy + self.normal * self.w_left[:, None]

    @property
    def right_edge(self):
        return self.xy - self.normal * self.w_right[:, None]

    @property
    def is_ccw(self):
        return signed_area(self.xy) > 0

    # ------------------------------------------------------------------ IO
    @classmethod
    def from_csv(cls, path, name=None):
        data = np.loadtxt(path, delimiter=',', comments='#')
        return cls(data[:, :2], data[:, 2], data[:, 3], name=name or str(path).split('/')[-1].split('.')[0])

    def to_csv(self, path):
        data = np.column_stack([self.xy, self.w_right, self.w_left])
        np.savetxt(path, data, delimiter=', ', fmt='%.6f',
                   header='x_m, y_m, w_tr_right_m, w_tr_left_m', comments='# ')

    # ------------------------------------------------------------------ operations
    def resampled(self, ds=None, n=None, smoothing=0.0):
        """Uniformly resampled (optionally smoothed) copy. Widths are interpolated."""
        xy, w = resample_closed(self.xy, ds, smoothing=smoothing,
                                extra=np.column_stack([self.w_right, self.w_left]), n=n)
        if smoothing > 0:
            # smoothing moves the reference line sideways: keep the WALLS where they were by
            # correcting both widths with the lateral shift relative to the original line
            s, shift, _ = self.project(xy)
            wr, wl = self.width_at(s)
            w = np.column_stack([wr + shift, wl - shift])
        return Track(xy, w[:, 0], w[:, 1], name=self.name)

    def reversed(self):
        return Track(self.xy[::-1], self.w_left[::-1], self.w_right[::-1], name=self.name)

    def rolled_to(self, idx):
        idx = int(idx) % self.n_points
        return Track(np.roll(self.xy, -idx, axis=0), np.roll(self.w_right, -idx),
                     np.roll(self.w_left, -idx), name=self.name)

    def project(self, pts):
        """Frenet projection of points onto the reference line.

        Returns (s, n, idx): arc length of the foot point, signed lateral offset (left > 0),
        and index of the reference segment start.
        """
        pts = np.atleast_2d(np.asarray(pts, float))
        _, near = self._tree.query(pts, k=1)
        N = self.n_points
        best_s = np.zeros(len(pts))
        best_n = np.zeros(len(pts))
        best_i = np.zeros(len(pts), dtype=int)
        best_d = np.full(len(pts), np.inf)
        for off in (-1, 0):
            i0 = (near + off) % N
            i1 = (i0 + 1) % N
            a = self.xy[i0]
            b = self.xy[i1]
            ab = b - a
            L2 = np.maximum(np.sum(ab * ab, axis=1), 1e-12)
            t = np.clip(np.sum((pts - a) * ab, axis=1) / L2, 0.0, 1.0)
            foot = a + ab * t[:, None]
            d = np.hypot(*(pts - foot).T)
            cross = ab[:, 0] * (pts[:, 1] - a[:, 1]) - ab[:, 1] * (pts[:, 0] - a[:, 0])
            sign = np.sign(cross)
            sign[sign == 0] = 1.0
            better = d < best_d
            best_d[better] = d[better]
            best_n[better] = (sign * d)[better]
            seg_len = np.sqrt(L2)
            best_s[better] = (self.s[i0] + t * seg_len)[better]
            best_i[better] = i0[better]
        return best_s, best_n, best_i

    def width_at(self, s):
        """Interpolated (w_right, w_left) at arc length s (periodic)."""
        s = np.mod(s, self.length)
        ss = np.concatenate([self.s, [self.length]])
        wr = np.interp(s, ss, np.concatenate([self.w_right, self.w_right[:1]]))
        wl = np.interp(s, ss, np.concatenate([self.w_left, self.w_left[:1]]))
        return wr, wl

    def summary(self):
        return dict(name=self.name, length_m=round(self.length, 2), n_points=self.n_points,
                    width_min_m=round(float(np.min(self.w_left + self.w_right)), 2),
                    width_max_m=round(float(np.max(self.w_left + self.w_right)), 2),
                    min_radius_m=round(float(1.0 / max(np.max(np.abs(self.kappa)), 1e-9)), 2))
