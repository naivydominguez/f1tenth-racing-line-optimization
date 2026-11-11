"""Closed-loop lap simulation: our Lab-5 pure pursuit following a racing line on a simulated car.

The controller is a direct port of pure_pursuit_car/scripts/pure_pursuit_node.py (forward-window
closest point, interpolated look-ahead point, gamma = 2 y / L^2, delta = atan(wheelbase * gamma)),
except that the speed set-point comes from the optimiser's speed profile (x speed_scale).

Metrics (proposal): mean/std lap time over N flying laps, boundary violations (any corner of the
car's footprint outside the track edge), laps completed. A violation = the car touched a wall,
so the run stops there (crash).
"""
import math
import numpy as np

from .vehicle import Car


class PurePursuit:
    def __init__(self, xy, v, p):
        self.wp = np.asarray(xy, float)
        self.v = np.asarray(v, float)
        c = p.controller
        self.L_min, self.L_gain, self.L_max = c.lookahead_min, c.lookahead_gain, c.lookahead_max
        self.speed_scale = c.speed_scale
        self.slip_comp = float(c.get('slip_compensation', 0.0))
        vp = p.vehicle
        self.lr = vp.lr
        self.ky_r = 1.0 / (vp.mu_stiffness * vp.C_Sr * 9.81)            # rear slip per unit a_y
        self.k_us = (1.0 / vp.C_Sf - 1.0 / vp.C_Sr) / (vp.mu_stiffness * 9.81)  # understeer gradient
        from raceline.geometry import heading_curvature
        self.kappa = heading_curvature(self.wp)[1]
        self.wheelbase = p.vehicle.lf + p.vehicle.lr
        self.max_steer = p.limits.delta_max
        self.idx = None
        n = len(self.wp)
        seg = np.hypot(*(np.roll(self.wp, -1, axis=0) - self.wp).T)
        self.window = int(max(60, 6.0 / max(np.mean(seg), 1e-3)))

    def closest(self, x, y):
        n = len(self.wp)
        if self.idx is None:
            self.idx = int(np.argmin(np.hypot(self.wp[:, 0] - x, self.wp[:, 1] - y)))
            return self.idx
        idx = (self.idx + np.arange(-5, self.window)) % n
        d = np.hypot(self.wp[idx, 0] - x, self.wp[idx, 1] - y)
        self.idx = int(idx[np.argmin(d)])
        return self.idx

    def lookahead(self, x, y, start, L):
        wp = self.wp
        n = len(wp)
        prev = wp[start]
        for k in range(1, n):
            i = (start + k) % n
            cur = wp[i]
            if math.hypot(cur[0] - x, cur[1] - y) >= L:
                dx, dy = cur[0] - prev[0], cur[1] - prev[1]
                fx, fy = prev[0] - x, prev[1] - y
                a = dx * dx + dy * dy
                b = 2 * (fx * dx + fy * dy)
                c = fx * fx + fy * fy - L * L
                disc = max(b * b - 4 * a * c, 0.0)
                t = (-b + math.sqrt(disc)) / (2 * a) if a > 1e-12 else 1.0
                t = min(max(t, 0.0), 1.0)
                return (prev[0] + t * dx, prev[1] + t * dy), i
            prev = cur
        i = (start + 1) % n
        return tuple(wp[i]), i

    def __call__(self, x, y, yaw, speed):
        idx = self.closest(x, y)
        L = min(max(self.L_min + self.L_gain * speed, self.L_min), self.L_max)
        goal, gi = self.lookahead(x, y, idx, L)
        dx, dy = goal[0] - x, goal[1] - y
        ff = 0.0
        if self.slip_comp > 0.0:
            # steady-state single-track feed-forward from the planned curvature: the body yaw
            # points inside the velocity by beta = lr*k - a_y/(mu C_Sr g), and the car needs
            # k_us * a_y more steering than geometry (understeer). Pure pursuit alone runs wide.
            k = 0.5 * (self.kappa[idx] + self.kappa[gi])
            ay = speed * speed * k
            beta = self.lr * k - self.ky_r * ay
            yaw = yaw + self.slip_comp * beta
            ff = self.slip_comp * self.k_us * ay
        c, s = math.cos(yaw), math.sin(yaw)
        yc = -s * dx + c * dy
        gamma = 2.0 * yc / (L * L)
        delta = max(-self.max_steer, min(self.max_steer, math.atan(self.wheelbase * gamma) + ff))
        v = min(self.v[idx], self.v[gi]) * self.speed_scale
        return v, delta


def _footprint_offsets(p):
    hl = 0.5 * p.vehicle.length
    hw = 0.5 * p.vehicle.width
    # gym body is centred between the axles; corners relative to the reference point
    off = 0.5 * (p.vehicle.lf - p.vehicle.lr)
    return [(hl + off, hw), (hl + off, -hw), (-hl + off, hw), (-hl + off, -hw)]


class GridCollision:
    """Wall check against the occupancy-grid map itself. Like f1tenth_gym, only dark (occupied)
    pixels are walls (gym: pixel <= 128); unknown SLAM cells inside the corridor are driveable."""

    def __init__(self, yaml_path, min_obstacle_area=0.05):
        import cv2
        from scipy import ndimage
        from raceline.map_io import read_map
        self.meta = read_map(yaml_path)
        img = self.meta['img']
        occ = (img >= 128) if self.meta.get('negate', 0) else (img < 128)
        if min_obstacle_area > 0:   # drop SLAM speckle (people / noise seen once while mapping)
            lab, n = ndimage.label(occ, structure=np.ones((3, 3)))
            if n:
                area = ndimage.sum(np.ones_like(lab), lab, index=np.arange(1, n + 1)) * self.meta['resolution'] ** 2
                small = np.concatenate([[False], area < min_obstacle_area])
                occ = occ & ~small[lab]
        self.dist = cv2.distanceTransform((~occ).astype(np.uint8), cv2.DIST_L2, 5) * self.meta['resolution']
        self.H, self.W = occ.shape
        self.res = self.meta['resolution']
        self.ox, self.oy = self.meta['origin'][:2]

    def clearance(self, pts):
        """Distance [m] from each point to the nearest non-free cell (<= 0 means inside a wall)."""
        out = []
        for x, y in pts:
            c = int((x - self.ox) / self.res)
            r = self.H - 1 - int((y - self.oy) / self.res)
            if 0 <= r < self.H and 0 <= c < self.W:
                out.append(self.dist[r, c] - 0.5 * self.res)
            else:
                out.append(-1.0)
        return min(out)


def run_laps(track, line_xy, line_v, p, laps=None, tire_model=None, mu_true=None, seed=0,
             record=False, max_time=None, collision_map=None):
    """Simulate `laps` flying laps (+1 standing-start lap that is discarded).

    track: Track (centre line + widths) used for lap progress (and for the wall check if no map).
    line_xy, line_v: racing line to follow (closed, ~0.1 m spacing) and its speed profile.
    collision_map: optional map YAML (or GridCollision) -> walls = occupied/unknown map cells.
    """
    if isinstance(collision_map, str):
        collision_map = GridCollision(collision_map)
    laps = laps or p.sim.laps
    tire_model = tire_model or p.sim.tire_model
    rng = np.random.default_rng(seed)
    dt = p.sim.dt
    car = Car(p, tire_model=tire_model, mu_true=mu_true, dt=dt, steer_delay=p.sim.steer_delay_steps)
    pp = PurePursuit(line_xy, line_v, p)
    # start on the racing line point closest to the track start, at rest, heading along the line
    s_line, _, _ = track.project(line_xy)
    i0 = int(np.argmin(np.minimum(s_line, track.length - s_line)))
    nxt = line_xy[(i0 + 1) % len(line_xy)]
    yaw0 = math.atan2(nxt[1] - line_xy[i0][1], nxt[0] - line_xy[i0][0])
    car.reset(line_xy[i0][0], line_xy[i0][1], yaw0 % (2 * math.pi), 0.0)

    ctrl_every = max(int(round(1.0 / (p.sim.pose_rate_hz * dt))), 1)
    corners = _footprint_offsets(p)
    L = track.length
    max_time = max_time or (laps + 1) * 60.0
    s_prev, _, _ = track.project([car.pose[:2]])
    s_prev = float(s_prev[0])
    lap_start = None
    lap_times = []
    t = 0.0
    step = 0
    cmd = (0.0, 0.0)
    crashed = False
    min_clear = np.inf
    hist = [] if record else None
    hint = None
    while t < max_time:
        x, y, yaw = car.pose
        if step % ctrl_every == 0:
            nx = x + rng.normal(0, p.sim.pose_noise_xy)
            ny = y + rng.normal(0, p.sim.pose_noise_xy)
            nyaw = yaw + rng.normal(0, p.sim.pose_noise_yaw)
            cmd = pp(nx, ny, nyaw, car.speed)
        car.step(*cmd)
        t += dt
        step += 1
        x, y, yaw = car.pose
        # footprint vs track edges
        c, s = math.cos(yaw), math.sin(yaw)
        pts = [(x + c * a - s * b, y + s * a + c * b) for a, b in corners]
        if collision_map is not None:
            mids = [(0.5 * (pts[0][0] + pts[1][0]), 0.5 * (pts[0][1] + pts[1][1])),
                    (0.5 * (pts[0][0] + pts[2][0]), 0.5 * (pts[0][1] + pts[2][1])),
                    (0.5 * (pts[1][0] + pts[3][0]), 0.5 * (pts[1][1] + pts[3][1]))]
            clear = collision_map.clearance(pts + mids)
            ss, _, _ = track.project(pts[:2])
        else:
            ss, nn, _ = track.project(pts)
            wr, wl = track.width_at(ss)
            clear = float(np.min(np.minimum(wl - nn, nn + wr)))
        min_clear = min(min_clear, clear)
        s_now = float(np.mean(ss[:2]) if abs(ss[0] - ss[1]) < L / 2 else ss[0])
        if record and step % 5 == 0:
            hist.append((t, x, y, car.speed, cmd[1], s_now))
        if clear < 0:
            crashed = True
            break
        # lap counting: crossing s = 0 forwards
        if s_prev > 0.75 * L and s_now < 0.25 * L and t > 1.0:
            if lap_start is not None:
                lap_times.append(t - lap_start)
            lap_start = t
            if len(lap_times) >= laps:
                break
        s_prev = s_now
    out = dict(lap_times=lap_times, laps_completed=len(lap_times), crashed=crashed,
               violations=int(crashed), min_clearance=float(min_clear), sim_time=t,
               mean_lap=float(np.mean(lap_times)) if lap_times else float('nan'),
               std_lap=float(np.std(lap_times)) if lap_times else float('nan'))
    if record:
        out['hist'] = np.array(hist)
    return out
