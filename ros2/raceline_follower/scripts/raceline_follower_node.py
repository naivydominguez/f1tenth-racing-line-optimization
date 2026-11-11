#!/usr/bin/env python3
"""
Racing-line follower (CS 498 F1TENTH final project, Team 1).

Our Lab-5 pure pursuit node with four additions for the racing-line project:
  * use_csv_speed      : follow the optimiser's speed profile (column 'v' of *_waypoints.csv)
                         instead of recomputing one from curvature
  * slip_compensation  : 0..1 steady-state side-slip / understeer feed-forward (0 = plain Lab 5)
  * lap logging        : every lap time is printed and appended to lap_log_path (CSV)
  * wall monitor       : if track_csv (x,y,w_right,w_left centre line) is given, the car's lateral
                         offset is checked every pose and "boundary violations" are counted per lap

Pose: /pf/pose/odom (particle filter, nav_msgs/Odometry) on the car, /ego_racecar/odom in the sim.
Waypoints: results/lines/<track>/<method>_waypoints.csv from the optimiser (header x,y,v,...).
UNTESTED ON ROS: the control law is the same code as sim/closed_loop.py (tested there), the ROS
plumbing is copied from our working Lab-5 node.
"""
import csv
import math
import os
import time

import numpy as np
import rclpy
from rclpy.node import Node
from ackermann_msgs.msg import AckermannDriveStamped
from geometry_msgs.msg import Point, PoseStamped
from nav_msgs.msg import Odometry
from visualization_msgs.msg import Marker, MarkerArray

try:
    from ament_index_python.packages import get_package_share_directory
except ImportError:  # pragma: no cover
    get_package_share_directory = None


# ---------------------------------------------------------------- pure math helpers
def load_waypoints(path):
    """CSV with a header containing at least x,y (optional v). Header-less: first two columns x,y.
    Returns (xy (N,2), v (N,) or None)."""
    rows, vs = [], []
    with open(path) as f:
        reader = csv.reader(f)
        first = next(reader)
        iv = None
        try:
            rows.append([float(first[0]), float(first[1])])
            ix, iy = 0, 1
        except ValueError:
            hdr = [h.strip().lower() for h in first]
            ix, iy = hdr.index('x'), hdr.index('y')
            iv = hdr.index('v') if 'v' in hdr else None
        for r in reader:
            if len(r) > max(ix, iy) and r[ix].strip():
                rows.append([float(r[ix]), float(r[iy])])
                if iv is not None:
                    vs.append(float(r[iv]))
    v = np.array(vs, dtype=float) if iv is not None and len(vs) == len(rows) else None
    return np.array(rows, dtype=float), v


def path_curvature(wp):
    """Signed curvature at each waypoint of a closed loop (heading change / arc length)."""
    prev, nxt = np.roll(wp, 1, axis=0), np.roll(wp, -1, axis=0)
    h1 = np.arctan2(wp[:, 1] - prev[:, 1], wp[:, 0] - prev[:, 0])
    h2 = np.arctan2(nxt[:, 1] - wp[:, 1], nxt[:, 0] - wp[:, 0])
    dh = (h2 - h1 + np.pi) % (2 * np.pi) - np.pi
    ds = 0.5 * (np.linalg.norm(wp - prev, axis=1) + np.linalg.norm(nxt - wp, axis=1))
    return dh / np.maximum(ds, 1e-6)


def speed_profile(wp, v_max, v_min, a_lat, a_decel, smooth_pts=5):
    """Lab-5 fallback: v = min(v_max, sqrt(a_lat/|k|)) + backward braking pass."""
    k = np.abs(path_curvature(wp))
    if smooth_pts > 1:
        k = np.max([np.roll(k, s) for s in range(-smooth_pts, smooth_pts + 1)], axis=0)
    v = np.minimum(v_max, np.sqrt(a_lat / np.maximum(k, 1e-6)))
    v = np.maximum(v, v_min)
    seg = np.linalg.norm(np.roll(wp, -1, axis=0) - wp, axis=1)
    n = len(wp)
    for _ in range(2):
        for i in range(n - 1, -1, -1):
            j = (i + 1) % n
            v[i] = min(v[i], math.sqrt(v[j] ** 2 + 2.0 * a_decel * seg[i]))
    return v


def closest_index(wp, x, y, last_idx=None, window=60):
    n = len(wp)
    if last_idx is None:
        return int(np.argmin(np.hypot(wp[:, 0] - x, wp[:, 1] - y)))
    idx = (last_idx + np.arange(-5, window)) % n
    d = np.hypot(wp[idx, 0] - x, wp[idx, 1] - y)
    return int(idx[np.argmin(d)])


def lookahead_point(wp, x, y, start_idx, L):
    n = len(wp)
    p = np.array([x, y])
    prev = wp[start_idx]
    for k in range(1, n):
        i = (start_idx + k) % n
        cur = wp[i]
        if np.linalg.norm(cur - p) >= L:
            d = cur - prev
            f = prev - p
            a = d @ d
            b = 2 * f @ d
            c = f @ f - L * L
            disc = max(b * b - 4 * a * c, 0.0)
            t = (-b + math.sqrt(disc)) / (2 * a) if a > 1e-12 else 1.0
            t = min(max(t, 0.0), 1.0)
            return prev + t * d, i
        prev = cur
    return wp[(start_idx + 1) % n], (start_idx + 1) % n


def to_vehicle_frame(goal, x, y, yaw):
    dx, dy = goal[0] - x, goal[1] - y
    c, s = math.cos(yaw), math.sin(yaw)
    return c * dx + s * dy, -s * dx + c * dy


def pure_pursuit_steer(y_c, L, wheelbase, max_steer, ff=0.0):
    gamma = 2.0 * y_c / (L * L)
    delta = math.atan(wheelbase * gamma) + ff
    return float(np.clip(delta, -max_steer, max_steer)), gamma


def slip_feedforward(kappa, speed, lr=0.17145, mu_s=1.0489, C_Sf=4.718, C_Sr=5.4562, g=9.81):
    """Steady-state single-track: body side-slip beta and extra (understeer) steering for a corner
    of curvature kappa at this speed. Same model/numbers as config/params.yaml (f1tenth_gym)."""
    ay = speed * speed * kappa
    beta = lr * kappa - ay / (mu_s * C_Sr * g)
    d_us = (1.0 / C_Sf - 1.0 / C_Sr) / (mu_s * g) * ay
    return beta, d_us


def yaw_from_quat(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def load_track(path):
    """Centre line CSV '# x_m, y_m, w_tr_right_m, w_tr_left_m' -> (xy, w_right, w_left)."""
    d = np.loadtxt(path, delimiter=',', comments='#')
    return d[:, :2], d[:, 2], d[:, 3]


def lateral_offset(track_xy, x, y, hint=None, window=80):
    """Signed offset (left +) of (x, y) from the closest centre-line segment, and its index."""
    n = len(track_xy)
    if hint is None:
        i = int(np.argmin(np.hypot(track_xy[:, 0] - x, track_xy[:, 1] - y)))
    else:
        idx = (hint + np.arange(-10, window)) % n
        i = int(idx[np.argmin(np.hypot(track_xy[idx, 0] - x, track_xy[idx, 1] - y))])
    a, b = track_xy[i], track_xy[(i + 1) % n]
    ab = b - a
    cross = ab[0] * (y - a[1]) - ab[1] * (x - a[0])
    return cross / max(np.linalg.norm(ab), 1e-9), i


# ---------------------------------------------------------------- ROS node
class RacelineFollower(Node):
    def __init__(self):
        super().__init__('raceline_follower')
        default_csv = ''
        if get_package_share_directory is not None:
            try:
                default_csv = os.path.join(get_package_share_directory('raceline_follower'),
                                           'waypoints', 'mintime_waypoints.csv')
            except Exception:
                pass

        p = {
            'waypoints_path': default_csv,
            'pose_topic': '/ego_racecar/odom',   # car: /pf/pose/odom
            'pose_msg_type': 'odom',
            'drive_topic': '/drive',
            'frame_id': 'map',
            'wheelbase': 0.33,
            'max_steer': 0.36,
            'lookahead_min': 0.4,
            'lookahead_gain': 0.1,
            'lookahead_max': 2.0,
            'use_csv_speed': True,   # follow the optimiser's v column
            'v_max': 4.0,            # fallback profile / hard cap on the command
            'v_min': 0.8,
            'a_lat': 4.0,
            'a_decel': 3.0,
            'speed_scale': 0.4,      # ramp up 0.4 -> 0.6 -> 0.8 -> 1.0 over test sessions
            'slip_compensation': 0.0,
            'steer_slowdown': False,
            'track_csv': '',         # optional: centre line with widths -> wall monitor
            'car_half_width': 0.155,
            'lap_log_path': os.path.expanduser('~/raceline_laps.csv'),
            'publish_markers': True,
        }
        for k, v in p.items():
            self.declare_parameter(k, v)
            setattr(self, k, self.get_parameter(k).value)

        if not self.waypoints_path or not os.path.isfile(self.waypoints_path):
            raise RuntimeError('waypoints_path not found: "%s"' % self.waypoints_path)
        self.wp, v_csv = load_waypoints(self.waypoints_path)
        if self.use_csv_speed and v_csv is not None:
            self.v_ref = v_csv
            src = 'optimiser (csv)'
        else:
            self.v_ref = speed_profile(self.wp, self.v_max, self.v_min, self.a_lat, self.a_decel)
            src = 'Lab-5 curvature profile'
        self.kappa = path_curvature(self.wp)
        self.get_logger().info('Loaded %d waypoints from %s; speed from %s (%.2f..%.2f m/s) x %.2f' % (
            len(self.wp), self.waypoints_path, src, self.v_ref.min(), self.v_ref.max(), self.speed_scale))

        self.track = None
        if self.track_csv and os.path.isfile(self.track_csv):
            self.track = load_track(self.track_csv)
            self.get_logger().info('Wall monitor on: %s' % self.track_csv)
        self.track_hint = None
        self.lap_violations = 0
        self.lap_min_clear = float('inf')
        self.lap_count = 0

        self.drive_pub = self.create_publisher(AckermannDriveStamped, self.drive_topic, 10)
        self.marker_pub = self.create_publisher(MarkerArray, '/raceline/waypoints', 1)
        self.target_pub = self.create_publisher(Marker, '/raceline/target', 10)
        msg_type = Odometry if self.pose_msg_type == 'odom' else PoseStamped
        self.create_subscription(msg_type, self.pose_topic, self.pose_callback, 10)
        if self.publish_markers:
            self.create_timer(1.0, self.publish_waypoint_markers)

        self.last_idx = None
        self.speed = 0.0
        self.lap_start = None
        self.prev_idx = None
        self.prev_violating = False

    # ------------------------------------------------------------ callbacks
    def pose_callback(self, msg):
        if isinstance(msg, Odometry):
            pose = msg.pose.pose
            self.speed = abs(msg.twist.twist.linear.x)
        else:
            pose = msg.pose
        x, y = pose.position.x, pose.position.y
        yaw = yaw_from_quat(pose.orientation)
        if not isinstance(msg, Odometry):
            self.speed = self._last_cmd_speed()

        idx = closest_index(self.wp, x, y, self.last_idx)
        self.last_idx = idx
        self._lap_timer(idx)
        if self.track is not None:
            self._wall_monitor(x, y)

        L = float(np.clip(self.lookahead_min + self.lookahead_gain * self.speed,
                          self.lookahead_min, self.lookahead_max))
        goal, goal_idx = lookahead_point(self.wp, x, y, idx, L)

        ff = 0.0
        if self.slip_compensation > 0.0:
            k = 0.5 * (self.kappa[idx] + self.kappa[goal_idx])
            beta, d_us = slip_feedforward(k, self.speed)
            yaw += self.slip_compensation * beta
            ff = self.slip_compensation * d_us
        _, y_c = to_vehicle_frame(goal, x, y, yaw)
        steer, _ = pure_pursuit_steer(y_c, L, self.wheelbase, self.max_steer, ff)

        v = float(min(self.v_ref[idx], self.v_ref[goal_idx])) * self.speed_scale
        if self.steer_slowdown:
            v *= 1.0 - 0.5 * min(abs(steer) / self.max_steer, 1.0)
        v = min(max(v, self.v_min * self.speed_scale), self.v_max)

        out = AckermannDriveStamped()
        out.header.stamp = self.get_clock().now().to_msg()
        out.header.frame_id = 'base_link'
        out.drive.steering_angle = steer
        out.drive.speed = v
        self.drive_pub.publish(out)
        self._cmd_speed = v

        if self.publish_markers:
            self.publish_target_marker(goal)

    def _last_cmd_speed(self):
        return getattr(self, '_cmd_speed', 0.0)

    def _wall_monitor(self, x, y):
        xy, wr, wl = self.track
        n, i = lateral_offset(xy, x, y, self.track_hint)
        self.track_hint = i
        clear = min(wl[i] - n, n + wr[i]) - self.car_half_width
        self.lap_min_clear = min(self.lap_min_clear, clear)
        violating = clear < 0.0
        if violating and not self.prev_violating:
            self.lap_violations += 1
            self.get_logger().warn('BOUNDARY VIOLATION at (%.2f, %.2f), clearance %.2f m' % (x, y, clear))
        self.prev_violating = violating

    def _lap_timer(self, idx):
        now = self.get_clock().now().nanoseconds * 1e-9
        n = len(self.wp)
        if self.prev_idx is not None and self.prev_idx > 0.9 * n and idx < 0.1 * n:
            if self.lap_start is not None:
                self.lap_count += 1
                lap = now - self.lap_start
                self.get_logger().info('Lap %d: %.3f s  violations %d  min clearance %.2f m  (speed_scale %.2f)' % (
                    self.lap_count, lap, self.lap_violations, self.lap_min_clear, self.speed_scale))
                self._log_lap(lap)
            self.lap_start = now
            self.lap_violations = 0
            self.lap_min_clear = float('inf')
        if self.lap_start is None and self.prev_idx is None:
            self.lap_start = now
        self.prev_idx = idx

    def _log_lap(self, lap):
        if not self.lap_log_path:
            return
        new = not os.path.isfile(self.lap_log_path)
        with open(self.lap_log_path, 'a') as f:
            if new:
                f.write('wall_time,waypoints,speed_scale,lap,lap_time_s,violations,min_clearance_m\n')
            f.write('%.3f,%s,%.3f,%d,%.4f,%d,%.3f\n' % (time.time(), os.path.basename(self.waypoints_path),
                                                       self.speed_scale, self.lap_count, lap,
                                                       self.lap_violations,
                                                       self.lap_min_clear if self.track is not None else float('nan')))

    # ------------------------------------------------------------ RViz
    def publish_waypoint_markers(self):
        stamp = self.get_clock().now().to_msg()
        line = Marker()
        line.header.frame_id = self.frame_id
        line.header.stamp = stamp
        line.ns, line.id = 'path', 0
        line.type, line.action = Marker.LINE_STRIP, Marker.ADD
        line.scale.x = 0.04
        line.color.r, line.color.g, line.color.b, line.color.a = 1.0, 0.4, 0.1, 1.0
        line.pose.orientation.w = 1.0
        for (wx, wy) in self.wp:
            line.points.append(Point(x=float(wx), y=float(wy), z=0.0))
        line.points.append(line.points[0])
        arr = MarkerArray()
        arr.markers = [line]
        self.marker_pub.publish(arr)

    def publish_target_marker(self, goal):
        m = Marker()
        m.header.frame_id = self.frame_id
        m.header.stamp = self.get_clock().now().to_msg()
        m.ns, m.id = 'target', 0
        m.type, m.action = Marker.SPHERE, Marker.ADD
        m.pose.position.x, m.pose.position.y = float(goal[0]), float(goal[1])
        m.pose.orientation.w = 1.0
        m.scale.x = m.scale.y = m.scale.z = 0.25
        m.color.r, m.color.g, m.color.b, m.color.a = 1.0, 0.1, 0.1, 1.0
        self.target_pub.publish(m)


def main(args=None):
    rclpy.init(args=args)
    node = RacelineFollower()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    stop = AckermannDriveStamped()
    for _ in range(5):
        node.drive_pub.publish(stop)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
