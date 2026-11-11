"""Vehicle models for closed-loop testing without ROS.

* 'gym'     : the f1tenth_gym single-track model, transcribed 1:1 from
              f1tenth_gym/gym/f110_gym/envs/dynamic_models.py (MIT License,
              Copyright (c) 2020 Joseph Auckley, Matthew O'Kelly, Aman Sinha, Hongrui Zheng),
              including its acceleration/steering constraints, its speed/steer 'pid', the 2-step
              steering delay buffer and RK4 at dt = 0.01 s (gym defaults). tests/test_gym_equivalence.py
              checks this transcription against the original functions.
              NOTE: its tyres are linear (no saturation) -> the gym car never slides/drifts.
* 'pacejka' : same chassis, but saturating tyres (simplified Pacejka with a friction ellipse,
              peak force = mu_true * Fz). Exceeding the real grip makes the car slide -> used to test
              what happens when the planner's mu is wrong.
"""
import math

G = 9.81


# ------------------------------------------------------------------------- f1tenth_gym (MIT)
def accl_constraints(vel, accl, v_switch, a_max, v_min, v_max):
    if vel > v_switch:
        pos_limit = a_max * v_switch / vel
    else:
        pos_limit = a_max
    if (vel <= v_min and accl <= 0) or (vel >= v_max and accl >= 0):
        accl = 0.
    elif accl <= -a_max:
        accl = -a_max
    elif accl >= pos_limit:
        accl = pos_limit
    return accl


def steering_constraint(steering_angle, steering_velocity, s_min, s_max, sv_min, sv_max):
    if (steering_angle <= s_min and steering_velocity <= 0) or (steering_angle >= s_max and steering_velocity >= 0):
        steering_velocity = 0.
    elif steering_velocity <= sv_min:
        steering_velocity = sv_min
    elif steering_velocity >= sv_max:
        steering_velocity = sv_max
    return steering_velocity


def vehicle_dynamics_ks(x, u_init, mu, C_Sf, C_Sr, lf, lr, h, m, I, s_min, s_max, sv_min, sv_max,
                        v_switch, a_max, v_min, v_max):
    lwb = lf + lr
    u0 = steering_constraint(x[2], u_init[0], s_min, s_max, sv_min, sv_max)
    u1 = accl_constraints(x[3], u_init[1], v_switch, a_max, v_min, v_max)
    return [x[3] * math.cos(x[4]), x[3] * math.sin(x[4]), u0, u1, x[3] / lwb * math.tan(x[2])]


def vehicle_dynamics_st(x, u_init, mu, C_Sf, C_Sr, lf, lr, h, m, I, s_min, s_max, sv_min, sv_max,
                        v_switch, a_max, v_min, v_max):
    g = 9.81
    u0 = steering_constraint(x[2], u_init[0], s_min, s_max, sv_min, sv_max)
    u1 = accl_constraints(x[3], u_init[1], v_switch, a_max, v_min, v_max)
    if abs(x[3]) < 0.5:
        lwb = lf + lr
        f_ks = vehicle_dynamics_ks(x[0:5], [u0, u1], mu, C_Sf, C_Sr, lf, lr, h, m, I, s_min, s_max,
                                   sv_min, sv_max, v_switch, a_max, v_min, v_max)
        return f_ks + [u1 / lwb * math.tan(x[2]) + x[3] / (lwb * math.cos(x[2]) ** 2) * u0, 0.0]
    return [x[3] * math.cos(x[6] + x[4]),
            x[3] * math.sin(x[6] + x[4]),
            u0,
            u1,
            x[5],
            -mu * m / (x[3] * I * (lr + lf)) * (lf ** 2 * C_Sf * (g * lr - u1 * h) + lr ** 2 * C_Sr * (g * lf + u1 * h)) * x[5]
            + mu * m / (I * (lr + lf)) * (lr * C_Sr * (g * lf + u1 * h) - lf * C_Sf * (g * lr - u1 * h)) * x[6]
            + mu * m / (I * (lr + lf)) * lf * C_Sf * (g * lr - u1 * h) * x[2],
            (mu / (x[3] ** 2 * (lr + lf)) * (C_Sr * (g * lf + u1 * h) * lr - C_Sf * (g * lr - u1 * h) * lf) - 1) * x[5]
            - mu / (x[3] * (lr + lf)) * (C_Sr * (g * lf + u1 * h) + C_Sf * (g * lr - u1 * h)) * x[6]
            + mu / (x[3] * (lr + lf)) * (C_Sf * (g * lr - u1 * h)) * x[2]]


def pid(speed, steer, current_speed, current_steer, max_sv, max_a, max_v, min_v):
    steer_diff = steer - current_steer
    if abs(steer_diff) > 1e-4:
        sv = (steer_diff / abs(steer_diff)) * max_sv
    else:
        sv = 0.0
    vel_diff = speed - current_speed
    if current_speed > 0.:
        if vel_diff > 0:
            kp = 10.0 * max_a / max_v
        else:
            kp = 10.0 * max_a / (-min_v)
    else:
        if vel_diff > 0:
            kp = 2.0 * max_a / max_v
        else:
            kp = 2.0 * max_a / (-min_v)
    return kp * vel_diff, sv


# ------------------------------------------------------------------------- saturating-tyre variant
def vehicle_dynamics_pacejka(x, u_init, gp, mu_true, C_pac=1.5):
    """State x = [x, y, delta, vx, yaw, yaw_rate, vy] (body-frame velocities).
    Same chassis/actuator limits as the gym model; lateral force = D sin(C atan(B alpha)),
    D = sqrt((mu_true Fz)^2 - Fx^2), initial slope = gym cornering stiffness."""
    (mu_s, C_Sf, C_Sr, lf, lr, h, m, I, s_min, s_max, sv_min, sv_max, v_switch, a_max, v_min, v_max) = gp
    L = lf + lr
    u0 = steering_constraint(x[2], u_init[0], s_min, s_max, sv_min, sv_max)
    a = accl_constraints(x[3], u_init[1], v_switch, a_max, v_min, v_max)
    de, vx, psi, om, vy = x[2], x[3], x[4], x[5], x[6]
    if abs(vx) < 0.5:   # kinematic at very low speed (as the gym does); lateral states relax to it
        om_k = vx / L * math.tan(de)
        return [vx * math.cos(psi) - vy * math.sin(psi), vx * math.sin(psi) + vy * math.cos(psi), u0, a,
                om, (om_k - om) / 0.02, (0.0 - vy) / 0.02]
    Fzf = m * (G * lr - a * h) / L
    Fzr = m * (G * lf + a * h) / L
    Fx = m * a
    Fxf = Fx * Fzf / (m * G)
    Fxr = Fx * Fzr / (m * G)
    Fxf = max(-mu_true * Fzf, min(mu_true * Fzf, Fxf))
    Fxr = max(-mu_true * Fzr, min(mu_true * Fzr, Fxr))
    af = de - math.atan2(vy + lf * om, vx)
    ar = -math.atan2(vy - lr * om, vx)
    Df = math.sqrt(max((mu_true * Fzf) ** 2 - Fxf ** 2, 1e-9))
    Dr = math.sqrt(max((mu_true * Fzr) ** 2 - Fxr ** 2, 1e-9))
    Bf = mu_s * C_Sf * Fzf / (Df * C_pac)
    Br = mu_s * C_Sr * Fzr / (Dr * C_pac)
    Fyf = Df * math.sin(C_pac * math.atan(Bf * af))
    Fyr = Dr * math.sin(C_pac * math.atan(Br * ar))
    cd, sd = math.cos(de), math.sin(de)
    dvx = (Fxf * cd - Fyf * sd + Fxr) / m + om * vy
    dvy = (Fxf * sd + Fyf * cd + Fyr) / m - om * vx
    dom = (lf * (Fxf * sd + Fyf * cd) - lr * Fyr) / I
    return [vx * math.cos(psi) - vy * math.sin(psi), vx * math.sin(psi) + vy * math.cos(psi), u0, dvx, om, dom, dvy]


class Car:
    """One simulated car. step(speed_cmd, steer_cmd) advances dt, like f1tenth_gym's RaceCar."""

    def __init__(self, p, tire_model='gym', mu_true=None, dt=0.01, steer_delay=2):
        v = p.vehicle
        lim = p.limits
        # actuator limits = gym defaults (the car can physically do this; the planner may use less)
        self.gp = (v.mu_stiffness, v.C_Sf, v.C_Sr, v.lf, v.lr, v.h_cg, v.mass, v.I_z,
                   -0.4189, 0.4189, -3.2, 3.2, lim.v_switch, lim.a_accel_max, -5.0, 20.0)
        self.tire_model = tire_model
        self.mu_true = mu_true if mu_true is not None else p.sim.mu_true
        self.dt = dt
        self.steer_delay = steer_delay
        self.buf = []
        self.state = [0.0] * 7

    def reset(self, x, y, yaw, v=0.0):
        self.state = [x, y, 0.0, v, yaw, 0.0, 0.0]
        self.buf = []

    @property
    def speed(self):
        if self.tire_model == 'pacejka':
            return math.hypot(self.state[3], self.state[6])
        return self.state[3]

    @property
    def pose(self):
        s = self.state
        return s[0], s[1], s[4]

    def _f(self, x, u):
        if self.tire_model == 'pacejka':
            return vehicle_dynamics_pacejka(x, u, self.gp, self.mu_true)
        return vehicle_dynamics_st(x, u, *self.gp)

    def step(self, speed_cmd, steer_cmd):
        gp = self.gp
        # steering delay buffer (gym: buffer of size 2)
        if len(self.buf) < self.steer_delay:
            steer = 0.0
            self.buf.insert(0, steer_cmd)
        else:
            steer = self.buf.pop()
            self.buf.insert(0, steer_cmd)
        x = self.state
        accl, sv = pid(speed_cmd, steer, x[3], x[2], gp[11], gp[13], gp[15], gp[14])
        u = (sv, accl)
        dt = self.dt
        k1 = self._f(x, u)
        k2 = self._f([a + dt / 2 * b for a, b in zip(x, k1)], u)
        k3 = self._f([a + dt / 2 * b for a, b in zip(x, k2)], u)
        k4 = self._f([a + dt * b for a, b in zip(x, k3)], u)
        self.state = [a + dt / 6 * (b + 2 * c + 2 * d + e) for a, b, c, d, e in zip(x, k1, k2, k3, k4)]
        if self.state[4] > 2 * math.pi:
            self.state[4] -= 2 * math.pi
        elif self.state[4] < 0:
            self.state[4] += 2 * math.pi
