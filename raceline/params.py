"""Parameter loading: nested YAML -> attribute access, with 'section.key=value' overrides."""
import copy
import os
import yaml

G = 9.81
DEFAULT_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'config', 'params.yaml')


class P(dict):
    """dict with attribute access (recursive)."""
    def __getattr__(self, k):
        try:
            v = self[k]
        except KeyError as e:
            raise AttributeError(k) from e
        return P(v) if isinstance(v, dict) and not isinstance(v, P) else v

    def __setattr__(self, k, v):
        self[k] = v


def _wrap(d):
    return P({k: _wrap(v) if isinstance(v, dict) else v for k, v in d.items()})


def load_params(path=None, overrides=None):
    with open(path or DEFAULT_PATH) as f:
        p = _wrap(yaml.safe_load(f))
    if overrides:
        p = with_overrides(p, overrides)
    return p


def with_overrides(p, overrides):
    """overrides: dict like {'limits.mu': 0.7} or list of 'limits.mu=0.7' strings."""
    p = _wrap(copy.deepcopy(dict(p)))
    if isinstance(overrides, (list, tuple)):
        kv = {}
        for s in overrides:
            k, v = s.split('=', 1)
            kv[k.strip()] = yaml.safe_load(v)
        overrides = kv
    for key, val in overrides.items():
        sec, k = key.split('.', 1)
        p[sec][k] = val
    return p


def wheelbase(p):
    return p.vehicle.lf + p.vehicle.lr


def kappa_max(p):
    """Tightest curvature the car can steer: tan(delta_max) / L."""
    import math
    return math.tan(p.limits.delta_max) / wheelbase(p)


def a_lat_max(p):
    return p.limits.friction_usage * p.limits.mu * G


def drift_speed(p, radius):
    """Max steady-state speed in a corner of the given radius before the planner's friction budget
    is used up (the 'max speed before drift' knob)."""
    import math
    return math.sqrt(a_lat_max(p) * radius)


def half_car_clearance(p):
    return 0.5 * p.vehicle.width + p.track.wall_margin
