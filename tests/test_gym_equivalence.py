"""Checks sim/vehicle.py against the original f1tenth_gym dynamic_models.py (if available).
Usage: python tests/test_gym_equivalence.py /path/to/f1tenth_gym/gym/f110_gym/envs
"""
import sys, types, importlib.util, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sim import vehicle as V

envs = sys.argv[1] if len(sys.argv) > 1 else os.environ.get('F1TENTH_GYM_ENVS', os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'f1tenth_gym', 'gym', 'f110_gym', 'envs'))
try:
    import numba  # noqa
except ImportError:   # run the original without numba
    nb = types.ModuleType('numba'); nb.njit = lambda *a, **k: (a[0] if a and callable(a[0]) else (lambda f: f))
    sys.modules['numba'] = nb
spec = importlib.util.spec_from_file_location('dm', os.path.join(envs, 'dynamic_models.py'))
dm = importlib.util.module_from_spec(spec); spec.loader.exec_module(dm)
gp = (1.0489, 4.718, 5.4562, 0.15875, 0.17145, 0.074, 3.74, 0.04712, -0.4189, 0.4189, -3.2, 3.2, 7.319, 9.51, -5.0, 20.0)
rng = np.random.default_rng(0)
worst = 0.0
for _ in range(20000):
    x = np.array([rng.uniform(-5, 5), rng.uniform(-5, 5), rng.uniform(-0.45, 0.45), rng.uniform(-1, 12),
                  rng.uniform(0, 6.28), rng.uniform(-6, 6), rng.uniform(-0.4, 0.4)])
    u = np.array([rng.uniform(-5, 5), rng.uniform(-12, 12)])
    a = np.array(dm.vehicle_dynamics_st(x, u, *gp))
    b = np.array(V.vehicle_dynamics_st(list(x), list(u), *gp))
    worst = max(worst, float(np.max(np.abs(a - b))))
    sp, st, cs, cst = rng.uniform(-2, 12), rng.uniform(-0.5, 0.5), rng.uniform(-2, 12), rng.uniform(-0.5, 0.5)
    pa = np.array(dm.pid(sp, st, cs, cst, 3.2, 9.51, 20.0, -5.0)); pb = np.array(V.pid(sp, st, cs, cst, 3.2, 9.51, 20.0, -5.0))
    worst = max(worst, float(np.max(np.abs(pa - pb))))
print(f'max abs difference vs f1tenth_gym over 20000 random states: {worst:.3e}')
assert worst < 1e-9
print('PASS')
