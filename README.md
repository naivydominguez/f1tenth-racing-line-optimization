# Finding the Fastest Racing Line — F1TENTH Team 1 (CS 498, Fall 2026)

Offline racing-line pipeline for our F1TENTH car, built and tested in simulation before deployment:

```
 SLAM / gym map ──► centre line + widths ──► min-curvature QP ──► time-optimal OCP ──► waypoints CSV ──► pure pursuit
 (slam_toolbox)     raceline/map_extract     raceline/min_curv     raceline/min_time    (x, y, v, ...)     ros2/raceline_follower
                          ▲                                                                                 │
 random test tracks ──────┘                             closed-loop check: sim/closed_loop (f1tenth_gym dynamics, 10 laps)
 raceline/track_gen
```

* **Track generator** — race-like random tracks (straights, corners, hairpins, chicanes, variable width),
  exported as gym maps (`_map.png/.yaml`) + centre lines in the f1tenth_racetracks format.
* **Map → centre line** — works on SLAM maps with leaks, alcoves and furniture (tested on our Siebel map).
* **Minimum-curvature line** — own implementation of Heilmeier et al. (2020), iterative QP (CasADi + qpOASES/OSQP).
  Warm start and fallback for the time-optimal solver.
* **Time-optimal line** — CasADi + IPOPT, direct multiple shooting with RK4, curvilinear coordinates,
  dynamic single-track model (the exact f1tenth_gym model) **plus a per-axle friction limit**, steering-rate limit,
  power limit. 2–6 s per solve on a laptop CPU; automatic retries, falls back to min-curvature if IPOPT fails.
* **Speed profile** — forward/backward pass with a friction ellipse (used for the baseline and min-curvature line).
* **Closed-loop simulator** — our Lab-5 pure pursuit on the f1tenth_gym single-track dynamics
  (transcribed and checked identical: `tests/test_gym_equivalence.py`), walls = the occupancy map, lap timing,
  localisation noise, optional saturating tyres (`--tire pacejka`) to test what happens when the floor has less grip.
* **Friction calibration** — circle-test analysis (`tools/calibrate_mu.py`) + ROS 2 logger.
* **ROS 2 package** `ros2/raceline_follower` — Lab-5 pure pursuit + optimiser speed profile, slip feed-forward,
  lap/violation logging. *ROS plumbing is untested (no ROS in our build environment); the control law is the one
  tested in `sim/closed_loop.py`.*

## Install

```bash
python3 -m pip install numpy scipy matplotlib scikit-image networkx pyyaml opencv-python casadi
```
(CasADi ships IPOPT, qpOASES and OSQP.) Python 3.10+.

## Quick start

```bash
# 1. make test tracks (one preset, all presets, or random layouts)
python scripts/generate_tracks.py --preset all --seed 1
python scripts/generate_tracks.py --random 10 --seed 0
python scripts/generate_tracks.py --length 80 --corners 12 --hairpins 2 --chicanes 1 --width 1.5 --seed 4

# 2. (real map) SLAM map -> centre line
python scripts/extract_centerline.py tracks/siebel2/siebel2.yaml --start 0 0 --yaw 3.10 \
       --max-half-width 1.5 --smoothing 0.1 --plot siebel_cl.png

# 3. plan the three lines (writes results/lines/<track>/<method>_waypoints.csv + lines.png)
python scripts/optimize.py tracks/technical_s1/technical_s1_centerline.csv
python scripts/optimize.py tracks/siebel2/siebel2_centerline.csv \
       --set limits.mu=0.7 limits.v_max=4 limits.delta_max=0.36 track.wall_margin=0.3

# 4. closed-loop check (10 laps, f1tenth_gym dynamics, walls from the map)
python scripts/simulate.py tracks/technical_s1/technical_s1_centerline.csv \
       results/lines/technical_s1/mintime_waypoints.csv --map tracks/technical_s1/technical_s1_map.yaml
python scripts/simulate.py ... --tire pacejka --mu-true 0.6     # floor with less grip than planned

# 5. everything used for the standup slides
python scripts/run_benchmark.py            # -> results/benchmark.json (~25 min on 2 cores)
python scripts/make_figures.py             # -> results/figures/*.png
```

Generated maps start at (0, 0) heading +x, so in `f1tenth_gym_ros/config/sim.yaml` set
`map_path: <abs path>/tracks/<name>/<name>_map` and `sx: 0.0, sy: 0.0, stheta: 0.0`.

## Results so far (simulation, `python scripts/run_benchmark.py`)

Closed loop: our pure pursuit on the f1tenth_gym dynamics, walls = the occupancy map, mean of 10 flying laps per line,
wall margin 0.2 m, 2 cm / 0.01 rad localisation noise. 30 tracks (the 9 below + 21 random tracks in `tracks/grid`):
**900 laps, 0 wall contacts.** Grid figure: `results/figures/fig_grid.png`.

| Track | Centre line | Min curvature | Min time | Saved (min time) |
|---|---|---|---|---|
| Technical (gen.) | 10.78 s | 9.65 s | 9.58 s | 11.1 % |
| Hairpins (gen.) | 11.13 s | 10.69 s | 10.74 s | 3.6 % |
| Chicanes (gen.) | 12.75 s | 10.42 s | 10.20 s | 20.0 % |
| Long (gen.) | 20.09 s | 17.39 s | 17.09 s | 14.9 % |
| Narrow (gen.) | 9.04 s | 8.50 s | 8.47 s | 6.3 % |
| Var. width (gen.) | 11.80 s | 10.23 s | 10.20 s | 13.5 % |
| Spielberg (f1tenth_racetracks) | 46.58 s | 43.97 s | 43.32 s | 7.0 % |
| Oschersleben (f1tenth_racetracks) | 38.85 s | 34.65 s | 34.45 s | 11.3 % |
| Siebel (our SLAM map, car-like limits) | 16.52 s | 14.37 s | 13.82 s | 16.3 % |

* Over all 30 tracks the time-optimal line saves 3.6–26.1 % vs the centre line (mean 12.8 %), ≥ 10 % on 20 of 30.
  Min-time vs min-curvature: −0.4 to +3.8 % (mean +1.0 %).
* 20 random tracks (planning only): IPOPT produced a valid line on 20/20 (18 first try), median 2.4 s.
* Solutions are validated: IPOPT sometimes "succeeds" on a kinked line that exploits the discretisation on hairpins
  (heading swinging > 60° within one 0.25 m interval). `mintime.xi_max` bounds the heading error and any line with a
  kink the car cannot steer, or slower than the min-curvature warm start, is rejected → retry → min-curvature fallback.
* Generated tracks are run end-to-end: the centre line is extracted from the rendered map (3.4–3.7 cm mean error).
* Friction mismatch (saturating tyres, true μ = 0.7, friction usage 100 %): planner μ 0.5 → all clean;
  0.6 → min-curvature clean, min-time crashed 3/3; 0.7 → 5 of 6 runs crashed.
* Wall margin (technical track, min-time): at 5 cm pose noise 0.15 m crashed 3/3 runs and 0.20 m 2/3; 0.30 m never;
  ≈ 1.4 % lap time per 5 cm.
* Circle-test calibration on the saturating-tyre sim: estimates 0.48 / 0.66 / 0.82 for true μ 0.5 / 0.7 / 0.9.
* Plain Lab-5 pure pursuit ran 10–20 cm wide in fast corners (body side-slip) and hit the wall; `controller.slip_compensation` fixes it in sim.
* The Oschersleben width file is 23 cm wider than its own map: benchmark B recomputes widths from the map (`widths_from_map`).

## Parameters to tune on the car (`config/params.yaml`)

Every `TUNE` line can also be overridden from the command line: `--set limits.mu=0.65`.

| Parameter | Default (sim) | What it does | How we set it on the car |
|---|---|---|---|
| `limits.mu` | 1.0 | tyre–floor friction; sets every corner speed | circle test: `circle_test_logger.py` → `tools/calibrate_mu.py` |
| `limits.friction_usage` | 0.85 | share of μ the planner uses → **max speed before drift** = √(usage·μ·g·R) | **start 0.7** (sim: 100 % of true μ always slid out, ~71 % never did); +0.05 per clean session |
| `limits.v_max` | 8 m/s | top-speed cap | start 3 m/s, raise with `speed_scale` |
| `limits.a_brake_max` | 8 m/s² | braking decel | straight-line stop: a = v²/(2·stopping distance) |
| `limits.a_accel_max`, `v_switch` | 9.51, 7.32 | motor accel & power limit | speed ramp from a rosbag |
| `limits.delta_max` | 0.4189 rad | max steering → min turn radius L/tan(δ) | our car clips at **0.36** → 0.88 m min radius |
| `limits.delta_rate_max` | 3.2 rad/s | servo rate (OCP constraint) | step command, or keep |
| `track.wall_margin` | 0.20 m | clearance kept beyond half the car width | = PF error + tracking error; **start 0.30** |
| `controller.lookahead_*` | 0.4 + 0.1·v | pure pursuit look-ahead | as in Lab 5; longer if it oscillates |
| `controller.speed_scale` | 1.0 | multiplier on the planned speeds | 0.4 → 0.6 → 0.8 → 1.0 after 5 clean laps each |
| `controller.slip_compensation` | 1.0 | side-slip / understeer feed-forward | start 0, raise if the car runs wide at corner exits |
| `vehicle.mass` etc. | gym values | dynamics in the OCP | weigh the car; keep the rest unless we identify the tyres |

Quick reference — max speed before drift √(usage·μ·g·R) at the car's starting usage 0.7:

| μ \ R | 1 m | 2 m | 4 m |
|---|---|---|---|
| 0.5 | 1.9 m/s | 2.6 m/s | 3.7 m/s |
| 0.7 | 2.2 m/s | 3.1 m/s | 4.4 m/s |
| 1.0 | 2.6 m/s | 3.7 m/s | 5.2 m/s |

## Deploying (ROS 2)

```bash
cp -r ros2/raceline_follower ~/sim_ws/src/      # or ~/f1tenth_ws/src on the car
cp results/lines/<track>/mintime_waypoints.csv ~/sim_ws/src/raceline_follower/waypoints/
colcon build --packages-select raceline_follower && source install/setup.bash
# sim (gym bridge running):
ros2 launch raceline_follower raceline_launch.py config:=$(ros2 pkg prefix raceline_follower)/share/raceline_follower/config/sim_params.yaml speed_scale:=1.0
# car (bringup + particle filter running, map = the one the line was planned on):
ros2 launch raceline_follower raceline_launch.py speed_scale:=0.4 track_csv:=/path/to/<track>_centerline.csv
# circle test (IMU yaw rate; NOT /odom yaw rate, which vesc_to_odom computes from the steering command)
ros2 run raceline_follower circle_test_logger.py --ros-args -p yaw_source:=imu -p imu_topic:=/sensors/imu/raw
```
Lap times, boundary violations and min clearance per lap are appended to `~/raceline_laps.csv`.

## Layout

```
config/params.yaml          all tunable parameters (documented)
raceline/                   track_gen, map_io, map_extract, min_curv, min_time, speed_profile, pipeline, calibration
sim/                        vehicle.py (f1tenth_gym ST model + saturating-tyre variant), closed_loop.py
scripts/                    CLI: generate_tracks, extract_centerline, optimize, simulate, run_benchmark, make_figures
tools/                      calibrate_mu.py, sim_circle_test.py
ros2/raceline_follower/     ROS 2 node + circle-test logger + launch/config
tests/                      gym-equivalence test, smoke tests
tracks/                     generated tracks (+ siebel2 SLAM map from Lab 5)
results/                    benchmark.json, figures/, lines/<track>/<method>_waypoints.csv
```

## Credits / licences

* f1tenth_gym single-track model and controller transcribed from github.com/f1tenth/f1tenth_gym (MIT).
* Methods: Heilmeier et al. 2020 (min curvature), Christ et al. 2019/2021 (time-optimal), Diehl et al. 2006
  (multiple shooting), CasADi (LGPL), IPOPT (EPL). No TUM code is vendored; the f1tenth_racetracks maps
  (GPL-3.0) are referenced, not included — clone github.com/f1tenth/f1tenth_racetracks next to this repo or set
  `F1TENTH_RACETRACKS=/path/to/f1tenth_racetracks` for benchmark B. `tests/test_gym_equivalence.py` likewise looks for
  github.com/f1tenth/f1tenth_gym next to this repo (or `F1TENTH_GYM_ENVS=.../gym/f110_gym/envs`).
* AI use: built with Claude (Anthropic); disclose per course policy.
