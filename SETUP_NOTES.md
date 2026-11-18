# Simulator setup and expert driver: what was wrong, what was measured, what to use

Simulator: SDSandbox `donkey_sim.exe` v25.10.06 (`Desktop\DonkeySimWin`), protocol version 2.
Track: `roboracingleague_1`. Every number below was measured, and `python check_setup.py` re-checks the key facts in about 25 s.

The simulator writes its own log to
`C:\Users\meddh\AppData\LocalLow\DonkeyCar\donkey_sim\Player.log`.
That log is how most of the problems were found. Check it whenever something looks strange.

## 0. How to check everything yourself (in this order)
**The scripts choose the track themselves** (`--scene NAME`; default `roboracingleague_1` in `config.py`). Whatever track is open in the simulator is left and the requested one is loaded, so don't pick tracks in the simulator's menu while a script runs.
1. Start `donkey_sim.exe` (the menu is fine). Then run `python check_setup.py --scene mini_monaco`. All 5 lines must say PASS.
2. `python experiment.py look --at 65:0.3 --images 1:0:0:1`. Open `logs/look/000009.jpg`: the right white line should be close to the image centre.
3. `python drive.py demo --scene mini_monaco --laps 2`. Watch the simulator: 2 laps of about 19 s, no teleports.
4. `python plot_track.py demo`, then open `logs/track.png`.
5. `python show_log.py demo 20`: a table of what the car measured and what it was told to do.
6. `python survey_tracks.py`: one lap on every track, with a summary (about 5 minutes).

## 0b. Why the car crashed on mini_monaco before
The scripts used to drive with the centre line saved for `roboracingleague_1`, whatever track was really open. The simulator never says which track is loaded. On `mini_monaco` the car followed the other track's path into the wall, and the off-road rule put it back on that same wrong path, over and over.
Now:
- `SimLink.start` always loads the requested track. After `exit_scene` it must **reconnect**, because leaving a track destroys the server the socket belonged to (Player.log: `JustOne removing instance.SandboxServer`).
- `track.load_track` keeps one geometry file per track (`logs/track_<scene>.json`), measured on first use.

## 1. Problems in the original setup, and their causes

| Symptom | Cause (evidence) | Fix |
|---|---|---|
| Images were 160x120, not the 255x255 requested | Configs were sent before the car existed. Player.log: `No delegates for msg type: cam_config` | Send `cam_config` **after** `car_loaded`, then check the image size (`SimLink.start`) |
| Simulator errors on every connection | `cam_config_b` / `lidar_config` with `"enabled": "false"` crash the handler (`NullReferenceException` in Player.log) | Never send them |
| Commands sent irregularly (0 to ~19 per frame) | `SDClient.send()` keeps **one** pending message: each new one overwrites the previous unsent one. Our code also ran in the network thread, with printing | `simlink.py`: the network thread only reads; we send exactly one command per frame |
| One exception (missing `camera/` folder) killed the run | Exceptions inside `SDClient`'s thread abort the connection | Work is done outside the network thread |
| `cte` jumps to 20–50 while the car sits on the road | The simulator updates `activeNode` by searching **near its previous value** only. After a reset, teleport or wild driving, it measures `cte` against the wrong segment (measured: −25, −34, +52). Also wrong just before node 0 and just before curves | `track.py` computes our own `cte` against **all** segments of the true path (`node_position`). It matches the simulator's `cte` to 3 decimals wherever the simulator is right |
| Messages glued together on the simulator side | Sending `"}\n"` makes the simulator's parser treat it as partial JSON | Send bare JSON, without a newline |
| Car stuck against a cone, never reset | The old reset rule used the unreliable simulator `cte` only; nothing detected "stuck" | `drive.py`: off-road (our `cte`) and stuck (speed ≈ 0 for 1 s) rules, each logged |

## 2. Measured facts

**Timing**
- Telemetry comes every **0.050 s of simulator time**, with 0 gaps. The simulator runs at real-time speed.
- **A command sent after frame N acts in frame N+1**: exactly a 1-frame (50 ms) delay, every time.
- Our processing takes about 0.8 ms per frame.

**Units and geometry**
- `node_position` gives the 147 centre-path nodes. Track length is 391 node units, and nodes are about 2.7 apart.
- **Node units = 8 × telemetry `pos` units.** The simulator's `cte` is in node units. Speed is in `pos` units/s.
- After `reset_car`, the car spawns at `pos` (2.983, −3.244), yaw 270 (facing −x), slightly *before* node 0.
- **Road:** the white lines are about **3.3 node units** from the centre line (images at offsets 0 to 3, `logs/road_width.png`). There is a wall just outside the left line on the top straight.

**Teleport** (`track.Track.pose_at`)
- `set_position` takes node units. The quaternion attached to each node does **not** face along the track, so we compute our own.
- Use `pos_y` 0.6. At 0 the car spawns inside the floor (`hit` = `floor`/`lines`) and spins about 150°.

**Directions**
- **steering > 0 turns right**: yaw increases, `gyro_y` > 0 (same sign as the steering).
- **cte > 0 means the car is right of the centre line**, so steering should be negative to correct.
- Yaw: 0 = +z, 90 = +x, 270 = −x. It wraps at 360.
- ±0.5 steering gives mirror-image results to about 1%, so the simulator is repeatable.
- At steering 0.5 the turning radius is about 1.45 pos units (11.5 node units).

**Throttle and brake** (run on the top straight from node 62)

| throttle | speed after 0.5 / 1 / 1.5 / 2 / 2.5 s | brake=1 stop |
|---|---|---|
| 0.25 | 0.30 / 0.58 / 0.83 / 1.06 / 1.24 | 0.5 s, 0.3 units |
| 0.5 | 0.59 / 1.14 / 1.64 / 2.08 / 2.43 | 0.9 s, 1.1 units |
| 0.75 | 0.89 / 1.71 / 2.43 / 3.05 / 3.49 | 1.25 s, 2.2 units |
| 1.0 | 1.18 / 2.26 / 3.17 / 3.85 / 4.28 | ran off the straight before stopping |

- **Leaving the road gives no `hit`** on this track. Only our own `cte` detects it.
- **Grip:** sideways acceleration (`gyro_y` × `speed`) up to about 5.7 pos/s² gives almost no sliding (|`vel_x`| < 0.03). At about 8.6 the car slides (|`vel_x`| up to 1.4) and leaves the road.
- `collision_with_starting_line` has `starting_line_index` 0 (bottom straight) **and** 1 (top straight). Laps are counted with our own track progress instead.

**Other**
- `hit` is reported only on the frame of contact. Seen values: `cone`, `floor`, `lines`.
- `racer_info` is ignored by the local simulator (`No delegates`). It probably matters only on a race server.
- `{}` messages about every 3 s are keep-alives.
- **`step_mode` synchronous froze this simulator build until a restart.** Don't use it.

## 3. The expert driver (`expert.py`, run by `drive.py`)
- **Steering: pure pursuit.** Aim at a point on the centre path 6 + 1.5 × speed node units ahead, and steer along the arc that reaches it. It accounts for the 1-frame delay by predicting where the car will be. It sees curves coming, unlike reacting to `cte`.
- **Speed: planned from the track.** The allowed speed at each node is √(grip ÷ curvature), with grip 5.5 (below the measured sliding limit). The plan is then lowered before each curve, so the car can brake in time at 2.0 pos/s². Throttle and brake follow the plan.
- **Safety rules:** off road (|our `cte`| > 3.3) or stuck → the car is put back on the centre line at the same place. Every rule is logged, never silent.

| setting | lap time | interventions | max \|cte\| |
|---|---|---|---|
| constant speed 1.0 | 40 s | 0 | 1.6 |
| constant speed 2.0 | 21 s | 0 | 1.8 |
| constant speed 3.0 | 18 s | **8** (slides off in curves) | 3.5 |
| **planned, top 4.0, grip 5.5 (default)** | **15.75 s** (laps 2 and 3 identical) | **0** | 1.9 |
| planned, top 5.0, grip 6.5 | 15.65 s | 0 | 2.4 |

Pushing harder barely helps: the straights are too short to reach the top speed, so the default is the safer setting.
This expert uses information the camera-only model will **not** have: the car's position and the true path. That is exactly why its commands are good *labels* for training.

Since then, the expert's settings are expressed in **pos units** (the car's physics units, the same on every track): look-ahead 0.75 + 0.19 s × speed, off-road at |cte| > 0.41. Every track's scale is measured automatically by placing the car on 8 nodes and fitting `node = scale × pos + offset`.

## 4. All tracks (one lap each, `survey_tracks.py`, default settings)

| track | nodes | length (pos) | scale | lap | max \|cte\| (pos) | interventions |
|---|---|---|---|---|---|---|
| roboracingleague_1 | 147 | 48.9 | 8.000 | 16.65 s (15.75 flying) | 0.24 | 0 |
| mini_monaco | 304 | 68.9 | 8.000 | 19.8 s (18.85 flying) | 0.24 | 0 |
| circuit_launch | 139 | 46.6 | 8.000 | 15.05 s | 0.30 | 0 |
| warren | 297 | 49.4 | 8.000 | 13.55 s | 0.11 | 0 |
| generated_track | 108 | 26.9 | 8.001 | 8.9 s | 0.17 | 0 |
| mountain_track | 250 | 64.4 | 8.019 | 17.2 s | 0.18 | 0 |
| waveshare | 24 | 8.3 | 7.998 | 5.55 s | 0.11 | 0 |
| warehouse | 173 | 43.9 | 8.000 | 12.95 s | 0.14 | 0 |
| sparkfun_avc | 502 | 131.0 | 8.000 | 33.3 s | 0.48 | 4: **the centre path goes through barrels** (`hit:Barrel`) |
| generated_road | 1000 | 327.9 | 8.000 | — | — | an open road, not a loop |

- The first lap starts from a standstill. "Flying" = later laps.
- On `circuit_launch` and `mountain_track` the car spawns more than 0.41 off the centre path. `drive.py` now starts it on the path instead.
- `mini_monaco`: the centre path is the yellow centre line; the white edges are about 0.5 pos units away (`logs/road_width_mini_monaco.png`).
- The road width was measured only on `roboracingleague_1` (0.41) and `mini_monaco` (~0.5). The same 0.41 off-road limit is used everywhere.

## 5. Still unknown
- How the car behaves with other cars on the track.
- Obstacles (sparkfun_avc barrels): path-following alone cannot avoid them.

## 6. Files

| File | Purpose |
|---|---|
| `my_client.py` | Teacher's original, unchanged (reference) |
| `config.py` | The one place for host, default scene and camera settings |
| `simlink.py` | Clean connection: reader thread + queue, `start()` (loads the requested track, configures and checks the camera), `control()`, `reset_car()` |
| `track.py` | Track geometry: measures each track (`load_track`), our own `cte`, distance along the track, `pose_at()` for teleporting |
| `check_setup.py` | Re-checks camera, frame rate, delay, geometry and signs on any track (PASS/FAIL) |
| `experiment.py` | Scripted command sequences → `logs/NAME.jsonl`. Options: `--scene`, `--images`, `--at NODE[:OFFSET_POS[:YAW]]` |
| `expert.py` | The automatic driver (pure pursuit + speed plan) |
| `drive.py` | Drives N laps on a track, with the expert or a trained CNN (`--model`), using the safety rules; logs every frame, the expert's labels included (+ images with `--images`) |
| `survey_tracks.py` | One lap (or `--laps N`) on every track with the expert or `--model`, plus a summary table |
| `record_all.py` | Records the training data (expert clean + noisy laps, DAgger rounds), see `ML_PIPELINE.md` |
| `dataset.py` | Cleaning rules, pre-processing, train/val/test split → `logs/dataset*.npz` |
| `model.py` | The CNN (`DriveNet`) and `ModelDriver` used by `drive.py --model` |
| `train.py` | Trains the CNN with augmentation → `models/NAME.pt` |
| `models/` | Trained models: v1, v2, v3 (training tracks only), race1, race2, **race3 (best, all tracks)**, race3_cam (camera only) |
| `show_log.py` | Print a log as a table |
| `plot_track.py` | Plot the track + car paths coloured by \|cte\| → `logs/track.png` |
| `camera/` | Needed only by the teacher's `my_client.py` |
| `logs/` | Recorded runs (`*.jsonl`), per-track geometry (`track_<scene>.json`), road-width images |

Example of a scripted test:
```
python experiment.py my_test --at 62:0.06:10 0.5:0:0:1 3:0:0.3
python show_log.py my_test 5
```
This places the car at node 62, 0.06 pos units right of centre, turned 10°, then runs 0.5 s braked and 3 s straight at throttle 0.3.
