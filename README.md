# Donkey Car: a CNN that drives from the camera

University ML project (teacher: Réda DEHAK). A car in the Donkey Car simulator (`donkey_sim.exe`, SDSandbox) must do fast, clean laps on tracks chosen by the teacher.

**Approach: behavioural cloning.**
1. An automatic **expert** driver drives perfectly. It knows the car's exact position and the track's centre line.
2. We record what the car's **camera sees**, together with what the expert **decides**.
3. A small **CNN** learns to make the same decisions from the image alone.

**Status:** the best model, `models/race3.pt`, drives 8 tracks at the expert's lap times (within about 0.5 s), with zero mistakes on 7 of them. One corner on `mountain_track` still fails. The details are in [ML_PIPELINE.md](ML_PIPELINE.md).

| Document | What it explains |
|---|---|
| [README.md](README.md) (this file) | Running it, the folder map, conventions, how to contribute |
| [SETUP_NOTES.md](SETUP_NOTES.md) | Simulator problems we found and fixed, measured physics, expert tuning |
| [ML_PIPELINE.md](ML_PIPELINE.md) | Data, cleaning, model, DAgger, all results, open problems |

---

## 1. Run it (about 10 minutes)

1. **Install** Python 3.10+ (tested on 3.14, CPU only), then:
   ```
   pip install -r requirements.txt
   ```
2. **Start the simulator.** Run `donkey_sim.exe` and leave it on its menu. Our code talks to it over TCP port 9091 and loads tracks itself.
3. **Check the setup.** This runs 5 PASS/FAIL tests: camera, frame rate, delay, geometry and signs.
   ```
   python check_setup.py
   ```
4. **Watch the trained CNN drive:**
   ```
   python drive.py demo --scene mini_monaco --model models/race3.pt --laps 3
   ```
5. **Watch the expert drive** (no `--model`):
   ```
   python drive.py demo --scene circuit_launch --laps 2
   ```
6. **Evaluate on all tracks** and print a summary table:
   ```
   python survey_tracks.py --model models/race3.pt --laps 2
   ```

Tracks in this simulator build: `roboracingleague_1`, `mini_monaco`, `circuit_launch`, `warren`, `generated_track`, `mountain_track`, `waveshare`, `warehouse`, `sparkfun_avc`, `generated_road`.

---

## 2. The pipeline

```
expert.py  --drives-->  record_all.py / drive.py  --images + labels-->  logs/data_*/
                                                                          |
                                                    dataset.py (clean, crop, split)
                                                                          |
                                                    train.py  -->  models/NAME.pt
                                                                          |
                         drive.py --model / survey_tracks.py  (the CNN drives, the expert grades)
                                                                          |
                                         record_all.py --dagger  (more data from the CNN's own mistakes)
```

---

## 3. Folder and file map

### Code

| File | Role |
|---|---|
| `my_client.py` | The teacher's original example client, **unchanged**. For reference only: our code doesn't use it |
| `config.py` | Host/port, default track, camera settings (160x120 JPG) |
| `simlink.py` | Talks to the simulator: connect, load a track, configure the camera, send `control(steering, throttle, brake)` |
| `track.py` | Track geometry: our own distance from the centre line (the simulator's `cte` is unreliable), teleporting the car |
| `check_setup.py` | Setup self-test |
| `expert.py` | The expert driver (pure pursuit + speed plan) |
| `drive.py` | Drives N laps (expert, or CNN with `--model`), applies the safety rules, logs every frame |
| `record_all.py` | Records the training data on every track (clean, noisy and DAgger runs) |
| `dataset.py` | Cleaning rules, crop/resize, train/val/test split → `logs/dataset*.npz` |
| `model.py` | The CNN (`DriveNet`) and `ModelDriver` (the CNN as a driver) |
| `train.py` | Training with augmentation → `models/NAME.pt` |
| `survey_tracks.py` | Evaluation: N laps on every track + summary |
| `experiment.py`, `show_log.py`, `plot_track.py` | Tools: scripted tests, print a log, plot driven paths |

### Folders

| Folder | Content | In git? |
|---|---|---|
| `models/` | Trained CNNs, 1.5 MB each. `race3.pt` = best; `race3_cam.pt` = camera only (no speed input); `v1`–`v3`, `race1`, `race2` = earlier versions, kept for comparison | yes |
| `logs/track_<scene>.json` | Measured geometry of each track (centre-line nodes, scale) | yes |
| `logs/*.txt` | Raw output of training and evaluation runs: the source of every number in the docs | yes |
| `logs/*.png` | Figures used in the docs | yes |
| `logs/data_<scene>_<kind>/` + `.jsonl` | **The recorded dataset**: one JPG per frame + one JSON line per frame (labels, telemetry). 29k images, ~170 MB | **no**: see §5 |
| `logs/dataset*.npz` | Pre-processed dataset cache, rebuilt by `dataset.py` | no |
| `logs/survey_*.jsonl`, other `logs/*.jsonl` | Per-frame logs of evaluation/experiment runs | no (re-created by running) |
| `camera/` | Where the teacher's `my_client.py` saves its images. The images you may see there come from one test run of that client. They aren't training data (one is a 0-byte file: the client was stopped while writing it) | only empty |

### The data runs: `clean`, `noisy`, `dagger1`–`dagger4`

Every run is named `data_<track>_<kind>`:

| kind | Who drives | Why |
|---|---|---|
| `clean` | The expert, 2 laps | Shows how to drive well |
| `noisy` | The expert, but random **steering pushes** are added (0.1–0.4 s, 0.15–0.5 steering). The label stays the expert's correct decision | Shows how to **recover** when off-centre. Without it, the CNN has never seen itself off-centre, and one small error grows into a crash |
| `dagger1` … `dagger4` | **The CNN** drives, 2 laps. The expert only writes down what it *would* do in every frame | **DAgger**: the CNN learns from the situations its *own* mistakes create. Round *k* is driven by the model trained after round *k−1* (v1 → dagger1, v3 → dagger2, race1 → dagger3, race2 → dagger4). It was the biggest improvement: 23 → 2 interventions |

`mini_monaco` and `warehouse` have no `dagger1`: they were held out, never trained on, to test unseen tracks. They were only added for the final race models.

**Why ~500–800 images per run?** Runs are fixed in laps, not frames. The camera gives 20 frames/s, so frames ≈ 2 laps × lap time × 20. For example, mini_monaco: 2 × 19.5 s × 20 ≈ 780. `waveshare` is tiny (5 s laps), so it gets 6 laps. `generated_road` is not a loop: it's recorded for 60 s, then excluded, because its images show no road.

---

## 4. Conventions (read before changing anything)

- **Commands**, sent once per frame as `control(steering, throttle, brake)`:
  - `steering` ∈ [−1, 1]: **positive = right, negative = left**. Measured: 0.5 gives a 1.44-unit turn radius. That's why mirror augmentation (flip the image, negate the steering) is valid.
  - `throttle` ∈ [0, 1] and `brake` ∈ [0, 1] are **two separate inputs**, not one signed number. Our code never sends both at once (`throttle_from_target`). So you can read them as one signed value `throttle − brake`, but the simulator receives two fields.
- **Telemetry:** `speed` (units/s), `pos_x/z`, `yaw` (0 = +z, 90 = +x), `gyro_y` (same sign as steering).
- **Our cte:** distance to the centre line in pos units, **positive = right of the path**.
- **Timing:** 20 Hz, and a command acts **one frame later**. The expert compensates by predicting the car's position 0.05 s ahead.
- **There is no PID controller.**
  - Steering uses **pure pursuit**, a geometric rule: aim at a point on the centre line 0.75 + 0.19 × speed units ahead, and steer along the circle arc that reaches it.
  - Speed is **planned** per track position (slow enough in curves for the grip, braking early), then followed by a **proportional controller with feed-forward**: `throttle = clip(0.25·v_target + (v_target − speed))`, `brake = clip(speed − v_target − 0.3)`. There is no integral or derivative term, and none was needed.
- **The CNN outputs** steering and a **target speed**, not a throttle. The throttle comes from the rule above.

---

## 5. Getting the data

The dataset isn't in git: 29k files, too big. Two options:

- **Download** `dehakgame_data.zip` (~170 MB) from the repository's **Releases** page. Unzip it into this folder; it creates `logs/data_*`.
- **Record your own** with the simulator running. The expert part takes ~15 min:
  ```
  python record_all.py                                # clean + noisy runs, all tracks
  python record_all.py --dagger models/race3.pt mine1 # a DAgger round with an existing model
  ```

Then:
```
python dataset.py --all          # clean + split (all tracks)
python train.py mymodel --all    # ~1 h on a 2-core laptop CPU, 20 epochs
python survey_tracks.py --model models/mymodel.pt --laps 2
```
Add `--no-speed` to `train.py` for a camera-only model. Without `--all`, `mini_monaco` and `warehouse` are held out as unseen test tracks.

---

## 6. The evaluation and what to work on

**What we know about the race:**
- 3 clean laps on each of 3 tracks: `circuit_launch`, `mini_monaco`, plus one chosen on the day.
- With the current model, both known tracks are clean (0 interventions in our runs).
- The surprise track can be any track, so the goal is **robustness on all tracks**, not only these two.

**Open problems, good tasks for the team:**
1. **The `mountain_track` corner** (segments 22–29): about 1 intervention per lap. It's a perception ambiguity: only the yellow centre line is visible, and there's glare. Ideas:
   - hard-example mining;
   - 2–3 stacked frames as input;
   - a wider camera FOV.
2. **Unseen-track robustness:** re-test by training without `--all` and driving on the held-out tracks. Ideas:
   - more augmentation (shadows, small rotations/shifts with steering correction);
   - more varied data.
3. **Speed:** the expert is limited to 4.0 units/s and a grip of 5.5. A faster expert means faster labels, but check the safety margin.
4. **Questions for the teacher:**
   - Is telemetry (speed) allowed? If not, use `race3_cam.pt`, and replace the throttle rule that reads `speed`.
   - Will there be other cars or obstacles?
   - What exactly counts as a "perfect" lap?

**How to collaborate:**
- Work on a branch, and open a pull request.
- Never commit `logs/data_*` or `.npz` files (`.gitignore` blocks them).
- Commit new models only when they beat `race3` in `survey_tracks.py`, and put the summary in the PR.
