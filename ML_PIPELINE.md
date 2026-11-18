# From expert driver to a CNN that drives from the camera

This follows `SETUP_NOTES.md` (simulator setup + the expert driver). Every number here comes from a log in `logs/`.

## The idea in one paragraph
An **expert** drives perfectly because it knows things a camera cannot: the car's exact position and the true centre line of the track. We record what it *sees* (camera images) together with what it *decides* (steering and target speed). Then we train a **CNN** to make the same decisions **from the image alone** (plus the current speed). This is **behavioural cloning**, a form of supervised learning. On race day, only the CNN drives.

## 1. Recording (`record_all.py`, which uses `drive.py`)
- **Tracks:**
  - **training:** roboracingleague_1, circuit_launch, warren, generated_track, mountain_track, waveshare;
  - **held out, used only to test generalisation:** mini_monaco, warehouse;
  - **excluded:** sparkfun_avc (the path goes through barrels) and generated_road (see section 2).
- **Per track:** 2 clean laps, plus 2 laps **with steering pushes** (`--noise 0.04`). A random steering offset of 0.15–0.5 is sent for 0.1–0.4 s, while the *label* stays the expert's decision. The car drifts, and the labels show how to come back. Without this the model never sees itself off-centre, and one small error grows into a crash.
- **Each frame stores:**
  - the image (160x120 JPG);
  - the labels: expert steering and **target speed** (the speed planned for this place on the track);
  - the command actually sent;
  - telemetry;
  - our own cte;
  - any intervention.

## 2. Cleaning (`dataset.py`)

| rule | why | frames removed (first dataset) |
|---|---|---|
| first 10 frames of each run | the car is still settling after being placed | 10 per run |
| 5 frames before → 15 after an intervention | the car is being teleported back, so image and label don't match | 439 |
| `hit` ≠ none, ±5 frames | collisions | 22 (DAgger runs) |
| speed < 0.1 | nothing to learn while standing still | 0 |
| **the whole track `generated_road`** | it shows **no road at all**, only sand to the horizon (`logs/generated_road_samples.png`). It was 30% of the first dataset | 2305 |

Frames during steering pushes are **kept on purpose**: they are the recovery examples.

- **Pre-processing:** remove the top 40 rows (sky, walls: different on every track), then halve the size, to 40x80 RGB.
- **Split:**
  - **test** = the held-out tracks;
  - **val** = the first 15% of each training track's lap, a stretch of curves the model never trained on;
  - **train** = the rest.

## 3. The model (`model.py`, `train.py`)
- **CNN:** 4 convolutions (24-32-48-64 filters, BatchNorm, ReLU), dropout, then 2 fully connected layers.
  - 383k parameters.
  - **4.4 ms per decision** on this laptop's CPU (the budget is 50 ms per frame).
- **Inputs:** the image, plus the current speed. One image can't show how fast the car is going, and the right steering depends on speed.
- **Outputs:**
  - **steering** (tanh, −1..1);
  - **target speed** (sigmoid × 5 pos/s): "how fast should I be going here", which the image *can* show (straight vs curve ahead).
  - The throttle then follows from target vs current speed, with the same simple rule the expert uses.
- **Loss:** MSE(steering) + MSE(target/5). Adam with a cosine learning-rate schedule, 20 epochs; the epoch with the best validation loss is kept.
- **Augmentation** (training batches only):
  - mirror the image with steering negated;
  - brightness, contrast and colour gain;
  - **colour-channel swaps and grey-scale**, so the network relies on line *shapes*, not colours, which differ on every track;
  - pixel noise.

## 4. DAgger: learning from its own mistakes
Training only on the expert's driving has a known flaw: the model makes small errors and reaches situations the expert never showed it. **DAgger** (Dataset Aggregation) fixes this:
1. The **model** drives.
2. The expert computes what it *would* do in every frame.
3. These frames are added to the dataset, and the model is retrained.

`python record_all.py --dagger models/X.pt ROUND`. The expert never steers during DAgger, except the safety teleports.

## 5. Results (one lap per track, interventions = times the safety rule had to put the car back)

| model | data | roboracing | mini_monaco* | circuit | warren | gen_track | mountain | waveshare | warehouse* | total |
|---|---|---|---|---|---|---|---|---|---|---|
| expert | — | 16.6 s / 0 | 19.9 s / 0 | 15.0 s / 0 | 13.6 s / 0 | 8.9 s / 0 | 17.2 s / 0 | 5.6 s / 0 | 13.0 s / 0 | 0 |
| v1 | expert laps, incl. generated_road | 17.1 / 0 | 22.0 / **0** | 15.9 / 1 | 14.1 / 1 | 10.2 / 7 | 18.0 / 1 | 4.1 / 0 | 15.9 / 13 | 23 |
| v2 | without generated_road, stronger colour aug | 18.0 / 1 | 21.3 / 1 | 16.7 / 2 | 14.5 / 2 | 9.6 / 2 | 19.0 / 5 | 4.2 / 0 | 15.3 / 15 | 28 |
| v3 | v2 + **DAgger round 1** | 17.5 / 0 | 20.8 / **0** | 15.6 / 1 | 13.8 / 0 | 9.3 / 0 | 17.6 / 2 | 4.9 / 0 | 14.9 / 17† | 20 |

\* never trained on (held out). † counted with a symmetric ±0.41 road; with the measured warehouse edges it is 3–4 per lap (see below).

What it shows:
- **DAgger made the biggest difference.** On the training tracks, interventions went from 10 (v1) and 13 (v2) to **3** (v3).
- **A lower offline error does not guarantee better driving.** v2 had lower test errors than v1 but drove slightly worse. Only closed-loop driving tells the truth.
- **Unseen-track generalisation works when the track is laid out like the training tracks:** mini_monaco, never seen, had 2 clean laps, the second in 19.7 s (expert: 18.8 s).
- **…and partly fails when it isn't.** On warehouse the expert's path is the **right lane** (between a yellow dashed centre line and a white edge, `logs/road_width_warehouse.png`). On the training tracks the path often follows the yellow line. The model therefore steers about 0.25 too far left and drifts over the yellow line toward the left edge, especially in right turns. This is a *convention* difference that no amount of colour augmentation fixes.
- **Measuring honestly needed the real road edges.** A single ±0.41 limit called "driving on the warehouse centre line" off-road. `drive.py` now uses measured edges per track (`ROAD`: warehouse −0.85..+0.35, mini_monaco ±0.5, circuit_launch ±0.75, mountain_track −0.65..+0.75).

## 6. The race model (trained on every track)
For the race, the model learns from all 8 tracks:
- noisy laps added on mini_monaco and warehouse;
- DAgger round 2 with v3 on all 8 tracks;
- `python dataset.py --all` and `python train.py race1 --all`.

Result: 13,590 training frames, validation steering MAE 0.084 (baseline 0.256). **2 laps per track** (best lap / interventions over both laps):

| model | roboracing | mini_monaco | circuit | warren | gen_track | mountain | waveshare | warehouse | total (16 laps) |
|---|---|---|---|---|---|---|---|---|---|
| expert (flying lap) | 15.75 | 18.8 | 14.0 | 12.55 | 8.1 | 16.25 | 4.5 | 12.25 | 0 |
| race1 | 16.20 / 0 | 18.95 / 0 | 14.40 / 5 | 12.75 / 2 | 8.40 / 2 | 16.60 / 2 | 5.05 / 0 | 12.55 / 8 | 19 |

race1 is nearly as fast as the expert, but it is less careful than v3 on some tracks. So one more DAgger round followed: race1 drove all 8 tracks with expert labels (dagger3), and the result was retrained as **race2** (17,243 frames, val steering MAE 0.078).

| model | roboracing | mini_monaco | circuit | warren | gen_track | mountain | waveshare | warehouse | total (16 laps) |
|---|---|---|---|---|---|---|---|---|---|
| race2 (±0.41 on circuit/mountain) | 16.00 / 0 | 19.20 / 0 | 14.50 / 2 | 12.80 / 0 | 8.10 / 0 | 16.95 / 3 | 4.10 / 0 | **11.85 / 0** | 5 |
| race2 (measured road edges) | | | **14.45 / 0** | | | 17.00 / 2 | | | **2** |

- The circuit_launch and mountain_track roads are much wider than 0.41: about ±0.75 and −0.65..+0.75 (`logs/road_width_circuit_launch.png`, `logs/road_width_mountain_track.png`). With their measured edges, circuit_launch is clean.
- **mountain_track has one real intervention per lap, at the same corner** (segments 22–29, a right-hander about 3 s into the lap). The model steers about +0.1 where the expert wants +0.5 to +0.7, and keeps a target of 4.0 where the expert plans 3.0. It enters too fast and runs wide.
- Earlier rows (v1–v3, race1) used ±0.41 on those two tracks, so their counts there are pessimistic.

**race3**: one more DAgger round (race2 driving all 8 tracks, `dagger4`) → 21,069 frames, val steering MAE 0.076. Measured road edges, 2 laps per track:

| model | roboracing | mini_monaco | circuit | warren | gen_track | mountain | waveshare | warehouse | total (16 laps) |
|---|---|---|---|---|---|---|---|---|---|
| expert (flying lap) | 15.75 | 18.8 | 14.0 | 12.55 | 8.1 | 16.25 | 4.5 | 12.25 | 0 |
| **race3** | 15.85 / 0 | **18.80** / 0 | 14.35 / 0 | 12.75 / 0 | 8.25 / 0 | 16.70 / 2 | 5.20 / 0 | 11.75 / 0 | **2** |
| race3 at 90 % speed (`--speed-scale 0.9`) | | | | | | 18.05 / 3 (3 laps) | | | |
| **race3_cam** (camera only, `--no-speed`) | 16.25 / 0 | 18.90 / 0 | 14.20 / 0 | 12.60 / 0 | 8.30 / 0 | 16.55 / 2 | 4.75 / 0 | 11.80 / 0 | **2** |

- **race3 matches the expert's lap times** within about 0.5 s on every track, with **zero interventions on 7 of 8 tracks**.
- **The speed input is not needed.** race3_cam was trained on the same data without the speed number (val steering MAE 0.075, vs 0.076 for race3). It drives the same: same lap times within about 0.4 s, and the same 2 interventions at the same mountain corner. The image already shows enough. The car accelerates the same way every time, so where the car is on the track says roughly how fast it is going.
- **The last failure is a perception ambiguity, not speed.** Driving at 90 % speed does not help. On that mountain_track right-hander (`logs/mountain_corner.png`) the camera sees only the yellow centre line: the right edge is out of view, and there is sun glare on the left. The expert's path is the right lane, but "follow the yellow line" looks just as plausible from the image. The model steers −0.14 where the expert steers +0.36. This is the same lane ambiguity as warehouse.
- Possible next steps:
  - over-sample the frames where the model is most wrong (hard-example mining);
  - give the network 2–3 stacked frames, so it sees where it is heading;
  - a wider camera field of view, so both road edges stay visible.

## 7. How to reproduce
```
python record_all.py                                  # expert data, all tracks (~15 min)
python dataset.py                                     # clean + split, held-out mode
python train.py v1                                    # ~4-7 min on CPU
python survey_tracks.py --model models/v1.pt          # one lap per track
python record_all.py --dagger models/v1.pt dagger1    # DAgger round
python dataset.py && python train.py v3               # retrain with DAgger data
python dataset.py --all && python train.py race3 --all   # race model: every track
python drive.py demo --scene mini_monaco --model models/race3.pt --laps 2
```

## 8. Open questions for the teacher / limits
- **Speed input:** race3 uses the car's current speed from telemetry. If only the camera is allowed, `models/race3_cam.pt` (trained with `--no-speed`) drives as well (section 6). The throttle rule still compares the target with the measured speed; a camera-only race would also need that replaced, for example by a fixed throttle per target.
- **Other cars** on the track have never been seen in training.
- **Obstacles** (sparkfun_avc barrels): the expert itself cannot avoid them, so neither can the model.
- **Lane convention:** if the race track asks the car to keep to one lane, as on warehouse, the model may pick the wrong one.
