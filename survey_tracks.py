"""
survey_tracks.py - drive one lap on several tracks and summarise.

    python survey_tracks.py [--model models/NAME.pt] [--laps N] [scene ...]

Without --model the expert drives; with it the CNN drives (the expert only
computes labels). Default tracks: every track with a closed loop and no
obstacles on the path. Logs: logs/survey_<model or expert>_<scene>.jsonl.
A failure on one track is reported, not fatal.
"""
import argparse
import os
import traceback

from drive import drive

ALL = ["roboracingleague_1", "mini_monaco", "circuit_launch", "warren", "generated_track",
       "mountain_track", "waveshare", "warehouse"]

p = argparse.ArgumentParser()
p.add_argument("scenes", nargs="*")
p.add_argument("--model")
p.add_argument("--laps", type=int, default=1)
p.add_argument("--speed-scale", type=float, default=1.0)
a = p.parse_args()
who = os.path.splitext(os.path.basename(a.model))[0] if a.model else "expert"
if a.speed_scale != 1.0:
    who += f"_x{a.speed_scale:g}"

results = {}
for scene in a.scenes or ALL:
    print(f"===== {scene}")
    try:
        laps, events = drive(f"survey_{who}_{scene}", scene=scene, laps=a.laps, max_seconds=40 * a.laps,
                             model=a.model, speed_scale=a.speed_scale)
        results[scene] = (laps[-1] if laps else "no full lap in time",
                          sum(e["type"] in ("off_road", "stuck") for e in events))
    except Exception as e:
        traceback.print_exc(limit=1)
        results[scene] = (f"ERROR: {e}", None)

print(f"\n===== summary ({who})")
for scene, (lap, interventions) in results.items():
    if isinstance(lap, dict):
        print(f"{scene:20s} lap {lap['seconds']:6.2f}s  mean|cte| {lap['mean_abs_cte_pos']:.3f}  "
              f"max|cte| {lap['max_abs_cte_pos']:.3f}  interventions {interventions}")
    else:
        print(f"{scene:20s} {lap}  interventions {interventions}")
