"""
dataset.py - turn recorded runs into clean training arrays.

Cleaning rules (each one counted and reported by `python dataset.py`):
  start      first 10 frames of a run (car settling after placement)
  intervened an off_road/stuck event, 5 frames before and 15 after
             (the car is being teleported back: images don't match labels)
  collision  hit != "none", 5 frames before and after
  standstill speed < 0.1 (nothing to learn about driving)
and whole tracks in SKIP_SCENES (see below).
Everything else is kept: the expert's runs are clean by construction, and
frames during steering pushes are KEPT on purpose (their labels show how to
recover).

Images: crop the top CROP_TOP rows (sky, walls: different on every track),
then halve the size: 120x160 -> 40x80 RGB.

Split (python dataset.py; with --all nothing is held out, for the race model):
  test  = TEST_TRACKS (never trained on: like the teacher's unknown track)
  val   = on each training track, the stretch of track from 0 to 15% of the
          lap (curves the model never trained on), all runs
  train = everything else
"""
import json
import os
import sys

import numpy as np
from PIL import Image

from record_all import TEST_TRACKS
from track import Track

CROP_TOP = 40          # rows removed from the top of the 120-row image
SIZE = (80, 40)        # (width, height) after resizing
MAX_SPEED = 5.0        # pos/s, scales speeds to about 0..1
VAL_FRACTION = 0.15
CACHE = "logs/dataset.npz"            # test tracks held out
CACHE_ALL = "logs/dataset_all.npz"    # every track trained on (final race model)
# generated_road shows NO road (plain sand to the horizon, logs/generated_road_samples.png):
# its labels cannot be learned from the images. It was 30% of the first dataset.
SKIP_SCENES = {"generated_road"}


def preprocess(image):
    """PIL image or HxWx3 uint8 array -> 3x40x80 uint8 array."""
    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)
    image = image.crop((0, CROP_TOP, image.width, image.height)).resize(SIZE, Image.BILINEAR)
    return np.array(image, dtype=np.uint8).transpose(2, 0, 1)   # a writable copy (torch needs it)


def clean(rows):
    """rows of one run -> (kept rows, {rule: frames removed})."""
    n = len(rows)
    reason = [None] * n

    def mark(lo, hi, why):
        for i in range(max(0, lo), min(n, hi)):
            reason[i] = reason[i] or why

    mark(0, 10, "start")
    for i, r in enumerate(rows):
        if r.get("event"):
            mark(i - 5, i + 16, "intervened")
        if r.get("hit", "none") != "none":
            mark(i - 5, i + 6, "collision")
        if r["speed"] < 0.1:
            mark(i, i + 1, "standstill")
    counts = {}
    for why in reason:
        if why:
            counts[why] = counts.get(why, 0) + 1
    return [r for r, why in zip(rows, reason) if why is None], counts


def runs():
    """Names of the recorded runs (logs/data_*.jsonl with an image folder)."""
    return sorted(f[:-6] for f in os.listdir("logs")
                  if f.startswith("data_") and f.endswith(".jsonl") and os.path.isdir(f"logs/{f[:-6]}")
                  and not any(f"_{s}_" in f for s in SKIP_SCENES))


def build(run_names=None, verbose=True, all_tracks=False):
    """
    Load, clean, preprocess and split. Returns dict of arrays, cached in CACHE.
    all_tracks=True (final race model): the test tracks are split like the
    others (val = first 15% of the lap) and nothing is held out; cached in
    CACHE_ALL.
    """
    holdout = set() if all_tracks else set(TEST_TRACKS)
    images, steering, target, speed, split, scene_of = [], [], [], [], [], []
    report = {}
    for name in run_names or runs():
        rows = [json.loads(l) for l in open(f"logs/{name}.jsonl")]
        scene = rows[0]["scene"]
        lap_length = Track(json.load(open(f"logs/track_{scene}.json"))).length   # my_s units
        kept, counts = clean(rows)
        report[name] = {"frames": len(rows), "kept": len(kept), **counts}
        for r in kept:
            images.append(preprocess(Image.open(f"logs/{name}/{r['frame']:06d}.jpg").convert("RGB")))
            steering.append(r["label_steering"])
            target.append(r["v_target"])
            speed.append(r["speed"])
            scene_of.append(scene)
            if scene in holdout:
                split.append("test")
            elif (r["my_s"] % lap_length) / lap_length < VAL_FRACTION and scene != "generated_road":
                split.append("val")
            else:
                split.append("train")
    data = {"images": np.stack(images), "steering": np.array(steering, np.float32),
            "target": np.array(target, np.float32), "speed": np.array(speed, np.float32),
            "split": np.array(split), "scene": np.array(scene_of)}
    np.savez_compressed(CACHE_ALL if all_tracks else CACHE, **data)
    if verbose:
        for name, r in report.items():
            print(f"{name:40s} {r}")
        for s in ("train", "val", "test"):
            sel = data["split"] == s
            print(f"{s:5s}: {sel.sum():6d} frames, tracks {sorted(set(data['scene'][sel]))}")
    return data


def load(all_tracks=False):
    path = CACHE_ALL if all_tracks else CACHE
    if not os.path.exists(path):
        return build(all_tracks=all_tracks)
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


if __name__ == "__main__":
    # python dataset.py [--all] [run ...]
    args = sys.argv[1:]
    build([a for a in args if a != "--all"] or None, all_tracks="--all" in args)
