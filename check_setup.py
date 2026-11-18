"""
check_setup.py - re-checks every fact the project relies on, in ~30 seconds.
Run it with the simulator open (menu or any track):

    python check_setup.py [--scene NAME]

It loads the track itself, re-measures its geometry (logs/track_<scene>.json)
and prints PASS/FAIL with the measured numbers. See SETUP_NOTES.md for why
each one matters.
"""
import argparse
import os

import numpy as np

from config import CAMERA, HOST, PORT, SCENE
from simlink import SimLink, decode_image
from track import load_track

parser = argparse.ArgumentParser()
parser.add_argument("--scene", default=SCENE)
scene = parser.parse_args().scene
failures = 0


def report(ok, name, detail):
    global failures
    failures += not ok
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")


def frames(link, n, steering=0.0, throttle=0.0, brake=0.0, steering_fn=None):
    """Answer the next n telemetry frames with one command each; return them."""
    got = []
    while len(got) < n:
        m = link.wait_for("telemetry", 2)
        if m is None:
            raise RuntimeError("no telemetry for 2 s")
        st = steering_fn(len(got)) if steering_fn else steering
        link.control(st, throttle, brake)
        m["sent_steering"] = st
        got.append(m)
    return got


os.makedirs("logs", exist_ok=True)
link = SimLink(HOST, PORT)

# 1. handshake + camera: configs sent after car_loaded, image size verified
first = link.start(scene, CAMERA)
shape = decode_image(first).shape
report(shape == (CAMERA["img_h"], CAMERA["img_w"], 3), "1 camera", f"{scene}: image {shape}")

# 2. timing: 10 s braked, steering flips every 20 frames
link.reset_car()
fr = frames(link, 200, brake=1.0, steering_fn=lambda i: 0.5 if (i // 20) % 2 else -0.5)
dt = np.diff([m["time"] for m in fr])
delays = []
for i in range(1, len(fr)):
    if fr[i]["sent_steering"] != fr[i - 1]["sent_steering"]:
        j = next((j for j in range(i, len(fr)) if abs(fr[j]["steering_angle"] - fr[i]["sent_steering"]) < 1e-3), None)
        if j is not None:
            delays.append(j - i)
report(abs(np.median(dt) - 0.05) < 1e-3 and (dt > 0.075).sum() == 0, "2 frame rate",
       f"step median {np.median(dt):.4f}s, gaps {(dt > 0.075).sum()}")
report(set(delays) == {1}, "3 command delay", f"frames until applied: {sorted(set(delays))}")

# 4. track geometry (re-measured every run) and our own cte, on nodes the
#    calibration did not use, centred and 0.12 pos units to the right
track = load_track(link, scene, first["totalNodes"], refresh=True)
errors = []
for node in (track.n // 7, track.n // 2 + 3, track.n - 5):
    for offset_pos in (0.0, 0.12):
        link.send(track.pose_at(node, offset_pos * track.scale))
        m = frames(link, 10, brake=1.0)[-1]
        errors.append(abs(track.locate(m["pos_x"], m["pos_z"])["cte"] / track.scale - offset_pos))
report(max(errors) < 0.02, "4 own cte after teleports",
       f"{track.n} nodes, length {track.length / track.scale:.1f} pos units, scale {track.scale:.3f}; "
       f"worst cte error {max(errors):.4f} pos units")

# 5. sign conventions: same start, steering +0.5 vs -0.5 for 1 s
result = {}
for st in (0.5, -0.5):
    link.send(track.pose_at(track.n // 3))
    frames(link, 10, brake=1.0)
    m = frames(link, 20, steering=st, throttle=0.3)[-1]
    result[st] = (m["gyro_y"], track.locate(m["pos_x"], m["pos_z"])["cte"])
ok = result[0.5][0] > 0 > result[-0.5][0] and result[0.5][1] > result[-0.5][1]
report(ok, "5 signs", f"steer +0.5 -> gyro_y {result[0.5][0]:+.2f}, cte {result[0.5][1]:+.2f}; "
       f"steer -0.5 -> gyro_y {result[-0.5][0]:+.2f}, cte {result[-0.5][1]:+.2f} "
       f"(+ steering must give gyro_y > 0 and the larger cte)")

link.reset_car()
frames(link, 5, brake=1.0)
link.close()
print("ALL CHECKS PASSED" if failures == 0 else f"{failures} CHECK(S) FAILED")
