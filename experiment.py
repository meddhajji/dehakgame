"""
Run a fixed, scripted command sequence and log every frame.

    python experiment.py NAME [--scene S] [--at NODE[:OFFSET[:YAW_DEG]]] [--images] SEGMENT [SEGMENT ...]

Each SEGMENT is  seconds:steering:throttle[:brake]   e.g.  2:0:0.3
The car is reset (or placed with --at; OFFSET in pos units, > 0 = right of
centre) first. One command is sent per telemetry frame, so the command
logged on a row is the one sent in answer to that frame; it acts from the
NEXT frame on (measured delay: 1 frame).

Each row also gets our own track measures (my_cte in pos units, my_seg,
my_s), which do not suffer from the simulator's activeNode losing track.

Output: logs/NAME.jsonl  (one JSON line per frame, without the image)
        logs/NAME/000123.jpg  (only with --images)
"""
import argparse
import base64
import json
import os
import time

from config import CAMERA, HOST, PORT, SCENE
from simlink import SimLink
from track import load_track


def parse(segment):
    parts = [float(x) for x in segment.split(":")]
    seconds, steering, throttle = parts[:3]
    brake = parts[3] if len(parts) > 3 else 0.0
    return seconds, steering, throttle, brake


def run(name, segments, scene=SCENE, at=None, save_images=False):
    link = SimLink(HOST, PORT)
    first = link.start(scene, CAMERA)
    track = load_track(link, scene, first["totalNodes"])
    os.makedirs("logs", exist_ok=True)
    if save_images:
        os.makedirs(f"logs/{name}", exist_ok=True)

    if at is None:
        link.reset_car()
    else:
        node, offset_pos, yaw = at
        link.send(track.pose_at(node, offset_pos * track.scale, yaw))
    for _ in range(10):        # settle under the hand brake, skipping frames from before the move
        link.wait_for("telemetry", 2)
        link.control(0, 0, 1.0)

    rows, events = [], []
    with open(f"logs/{name}.jsonl", "w") as f:
        for seg_index, (seconds, steering, throttle, brake) in enumerate(map(parse, segments)):
            done = 0
            while done < round(seconds / 0.05):
                m = link.get(2.0)
                if m is None:
                    raise RuntimeError("no telemetry for 2 s: was the track changed or the simulator closed?")
                kind = m.get("msg_type")
                if kind != "telemetry":
                    if kind not in (None, "car_loaded"):
                        events.append({"after_frame": len(rows),
                                       **{k: v for k, v in m.items() if not k.startswith("_")}})
                    continue
                link.control(steering, throttle, brake)
                row = {k: v for k, v in m.items() if k not in ("image", "imageb", "lidar")}
                where = track.locate(m["pos_x"], m["pos_z"])
                row.update(scene=scene, frame=len(rows), segment=seg_index, cmd_steering=steering,
                           cmd_throttle=throttle, cmd_brake=brake,
                           my_cte=where["cte"] / track.scale, my_seg=where["segment"], my_s=where["s"])
                if save_images:
                    with open(f"logs/{name}/{len(rows):06d}.jpg", "wb") as img:
                        img.write(base64.b64decode(m["image"]))
                f.write(json.dumps(row) + "\n")
                rows.append(row)
                done += 1
        for e in events:
            f.write(json.dumps({"event": True, **e}) + "\n")
    link.control(0, 0, 1.0)
    time.sleep(0.2)
    link.close()
    return rows, events


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("name")
    p.add_argument("segments", nargs="+", help="seconds:steering:throttle[:brake]")
    p.add_argument("--scene", default=SCENE)
    p.add_argument("--at", help="NODE[:OFFSET_POS[:YAW_DEG]]")
    p.add_argument("--images", action="store_true")
    a = p.parse_args()
    at = None
    if a.at:
        parts = [float(x) for x in a.at.split(":")]
        at = (int(parts[0]), *(parts[1:] + [0.0, 0.0])[:2])
    rows, events = run(a.name, a.segments, scene=a.scene, at=at, save_images=a.images)
    hits = [r["frame"] for r in rows if r.get("hit", "none") != "none"]
    print(f"{len(rows)} frames -> logs/{a.name}.jsonl; events: {[e['msg_type'] for e in events]}; "
          f"hit frames: {hits[:10]}{'...' if len(hits) > 10 else ''}")
