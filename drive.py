"""
drive.py - drive a track (expert or trained model), with safety rules, and log everything.

    python drive.py NAME [--scene S] [--laps N] [--model models/X.pt]
                         [--speed V] [--grip G] [--noise P] [--images]

Without --model the expert drives. With --model the CNN drives from the camera
image (+ speed), and the expert only computes what it WOULD do (the label).

Per frame it logs telemetry, the expert's decision (the LABEL: label_steering,
label_throttle, label_brake, v_target), the command actually sent (applied_*)
and our own cte/progress.

--noise P : with probability P per frame, start a 0.1-0.4 s "push": a random
            steering offset (0.15-0.5, random side) is added to what is SENT,
            while the label stays the expert's decision. The car drifts off its
            line and the expert's corrections become recovery examples.
            Logged as perturbed=True.

Safety rules (each one logged as an event, never silent):
  * off road  : our cte outside the road -> car put back on the centre line
                (measured edges in ROAD, else |cte| > --off-road)
  * stuck     : speed < 0.05 for 1 s     -> put back (e.g. pressed against a wall)
  * collision : hit != "none"            -> logged
A lap = our own track progress grew by one track length since the start.
"""
import argparse
import base64
import json
import os
import random
import time

from config import CAMERA, HOST, PORT, SCENE
from expert import Expert
from simlink import SimLink, decode_image
from track import load_track


def ModelDriver(path, speed_scale=1.0):
    from model import ModelDriver as _ModelDriver   # torch only imported when a model drives
    return _ModelDriver(path, speed_scale)

STUCK_FRAMES = 20     # 1 s

# Road edges as (left, right) cte limits in pos units, measured from camera images
# at known offsets (logs/road_width*.png). The path is not always the road's centre:
# on warehouse it is the RIGHT lane (yellow centre line at -0.35, left edge -0.85).
# Tracks not listed use +-off_road (0.41 = the white lines on roboracingleague_1).
ROAD = {"warehouse": (-0.85, 0.35), "mini_monaco": (-0.5, 0.5),
        "circuit_launch": (-0.75, 0.75), "mountain_track": (-0.65, 0.75)}


def settle(link, frames=10):
    """Answer `frames` telemetry frames with the hand brake; return the last one."""
    m = None
    for _ in range(frames):
        m = link.wait_for("telemetry", 2)
        link.control(0, 0, 1.0)
    return m


def drive(name, scene=SCENE, laps=2, max_seconds=300, save_images=False, off_road=0.41,
          noise=0.0, seed=0, model=None, speed_scale=1.0, **expert_args):
    rng = random.Random(seed)
    driver = ModelDriver(model, speed_scale) if model else None
    road_left, road_right = ROAD.get(scene, (-off_road, off_road))
    link = SimLink(HOST, PORT)
    first = link.start(scene, CAMERA)
    track = load_track(link, scene, first["totalNodes"])
    print(f"{scene}: {track.n} nodes, length {track.length / track.scale:.1f} pos units, "
          f"scale {track.scale:.3f}")
    expert = Expert(track, **expert_args)
    link.reset_car()
    m = settle(link)
    # some tracks spawn the car off the centre path (circuit_launch, mountain_track:
    # > 0.41 away, an "off road" at frame 0): start from the path instead
    where = track.locate(m["pos_x"], m["pos_z"])
    if abs(where["cte"]) / track.scale > 0.1:
        link.send(track.pose_at(where["segment"]))
        settle(link)
    os.makedirs("logs", exist_ok=True)
    if save_images:
        os.makedirs(f"logs/{name}", exist_ok=True)

    progress, last_s, lap_start_frame, frame = 0.0, None, 0, 0
    stuck, events, lap_stats = 0, [], []
    lap_cte = []
    push_frames, push, cooldown = 0, 0.0, 0
    t_end = time.perf_counter() + max_seconds
    with open(f"logs/{name}.jsonl", "w") as f:
        while len(lap_stats) < laps and time.perf_counter() < t_end:
            m = link.get(2.0)
            if m is None:
                raise RuntimeError("no telemetry for 2 s: was the track changed or the simulator closed?")
            if m.get("msg_type") != "telemetry":
                if m.get("msg_type") not in (None, "car_loaded"):
                    events.append({"frame": frame, "type": m["msg_type"]})
                continue

            label_steering, label_throttle, label_brake, info = expert.act(m)
            steering, throttle, brake = label_steering, label_throttle, label_brake
            model_v_target, think_ms = None, None
            if driver is not None:
                t0 = time.perf_counter()
                steering, throttle, brake, model_v_target = driver.act(decode_image(m), m["speed"])
                think_ms = (time.perf_counter() - t0) * 1000
            if push_frames == 0 and noise > 0 and frame > 20 and rng.random() < noise:
                # 0.3-0.8 for 0.2-0.6 s pushed the car off road 4 times per lap
                push_frames = rng.randint(2, 8)
                push = rng.choice((-1, 1)) * rng.uniform(0.15, 0.5)
            if push_frames > 0:
                push_frames -= 1
                steering = max(-1.0, min(1.0, steering + push))

            cte_pos = info["cte"] / track.scale
            event = None
            cooldown = max(0, cooldown - 1)
            if not road_left <= cte_pos <= road_right and cooldown == 0:
                event = "off_road"
            stuck = stuck + 1 if (m["speed"] < 0.05 and frame > 40) else 0
            if stuck > STUCK_FRAMES:
                event = "stuck"
            if m["hit"] != "none":
                events.append({"frame": frame, "type": f"hit:{m['hit']}"})
            if event:
                events.append({"frame": frame, "type": event})
                link.send(track.pose_at(info["segment"]))   # back on the centre line, same place
                # the move shows in telemetry ~1 frame later: don't count it twice
                stuck, push_frames, cooldown = 0, 0, 3
                steering, throttle, brake = 0.0, 0.0, 1.0
            link.control(steering, throttle, brake)

            # progress along the track (handles the wrap at the start line)
            if last_s is not None:
                ds = info["s"] - last_s
                ds -= track.length * round(ds / track.length)
                progress += ds
            last_s = info["s"]
            lap_cte.append(abs(cte_pos))
            if progress >= track.length * (len(lap_stats) + 1):
                lap_events = [e for e in events if e["frame"] >= lap_start_frame and e["type"] in ("off_road", "stuck")]
                lap_stats.append({"lap": len(lap_stats) + 1,
                                  "seconds": round((frame - lap_start_frame) * 0.05, 2),
                                  "mean_abs_cte_pos": round(sum(lap_cte) / len(lap_cte), 3),
                                  "max_abs_cte_pos": round(max(lap_cte), 3),
                                  "interventions": len(lap_events)})
                print("lap", lap_stats[-1])
                lap_start_frame, lap_cte = frame, []

            row = {k: v for k, v in m.items() if k not in ("image", "imageb", "lidar")}
            row.update(scene=scene, frame=frame, lap=len(lap_stats),
                       label_steering=label_steering, label_throttle=label_throttle,
                       label_brake=label_brake, v_target=info["v_target"],
                       applied_steering=steering, applied_throttle=throttle, applied_brake=brake,
                       perturbed=push_frames > 0 or (steering != label_steering and not event),
                       my_cte=cte_pos, my_s=info["s"], my_seg=info["segment"],
                       progress=progress, event=event, model_v_target=model_v_target, think_ms=think_ms)
            if save_images:
                with open(f"logs/{name}/{frame:06d}.jpg", "wb") as img:
                    img.write(base64.b64decode(m["image"]))
            f.write(json.dumps(row) + "\n")
            frame += 1
    link.control(0, 0, 1.0)
    time.sleep(0.2)
    link.close()
    print("events:", [e for e in events if e["type"] != "collision_with_starting_line"])
    return lap_stats, events


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("name")
    p.add_argument("--scene", default=SCENE)
    p.add_argument("--laps", type=int, default=2)
    # defaults = best measured setting on roboracingleague_1: 15.75 s laps, 0 interventions
    # (grip 6.5 / speed 5 was no faster: the straights are too short to use it)
    p.add_argument("--speed", type=float, default=4.0, help="top speed (pos/s)")
    p.add_argument("--grip", type=float, default=5.5,
                   help="max lateral accel (pos/s^2) used to slow down in curves; 0 = constant speed")
    p.add_argument("--off-road", type=float, default=0.41,
                   help="|cte| (pos units) that counts as off road; 0.41 = white lines on roboracingleague_1")
    p.add_argument("--noise", type=float, default=0.0, help="probability per frame of a steering push")
    p.add_argument("--images", action="store_true")
    p.add_argument("--model", help="models/NAME.pt: the CNN drives; the expert only labels")
    p.add_argument("--speed-scale", type=float, default=1.0,
                   help="model only: drive at this fraction of its predicted target speed")
    a = p.parse_args()
    drive(a.name, scene=a.scene, laps=a.laps, save_images=a.images, off_road=a.off_road,
          noise=a.noise, model=a.model, speed_scale=a.speed_scale, target_speed=a.speed,
          lateral_accel=a.grip or None)
