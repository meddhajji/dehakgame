"""
Print chosen columns of an experiment log as a table.

    python show_log.py NAME [every_n_frames] [col,col,...]
"""
import json
import sys

name = sys.argv[1]
every = int(sys.argv[2]) if len(sys.argv) > 2 else 5
cols = (sys.argv[3].split(",") if len(sys.argv) > 3 else
        ["frame", "time", "cmd_steering", "cmd_throttle", "steering_angle", "speed",
         "cte", "activeNode", "pos_x", "pos_z", "yaw", "gyro_y", "vel_x", "hit"])

rows, events = [], []
for line in open(f"logs/{name}.jsonl"):
    r = json.loads(line)
    (events if r.get("event") else rows).append(r)


def fmt(v):
    return f"{v:9.3f}" if isinstance(v, float) else f"{str(v):>9}"


print(" ".join(f"{c[:9]:>9}" for c in cols))
for r in rows[::every]:
    print(" ".join(fmt(r.get(c, "")) for c in cols))
for e in events:
    print("event:", e)
