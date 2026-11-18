"""
Plot the track centre path and the car path of one or more logs (same track),
coloured by |our cte| in pos units. Red crosses = interventions.
    python plot_track.py LOG [LOG ...]      -> logs/track.png
"""
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import SCENE
from track import Track

logs = {}
for name in sys.argv[1:]:
    rows = [json.loads(l) for l in open(f"logs/{name}.jsonl")]
    logs[name] = [r for r in rows if r.get("event") is not True and "pos_x" in r]
scene = next((rows[0].get("scene") for rows in logs.values() if rows), None) or SCENE
track = Track(json.load(open(f"logs/track_{scene}.json")))

fig, ax = plt.subplots(figsize=(9, 7))
loop = list(track.p) + [track.p[0]]
ax.plot([p[0] for p in loop], [p[1] for p in loop], "-", color="gray", lw=8, alpha=0.25, label="centre path")
for i in range(0, track.n, 10):
    ax.annotate(str(i), track.p[i], fontsize=7, color="gray")

sc = None
for name, rows in logs.items():
    xz = [track.to_node(r["pos_x"], r["pos_z"]) for r in rows]
    c = [abs(r.get("my_cte", 0)) for r in rows]
    sc = ax.scatter([p[0] for p in xz], [p[1] for p in xz], c=c, s=3, cmap="viridis", vmin=0, vmax=0.41)
    bad = [p for p, r in zip(xz, rows) if r.get("event")]
    ax.plot([p[0] for p in bad], [p[1] for p in bad], "rx", ms=8)
if sc is not None:
    fig.colorbar(sc, ax=ax, label="|cte| (pos units; white line at ~0.41 on roboracingleague_1)")
ax.set_aspect("equal")
ax.set_title(f"{scene}: " + ", ".join(logs))
fig.savefig("logs/track.png", dpi=110, bbox_inches="tight")
print("saved logs/track.png")
