"""
track.py - our own track geometry, computed from the simulator's node list.

The simulator's cte is measured against the segment near `activeNode`, and
activeNode is only updated by searching around its previous value. When it
loses track (resets, teleports, cutting corners) the simulator's cte becomes
meaningless (measured: -25, -34, +19 while sitting on the road).
Here we search ALL segments every time, so it cannot lose track.

Units: node coordinates (used by node_position and set_position) are NOT the
same as telemetry pos. On roboracingleague_1, node = 8 x pos. Because this may
differ per track, load_track() measures it for every scene by placing the car
on several nodes and fitting  node = scale * pos + offset.
All outputs are in node units (same unit as the simulator's cte).
"""
import json
import os
import time

import numpy as np


class Track:
    def __init__(self, data):
        self.scene = data["scene"]
        self.scale = data["scale"]                                         # node units per pos unit
        self.offset = np.array(data["offset"])                             # node = scale*pos + offset
        nodes = data["nodes"]
        self.y = np.array([n["pos_y"] for n in nodes])
        self.p = np.array([[n["pos_x"], n["pos_z"]] for n in nodes])     # (N, 2), closed loop
        self.n = len(self.p)
        self.seg = np.roll(self.p, -1, axis=0) - self.p                    # segment i: node i -> i+1
        self.seg_len = np.linalg.norm(self.seg, axis=1)
        self.dir = self.seg / self.seg_len[:, None]
        self.s0 = np.concatenate([[0], np.cumsum(self.seg_len)[:-1]])     # distance at node i
        self.length = self.seg_len.sum()
        self.heading = np.arctan2(self.dir[:, 1], self.dir[:, 0])          # radians, per segment

    def to_node(self, pos_x, pos_z):
        return np.array([pos_x, pos_z]) * self.scale + self.offset

    def locate(self, pos_x, pos_z):
        """Telemetry position -> dict(cte, segment, s, heading)."""
        q = self.to_node(pos_x, pos_z)
        rel = q - self.p                                                   # (N, 2)
        t = np.clip((rel * self.dir).sum(1) / self.seg_len, 0, 1)
        closest = self.p + self.seg * t[:, None]
        d = np.linalg.norm(q - closest, axis=1)
        i = int(np.argmin(d))
        cross = self.dir[i, 0] * rel[i, 1] - self.dir[i, 1] * rel[i, 0]   # side of the path
        # sign chosen to match the simulator's cte where both agree
        return {"cte": float(-np.sign(cross) * d[i]), "segment": i,
                "s": float(self.s0[i] + t[i] * self.seg_len[i]),
                "heading": float(self.heading[i])}

    def point_at(self, s):
        """Point on the centre path at distance s (node units, wraps around the lap)."""
        s %= self.length
        i = int(np.searchsorted(self.s0, s, side="right") - 1)
        return self.p[i] + self.dir[i] * (s - self.s0[i])

    def pose_at(self, node, offset=0.0, yaw_offset_deg=0.0):
        """
        A set_position message placing the car at `node`, facing along the
        track, shifted `offset` node units sideways (sign = our cte sign) and
        turned `yaw_offset_deg` degrees. Unity yaw: 0 = +z, 90 = +x.
        The node's own quaternion from the simulator does NOT face the track.
        """
        x, z = self.p[node]
        dx, dz = self.dir[node]
        # cte = -sign(cross): a positive offset is to the right of the direction
        x, z = x + offset * dz, z - offset * dx
        yaw = np.degrees(np.arctan2(dx, dz)) + yaw_offset_deg
        half = np.radians(yaw) / 2
        # dropped from 0.6 above the node: at node height the car spawns inside
        # the floor, collides ("floor"/"lines") and spins ~150 degrees.
        return {"msg_type": "set_position", "pos_x": f"{x:.4f}", "pos_y": f"{self.y[node] + 0.6:.4f}",
                "pos_z": f"{z:.4f}", "Qx": "0", "Qy": f"{np.sin(half):.6f}", "Qz": "0",
                "Qw": f"{np.cos(half):.6f}"}


def fetch_nodes(link, total):
    """Ask the simulator for all `total` path nodes."""
    for i in range(total):
        link.send({"msg_type": "node_position", "index": str(i)})
        time.sleep(0.005)
    nodes = {}
    end = time.perf_counter() + 3
    while len(nodes) < total and time.perf_counter() < end:
        m = link.get(0.2)
        if m is not None and m.get("msg_type") == "node_position":
            nodes[int(m["index"])] = {k: m[k] for k in ("pos_x", "pos_y", "pos_z", "Qx", "Qy", "Qz", "Qw")}
    if len(nodes) < total:
        raise RuntimeError(f"only {len(nodes)}/{total} nodes received")
    return [nodes[i] for i in range(total)]


def _settled(link, frames=10):
    m = None
    for _ in range(frames):
        m = link.wait_for("telemetry", 2)
        if m is None:
            raise RuntimeError("no telemetry for 2 s")
        link.control(0, 0, 1.0)
    return m


def calibrate(link, scene, total):
    """Fetch nodes and measure node = scale * pos + offset by teleporting onto 8 nodes."""
    nodes = fetch_nodes(link, total)
    tmp = Track({"scene": scene, "scale": 1.0, "offset": [0, 0], "nodes": nodes})
    pos, node = [], []
    for i in np.linspace(0, total - 1, 8).astype(int):
        link.send(tmp.pose_at(i))
        m = _settled(link)
        pos.append([m["pos_x"], m["pos_z"]])
        node.append(tmp.p[i])
    pos, node = np.array(pos), np.array(node)
    # one scale for both axes, one offset per axis: least squares
    A = np.zeros((2 * len(pos), 3))
    A[0::2, 0], A[0::2, 1] = pos[:, 0], 1
    A[1::2, 0], A[1::2, 2] = pos[:, 1], 1
    (scale, ox, oz), *_ = np.linalg.lstsq(A, node.reshape(-1), rcond=None)
    residual = np.abs(A @ np.array([scale, ox, oz]) - node.reshape(-1)).max()
    link.reset_car()
    _settled(link)
    return {"scene": scene, "scale": float(scale), "offset": [float(ox), float(oz)],
            "fit_max_error": float(residual), "nodes": nodes}


def load_track(link, scene, total_nodes, refresh=False):
    """Track geometry for `scene`: from logs/track_<scene>.json, or measured now."""
    path = f"logs/track_{scene}.json"
    if refresh or not os.path.exists(path):
        os.makedirs("logs", exist_ok=True)
        data = calibrate(link, scene, total_nodes)
        json.dump(data, open(path, "w"))
    return Track(json.load(open(path)))
