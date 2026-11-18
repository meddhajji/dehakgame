"""
expert.py - the automatic "teacher" driver that produces training labels.

It uses information the final camera-only model will NOT have (car position
and the true centre path), which is exactly why its decisions are good labels.

Steering - pure pursuit: aim at a point on the centre path a lookahead ahead,
           steer along the circle arc that reaches it. Unlike reacting to cte,
           it sees curves coming.
Speed    - a planned target speed per node: slower in curves (grip limit),
           braking early enough before them; throttle/brake follow it.

All settings are in telemetry pos units (the car's physics units), which are
the same on every track; track.scale converts them to the track's node units.
Measured facts used here (SETUP_NOTES.md):
  * steering > 0 turns right; yaw 0 = +z, 90 = +x
  * steering 0.5 -> turn radius 1.44 pos units
  * commands act one frame (0.05 s) later
"""
import math

import numpy as np

CURVATURE_PER_STEERING = 1 / (1.44 * 0.5)   # 1/pos units, measured at steering 0.5


class Expert:
    def __init__(self, track, target_speed=4.0, lookahead=0.75, lookahead_time=0.19,
                 steering_gain=1.0, speed_gain=1.0, lateral_accel=5.5, brake_accel=2.0):
        """
        target_speed  : top speed (pos units/s)
        lookahead     : aim this far ahead at standstill (pos units) ...
        lookahead_time: ... plus speed x this many seconds
        lateral_accel : slow down in curves so that speed^2 x curvature stays
                        below it (pos/s^2; measured: no sliding up to ~5.7,
                        sliding and off-road at ~8.6). None = constant speed.
        brake_accel   : deceleration assumed when planning to slow down before
                        a curve (pos/s^2; measured full brake ~2.8).
        """
        self.track = track
        self.target_speed = target_speed
        self.lookahead = lookahead
        self.lookahead_time = lookahead_time
        self.steering_gain = steering_gain
        self.speed_gain = speed_gain
        self.plan = self._speed_plan(lateral_accel, brake_accel)

    def _speed_plan(self, lateral_accel, brake_accel):
        """Allowed speed at each node (pos units/s), including braking ahead of curves."""
        tr = self.track
        if lateral_accel is None:
            return np.full(tr.n, self.target_speed)
        # curvature per segment, smoothed over neighbours (nodes are coarse), in 1/pos units
        turn = np.angle(np.exp(1j * (tr.heading - np.roll(tr.heading, 1))))
        kappa = np.abs(turn) / tr.seg_len
        kappa = np.convolve(np.concatenate([kappa[-2:], kappa, kappa[:2]]), np.ones(5) / 5, "valid")
        kappa_pos = kappa * tr.scale
        v = np.minimum(self.target_speed, np.sqrt(lateral_accel / np.maximum(kappa_pos, 1e-6)))
        # going backwards around the loop (twice, it is closed): we must be able to
        # brake from v[i] down to v[i+1] over the segment length
        seg_pos = tr.seg_len / tr.scale
        for _ in range(2):
            for i in reversed(range(tr.n)):
                j = (i + 1) % tr.n
                v[i] = min(v[i], math.sqrt(v[j] ** 2 + 2 * brake_accel * seg_pos[i]))
        return v

    def act(self, m):
        """Telemetry message -> (steering, throttle, brake, info dict)."""
        tr = self.track
        speed = m["speed"]
        yaw = math.radians(m["yaw"])
        fwd = np.array([math.sin(yaw), math.cos(yaw)])    # (x, z)
        right = np.array([math.cos(yaw), -math.sin(yaw)])

        # where the car will be when this command takes effect (1 frame later), pos units
        here_pos = np.array([m["pos_x"], m["pos_z"]]) + fwd * speed * 0.05
        where = tr.locate(*here_pos)
        here = tr.to_node(*here_pos)

        ld = (self.lookahead + self.lookahead_time * speed) * tr.scale      # node units
        d = tr.point_at(where["s"] + ld) - here
        alpha = math.atan2(d @ right, d @ fwd)            # > 0: target is to the right
        curvature = 2 * math.sin(alpha) / max(math.hypot(*d), 1e-6) * tr.scale   # 1/pos units
        steering = float(np.clip(self.steering_gain * curvature / CURVATURE_PER_STEERING, -1, 1))

        # plan value 0.1 s ahead: the car needs time to change speed
        ahead_seg = int(np.searchsorted(tr.s0, (where["s"] + speed * 0.1 * tr.scale) % tr.length, "right") - 1)
        v_target = float(min(self.plan[where["segment"]], self.plan[ahead_seg]))
        err = v_target - speed
        throttle = float(np.clip(0.25 * v_target + self.speed_gain * err, 0, 1))
        brake = float(np.clip(-err - 0.3, 0, 1))          # brake only when clearly too fast
        if brake > 0:
            throttle = 0.0
        return steering, throttle, brake, {"cte": where["cte"], "s": where["s"],
                                           "segment": where["segment"], "alpha": alpha,
                                           "v_target": v_target}
