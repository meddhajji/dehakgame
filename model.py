"""
model.py - the CNN that replaces the expert.

Input : camera image (3x40x80 after dataset.preprocess) and, optionally, the
        current speed (one number). One image cannot show how fast the car is
        going, and the right steering depends on speed (the expert looks
        further ahead when faster), so speed is given when allowed.
Output: steering (-1..1) and target speed (pos/s): "how fast should I be
        going here", which an image CAN show (straight vs curve ahead).
        The throttle is then a simple rule on target vs current speed, the
        same rule the expert uses (throttle_from_target).

Architecture: 4 conv layers (each halves the size or keeps it), then 2 fully
connected layers. Small on purpose: it must run well under the 50 ms between
frames on a 2-core laptop CPU.
"""
import numpy as np
import torch
from torch import nn

from dataset import MAX_SPEED


class DriveNet(nn.Module):
    def __init__(self, use_speed=True):
        super().__init__()
        self.use_speed = use_speed
        self.features = nn.Sequential(
            nn.Conv2d(3, 24, 5, stride=2, padding=2), nn.BatchNorm2d(24), nn.ReLU(),   # 20x40
            nn.Conv2d(24, 32, 5, stride=2, padding=2), nn.BatchNorm2d(32), nn.ReLU(),  # 10x20
            nn.Conv2d(32, 48, 3, stride=2, padding=1), nn.BatchNorm2d(48), nn.ReLU(),  # 5x10
            nn.Conv2d(48, 64, 3, stride=1, padding=1), nn.BatchNorm2d(64), nn.ReLU(),  # 5x10
            nn.Flatten(), nn.Dropout(0.3),
        )
        self.head = nn.Sequential(
            nn.Linear(64 * 5 * 10 + int(use_speed), 100), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(100, 2),
        )

    def forward(self, images, speed):
        """images: (B,3,40,80) float 0..1; speed: (B,) normalised by MAX_SPEED."""
        x = self.features(images)
        if self.use_speed:
            x = torch.cat([x, speed[:, None]], dim=1)
        out = self.head(x)
        steering = torch.tanh(out[:, 0])
        target = torch.sigmoid(out[:, 1])          # 0..1 = 0..MAX_SPEED
        return steering, target


def throttle_from_target(v_target, speed):
    """Same rule as expert.Expert.act: (throttle, brake) to follow a target speed."""
    err = v_target - speed
    throttle = float(np.clip(0.25 * v_target + err, 0, 1))
    brake = float(np.clip(-err - 0.3, 0, 1))
    return (0.0, brake) if brake > 0 else (throttle, 0.0)


class ModelDriver:
    """Drives from the camera image (+ speed) with a trained DriveNet."""

    def __init__(self, path, speed_scale=1.0):
        """speed_scale < 1: drive at that fraction of the predicted target speed (safety margin)."""
        self.speed_scale = speed_scale
        ckpt = torch.load(path, map_location="cpu")
        self.net = DriveNet(use_speed=ckpt["use_speed"])
        self.net.load_state_dict(ckpt["state"])
        self.net.eval()
        torch.set_num_threads(1)     # leave the other core to the simulator

    def act(self, image_array, speed):
        """HxWx3 uint8 camera image, speed (pos/s) -> steering, throttle, brake, v_target."""
        from dataset import preprocess
        x = torch.from_numpy(preprocess(image_array)).float()[None] / 255.0
        with torch.no_grad():
            st, tg = self.net(x, torch.tensor([speed / MAX_SPEED], dtype=torch.float32))
        v_target = float(tg[0]) * MAX_SPEED
        throttle, brake = throttle_from_target(v_target * self.speed_scale, speed)
        return float(st[0]), throttle, brake, v_target
