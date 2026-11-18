"""
train.py - train DriveNet on the cleaned dataset.

    python train.py NAME [--epochs 20] [--no-speed]

Augmentation (training batches only, new random values every batch):
  * horizontal flip with probability 0.5, steering label negated
    (the road mirrored = the opposite turn; target speed unchanged)
  * colour channels shuffled (p=0.3), grey-scale (p=0.2), brightness x0.6-1.4,
    contrast x0.6-1.4, per-channel colour gain x0.7-1.3
    (every track has different light and colours)
  * small pixel noise
Loss: MSE(steering) + MSE(target speed / MAX_SPEED).
Keeps the epoch with the best validation loss -> models/NAME.pt
Reports errors in real units, with a "predict the training average" baseline.
"""
import argparse
import os
import time

import numpy as np
import torch
from torch import nn

from dataset import MAX_SPEED, load
from model import DriveNet

torch.set_num_threads(2)


def augment(x, steering):
    """x: (B,3,H,W) float 0..1."""
    b = x.shape[0]
    flip = torch.rand(b) < 0.5
    x = torch.where(flip[:, None, None, None], x.flip(3), x)
    steering = torch.where(flip, -steering, steering)
    # colours differ on every track (warehouse is yellow-tinted): make colour unreliable
    # so the network relies on the SHAPE of lines and road edges
    perm = torch.rand(b) < 0.3
    x = torch.where(perm[:, None, None, None], x[:, torch.randperm(3)], x)       # swap channels
    grey = torch.rand(b) < 0.2
    x = torch.where(grey[:, None, None, None], x.mean(1, keepdim=True).expand_as(x), x)
    mean = x.mean(dim=(1, 2, 3), keepdim=True)
    x = (x - mean) * torch.empty(b, 1, 1, 1).uniform_(0.6, 1.4) + mean          # contrast
    x = x * torch.empty(b, 1, 1, 1).uniform_(0.6, 1.4)                           # brightness
    x = x * torch.empty(b, 3, 1, 1).uniform_(0.7, 1.3)                           # colour
    x = x + torch.randn_like(x) * 0.02
    return x.clamp(0, 1), steering


def tensors(data, split):
    sel = data["split"] == split
    return (torch.from_numpy(data["images"][sel]), torch.from_numpy(data["steering"][sel]),
            torch.from_numpy(data["target"][sel] / MAX_SPEED), torch.from_numpy(data["speed"][sel] / MAX_SPEED),
            data["scene"][sel])


def evaluate(net, imgs, st, tg, sp, batch=512):
    """-> (loss, steering MAE, target speed MAE in pos/s)."""
    net.eval()
    ps, pt = [], []
    with torch.no_grad():
        for i in range(0, len(imgs), batch):
            a, b = net(imgs[i:i + batch].float() / 255.0, sp[i:i + batch].float())
            ps.append(a)
            pt.append(b)
    ps, pt = torch.cat(ps), torch.cat(pt)
    loss = float(((ps - st) ** 2).mean() + ((pt - tg) ** 2).mean())
    return loss, float((ps - st).abs().mean()), float((pt - tg).abs().mean() * MAX_SPEED)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("name")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--no-speed", action="store_true", help="camera only: no speed input")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--all", action="store_true", help="train on every track (final race model)")
    a = p.parse_args()

    torch.manual_seed(0)
    data = load(all_tracks=a.all)
    train, val, test = tensors(data, "train"), tensors(data, "val"), tensors(data, "test")
    print(f"train {len(train[0])}, val {len(val[0])}, test {len(test[0])} frames")
    net = DriveNet(use_speed=not a.no_speed)
    print(f"parameters: {sum(p.numel() for p in net.parameters()):,}")
    opt = torch.optim.Adam(net.parameters(), lr=a.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs)

    # baseline: always predict the training averages
    ms, mt = float(train[1].mean()), float(train[2].mean())
    for nm, d in (("val", val), ("test", test)):
        if len(d[0]) == 0:
            continue
        print(f"baseline {nm}: steering MAE {float((d[1] - ms).abs().mean()):.3f}, "
              f"target speed MAE {float((d[2] - mt).abs().mean() * MAX_SPEED):.2f} pos/s")

    os.makedirs("models", exist_ok=True)
    best = float("inf")
    for epoch in range(1, a.epochs + 1):
        net.train()
        t0, total, n = time.perf_counter(), 0.0, 0
        order = torch.randperm(len(train[0]))
        for i in range(0, len(order), a.batch):
            idx = order[i:i + a.batch]
            x, st = augment(train[0][idx].float() / 255.0, train[1][idx])
            ps, pt = net(x, train[3][idx].float())
            loss = nn.functional.mse_loss(ps, st) + nn.functional.mse_loss(pt, train[2][idx].float())
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.detach()) * len(idx)
            n += len(idx)
        sched.step()
        vl, vs, vt = evaluate(net, *val[:4])
        mark = ""
        if vl < best:
            best = vl
            torch.save({"state": net.state_dict(), "use_speed": not a.no_speed, "epoch": epoch}, f"models/{a.name}.pt")
            mark = "  <- saved"
        print(f"epoch {epoch:2d}  train loss {total / n:.4f}  val loss {vl:.4f}  "
              f"val steering MAE {vs:.3f}  val speed MAE {vt:.2f}  ({time.perf_counter() - t0:.0f}s){mark}")

    ckpt = torch.load(f"models/{a.name}.pt")
    net.load_state_dict(ckpt["state"])
    print(f"best epoch {ckpt['epoch']}")
    for scene in sorted(set(test[4])):
        sel = torch.from_numpy(test[4] == scene)
        _, s, t = evaluate(net, *(x[sel] for x in test[:4]))
        print(f"test {scene}: steering MAE {s:.3f}, target speed MAE {t:.2f} pos/s")


if __name__ == "__main__":
    main()
