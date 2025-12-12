#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import math
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def quat_normalize(q: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(q, axis=-1, keepdims=True)
    return q / n


def quat_xyzw_to_rot(q: np.ndarray) -> np.ndarray:
    """q: (...,4) with (x,y,z,w)"""
    x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z

    R = np.empty(q.shape[:-1] + (3, 3), dtype=float)
    R[..., 0, 0] = 1 - 2 * (yy + zz)
    R[..., 0, 1] = 2 * (xy - wz)
    R[..., 0, 2] = 2 * (xz + wy)

    R[..., 1, 0] = 2 * (xy + wz)
    R[..., 1, 1] = 1 - 2 * (xx + zz)
    R[..., 1, 2] = 2 * (yz - wx)

    R[..., 2, 0] = 2 * (xz - wy)
    R[..., 2, 1] = 2 * (yz + wx)
    R[..., 2, 2] = 1 - 2 * (xx + yy)
    return R


def rot_to_quat_xyzw(R: np.ndarray) -> np.ndarray:
    """R: (N,3,3) -> q: (N,4) with (x,y,z,w), robust branch implementation."""
    N = R.shape[0]
    q = np.empty((N, 4), dtype=float)

    tr = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]
    m0 = tr > 0.0

    # case 0: trace > 0
    if np.any(m0):
        S = np.sqrt(tr[m0] + 1.0) * 2.0
        qw = 0.25 * S
        qx = (R[m0, 2, 1] - R[m0, 1, 2]) / S
        qy = (R[m0, 0, 2] - R[m0, 2, 0]) / S
        qz = (R[m0, 1, 0] - R[m0, 0, 1]) / S
        q[m0] = np.stack([qx, qy, qz, qw], axis=1)

    # case 1: trace <= 0
    m1 = ~m0
    if np.any(m1):
        Rm = R[m1]
        r00, r11, r22 = Rm[:, 0, 0], Rm[:, 1, 1], Rm[:, 2, 2]
        idx = np.argmax(np.stack([r00, r11, r22], axis=1), axis=1)

        q_sub = np.empty((Rm.shape[0], 4), dtype=float)

        for i in (0, 1, 2):
            mi = idx == i
            if not np.any(mi):
                continue
            Ri = Rm[mi]

            if i == 0:
                S = np.sqrt(1.0 + Ri[:, 0, 0] - Ri[:, 1, 1] - Ri[:, 2, 2]) * 2.0
                qx = 0.25 * S
                qy = (Ri[:, 0, 1] + Ri[:, 1, 0]) / S
                qz = (Ri[:, 0, 2] + Ri[:, 2, 0]) / S
                qw = (Ri[:, 2, 1] - Ri[:, 1, 2]) / S
            elif i == 1:
                S = np.sqrt(1.0 + Ri[:, 1, 1] - Ri[:, 0, 0] - Ri[:, 2, 2]) * 2.0
                qy = 0.25 * S
                qx = (Ri[:, 0, 1] + Ri[:, 1, 0]) / S
                qz = (Ri[:, 1, 2] + Ri[:, 2, 1]) / S
                qw = (Ri[:, 0, 2] - Ri[:, 2, 0]) / S
            else:
                S = np.sqrt(1.0 + Ri[:, 2, 2] - Ri[:, 0, 0] - Ri[:, 1, 1]) * 2.0
                qz = 0.25 * S
                qx = (Ri[:, 0, 2] + Ri[:, 2, 0]) / S
                qy = (Ri[:, 1, 2] + Ri[:, 2, 1]) / S
                qw = (Ri[:, 1, 0] - Ri[:, 0, 1]) / S

            q_sub[mi] = np.stack([qx, qy, qz, qw], axis=1)

        q[m1] = q_sub

    return quat_normalize(q)


def compute_R_BD(t: np.ndarray, omega: float, alpha: float) -> np.ndarray:
    ct = np.cos(omega * t)
    st = np.sin(omega * t)
    ca = math.cos(alpha)
    sa = math.sin(alpha)

    R = np.zeros((t.shape[0], 3, 3), dtype=float)
    R[:, 0, 0] = ct
    R[:, 0, 1] = -st * ca
    R[:, 0, 2] = st * sa
    R[:, 1, 0] = st
    R[:, 1, 1] = ct * ca
    R[:, 1, 2] = -ct * sa
    R[:, 2, 0] = 0.0
    R[:, 2, 1] = sa
    R[:, 2, 2] = ca
    return R


def enforce_qw_ge0(q: np.ndarray) -> np.ndarray:
    q2 = q.copy()
    m = q2[:, 3] < 0.0
    q2[m] *= -1.0
    return q2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tracking", default="tracking.csv")
    ap.add_argument("--out_csv", default="solutions/q1_quaternion_world.csv")
    ap.add_argument("--out_png", default="solutions/q1_quaternion_world.png")
    ap.add_argument("--omega", type=float, default=0.5)
    ap.add_argument("--alpha", type=float, default=math.pi / 12.0)
    args = ap.parse_args()

    df = pd.read_csv(args.tracking)
    t = df["t"].to_numpy()
    qWB = df[["qx", "qy", "qz", "qw"]].to_numpy()
    qWB = quat_normalize(qWB)

    # {}^W R_B from tracking quaternion
    RWB = quat_xyzw_to_rot(qWB)

    # {}^B R_D from the given formula
    RBD = compute_R_BD(t, args.omega, args.alpha)

    # {}^W R_D = {}^W R_B * {}^B R_D
    RWD = RWB @ RBD

    # back to quaternion
    qWD = rot_to_quat_xyzw(RWD)
    qWD = enforce_qw_ge0(qWD)  # required by the homework statement
    qWD = quat_normalize(qWD)  # final safety

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    out = pd.DataFrame({
        "t": t,
        "qx": qWD[:, 0],
        "qy": qWD[:, 1],
        "qz": qWD[:, 2],
        "qw": qWD[:, 3],
    })
    out.to_csv(args.out_csv, index=False)

    # plot
    plt.figure(figsize=(10, 5))
    plt.plot(t, qWD[:, 0], label="qx")
    plt.plot(t, qWD[:, 1], label="qy")
    plt.plot(t, qWD[:, 2], label="qz")
    plt.plot(t, qWD[:, 3], label="qw")
    plt.xlabel("t [s]")
    plt.ylabel("quaternion component")
    plt.title("End-effector orientation in world frame ($q_{WD}$)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.out_png, dpi=250)
    print(f"[OK] saved: {args.out_csv}")
    print(f"[OK] saved: {args.out_png}")


if __name__ == "__main__":
    main()
