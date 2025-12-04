#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, csv, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")    # tránh lỗi display khi chạy headless
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa

# ======== cấu hình obstacle (tuỳ chỉnh) ========
OBSTACLE_CENTER = np.array([0.50, -0.20, 0.60], dtype=float)
OBSTACLE_HALF   = np.array([0.05,  0.125, 0.60], dtype=float)

# ---------------------------------------
def load_csv(path):
    rows = []
    with open(path, "r") as f:
        rd = csv.DictReader(f)
        for r in rd:
            rows.append(r)
    return rows

def to_float(x, default=np.nan):
    try:
        return float(x)
    except Exception:
        return default

def group_by_episode(rows):
    eps = {}
    for r in rows:
        ep = int(r["ep"]) if r["ep"] != "" else -1
        eps.setdefault(ep, []).append(r)
    # sort each episode by step
    for k in eps:
        eps[k].sort(key=lambda rr: int(rr["step"]))
    return eps

# ---------------------------------------
def plot_learning_curves(train_csv, random_csv=None, out_png="learning_curves.png"):
    tr = load_csv(train_csv)
    tr_eps = group_by_episode(tr)

    def ep_stats(ep_rows):
        rewards = [to_float(r["reward"], 0.0) for r in ep_rows]
        done    = any(int(r["done"])==1 for r in ep_rows)
        collide = any(to_float(r.get("signed", 1.0)) < 0.0 for r in ep_rows)
        dist_vals = [to_float(r.get("dist_out", np.nan)) for r in ep_rows]
        min_d = np.nanmin(dist_vals)
        return np.nansum(rewards), int(done), int(collide), float(min_d)

    # RL:
    ep_ids = sorted(tr_eps.keys())
    rl_ret, rl_done, rl_coll, rl_mind = [], [], [], []
    for e in ep_ids:
        R, D, C, M = ep_stats(tr_eps[e])
        rl_ret.append(R)
        rl_done.append(D)
        rl_coll.append(C)
        rl_mind.append(M)

    # vẽ
    fig, ax = plt.subplots(2,2, figsize=(11,8))
    ax = ax.ravel()

    ax[0].plot(ep_ids, rl_ret, label="RL")
    ax[0].set_title("Tổng reward / episode")
    ax[0].set_xlabel("Episode"); ax[0].set_ylabel("Return"); ax[0].grid(True)

    ax[1].plot(ep_ids, rl_mind, label="RL")
    ax[1].set_title("Khoảng cách tối thiểu tới vật cản")
    ax[1].set_xlabel("Episode"); ax[1].set_ylabel("Min dist (m)"); ax[1].grid(True)

    # Rolling mean
    W = max(1, len(ep_ids)//10)
    def rolling_mean(x, w):
        if w<=1: return x
        out=[]
        for i in range(len(x)):
            lo=max(0,i-w+1)
            out.append(np.mean(x[lo:i+1]))
        return out

    ax[2].plot(ep_ids, rolling_mean(rl_done, W), label="RL")
    ax[2].set_title("Tỉ lệ đạt mục tiêu (rolling)")
    ax[2].set_xlabel("Episode"); ax[2].set_ylabel("Rate"); ax[2].grid(True)

    ax[3].plot(ep_ids, rolling_mean(rl_coll, W), label="RL")
    ax[3].set_title("Tỉ lệ va chạm (rolling)")
    ax[3].set_xlabel("Episode"); ax[3].set_ylabel("Rate"); ax[3].grid(True)

    # baseline no-RL random
    if random_csv is not None and os.path.isfile(random_csv):
        print(f"[INFO] Using baseline random: {random_csv}")
        rd = load_csv(random_csv)
        rd_eps = group_by_episode(rd)
        epb = sorted(rd_eps.keys())

        br, bd, bc, bm = [], [], [], []
        for e in epb:
            R,D,C,M = ep_stats(rd_eps[e])
            br.append(R); bd.append(D); bc.append(C); bm.append(M)

        ax[0].plot(epb, br, label="Random")
        ax[1].plot(epb, bm, label="Random")
        ax[2].plot(epb, rolling_mean(bd, W), label="Random")
        ax[3].plot(epb, rolling_mean(bc, W), label="Random")

    for k in range(4):
        ax[k].legend(loc="best")

    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    print(f"[SAVE] {out_png}")

# ---------------------------------------
def plot_trajectories(csv_path, out_png="traj_3d.png",
                      episodes=None, max_points=3000,
                      mark_intersections=True):

    rows = load_csv(csv_path)
    if episodes is not None:
        rows = [r for r in rows if int(r["ep"]) in set(episodes)]

    xs, ys, zs = [], [], []
    for r in rows[:max_points]:
        x = to_float(r.get("ee_x"))
        y = to_float(r.get("ee_y"))
        z = to_float(r.get("ee_z"))
        if not (np.isnan(x) or np.isnan(y) or np.isnan(z)):
            xs.append(x); ys.append(y); zs.append(z)

    if len(xs)==0:
        print("No valid EE points.")
        return

    xs = np.array(xs, float)
    ys = np.array(ys, float)
    zs = np.array(zs, float)

    # mask obstacle
    cx, cy, cz = OBSTACLE_CENTER
    hx, hy, hz = OBSTACLE_HALF
    inside_mask = (
        (np.abs(xs - cx) <= hx) &
        (np.abs(ys - cy) <= hy) &
        (np.abs(zs - cz) <= hz)
    )

    fig = plt.figure(figsize=(8,7))
    ax = fig.add_subplot(111, projection='3d')

    ax.plot(xs, ys, zs, linewidth=1.5, label="Quỹ đạo EE")

    if mark_intersections and np.any(inside_mask):
        ax.scatter(xs[inside_mask], ys[inside_mask], zs[inside_mask],
                   s=18, c='r', depthshade=False, label="Điểm giao")

    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    ax.set_title("Quỹ đạo End Effector thuật toán D4PG ")

    # vẽ hộp obstacle
    from itertools import product, combinations
    X = [cx-hx, cx+hx]; Y = [cy-hy, cy+hy]; Z = [cz-hz, cz+hz]
    pts = np.array(list(product(X,Y,Z)))
    for s,e in combinations(pts,2):
        if np.sum(np.abs(s-e) > 1e-9) == 1:
            ax.plot(*zip(s,e), linewidth=1.0, alpha=0.8)

    # ---------------------------
    # Fix set_box_aspect cho mọi Matplotlib
    # ---------------------------
    if hasattr(ax, "set_box_aspect"):
        ax.set_box_aspect((1,1,1))
    else:
        xmin, xmax = np.nanmin(xs), np.nanmax(xs)
        ymin, ymax = np.nanmin(ys), np.nanmax(ys)
        zmin, zmax = np.nanmin(zs), np.nanmax(zs)

        max_range = max(xmax-xmin, ymax-ymin, zmax-zmin)
        if max_range <= 0 or not np.isfinite(max_range):
            max_range = 1.0

        xmid = (xmax + xmin) / 2.0
        ymid = (ymax + ymin) / 2.0
        zmid = (zmax + zmin) / 2.0

        ax.set_xlim(xmid-max_range/2, xmid+max_range/2)
        ax.set_ylim(ymid-max_range/2, ymid+max_range/2)
        ax.set_zlim(zmid-max_range/2, zmid+max_range/2)

    ax.legend(loc="best")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    print(f"[SAVE] {out_png}")

# ---------------------------------------
if __name__ == "__main__":
    LOG_DIR = os.path.expanduser("~/TiepDB_robotic_arm_ws_7_5_trieu/ros2_ws/rl_logs")

    # tìm file RL mới nhất
    files = sorted(glob.glob(os.path.join(LOG_DIR, "20251118_090720_d4pg_train.csv"))) #no_rl_random // ddpg_train
    if not files:
        print("No training logs found.")
        exit(0)

    train_csv = files[-1]

    # tìm baseline random nếu có
    rand_list = sorted(glob.glob(os.path.join(LOG_DIR, "*no_rl_random*.csv")))
    rand_csv = rand_list[-1] if rand_list else None

    plot_learning_curves(
        train_csv,
        random_csv=rand_csv,
        out_png=os.path.join(LOG_DIR, "learning_curves_D4PG_train_RL.png")
    )

    plot_trajectories(
        train_csv,
        out_png=os.path.join(LOG_DIR, "traj_3d_D4PG_train_RL.png"),
        episodes=None,
        max_points=500
    )

