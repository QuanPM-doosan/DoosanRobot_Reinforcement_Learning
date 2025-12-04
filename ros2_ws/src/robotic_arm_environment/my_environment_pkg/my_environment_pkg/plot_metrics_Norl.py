#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, csv, glob, math
import numpy as np
import matplotlib.pyplot as plt

# ======== cấu hình obstacle (tuỳ chỉnh) ========
OBSTACLE_CENTER = np.array([0.50, -0.20, 0.60], dtype=float)  # chỉnh theo mô phỏng
OBSTACLE_HALF   = np.array([0.05,  0.125, 0.60], dtype=float) # 0.10 x 0.25 x 1.20

# ======== thư mục log ========
LOG_DIR = os.path.expanduser("~/QuanPM_robotic_arm_ws/ros2_ws/rl_logs")

# ----------------- utils -----------------
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
        ep = r.get("ep", "")
        try:
            ep = int(ep) if ep != "" else -1
        except Exception:
            ep = -1
        eps.setdefault(ep, []).append(r)
    # sort theo step
    for k in eps:
        eps[k].sort(key=lambda rr: int(rr.get("step", 0) or 0))
    return eps

def rolling_mean(x, w):
    if w <= 1:
        return x
    out = []
    for i in range(len(x)):
        lo = max(0, i - w + 1)
        out.append(np.mean(x[lo:i+1]))
    return out

def newest_random_csv():
    files = sorted(glob.glob(os.path.join(LOG_DIR, "*no_rl_random*.csv")))
    return files[-1] if files else None

# ----------------- plots -----------------
def plot_learning_curves_norl(random_csv, out_png="learning_curves_norl.png"):
    rows = load_csv(random_csv)
    eps = group_by_episode(rows)
    ep_ids = sorted(eps.keys())

    def ep_stats(ep_rows):
        rewards = [to_float(r.get("reward"), 0.0) for r in ep_rows]
        done    = any(int(r.get("done", 0)) == 1 for r in ep_rows)
        # collision nếu signed < 0 ở bất kì step
        collide = any(to_float(r.get("signed", np.nan)) < 0.0 for r in ep_rows)

        dlist = [to_float(r.get("dist_out", np.nan)) for r in ep_rows]
        dfin  = [d for d in dlist if not (d is None or (isinstance(d, float) and np.isnan(d)))]
        min_d = float(np.min(dfin)) if dfin else np.nan

        return np.nansum(rewards), int(done), int(collide), min_d

    ret, done_rate, coll_rate, min_dist = [], [], [], []
    for e in ep_ids:
        R, D, C, M = ep_stats(eps[e])
        ret.append(R)
        done_rate.append(D)
        coll_rate.append(C)
        min_dist.append(M)

    fig, ax = plt.subplots(2, 2, figsize=(11, 8))
    ax = ax.ravel()

    ax[0].plot(ep_ids, ret, label="No-RL (random)")
    ax[0].set_title("Tổng reward / episode")
    ax[0].set_xlabel("Episode"); ax[0].set_ylabel("Return"); ax[0].grid(True)

    ax[1].plot(ep_ids, min_dist, label="No-RL (random)")
    ax[1].set_title("Khoảng cách tối thiểu tới vật cản / episode")
    ax[1].set_xlabel("Episode"); ax[1].set_ylabel("Min dist (m)"); ax[1].grid(True)

    # rolling mean cho tỉ lệ (cửa sổ = 10% tổng số ep, tối thiểu 1)
    W = max(1, len(ep_ids)//10)
    ax[2].plot(ep_ids, rolling_mean(done_rate, W), label="No-RL (random)")
    ax[2].set_title("Tỉ lệ đạt mục tiêu (rolling)")
    ax[2].set_xlabel("Episode"); ax[2].set_ylabel("Rate"); ax[2].grid(True)

    ax[3].plot(ep_ids, rolling_mean(coll_rate, W), label="No-RL (random)")
    ax[3].set_title("Tỉ lệ va chạm (rolling)")
    ax[3].set_xlabel("Episode"); ax[3].set_ylabel("Rate"); ax[3].grid(True)

    for k in range(4):
        ax[k].legend(loc="best")

    plt.tight_layout()
    out_path = os.path.join(LOG_DIR, out_png)
    plt.savefig(out_path, dpi=150)
    print(f"[SAVE] {out_path}")

def plot_trajectories_norl(random_csv, out_png="traj_3d_norl.png",
                           include_phase=None,      # None = lấy tất cả phase (random)
                           episodes=None,           # ví dụ [1,2,3]
                           n_per_episode=8,         # MẶC ĐỊNH: 8 điểm/episode
                           max_points=None,         # NEW: giới hạn tổng số điểm (downsample đều)
                           mark_intersections=True  # tô đỏ các điểm nằm trong obstacle
                           ):
    rows = load_csv(random_csv)

    # lọc phase nếu muốn (thường phase="random")
    if include_phase is not None:
        rows = [r for r in rows if (r.get("phase") or "").lower() == include_phase.lower()]

    # gom theo episode và chọn danh sách episode
    eps = group_by_episode(rows)
    ep_ids = sorted(eps.keys())
    if episodes is not None:
        sel = set(episodes)
        ep_ids = [e for e in ep_ids if e in sel]

    xs, ys, zs = [], [], []
    for ep in ep_ids:
        ep_rows = eps[ep]
        # giữ các point có đủ toạ độ
        valid_rows = []
        for r in ep_rows:
            vals = [to_float(r.get("ee_x")), to_float(r.get("ee_y")), to_float(r.get("ee_z"))]
            if not any(np.isnan(v) for v in vals):
                valid_rows.append(r)
        if not valid_rows:
            continue

        # chọn đúng n_per_episode điểm cách đều theo step
        k = min(n_per_episode, len(valid_rows))
        idx = np.linspace(0, len(valid_rows)-1, num=k, dtype=int)
        for i in idx:
            r = valid_rows[i]
            xs.append(to_float(r["ee_x"]))
            ys.append(to_float(r["ee_y"]))
            zs.append(to_float(r["ee_z"]))

    if len(xs) == 0:
        print("No valid points to plot.")
        return

    xs = np.array(xs, dtype=float)
    ys = np.array(ys, dtype=float)
    zs = np.array(zs, dtype=float)

    # NEW: giới hạn tổng số điểm nếu cần (downsample đều)
    if (max_points is not None) and (len(xs) > max_points):
        sel_idx = np.linspace(0, len(xs)-1, num=max_points, dtype=int)
        xs, ys, zs = xs[sel_idx], ys[sel_idx], zs[sel_idx]

    # mask điểm nằm trong obstacle (AABB test)
    cx, cy, cz = OBSTACLE_CENTER
    hx, hy, hz = OBSTACLE_HALF
    inside_mask = (
        (np.abs(xs - cx) <= hx) &
        (np.abs(ys - cy) <= hy) &
        (np.abs(zs - cz) <= hz)
    )

    # vẽ
    from mpl_toolkits.mplot3d import Axes3D  # noqa
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection='3d')

    # vẽ tập điểm rời rạc (scatter) để quỹ đạo gọn gàng
    ax.scatter(xs, ys, zs, s=12, label="EE samples")

    if mark_intersections and np.any(inside_mask):
        ax.scatter(xs[inside_mask], ys[inside_mask], zs[inside_mask],
                   s=28, c='r', depthshade=False, label="Điểm giao object")

    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    ax.set_title(f"Quỹ đạo EE (No-RL) — {n_per_episode} điểm/episode")

    # vẽ hộp obstacle
    from itertools import product, combinations
    X = [cx-hx, cx+hx]; Y = [cy-hy, cy+hy]; Z = [cz-hz, cz+hz]
    pts = np.array(list(product(X, Y, Z)))
    for s, e in combinations(pts, 2):
        if np.sum(np.abs(s-e) > 1e-9) == 1:
            ax.plot(*zip(s, e), linewidth=1.0, alpha=0.8)

    # Fallback cho matplotlib cũ (thiếu set_box_aspect)
    try:
        ax.set_box_aspect([1, 1, 1])
    except Exception:
        xmin, xmax = np.nanmin(xs), np.nanmax(xs)
        ymin, ymax = np.nanmin(ys), np.nanmax(ys)
        zmin, zmax = np.nanmin(zs), np.nanmax(zs)
        max_range = max(xmax - xmin, ymax - ymin, zmax - zmin)
        if max_range == 0: max_range = 1.0
        xmid = (xmax + xmin) / 2.0
        ymid = (ymax + ymin) / 2.0
        zmid = (zmax + zmin) / 2.0
        ax.set_xlim(xmid - max_range/2, xmid + max_range/2)
        ax.set_ylim(ymid - max_range/2, ymid + max_range/2)
        ax.set_zlim(zmid - max_range/2, zmid + max_range/2)

    ax.legend(loc="best")
    plt.tight_layout()
    out_path = os.path.join(LOG_DIR, out_png)
    plt.savefig(out_path, dpi=150)
    print(f"[SAVE] {out_path}")

def main():
    os.makedirs(LOG_DIR, exist_ok=True)
    rand_csv = newest_random_csv()
    if not rand_csv:
        print(f"[ERR] Không tìm thấy CSV baseline no-RL trong {LOG_DIR} (pattern: *no_rl_random*.csv)")
        return

    print(f"[INFO] Using baseline CSV: {rand_csv}")
    plot_learning_curves_norl(rand_csv, out_png="learning_curves_norl.png")
    plot_trajectories_norl(rand_csv,
                           out_png="traj_3d_norl.png",
                           include_phase=None,     # hoặc "random"
                           episodes=None,          # hoặc [1,2,3]
                           n_per_episode=8,        # GIỮ 8 điểm/episode
                           max_points=3000,        # NEW: giới hạn tổng số điểm (tuỳ chỉnh)
                           mark_intersections=True)

if __name__ == "__main__":
    main()

