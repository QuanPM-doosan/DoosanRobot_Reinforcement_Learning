#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ============================================================
#  Auto pick-place (SMOOTH) with elevated via-points (A_up, B_up)
#  - Multi-point trajectory (mượt, ít reject)
#  - First point time_from_start > 0 (fix error_code=NA)
#  - Xác nhận đạt đích bằng /joint_states
#  - Né obstacle đơn giản bằng VIA nâng cao (giảm joint3)
#  YÊU CẦU: controller yaml đã nới goal_time / tolerances như đã hướng dẫn
# ============================================================

import os
import sys
import time
import math
import numpy as np

import rclpy
from rclpy.duration import Duration
from trajectory_msgs.msg import JointTrajectoryPoint
from control_msgs.action import FollowJointTrajectory

JOINT_ORDER = ['joint1','joint2','joint3','joint4','joint5','joint6']
IMPORT_NAME = 'jog_to_marker'  # đổi nếu file class Jogger của bạn khác tên

# ---- Poses ----
Q_HOME = np.array([0.0, 0.0,  0.0,   0.0,  -0.0,   0.0], dtype=float)
Q_A    = np.array([0.0, 0.0, -1.5,   0.0,  -1.57,  0.0], dtype=float)
Q_B    = np.array([ 3.9988, 0.0, -1.57, 0.0103, -1.599, -0.0956], dtype=float)

# ---- Via (nâng) để né obstacle (đơn giản, không cần IK) ----
LIFT_RAD = 0.45   # 0.25–0.45 tuỳ scene

# ---- Sampling & hạn chế ----
POINT_DT      = 0.10     # s/điểm (0.06–0.12 ổn)
MAX_STEP      = 0.05     # rad tối đa/điểm/khoảng
GOAL_TIME_TOL = 1.5      # giây: goal_time_tolerance cho action

# Giới hạn khớp (siết bớt để tránh đâm sàn)
JOINT_MIN = np.array([-2.9, -2.5, -2.8, -3.2, -2.2, -3.5], dtype=float)
JOINT_MAX = np.array([ 2.9,  2.5,  2.2,  3.2,  2.2,  3.5], dtype=float)

OBSTACLE_NAME = 'obstacle_box_1'  # để log/debug

# ----------------- Import Jogger -----------------
def import_jogger():
    try:
        mod = __import__(IMPORT_NAME, fromlist=['Jogger'])
        return getattr(mod, 'Jogger')
    except Exception:
        here = os.path.dirname(os.path.abspath(__file__))
        if here not in sys.path:
            sys.path.append(here)
        mod = __import__(IMPORT_NAME, fromlist=['Jogger'])
        return getattr(mod, 'Jogger')

# ----------------- Toán học góc -----------------
def ang_wrap(d):
    """Đưa hiệu góc về [-pi, pi] để đi đường ngắn nhất."""
    return (d + np.pi) % (2*np.pi) - np.pi

def clamp_joint_limits(q):
    return np.minimum(np.maximum(q, JOINT_MIN), JOINT_MAX)

def interp_traj(q_start, q_end, max_step=MAX_STEP):
    """Sinh danh sách điểm joint-space tuyến tính, bước không vượt quá max_step (rad)."""
    q_start = np.array(q_start, dtype=float).copy()
    q_end   = np.array(q_end,   dtype=float).copy()
    dq = ang_wrap(q_end - q_start)
    steps = int(max(1, math.ceil(np.max(np.abs(dq)) / max_step)))

    out = []
    for k in range(1, steps+1):
        alpha = k / steps
        q_k = q_start + alpha * dq
        q_k = clamp_joint_limits(q_k)
        out.append(q_k)
    return out

# ----------------- Wait tới đích thật -----------------
def wait_until_reached(node, q_goal, timeout=5.0, tol=0.05):
    """Đợi tới khi /joint_states gần q_goal (max|Δq| < tol), hoặc hết timeout."""
    t0 = time.time()
    while rclpy.ok() and (time.time() - t0) < timeout:
        rclpy.spin_once(node, timeout_sec=0.02)
        q_cur = node.current_positions()
        if q_cur is None:
            continue
        err = np.max(np.abs(ang_wrap(np.array(q_cur) - np.array(q_goal))))
        if err < tol:
            return True, err
    # trả về sai số cuối để log
    q_cur = node.current_positions() or q_goal
    err = np.max(np.abs(ang_wrap(np.array(q_cur) - np.array(q_goal))))
    return False, err

# ----------------- Gửi quỹ đạo nhiều điểm -----------------
def send_trajectory(node, q_list, point_dt=POINT_DT, goal_time_tol=GOAL_TIME_TOL, wait_joint_check=True, reach_tol=0.05):
    goal = FollowJointTrajectory.Goal()
    goal.goal_time_tolerance = Duration(seconds=float(goal_time_tol)).to_msg()
    goal.trajectory.joint_names = JOINT_ORDER

    # ⚠️ BẮT ĐẦU TỪ >0 GIÂY (fix reject)
    t = float(point_dt)
    for q in q_list:
        pt = JointTrajectoryPoint()
        pt.positions = [float(x) for x in q]
        # có thể thêm vel/acc = [0.0]*6 nếu muốn; controller nội suy vẫn ổn
        pt.time_from_start = Duration(seconds=float(t)).to_msg()
        goal.trajectory.points.append(pt)
        t += float(point_dt)

    node.get_logger().info(f"Gửi TRAJ {len(q_list)} điểm, tổng ~{(len(q_list))*point_dt:.2f}s")
    fut = node.ac.send_goal_async(goal)
    rclpy.spin_until_future_complete(node, fut)
    gh = fut.result()
    if gh is None or not gh.accepted:
        node.get_logger().error('⚠ Trajectory goal bị từ chối.')
        return False

    # Chờ kết quả từ action
    resf = gh.get_result_async()
    rclpy.spin_until_future_complete(node, resf)
    res = resf.result()
    ok = (res is not None and getattr(res, 'error_code', 1) == 0)
    if not ok:
        node.get_logger().warn(f'⚠ Lỗi thực thi quỹ đạo. error_code={getattr(res,"error_code","NA")}')

    # (Tuỳ chọn) xác nhận theo /joint_states để chắc chắn
    if wait_joint_check and len(q_list) > 0:
        reached, err = wait_until_reached(node, q_list[-1], timeout=max(3.0, len(q_list)*point_dt*1.2), tol=reach_tol)
        tag = "✓ TỚI điểm cuối theo /joint_states" if reached else "⚠ CHƯA tới theo /joint_states"
        node.get_logger().info(f"{tag} | max|Δq|={err:.3f} rad")

    return ok

# ----------------- Kế hoạch A→B với VIA nâng cao -----------------
def plan_A_to_B_with_via(qA, qB, lift=LIFT_RAD):
    """Lộ trình: A -> A_up -> B_up -> B (đi vòng trên cao)."""
    qA_up = qA.copy()
    qB_up = qB.copy()
    # Giảm joint3 (âm thêm) để “nâng” EE
    qA_up[2] = clamp_joint_limits(qA_up)[2] - abs(lift)
    qB_up[2] = clamp_joint_limits(qB_up)[2] - abs(lift)

    seg1 = interp_traj(qA,    qA_up)  # A -> A_up
    seg2 = interp_traj(qA_up, qB_up)  # A_up -> B_up
    seg3 = interp_traj(qB_up, qB)     # B_up -> B
    return seg1 + seg2 + seg3

# ----------------- Log sai số tới mục tiêu -----------------
def log_err_to(node, tag, q_goal):
    q_cur = node.current_positions()
    if q_cur is None:
        node.get_logger().info(f"[{tag}] chưa có /joint_states.")
        return None
    err = np.max(np.abs(ang_wrap(np.array(q_cur) - np.array(q_goal))))
    node.get_logger().info(f"[{tag}] max|Δq|={err:.3f} rad")
    return err

# ----------------- MAIN -----------------
def main():
    Jogger = import_jogger()

    rclpy.init()
    node = Jogger()  # có action client + service set_entity_state

    try:
        # 0) Đọc vị trí hiện tại nếu có; nếu chưa thì dùng HOME
        q_now = node.current_positions() or Q_HOME

        # 1) Về HOME (mượt)
        traj_home = interp_traj(q_now, Q_HOME)
        send_trajectory(node, traj_home, point_dt=POINT_DT)
        log_err_to(node, "HOME", Q_HOME)

        # 2) Tới Pose A (mượt)
        traj_to_A = interp_traj(Q_HOME, Q_A)
        send_trajectory(node, traj_to_A, point_dt=POINT_DT)
        log_err_to(node, "→A", Q_A)

        # 3) Grab (SNAP→ATTACH)
        node.do_grab()
        time.sleep(0.2)

        # 4) Log obstacle
        obs_p, _ = node.get_model_pose(OBSTACLE_NAME)
        if obs_p is not None:
            node.get_logger().info(f"[Obstacle] {OBSTACLE_NAME} at world xyz = {np.round(obs_p,3)}")
        else:
            node.get_logger().info(f"[Obstacle] {OBSTACLE_NAME} not found in /gazebo/model_states")

        # 5) A → B qua VIA (mượt)
        q_list_AB = plan_A_to_B_with_via(Q_A, Q_B, lift=LIFT_RAD)
        send_trajectory(node, q_list_AB, point_dt=POINT_DT, reach_tol=0.04)
        log_err_to(node, "→B", Q_B)

        # 6) Drop
        node.do_drop()
        time.sleep(0.2)

        # 7) Về HOME (mượt)
        traj_back = interp_traj(Q_B, Q_HOME)
        send_trajectory(node, traj_back, point_dt=POINT_DT)
        log_err_to(node, "BACK", Q_HOME)

        node.get_logger().info("✓ Hoàn tất chu trình pick→via→place→home (smooth).")

    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

