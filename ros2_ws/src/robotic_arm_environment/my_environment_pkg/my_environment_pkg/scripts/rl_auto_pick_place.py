#!/usr/bin/env python3
# rl_auto_pick_place.py
# Home -> Pose A -> GRAB -> (avoid obstacle_box_1) -> Pose B -> DROP -> Home

import time, sys, pathlib, importlib.util
import numpy as np
import rclpy

_THIS_DIR = pathlib.Path(__file__).parent.resolve()
sys.path.insert(0, str(_THIS_DIR))

def load_jogger_class():
    for p in _THIS_DIR.glob("*.py"):
        if p.name == pathlib.Path(__file__).name:
            continue
        spec = importlib.util.spec_from_file_location(p.stem, str(p))
        try:
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            if hasattr(mod, "Jogger"):
                return mod.Jogger
        except Exception:
            pass
    raise ImportError("Không tìm thấy class Jogger trong cùng thư mục.")
Jogger = load_jogger_class()

def _spin_for(node, sec=0.02):
    t_end = time.time() + sec
    while rclpy.ok() and time.time() < t_end:
        rclpy.spin_once(node, timeout_sec=0.01)

def _wait_for_joints(node, timeout=10.0):
    t0 = time.time()
    while rclpy.ok() and not node.has_joints:
        rclpy.spin_once(node, timeout_sec=0.1)
        if time.time() - t0 > timeout:
            node.get_logger().warn('Chưa thấy /joint_states sau 10s.')
            break

def _go(node: "Jogger", q, tsec=2.0):
    ok = node.go_to_pose(q, tsec)
    if not ok:
        node.get_logger().error(f'Không đi tới pose: {np.round(q,4)}')
    return ok

def _current_q(node: "Jogger"):
    q = node.current_positions()
    return np.array(q, dtype=float) if q is not None else None

def _ee_p(node: "Jogger"):
    p, _ = node.ee_pose()
    return np.array(p, dtype=float) if p is not None else None

def _model_p(node: "Jogger", name: str):
    p, _ = node.get_model_pose(name)
    return np.array(p, dtype=float) if p is not None else None

def heuristic_delta_q(q, q_goal, ee_p, obs_p,
                      step=0.06, avoid_radius=0.30, avoid_axis=0):
    dq_to_goal = q_goal - q
    dq = np.clip(dq_to_goal, -step, step)
    if ee_p is not None and obs_p is not None:
        d = np.linalg.norm(ee_p - obs_p)
        if d < avoid_radius:
            # Né đơn giản bằng joint0 (quay đế) sang một phía
            side = 1.0 if (ee_p[1] - obs_p[1]) >= 0.0 else -1.0
            dq[avoid_axis] += 0.05 * side
            dq = np.clip(dq, -step, step)
    return dq

def main():
    rclpy.init()
    node = Jogger()
    try:
        obstacle_name = 'obstacle_box_1'  # phải trùng tên bạn spawn
        q_home = np.array([0,0,0,0,0,0], dtype=float)
        q_a    = np.array([0,0,-1.5,0,-1.57,0], dtype=float)
        q_b    = np.array([ 3.9988, -0.5012, -0.9967,  0.0103, -1.599 , -0.0956], dtype=float)

        max_steps_to_b = 180
        step_time      = 0.18
        step_size      = 0.04
        tol_goal_q     = 0.03
        avoid_radius   = 0.30  # có thể tăng 0.25 nếu muốn né xa hơn

        _wait_for_joints(node)
        node.get_logger().info('=== BẮT ĐẦU PICK-PLACE + AVOID (heuristic) ===')

        # 1) Home
        _go(node, q_home.tolist(), 2.0); _spin_for(node, 0.25)

        # 2) Pose A
        _go(node, q_a.tolist(), max(1.5, getattr(node, "pose_a_time", 2.0))); _spin_for(node, 0.25)

        # 3) Grab
        node.do_grab(); _spin_for(node, 0.3)

        # 4) Điều hướng A->B có né obstacle
        q = _current_q(node)
        for k in range(max_steps_to_b):
            ee  = _ee_p(node)
            obs = _model_p(node, obstacle_name)
            if np.linalg.norm(q_b - q) < tol_goal_q:
                node.get_logger().info(f'Đã tới gần Pose B sau {k} bước.')
                break
            dq = heuristic_delta_q(q, q_b, ee, obs,
                                   step=step_size, avoid_radius=avoid_radius, avoid_axis=0)
            q_next = q + dq
            _go(node, q_next.tolist(), step_time)
            q = _current_q(node)
            _spin_for(node, 0.05)

        # 5) Chốt vào B (snap-to nhỏ)
        _go(node, q_b.tolist(), 1.5); _spin_for(node, 0.2)

        # 6) Drop
        node.do_drop(); _spin_for(node, 0.3)

        # 7) Home
        _go(node, q_home.tolist(), 2.0)
        node.get_logger().info('✓ Hoàn tất.')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

