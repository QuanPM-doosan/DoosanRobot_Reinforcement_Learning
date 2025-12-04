#!/usr/bin/env python3
# Auto Pick & Place for Doosan (ROS 2 Foxy)
# Sequence:
#   Home [0 0 0 0 0 0] -> Pose A [0 0 -1.5 0 -1.57 0] -> GRAB ->
#   Pose B [3.9988 -0.5012 -0.9967 0.0103 -1.599 -0.0956] -> DROP -> Home

import time
import numpy as np
import rclpy

# ==== Dynamic loader: tìm file có class Jogger trong cùng thư mục ====
import sys, pathlib, importlib.util

_THIS_DIR = pathlib.Path(__file__).parent.resolve()
sys.path.insert(0, str(_THIS_DIR))  # để import các file cùng thư mục khi cần

def load_jogger_class():
    # ưu tiên tên phổ biến
    candidates = [
        "jog_doosan_marker.py",
        "doosan_jog_to_marker.py",
        "jogger.py",
        "jogger_doosan.py",
    ]
    tried = set()

    # 1) thử các tên phổ biến trước
    for name in candidates:
        p = _THIS_DIR / name
        if p.exists() and p.is_file():
            spec = importlib.util.spec_from_file_location(p.stem, str(p))
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            if hasattr(mod, "Jogger"):
                return mod.Jogger
            tried.add(str(p))

    # 2) quét toàn bộ *.py (trừ bản thân file này)
    for p in _THIS_DIR.glob("*.py"):
        if p.name == pathlib.Path(__file__).name:
            continue
        if str(p) in tried:
            continue
        try:
            spec = importlib.util.spec_from_file_location(p.stem, str(p))
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            if hasattr(mod, "Jogger"):
                return mod.Jogger
        except Exception:
            continue

    raise ImportError(
        "Không tìm thấy lớp Jogger trong cùng thư mục.\n"
        f"Đặt file chứa class Jogger cạnh file này (vd: jog_doosan_marker.py) rồi chạy lại.\n"
        f"Thư mục đã quét: {_THIS_DIR}"
    )

Jogger = load_jogger_class()

# ==== Helpers ====
def _wait_for_joints(node, timeout=10.0):
    t0 = time.time()
    while rclpy.ok() and not node.has_joints:
        rclpy.spin_once(node, timeout_sec=0.1)
        if time.time() - t0 > timeout:
            node.get_logger().warn('Chưa thấy /joint_states sau 10s. Vẫn tiếp tục chạy.')
            break

def _go(node: "Jogger", q, tsec=4.0):
    ok = node.go_to_pose(q, tsec)
    if not ok:
        node.get_logger().error(f'Không đi được tới pose: {np.round(q, 4)}')
    return ok

def _tick(node: "Jogger", duration_s: float):
    t_end = time.time() + float(duration_s)
    while rclpy.ok() and time.time() < t_end:
        rclpy.spin_once(node, timeout_sec=0.02)

def main():
    rclpy.init()
    node = Jogger()
    try:
        # ====== Waypoints ======
        q_home = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        q_a    = [0.0, 0.0, -1.5, 0.0, -1.57, 0.0]
        q_b    = [3.9988, -0.5012, -0.9967, 0.0103, -1.599, -0.0956]

        # ====== Chuẩn bị ======
        _wait_for_joints(node, timeout=10.0)
        node.get_logger().info('=== BẮT ĐẦU KỊCH BẢN TỰ ĐỘNG ===')

        # 1) Về Home
        node.get_logger().info('1) Về Home [0,0,0,0,0,0]')
        _go(node, q_home, 2.0)
        _tick(node, 0.3)

        # 2) Tới Pose A
        node.get_logger().info('2) Tới Pose A [0,0,-1.5,0,-1.57,0]')
        t_pose_a = getattr(node, "pose_a_time", 2.0) or 2.0
        _go(node, q_a, t_pose_a)

        # 3) GRAB
        node.get_logger().info('3) GRAB vật (SNAP rồi ATTACH)')
        node.do_grab()
        _tick(node, 0.3)

        # 4) Đi tới Pose B (attach timer trong Jogger sẽ giữ vật bám link6)
        node.get_logger().info('4) Di chuyển tới Pose B mang theo vật')
        _go(node, q_b, 2.5)

        # 5) DROP
        node.get_logger().info('5) DROP (nhả vật)')
        node.do_drop()
        _tick(node, 0.3)

        # 6) Quay về Home
        node.get_logger().info('6) Quay về Home')
        _go(node, q_home, 2.0)

        node.get_logger().info('✓ Hoàn tất kịch bản tự động.')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

