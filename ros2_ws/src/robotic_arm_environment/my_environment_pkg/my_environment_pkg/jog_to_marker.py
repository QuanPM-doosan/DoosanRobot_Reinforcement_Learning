#!/usr/bin/env python3
# Jog cánh tay Doosan để dò tới marker_A (không cần MoveIt)
# ROS 2 Foxy

import sys, time, math
import rclpy
import numpy as np
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.duration import Duration

from sensor_msgs.msg import JointState
from gazebo_msgs.msg import ModelStates
from trajectory_msgs.msg import JointTrajectoryPoint
from control_msgs.action import FollowJointTrajectory

import tf2_ros
from tf2_ros import TransformException

JOINT_ORDER = ['joint1','joint2','joint3','joint4','joint5','joint6']  # thứ tự controller
EE_LINK = 'link6'
WORLD = 'world'
TARGET_MODEL = 'marker_A'  # đổi nếu bạn muốn dò tới object khác

class Jogger(Node):
    def __init__(self):
        super().__init__('doosan_jog_to_marker')

        # TF
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Sub
        self.joint_sub = self.create_subscription(JointState, '/joint_states', self.on_joint, 10)
        self.model_sub = self.create_subscription(ModelStates, '/gazebo/model_states', self.on_models, 10)

        # Action client
        self.ac = ActionClient(self, FollowJointTrajectory, '/joint_trajectory_controller/follow_joint_trajectory')

        # State
        self.joint_map = {}   # name -> pos
        self.has_joints = False
        self.target_xyz = None
        self.has_target = False

        # Wait for server (non-blocking)
        self.get_logger().info('Đang chờ action server /joint_trajectory_controller/follow_joint_trajectory ...')
        self.ac.wait_for_server()
        self.get_logger().info('OK: server sẵn sàng.')

    # --- callbacks ---
    def on_joint(self, msg: JointState):
        for n, p in zip(msg.name, msg.position):
            self.joint_map[n] = p
        # đủ 6 khớp?
        self.has_joints = all(n in self.joint_map for n in JOINT_ORDER)

    def on_models(self, msg: ModelStates):
        try:
            i = msg.name.index(TARGET_MODEL)
            p = msg.pose[i].position
            self.target_xyz = np.array([p.x, p.y, p.z], dtype=float)
            self.has_target = True
        except ValueError:
            self.has_target = False

    # --- helpers ---
    def current_positions(self):
        if not self.has_joints:
            return None
        return [float(self.joint_map[n]) for n in JOINT_ORDER]

    def send_goal(self, q, tsec=2.0):
        pt = JointTrajectoryPoint()
        pt.positions = list(q)
        #pt.velocities = [0.0]*6
        #pt.accelerations = [0.0]*6
        pt.time_from_start = Duration(seconds=float(tsec)).to_msg()

        from control_msgs.action import FollowJointTrajectory
        goal = FollowJointTrajectory.Goal()
        goal.goal_time_tolerance = Duration(seconds=1.0).to_msg()
        goal.trajectory.joint_names = JOINT_ORDER
        goal.trajectory.points = [pt]

        self.get_logger().info(f'Gửi quỹ đạo: {np.round(q,3)} trong {tsec}s')
        fut = self.ac.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, fut)
        gh = fut.result()
        if not gh or not gh.accepted:
            self.get_logger().error('Goal bị từ chối.')
            return False

        resf = gh.get_result_async()
        rclpy.spin_until_future_complete(self, resf)
        res = resf.result()
        ok = (res is not None and res.result.error_code == 0)
        if ok:
            self.get_logger().info('✓ Hoàn thành.')
        else:
            self.get_logger().warn('⚠ Có lỗi từ controller.')
        return ok

    def ee_xyz(self):
        try:
            tr = self.tf_buffer.lookup_transform(WORLD, EE_LINK, rclpy.time.Time())
            x = tr.transform.translation.x
            y = tr.transform.translation.y
            z = tr.transform.translation.z
            return np.array([x,y,z], dtype=float)
        except TransformException as ex:
            return None

    def print_status(self):
        q = self.current_positions()
        ee = self.ee_xyz()
        tgt = self.target_xyz if self.has_target else None
        print('\n=== TRẠNG THÁI HIỆN TẠI ===')
        if q is not None:
            print('Khớp (rad):', np.round(q, 4))
        else:
            print('Khớp: (chưa có dữ liệu /joint_states)')

        if ee is not None:
            print('EE (world):', np.round(ee, 4))
        else:
            print('EE: (chưa có TF world→link6)')

        if tgt is not None:
            print(f'{TARGET_MODEL} (world):', np.round(tgt, 4))
        else:
            print(f'{TARGET_MODEL}: (chưa thấy trong /gazebo/model_states)')

        if ee is not None and tgt is not None:
            d = np.linalg.norm(ee - tgt)
            print('Khoảng cách EE ↔ target:', round(float(d), 4), 'm')

    # --- interactive loop ---
    def loop(self):
        # chờ dữ liệu ban đầu
        t0 = time.time()
        while rclpy.ok() and (not self.has_joints):
            if time.time() - t0 > 10.0:
                self.get_logger().warn('Chưa có /joint_states. Kiểm tra controller.')
                break
            rclpy.spin_once(self, timeout_sec=0.1)

        print("""
HƯỚNG DẪN:
  - Nhập: <joint_index> <delta_rad>
      vd: '1 0.1'  (tăng joint1 thêm +0.1 rad)
          '3 -0.05' (giảm joint3 0.05 rad)
  - 'home'      → đưa tất cả khớp về 0
  - 'set'       → nhập trực tiếp 6 giá trị (rad) cho 6 khớp
  - 'show'      → in trạng thái (EE, marker, khoảng cách)
  - 'save'      → in mảng khớp hiện tại để copy vào code
  - 'q'         → thoát
""")

        # in trạng thái đầu
        self.print_status()

        while rclpy.ok():
            try:
                cmd = input('\nLệnh > ').strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not cmd:
                continue

            if cmd in ('q','quit','exit'):
                break

            if cmd == 'show':
                self.print_status()
                continue

            if cmd == 'home':
                q = [0.0]*6
                self.send_goal(q)
                self.print_status()
                continue

            if cmd == 'save':
                q = self.current_positions()
                if q is None:
                    print('Chưa có khớp.')
                else:
                    arr = ', '.join([f'{v:.6f}' for v in q])
                    print('Copy pose khớp hiện tại:\n[', arr, ']')
                continue

            if cmd == 'set':
                try:
                    s = input('Nhập 6 giá trị rad, cách nhau bởi khoảng trắng:\n> ').strip().split()
                    if len(s) != 6:
                        print('Cần đúng 6 số.')
                        continue
                    q = [float(x) for x in s]
                except Exception:
                    print('Sai định dạng.')
                    continue
                self.send_goal(q)
                self.print_status()
                continue

            # định dạng: "<idx> <delta>"
            parts = cmd.split()
            if len(parts) == 2 and parts[0].isdigit():
                idx = int(parts[0])
                try:
                    delta = float(parts[1])
                except Exception:
                    print('Delta phải là số (rad).')
                    continue
                if not (1 <= idx <= 6):
                    print('Joint index phải 1..6.')
                    continue
                base = self.current_positions()
                if base is None:
                    print('Chưa có khớp.')
                    continue
                q = base[:]
                q[idx-1] += delta
                self.send_goal(q)
                self.print_status()
                continue

            print('Lệnh không hợp lệ. Gõ "show", "home", "set", "save", hoặc "<joint> <delta_rad>".')

def main():
    rclpy.init()
    node = Jogger()
    try:
        node.loop()
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

