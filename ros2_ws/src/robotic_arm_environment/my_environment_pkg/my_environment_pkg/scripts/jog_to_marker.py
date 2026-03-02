#!/usr/bin/env python3
# Jog Doosan + dò tới marker_A, tích hợp pick/drop + snap/grab_snap/grab_at_A
# BẢN FIX: bám chắc theo KHUNG link6 + tăng tần số bám (ROS 2 Foxy)

import time
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.duration import Duration

from sensor_msgs.msg import JointState
from gazebo_msgs.msg import ModelStates, LinkStates
from gazebo_msgs.srv import SetEntityState
from trajectory_msgs.msg import JointTrajectoryPoint
from control_msgs.action import FollowJointTrajectory
from geometry_msgs.msg import Pose

import tf2_ros
from tf2_ros import TransformException

JOINT_ORDER = ['joint1','joint2','joint3','joint4','joint5','joint6']
WORLD = 'world'

class Jogger(Node):
    def __init__(self):
        super().__init__('doosan_jog_to_marker')

        # ================== Parameters ==================
        self.declare_parameter('target_model', 'marker_A')      # để so khoảng cách
        self.declare_parameter('pick_model',   'pick_object')   # vật cần pick/drop
        # Nếu link trong Gazebo là 'doosan::link6' => ee_link_name='link6', ee_model_prefix='doosan::'
        self.declare_parameter('ee_link_name', 'link6')
        self.declare_parameter('ee_model_prefix', '')
        # Pose A để “đưa tay tới A rồi grab”
        self.declare_parameter('pose_a', [0.0, 0.0, -1.5, 0.0, -1.57, 0.0])
        self.declare_parameter('pose_a_time', 2.0)
        # Bù chiều dài tool theo trục Z (m) khi snap/grab_snap/grab_at_A
        self.declare_parameter('grasp_offset_z', 0.0)

        self.target_model    = self.get_parameter('target_model').value
        self.pick_model      = self.get_parameter('pick_model').value
        self.ee_link_name    = self.get_parameter('ee_link_name').value
        self.ee_model_prefix = self.get_parameter('ee_model_prefix').value
        self.pose_a          = list(self.get_parameter('pose_a').value)
        self.pose_a_time     = float(self.get_parameter('pose_a_time').value)
        self.grasp_offset_z  = float(self.get_parameter('grasp_offset_z').value)

        # ================== TF & Subscribers ==================
        self.tf_buffer   = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.joint_sub = self.create_subscription(JointState, '/joint_states', self.on_joint, 10)
        self.model_sub = self.create_subscription(ModelStates, '/gazebo/model_states', self.on_models, 10)
        self.link_sub  = self.create_subscription(LinkStates,  '/gazebo/link_states',  self.on_links, 10)

        # ================== Action client (Jog) ==================
        self.ac = ActionClient(self, FollowJointTrajectory,
                               '/joint_trajectory_controller/follow_joint_trajectory')
        self.get_logger().info('Đang chờ action server /joint_trajectory_controller/follow_joint_trajectory ...')
        self.ac.wait_for_server()
        self.get_logger().info('OK: server sẵn sàng.')

        # ================== SetEntityState client (pick/drop/snap) ==================
        self.set_state_cli = self.create_client(SetEntityState, '/gazebo/set_entity_state')
        self.get_logger().info('Chờ dịch vụ /gazebo/set_entity_state ...')
        self.set_state_cli.wait_for_service()
        self.get_logger().info('OK: /gazebo/set_entity_state sẵn sàng.')

        # ================== States ==================
        self.joint_map  = {}
        self.has_joints = False

        self._model_names = []
        self._model_poses = []
        self.has_models   = False

        self.link_names = []
        self.link_poses = []
        self.has_links  = False

        # Pick/Attach state
        self.attached = False
        self.attach_offset_world = np.array([0.0, 0.0, 0.0])  # (không dùng khi bám theo khung link6)
        # Tăng tần số bám: 200 Hz (0.005s)
        self.attach_timer = self.create_timer(0.002, self._attach_follow_step)

    # ================== Callbacks ==================
    def on_joint(self, msg: JointState):
        for n, p in zip(msg.name, msg.position):
            self.joint_map[n] = p
        self.has_joints = all(n in self.joint_map for n in JOINT_ORDER)

    def on_models(self, msg: ModelStates):
        self._model_names = list(msg.name)
        self._model_poses = list(msg.pose)
        self.has_models = True

    def on_links(self, msg: LinkStates):
        self.link_names = list(msg.name)
        self.link_poses = list(msg.pose)
        self.has_links  = True

    # ================== Helpers: current joints ==================
    def current_positions(self):
        if not self.has_joints:
            return None
        return [float(self.joint_map[n]) for n in JOINT_ORDER]

    # ================== Helpers: EE pose ==================
    def ee_xyz_via_tf(self):
        try:
            tr = self.tf_buffer.lookup_transform(WORLD, self.ee_link_name, rclpy.time.Time())
            p = tr.transform.translation
            q = tr.transform.rotation
            return np.array([p.x, p.y, p.z], dtype=float), np.array([q.x, q.y, q.z, q.w], dtype=float)
        except TransformException:
            return None, None

    def ee_xyz_via_links(self):
        if not self.has_links:
            return None, None
        want = f"{self.ee_model_prefix}{self.ee_link_name}"
        try:
            i = self.link_names.index(want)
        except ValueError:
            cand = [k for k, nm in enumerate(self.link_names) if nm.endswith(self.ee_link_name)]
            if not cand:
                return None, None
            i = cand[0]
        p = self.link_poses[i].position
        o = self.link_poses[i].orientation
        return np.array([p.x, p.y, p.z], dtype=float), np.array([o.x, o.y, o.z, o.w], dtype=float)

    def ee_pose(self):
        pos, quat = self.ee_xyz_via_tf()
        if pos is not None:
            return pos, quat
        return self.ee_xyz_via_links()

    # ================== Helpers: model pose ==================
    def get_model_pose(self, model_name):
        if not self.has_models:
            return None, None
        try:
            i = self._model_names.index(model_name)
        except ValueError:
            return None, None
        p = self._model_poses[i].position
        o = self._model_poses[i].orientation
        return np.array([p.x, p.y, p.z], dtype=float), np.array([o.x, o.y, o.z, o.w], dtype=float)

    # ================== Jog send (đÃ SỬA CHO FOXY) ==================
    def send_goal(self, q, tsec=2.0):
        pt = JointTrajectoryPoint()
        pt.positions = list(q)
        #pt.velocities = [0.0]*6
        #pt.accelerations = [0.0]*6
        pt.time_from_start = Duration(seconds=float(tsec)).to_msg()

        goal = FollowJointTrajectory.Goal()
        goal.goal_time_tolerance = Duration(seconds=0.8).to_msg()
        goal.trajectory.joint_names = JOINT_ORDER
        goal.trajectory.points = [pt]

        self.get_logger().info(f'Gửi quỹ đạo: {np.round(q,3)} trong {tsec}s')

        # Gửi goal
        fut = self.ac.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, fut)
        gh = fut.result()
        if gh is None or not gh.accepted:
            self.get_logger().error('Goal bị từ chối.')
            return False

        # Chờ kết quả (Foxy: FollowJointTrajectory_Result, đọc trực tiếp .error_code)
        resf = gh.get_result_async()
        rclpy.spin_until_future_complete(self, resf)
        res = resf.result()

        ok = (res is not None and getattr(res, 'error_code', 1) == 0)
        if ok:
            self.get_logger().info('✓ Hoàn thành.')
        else:
            err = getattr(res, 'error_string', '')
            self.get_logger().warn(f'⚠ Controller trả về lỗi. error_code={getattr(res, "error_code", "NA")} {err}')
        return ok

    def go_to_pose(self, q, tsec=None):
        if tsec is None:
            tsec = 2.0
        return self.send_goal(q, tsec)

    # ================== Teleport model tới EE (theo world) ==================
    def teleport_model_to_ee(self, model_name, offset_world=np.zeros(3)):
        ee_p, ee_q = self.ee_pose()
        if ee_p is None:
            print('⚠ Không có EE pose (TF/link_states).')
            return False

        want = ee_p + offset_world

        req = SetEntityState.Request()
        req.state.name = model_name
        req.state.reference_frame = WORLD
        pose = Pose()
        pose.position.x = float(want[0])
        pose.position.y = float(want[1])
        pose.position.z = float(want[2])
        pose.orientation.x = float(ee_q[0])
        pose.orientation.y = float(ee_q[1])
        pose.orientation.z = float(ee_q[2])
        pose.orientation.w = float(ee_q[3])
        req.state.pose = pose

        fut = self.set_state_cli.call_async(req)
        rclpy.spin_until_future_complete(self, fut, timeout_sec=1.0)
        ok = (fut.result() is not None and fut.result().success)
        print('SNAP model → EE:', 'OK' if ok else 'FAIL')
        return ok

    # ================== Attach/Detach logic ==================
    def _attach_follow_step(self):
        """Nếu đang attached: bám theo KHUNG link6 (ổn định, không rơi)."""
        if not self.attached:
            return

        # Offset LOCAL theo trục Z của tool (đơn giản): dùng grasp_offset_z để ôm sát mỏ kẹp.
        off = np.array([0.0, 0.0, self.grasp_offset_z], dtype=float)

        req = SetEntityState.Request()
        req.state.name = self.pick_model
        # BÁM TRONG KHUNG link6 (vd: 'doosan::link6')
        req.state.reference_frame = f"{self.ee_model_prefix}{self.ee_link_name}"

        pose = Pose()
        pose.position.x = float(off[0])
        pose.position.y = float(off[1])
        pose.position.z = float(off[2])
        # Orientation = identity trong khung link6 → cùng hướng với tool
        pose.orientation.x = 0.0
        pose.orientation.y = 0.0
        pose.orientation.z = 0.0
        pose.orientation.w = 1.0

        req.state.pose = pose
        self.set_state_cli.call_async(req)

    def do_grab(self):
        """Auto-SNAP về mũi tool rồi attach; bám theo khung link6 với offset local Z = grasp_offset_z."""
        # Kéo vật tới đúng EE (nếu đang xa), dùng offset world Z = grasp_offset_z
        off_world = np.array([0.0, 0.0, self.grasp_offset_z], dtype=float)
        if not self.teleport_model_to_ee(self.pick_model, off_world):
            print('⚠ SNAP thất bại, không thể GRAB.')
            return
        # Bật trạng thái bám (theo khung link6 ở _attach_follow_step)
        self.attached = True
        print('✅ GRAB: SNAP→ATTACH (bám theo khung link6), grasp_offset_z =', self.grasp_offset_z)

    def do_drop(self):
        """Ngắt attached và thả vật xuống 2 cm để dễ quan sát."""
        if not self.attached:
            print('Đã ở trạng thái detach.')
            return
        self.attached = False

        obj_p, obj_q = self.get_model_pose(self.pick_model)
        if obj_p is not None:
            req = SetEntityState.Request()
            req.state.name = self.pick_model
            req.state.reference_frame = WORLD
            pose = Pose()
            pose.position.x = float(obj_p[0])
            pose.position.y = float(obj_p[1])
            pose.position.z = float(obj_p[2] - 0.02)
            pose.orientation.x = float(obj_q[0])
            pose.orientation.y = float(obj_q[1])
            pose.orientation.z = float(obj_q[2])
            pose.orientation.w = float(obj_q[3])
            req.state.pose = pose
            self.set_state_cli.call_async(req)
        print('🟦 DROP: detached.')

    def do_grab_snap(self):
        """SNAP vật tới EE rồi attach (giống grab nhưng viết tách lệnh)."""
        off = np.array([0.0, 0.0, self.grasp_offset_z], dtype=float)
        if not self.teleport_model_to_ee(self.pick_model, off):
            print('⚠ Không thể SNAP vật tới EE.')
            return
        self.attached = True
        print('✅ GRAB_SNAP: attached (bám theo khung link6), grasp_offset_z =', self.grasp_offset_z)

    def do_grab_at_A(self):
        """Đi tới Pose A, SNAP vật tới EE, rồi attach."""
        if not (isinstance(self.pose_a, (list, tuple)) and len(self.pose_a) == 6):
            print('⚠ pose_a không hợp lệ. Dùng -p pose_a:="[...6 số...]"')
            return
        print('→ Đi tới Pose A:', np.round(self.pose_a, 4))
        if not self.go_to_pose(self.pose_a, self.pose_a_time):
            print('⚠ Không đi được tới Pose A.')
            return
        off = np.array([0.0, 0.0, self.grasp_offset_z], dtype=float)
        if not self.teleport_model_to_ee(self.pick_model, off):
            print('⚠ Không thể SNAP vật tới EE tại Pose A.')
            return
        self.attached = True
        print('✅ GRAB_AT_A: attached tại Pose A (bám theo khung link6), grasp_offset_z =', self.grasp_offset_z)

    # ================== Status ==================
    def print_status(self):
        q = self.current_positions()
        ee_p, _ = self.ee_pose()
        tgt_p, _ = self.get_model_pose(self.target_model)
        pick_p, _ = self.get_model_pose(self.pick_model)

        print('\n=== TRẠNG THÁI HIỆN TẠI ===')
        if q is not None: print('Khớp (rad):', np.round(q, 4))
        else:             print('Khớp: (chưa có /joint_states)')
        if ee_p is not None: print('EE (world):', np.round(ee_p, 4))
        else:                print('EE: (chưa có TF và chưa tìm thấy trong /gazebo/link_states)')
        if tgt_p is not None: print(f'{self.target_model} (world):', np.round(tgt_p, 4))
        else:                 print(f'{self.target_model}: (chưa thấy trong /gazebo/model_states)')
        if pick_p is not None: print(f'{self.pick_model} (world):', np.round(pick_p, 4))
        else:                  print(f'{self.pick_model}: (chưa thấy trong /gazebo/model_states)')

        if ee_p is not None and tgt_p is not None:
            d = np.linalg.norm(ee_p - tgt_p)
            print('Khoảng cách EE ↔ target:', round(float(d), 4), 'm')
        if ee_p is not None and pick_p is not None:
            d2 = np.linalg.norm(ee_p - pick_p)
            print('Khoảng cách EE ↔ pick_object:', round(float(d2), 4), 'm')

        print('Attached:', self.attached, ' | grasp_offset_z:', self.grasp_offset_z)

    # ================== Interactive loop ==================
    def loop(self):
        # Chuẩn hoá dấu trừ unicode → '-'
        def normalize_minus(s: str) -> str:
            return (s.replace('–','-').replace('—','-').replace('−','-')
                      .replace('﹣','-').replace('―','-'))

        # chờ /joint_states
        t0 = time.time()
        while rclpy.ok() and (not self.has_joints):
            if time.time() - t0 > 10.0:
                self.get_logger().warn('Chưa có /joint_states. Kiểm tra controller.')
                break
            rclpy.spin_once(self, timeout_sec=0.1)

        print(f"""
HƯỚNG DẪN:
  - '<joint> <delta_rad>' : jog 1 khớp (vd: '3 -0.5')
  - 'set'                 : nhập 6 giá trị tuyệt đối (rad)
  - Nhập trực tiếp 6 số   : coi như 'set' (vd: '0 0 -1 0 -1 0')
  - 'delta a b c d e f'   : cộng 6 delta vào khớp hiện tại
  - 'grab'                : auto-SNAP → attach (bám theo khung link6)
  - 'drop'                : nhả vật (detach, thả xuống 2 cm)
  - 'snap'                : teleport vật tới đúng EE (có bù grasp_offset_z), CHƯA attach
  - 'grab_snap'           : snap rồi attach luôn (bám theo khung link6)
  - 'grab_at_A'           : đi tới Pose A → snap → attach
  - 'show' | 'home' | 'save' | 'q'
Gợi ý:
  - Nếu EE không hiện, script sẽ fallback đọc từ /gazebo/link_states.
  - Nếu tên link EE trong Gazebo là 'doosan::link6' → chạy thêm:
      --ros-args -p ee_link_name:=link6 -p ee_model_prefix:=doosan::
""")
        self.print_status()

        while rclpy.ok():
            try:
                raw = input('\nLệnh > ')
            except (EOFError, KeyboardInterrupt):
                break
            cmd = normalize_minus(raw.strip())
            if not cmd:
                continue

            if cmd in ('q','quit','exit'):
                break
            if cmd == 'show':
                self.print_status(); continue
            if cmd == 'home':
                q = [0.0]*6; self.send_goal(q); self.print_status(); continue
            if cmd == 'save':
                q = self.current_positions()
                if q is None: print('Chưa có khớp.')
                else:
                    arr = ', '.join([f'{v:.6f}' for v in q])
                    print('Copy pose khớp hiện tại:\n[', arr, ']')
                continue
            if cmd == 'grab':
                self.do_grab(); self.print_status(); continue
            if cmd == 'drop':
                self.do_drop(); self.print_status(); continue
            if cmd == 'snap':
                off = np.array([0.0, 0.0, self.grasp_offset_z], dtype=float)
                self.teleport_model_to_ee(self.pick_model, off); self.print_status(); continue
            if cmd == 'grab_snap':
                self.do_grab_snap(); self.print_status(); continue
            if cmd == 'grab_at_A':
                self.do_grab_at_A(); self.print_status(); continue

            if cmd == 'set':
                try:
                    s = input('Nhập 6 giá trị rad, cách nhau bởi khoảng trắng:\n> ')
                    s = normalize_minus(s).strip().split()
                    if len(s) != 6:
                        print('Cần đúng 6 số.'); continue
                    q = [float(x) for x in s]
                except Exception:
                    print('Sai định dạng.'); continue
                self.send_goal(q); self.print_status(); continue

            # delta a b c d e f
            if cmd.lower().startswith('delta'):
                parts = cmd.split()
                if len(parts) != 7:
                    print("Cú pháp: delta d1 d2 d3 d4 d5 d6 (đơn vị rad)")
                    continue
                base = self.current_positions()
                if base is None:
                    print('Chưa có khớp.'); continue
                try:
                    deltas = [float(x) for x in parts[1:]]
                except Exception:
                    print('Delta phải là số.'); continue
                q = [b + d for b, d in zip(base, deltas)]
                self.send_goal(q); self.print_status(); continue

            # 6 số → set
            toks = cmd.split()
            if len(toks) == 6:
                try:
                    q = [float(x) for x in toks]
                except Exception:
                    print('Sai định dạng 6 số.'); continue
                self.send_goal(q); self.print_status(); continue

            # "<joint> <delta>"
            if len(toks) == 2 and toks[0].isdigit():
                idx = int(toks[0])
                try:
                    delta = float(toks[1])
                except Exception:
                    print('Delta phải là số (rad).'); continue
                if not (1 <= idx <= 6):
                    print('Joint index phải 1..6.'); continue
                base = self.current_positions()
                if base is None:
                    print('Chưa có khớp.'); continue
                q = base[:]; q[idx-1] += delta
                self.send_goal(q); self.print_status(); continue

            print('Lệnh không hợp lệ. Gõ "show", "home", "set", "save", "delta ...", '
                  '"grab", "drop", "snap", "grab_snap", "grab_at_A" hoặc "<joint> <delta_rad>".')

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

