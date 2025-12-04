'''
Author: David Valencia (gốc) + chỉnh sửa
Mô tả:
- Môi trường RL/điều khiển cánh tay.
- Đọc JointState, ModelStates; gửi FollowJointTrajectory.
- [BỔ SUNG] Fallback quaternion (không cần tf_transformations), guard MoveIt,
  và hàm IK tới marker_A (ik_to_marker_A).

Lưu ý:
- ik_to_marker_A() cần MoveIt (move_group) + service /compute_ik đang chạy.
- Nếu chưa cài MoveIt, code vẫn chạy phần khác; gọi IK sẽ báo lỗi hướng dẫn cài.
'''

import os
import sys
import time
import math
import rclpy
import random
import numpy as np
import message_filters

from rclpy.node import Node
from rclpy.duration import Duration

from sensor_msgs.msg import JointState
from gazebo_msgs.msg import ModelStates
from gazebo_msgs.srv import SetEntityState

import tf2_ros
from tf2_ros import TransformException

from rclpy.action        import ActionClient
from trajectory_msgs.msg import JointTrajectoryPoint
from control_msgs.action import FollowJointTrajectory

# =========================
# Fallback quaternion_from_euler (không cần gói ngoài)
# =========================
try:
    from tf_transformations import quaternion_from_euler
except Exception:
    try:
        from tf.transformations import quaternion_from_euler
    except Exception:
        def quaternion_from_euler(roll, pitch, yaw):
            cr, sr = math.cos(roll*0.5),  math.sin(roll*0.5)
            cp, sp = math.cos(pitch*0.5), math.sin(pitch*0.5)
            cy, sy = math.cos(yaw*0.5),   math.sin(yaw*0.5)
            w = cr*cp*cy + sr*sp*sy
            x = sr*cp*cy - cr*sp*sy
            y = cr*sp*cy + sr*cp*sy
            z = cr*cp*sy - sr*cp*cy
            return (x, y, z, w)

# =========================
# Guard MoveIt (để code không crash nếu chưa cài)
# =========================
_HAS_MOVEIT = True
try:
    from geometry_msgs.msg import PoseStamped
    from moveit_msgs.srv import GetPositionIK
except Exception:
    _HAS_MOVEIT = False


class MyRLEnvironmentNode(Node):

    def __init__(self):
        super().__init__('node_main_rl_environment')
        print("initializing.....")

        # TF buffer/listener
        self.tf_buffer   = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Client reset sphere
        self.client_reset_sphere = self.create_client(SetEntityState, '/gazebo/set_entity_state')
        while not self.client_reset_sphere.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('sphere reset-service not available, waiting...')
        self.request_sphere_reset = SetEntityState.Request()

        # Action client điều khiển quỹ đạo
        self.trajectory_action_client = ActionClient(
            self, FollowJointTrajectory, '/joint_trajectory_controller/follow_joint_trajectory'
        )

        # Subscriptions
        self.joint_state_subscription  = message_filters.Subscriber(self, JointState,   '/joint_states')
        self.target_point_subscription = message_filters.Subscriber(self, ModelStates,  '/gazebo/model_states')

        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.joint_state_subscription, self.target_point_subscription],
            queue_size=10, slop=0.1, allow_headerless=True
        )
        self.ts.registerCallback(self.initial_callback)

        # Hai pose mẫu (demo)
        self.pose_a = [0.0, 0.0, -1.5, 0.0, -1.57, 0.0]
        self.pose_b = [3.9988, -0.5012, -0.9967,  0.0103, -1.599,  -0.0956]
        self._toggle = True

        # Tham số cho IK / target
        self.declare_parameter('target_model_name', 'marker_A')
        self.declare_parameter('group_name',        'doosan_arm')  # sửa theo MoveIt group của bạn
        self.declare_parameter('ee_link',           'link6')       # end-effector link
        self.declare_parameter('ref_frame',         'world')
        self.declare_parameter('tcp_offset_z',      0.0)           # nếu đầu tool nhô ra theo +Z world

        self.target_model_name = self.get_parameter('target_model_name').get_parameter_value().string_value
        self.group_name        = self.get_parameter('group_name').get_parameter_value().string_value
        self.ee_link           = self.get_parameter('ee_link').get_parameter_value().string_value
        self.ref_frame         = self.get_parameter('ref_frame').get_parameter_value().string_value
        self.tcp_offset_z      = float(self.get_parameter('tcp_offset_z').get_parameter_value().double_value)

        # IK client (nếu có MoveIt)
        if _HAS_MOVEIT:
            self.ik_client = self.create_client(GetPositionIK, '/compute_ik')
            if not self.ik_client.wait_for_service(timeout_sec=1.0):
                self.get_logger().warn("MoveIt IK service '/compute_ik' chưa sẵn sàng. Hãy chạy move_group.")
        else:
            self.get_logger().warn("MoveIt chưa khả dụng: IK bị vô hiệu. "
                                   "Cài: 'sudo apt install ros-foxy-moveit ros-foxy-moveit-msgs'.")

        # Cờ & bộ đệm
        self._has_joint_state = False
        self._has_target_pose = False
        self.target_x = self.target_y = self.target_z = None
        self.robot_x  = self.robot_y  = self.robot_z  = None

        # Tên frame
        self.reference_frame = 'world'
        self.child_frame     = 'link6'

    # ----------------------------
    # Callbacks & TF
    # ----------------------------
    def initial_callback(self, joint_state_msg, model_states_msg):
        # joint order đến: ['joint2', 'joint3', 'joint1', 'joint4', 'joint5', 'joint6']
        self.joint_1_pos = joint_state_msg.position[2]
        self.joint_2_pos = joint_state_msg.position[0]
        self.joint_3_pos = joint_state_msg.position[1]
        self.joint_4_pos = joint_state_msg.position[3]
        self.joint_5_pos = joint_state_msg.position[4]
        self.joint_6_pos = joint_state_msg.position[5]

        self.joint_1_vel = joint_state_msg.velocity[2]
        self.joint_2_vel = joint_state_msg.velocity[0]
        self.joint_3_vel = joint_state_msg.velocity[1]
        self.joint_4_vel = joint_state_msg.velocity[3]
        self.joint_5_vel = joint_state_msg.velocity[4]
        self.joint_6_vel = joint_state_msg.velocity[5]
        self._has_joint_state = True

        # (Giữ logic sphere cho reward nếu còn dùng)
        try:
            s_idx = model_states_msg.name.index('my_sphere')
            self.pos_sphere_x = model_states_msg.pose[s_idx].position.x
            self.pos_sphere_y = model_states_msg.pose[s_idx].position.y
            self.pos_sphere_z = model_states_msg.pose[s_idx].position.z
        except ValueError:
            pass

        # Lấy pose của target model (marker_A) theo world
        try:
            t_idx = model_states_msg.name.index(self.target_model_name)
            self.target_x = model_states_msg.pose[t_idx].position.x
            self.target_y = model_states_msg.pose[t_idx].position.y
            self.target_z = model_states_msg.pose[t_idx].position.z
            self._has_target_pose = True
        except ValueError:
            self._has_target_pose = False

        # EE pose (world)
        self.robot_x, self.robot_y, self.robot_z = self.get_end_effector_xyz()

    def get_end_effector_xyz(self):
        try:
            now  = rclpy.time.Time()
            trans = self.tf_buffer.lookup_transform(self.reference_frame, self.child_frame, now)
        except TransformException as ex:
            self.get_logger().info(f'Could not transform {self.reference_frame} to {self.child_frame}: {ex}')
            return None, None, None
        else:
            x = trans.transform.translation.x
            y = trans.transform.translation.y
            z = trans.transform.translation.z
            return round(x, 3), round(y, 3), round(z, 3)

    # ----------------------------
    # Reset môi trường (giữ nguyên)
    # ----------------------------
    def reset_environment_request(self):
        sphere_position_x = random.uniform(0.05, 1.05)
        sphere_position_y = random.uniform(-0.5, 0.5)
        sphere_position_z = random.uniform(0.05, 1.05)

        self.request_sphere_reset.state.name = 'my_sphere'
        self.request_sphere_reset.state.reference_frame = 'world'
        self.request_sphere_reset.state.pose.position.x = sphere_position_x
        self.request_sphere_reset.state.pose.position.y = sphere_position_y
        self.request_sphere_reset.state.pose.position.z = sphere_position_z

        self.future_sphere_reset = self.client_reset_sphere.call_async(self.request_sphere_reset)
        self.get_logger().info('Reseting sphere to new position...')
        rclpy.spin_until_future_complete(self, self.future_sphere_reset)

        sphere_service_response = self.future_sphere_reset.result()
        if sphere_service_response and sphere_service_response.success:
            self.get_logger().info("Sphere Moved to a New Position Success")
        else:
            self.get_logger().info("Sphere Reset Request failed")

        # đưa robot về pose_a như "home"
        home_point_msg = JointTrajectoryPoint()
        home_point_msg.positions     = self.pose_a
        home_point_msg.velocities    = [0.0]*6
        home_point_msg.accelerations = [0.0]*6
        home_point_msg.time_from_start = Duration(seconds=2).to_msg()

        joint_names   = ['joint1','joint2','joint3','joint4','joint5','joint6']
        home_goal_msg = FollowJointTrajectory.Goal()
        home_goal_msg.goal_time_tolerance    = Duration(seconds=1).to_msg()
        home_goal_msg.trajectory.joint_names = joint_names
        home_goal_msg.trajectory.points      = [home_point_msg]

        self.trajectory_action_client.wait_for_server()
        send_home_goal_future = self.trajectory_action_client.send_goal_async(home_goal_msg)
        rclpy.spin_until_future_complete(self, send_home_goal_future)
        goal_reset_handle = send_home_goal_future.result()

        if not goal_reset_handle or not goal_reset_handle.accepted:
            self.get_logger().info(' Home-Goal rejected ')
            return
        self.get_logger().info('Moving robot to home position...')

        get_reset_result = goal_reset_handle.get_result_async()
        rclpy.spin_until_future_complete(self, get_reset_result)

        if get_reset_result.result() and get_reset_result.result().result.error_code == 0:
            self.get_logger().info('Robot in Home position without problems')
        else:
            self.get_logger().info('There was a problem with the action')

    # ----------------------------
    # Gửi quỹ đạo khớp
    # ----------------------------
    def action_step_service(self, action_values):
        point_msg = JointTrajectoryPoint()
        point_msg.positions     = list(action_values)
        point_msg.velocities    = [0.0]*6
        point_msg.accelerations = [0.0]*6
        point_msg.time_from_start = Duration(seconds=2.0).to_msg()

        joint_names = ['joint1','joint2','joint3','joint4','joint5','joint6']
        goal_msg    = FollowJointTrajectory.Goal()
        goal_msg.goal_time_tolerance = Duration(seconds=1).to_msg()
        goal_msg.trajectory.joint_names = joint_names
        goal_msg.trajectory.points      = [point_msg]

        self.trajectory_action_client.wait_for_server()
        self.send_goal_future = self.trajectory_action_client.send_goal_async(goal_msg)
        rclpy.spin_until_future_complete(self, self.send_goal_future)
        goal_handle = self.send_goal_future.result()

        if not goal_handle or not goal_handle.accepted:
            self.get_logger().info('Action-Goal rejected')
            return False

        get_result = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, get_result)
        if get_result.result() and get_result.result().result.error_code == 0:
            self.get_logger().info('Hành động đã hoàn thành')
            return True
        else:
            self.get_logger().info('Có vấn đề với action')
            return False

    # ----------------------------
    # Demo hành vi 2 pose A/B
    # ----------------------------
    def generate_action_funct(self):
        action = self.pose_a if self._toggle else self.pose_b
        self._toggle = not self._toggle
        return action

    # ----------------------------
    # Reward cũ theo my_sphere (giữ nguyên)
    # ----------------------------
    def calculate_reward_funct(self):
        try:
            robot_end_position    = np.array((self.robot_x, self.robot_y, self.robot_z))
            target_point_position = np.array((self.pos_sphere_x, self.pos_sphere_y, self.pos_sphere_z))
        except Exception:
            self.get_logger().info('could not calculate the distance yet, trying again...')
            return
        else:
            distance = np.linalg.norm(robot_end_position - target_point_position)
            if distance <= 0.05:
                self.get_logger().info('Goal Reached')
                return 10, True
            else:
                return -1, False

    def state_space_funct(self):
        try:
            state = [
                self.robot_x, self.robot_y, self.robot_z,
                self.joint_1_pos, self.joint_2_pos, self.joint_3_pos, self.joint_4_pos, self.joint_5_pos, self.joint_6_pos,
                self.pos_sphere_x, self.pos_sphere_y, self.pos_sphere_z
            ]
        except Exception:
            self.get_logger().info('-------node not ready yet, Still getting values------------------')
            return
        else:
            return state

    # =======================================================
    # IK tới marker_A (cần MoveIt); nếu chưa có MoveIt -> raise
    # =======================================================
    def _wait_for_states(self, timeout=15.0):
        t0 = time.time()
        while rclpy.ok() and (not self._has_joint_state or not self._has_target_pose):
            if time.time() - t0 > timeout:
                raise RuntimeError(f"Timeout chờ JointStates hoặc pose của '{self.target_model_name}'")
            rclpy.spin_once(self, timeout_sec=0.1)

    def _ee_to_target_distance(self, use_tcp=True):
        if self.robot_x is None or self.target_x is None:
            return None
        ee  = np.array((self.robot_x, self.robot_y, self.robot_z))
        tgt = np.array((self.target_x, self.target_y, self.target_z + (self.tcp_offset_z if use_tcp else 0.0)))
        return float(np.linalg.norm(ee - tgt))

    def ik_to_marker_A(self, pitch_down=True, yaw=0.0, pos_tol=0.02, ik_timeout=1.0):
        """
        Tính IK bằng MoveIt để đưa EE (link6) tới pose tại marker_A.
        - pitch_down=True => định hướng quay mặt xuống (pitch = pi).
        - yaw: quay quanh trục Z (world) nếu cần định hướng.
        - pos_tol: dung sai khoảng cách cuối (m).
        """
        if not _HAS_MOVEIT:
            raise RuntimeError("IK đang bị vô hiệu vì thiếu MoveIt. Cài: 'sudo apt install ros-foxy-moveit ros-foxy-moveit-msgs'.")

        # Chờ dữ liệu
        self._wait_for_states(timeout=15.0)
        if not self.ik_client.service_is_ready():
            if not self.ik_client.wait_for_service(timeout_sec=2.0):
                raise RuntimeError("Service '/compute_ik' chưa sẵn sàng. Hãy chạy move_group (MoveIt).")

        # Pose mục tiêu theo world
        ps = PoseStamped()
        ps.header.frame_id = self.ref_frame
        ps.header.stamp    = self.get_clock().now().to_msg()
        ps.pose.position.x = float(self.target_x)
        ps.pose.position.y = float(self.target_y)
        ps.pose.position.z = float(self.target_z + self.tcp_offset_z)

        roll  = 0.0
        pitch = math.pi if pitch_down else 0.0
        qx, qy, qz, qw = quaternion_from_euler(roll, pitch, float(yaw))
        ps.pose.orientation.x = qx
        ps.pose.orientation.y = qy
        ps.pose.orientation.z = qz
        ps.pose.orientation.w = qw

        # Yêu cầu IK (seed từ trạng thái hiện tại)
        req = GetPositionIK.Request()
        req.ik_request.group_name   = self.group_name
        req.ik_request.ik_link_name = self.ee_link
        req.ik_request.pose_stamped = ps
        req.ik_request.timeout      = Duration(seconds=float(ik_timeout)).to_msg()
        req.ik_request.attempts     = 5

        seed = JointState()
        seed.name     = ['joint1','joint2','joint3','joint4','joint5','joint6']
        seed.position = [self.joint_1_pos, self.joint_2_pos, self.joint_3_pos,
                         self.joint_4_pos, self.joint_5_pos, self.joint_6_pos]
        req.ik_request.robot_state.joint_state = seed

        fut = self.ik_client.call_async(req)
        rclpy.spin_until_future_complete(self, fut)
        res = fut.result()
        if res is None:
            raise RuntimeError("Không nhận được phản hồi từ '/compute_ik'.")
        if res.error_code.val != 1:  # SUCCESS = 1
            raise RuntimeError(f"IK thất bại (error_code={res.error_code.val}). Kiểm tra reachable/định hướng/tcp_offset_z.")

        # Lấy nghiệm khớp theo thứ tự controller
        sol_js = res.solution.joint_state
        name_to_pos = {n: p for n, p in zip(sol_js.name, sol_js.position)}
        joint_order = ['joint1','joint2','joint3','joint4','joint5','joint6']
        try:
            q_cmd = [float(name_to_pos[n]) for n in joint_order]
        except KeyError as e:
            raise RuntimeError(f"Thiếu khớp trong nghiệm IK: {e}")

        self.get_logger().info(f"[IK] Đi tới {self.target_model_name} @ ({ps.pose.position.x:.3f}, {ps.pose.position.y:.3f}, {ps.pose.position.z:.3f})")
        ok = self.action_step_service(q_cmd)

        # Đợi cập nhật TF rồi kiểm tra khoảng cách cuối
        t0 = time.time()
        while rclpy.ok() and time.time() - t0 < 3.0:
            rclpy.spin_once(self, timeout_sec=0.1)

        dist = self._ee_to_target_distance(use_tcp=True)
        if dist is None:
            self.get_logger().warn("[IK] Không đọc được khoảng cách EE->A sau khi di chuyển.")
            return ok

        self.get_logger().info(f"[IK] Khoảng cách EE->A = {dist:.3f} m")
        if dist <= float(pos_tol):
            self.get_logger().info("[IK] ✅ ĐÃ CHẠM Điểm A (trong dung sai).")
            return True
        else:
            self.get_logger().warn("[IK] ❌ Chưa chạm Điểm A. Hãy tăng tcp_offset_z, đổi yaw/pitch, "
                                   "hoặc di chuyển marker_A vào vùng làm việc.")
            return False
