"""
Author: David Valencia
Date: 07/04/2022
Modification:
  - 07/11/2025 (TF-safe, sanitize JointTrajectoryPoint, AABB obstacle avoidance, logs)
  - 07/11/2025 (Chỉ đo khoảng cách với link6 – End Effector)
  - 07/11/2025 (Multi-point goal per action để controller nội suy mượt hơn)
  - 07/11/2025 (step_seconds = 1.2 để mượt hơn)

Describer:
    - State 15: [EE(3), joints(6), target(3), obstacle center(3)]
    - Reward: hướng tới đích + phạt khi sát mặt hộp (AABB + margin)
    - Tránh vật: shield + emergency lift (ở run_environment)
"""

import time
import rclpy
import random
import numpy as np
import message_filters
from rclpy.node import Node
from sensor_msgs.msg import JointState
from gazebo_msgs.msg import ModelStates
from gazebo_msgs.srv import SetEntityState

import tf2_ros
from tf2_ros import TransformException

from rclpy.action import ActionClient
from trajectory_msgs.msg import JointTrajectoryPoint
from control_msgs.action import FollowJointTrajectory
from rclpy.duration import Duration


def _to_float_list6(vec):
    arr = np.asarray(vec, dtype=np.float64).reshape(-1)
    if arr.size != 6:
        raise ValueError(f"q_target phải có 6 phần tử, hiện có {arr.size}: {arr}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"q_target chứa giá trị không hữu hạn: {arr}")
    return [float(v) for v in arr]


# Joint limits
J_MIN = np.array([-np.pi, -0.57595, -2.51327, -np.pi, -np.pi, -np.pi], dtype=np.float32)
J_MAX = np.array([+np.pi, +0.57595, +2.51327, +np.pi, +np.pi, +np.pi], dtype=np.float32)


def scale_action_unit_to_joint(a_unit: np.ndarray) -> np.ndarray:
    a = np.clip(a_unit, -1.0, 1.0)
    return J_MIN + (a + 1.0) * 0.5 * (J_MAX - J_MIN)


# Obstacle (0.10 x 0.25 x 1.20 m) → half = (0.05, 0.125, 0.60)
OBS_HALF = np.array([0.05, 0.125, 0.60], dtype=np.float32)
SAFETY_MARGIN = 0.15
WARN_EPS_SURF = 0.08
HARD_EPS_SURF = 0.00


def point_to_aabb_distance(p, c, h):
    """
    p: điểm (x,y,z)
    c: tâm hộp
    h: half-extents + margin
    return: (dist_outside>=0, signed)
      - dist_outside = 0 nếu đã chạm/ở trong
      - signed âm nếu ở trong, dương nếu ngoài (chính là dist_outside)
    """
    d = np.abs(p - c) - h
    outside = np.maximum(d, 0.0)
    dist_outside = float(np.linalg.norm(outside))
    if np.all(d <= 0):
        signed = float(np.max(d))  # âm
    else:
        signed = dist_outside
    return dist_outside, signed


class MyRLEnvironmentNode(Node):
    def __init__(self):
        super().__init__('node_main_rl_environment')
        print("initializing.....")

        # TF
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.robot_x = 0.0
        self.robot_y = 0.0
        self.robot_z = 0.0

        # Reset sphere
        self.client_reset_sphere = self.create_client(SetEntityState, '/gazebo/set_entity_state')
        while not self.client_reset_sphere.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('sphere reset-service not available, waiting...')
        self.request_sphere_reset = SetEntityState.Request()

        # Action client
        self.trajectory_action_client = ActionClient(
            self, FollowJointTrajectory, '/joint_trajectory_controller/follow_joint_trajectory'
        )

        # Subs
        self.joint_state_subscription = message_filters.Subscriber(self, JointState, '/joint_states')
        self.model_states_subscription = message_filters.Subscriber(self, ModelStates, '/gazebo/model_states')
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.joint_state_subscription, self.model_states_subscription],
            queue_size=10, slop=0.1, allow_headerless=True
        )
        self.ts.registerCallback(self.initial_callback)

        # Mỗi bước dài hơn cho mượt
        self.step_seconds = 1.2

        # Obstacle & target
        self.obstacle_available = False
        self.obs_x = 0.0
        self.obs_y = 0.0
        self.obs_z = 0.0
        self.pos_sphere_x = 0.0
        self.pos_sphere_y = 0.0
        self.pos_sphere_z = 0.0

    # ---------- TF ----------
    def _lookup_tf_point(self, frame: str):
        try:
            now = rclpy.time.Time()
            target = 'world'
            source = frame
            try:
                can = self.tf_buffer.can_transform(target, source, now, timeout=rclpy.duration.Duration(seconds=0.3))
            except TypeError:
                can = self.tf_buffer.can_transform(target, source, now)
            if not can:
                return None
            trans = self.tf_buffer.lookup_transform(target, source, now)
        except TransformException:
            return None
        p = trans.transform.translation
        return (float(p.x), float(p.y), float(p.z))

    def get_end_effector_transformation(self):
        pt = self._lookup_tf_point('link6')
        if pt is None:
            return None
        x, y, z = pt
        return round(x, 3), round(y, 3), round(z, 3)

    # ---------- Callbacks ----------
    def initial_callback(self, joint_state_msg, model_states_msg):
        # order: ['joint2','joint3','joint1','joint4','joint5','joint6']
        self.joint_1_pos = joint_state_msg.position[2]
        self.joint_2_pos = joint_state_msg.position[0]
        self.joint_3_pos = joint_state_msg.position[1]
        self.joint_4_pos = joint_state_msg.position[3]
        self.joint_5_pos = joint_state_msg.position[4]
        self.joint_6_pos = joint_state_msg.position[5]

        try:
            i = model_states_msg.name.index('my_sphere')
            self.pos_sphere_x = model_states_msg.pose[i].position.x
            self.pos_sphere_y = model_states_msg.pose[i].position.y
            self.pos_sphere_z = model_states_msg.pose[i].position.z
        except ValueError:
            pass

        try:
            j = model_states_msg.name.index('obstacle_box_1')
            self.obs_x = model_states_msg.pose[j].position.x
            self.obs_y = model_states_msg.pose[j].position.y
            self.obs_z = model_states_msg.pose[j].position.z
            if not self.obstacle_available:
                print(f"[RL/OBS] obstacle present @ (x,y,z)=({self.obs_x:.3f},{self.obs_y:.3f},{self.obs_z:.3f})")
            self.obstacle_available = True
        except ValueError:
            if self.obstacle_available:
                print("[RL/OBS] obstacle missing in /gazebo/model_states (was present before)")
            self.obstacle_available = False

        ee = self.get_end_effector_transformation()
        if ee is not None:
            self.robot_x, self.robot_y, self.robot_z = ee

    # ---------- Reset ----------
    def reset_environment_request(self):
        # warm-up TF
        start = time.time()
        while time.time() - start < 1.0:
            ee = self.get_end_effector_transformation()
            if ee is not None:
                self.robot_x, self.robot_y, self.robot_z = ee
                break
            rclpy.spin_once(self, timeout_sec=0.02)
            time.sleep(0.02)

        # spawn sphere: tránh vùng cấm
        for _ in range(60):
            sx = random.uniform(0.10, 1.00)
            sy = random.uniform(-0.45, 0.45)
            sz = random.uniform(0.10, 1.00)
            if not self.obstacle_available:
                break
            d_spawn, s_spawn = point_to_aabb_distance(
                np.array([sx, sy, sz], dtype=np.float32),
                np.array([self.obs_x, self.obs_y, self.obs_z], dtype=np.float32),
                OBS_HALF + SAFETY_MARGIN
            )
            if s_spawn >= 0.0 and d_spawn >= 0.10:
                break

        self.request_sphere_reset.state.name = 'my_sphere'
        self.request_sphere_reset.state.reference_frame = 'world'
        self.request_sphere_reset.state.pose.position.x = sx
        self.request_sphere_reset.state.pose.position.y = sy
        self.request_sphere_reset.state.pose.position.z = sz
        fut = self.client_reset_sphere.call_async(self.request_sphere_reset)
        self.get_logger().info('Reseting sphere to new position...')
        rclpy.spin_until_future_complete(self, fut)
        if fut.result() and fut.result().success:
            self.get_logger().info("Sphere Moved to a New Position Success")
        else:
            self.get_logger().info("Sphere Reset Request failed")

        # về home
        pt = JointTrajectoryPoint()
        pt.positions = [0.0] * 6
        pt.velocities = [0.0] * 6
        pt.accelerations = [0.0] * 6
        pt.time_from_start = Duration(seconds=2).to_msg()

        names = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']
        goal = FollowJointTrajectory.Goal()
        goal.goal_time_tolerance = Duration(seconds=1).to_msg()
        goal.trajectory.joint_names = names
        goal.trajectory.points = [pt]

        self.trajectory_action_client.wait_for_server()
        send_f = self.trajectory_action_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, send_f)
        h = send_f.result()
        if not h.accepted:
            self.get_logger().info('Home-Goal rejected')
            return
        self.get_logger().info('Moving robot to home position...')
        res_f = h.get_result_async()
        rclpy.spin_until_future_complete(self, res_f)
        if res_f.result().result.error_code == 0:
            self.get_logger().info('Robot in Home position without problems')
        else:
            self.get_logger().info('There was a problem with the action')

    # ---------- Gửi goal: multi-point ----------
    def _send_trajectory_points(self, points):
        names = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']
        goal = FollowJointTrajectory.Goal()
        goal.goal_time_tolerance = Duration(seconds=1).to_msg()
        goal.trajectory.joint_names = names
        goal.trajectory.points = points

        self.get_logger().info('Waiting for action server to move the robot...')
        self.trajectory_action_client.wait_for_server()
        self.get_logger().info('Sending goal-action request...')
        send_f = self.trajectory_action_client.send_goal_async(goal)
        self.get_logger().info('Checking if the goal is accepted...')
        rclpy.spin_until_future_complete(self, send_f)

        h = send_f.result()
        if not h.accepted:
            self.get_logger().info('Action-Goal rejected')
            return
        self.get_logger().info('Action-Goal accepted')
        self.get_logger().info('Checking the response from action-service...')
        res_f = h.get_result_async()
        rclpy.spin_until_future_complete(self, res_f)
        if res_f.result().result.error_code == 0:
            self.get_logger().info('Action Completed without problem')
        else:
            self.get_logger().info('There was a problem with the action')

    def action_step_service_chunked(self, q_target, chunks=2):
        """1 goal nhiều điểm: q_now -> ... -> q_target."""
        try:
            q_now = np.array([
                self.joint_1_pos, self.joint_2_pos, self.joint_3_pos,
                self.joint_4_pos, self.joint_5_pos, self.joint_6_pos
            ], dtype=np.float64)
        except Exception:
            q_now = np.zeros(6, dtype=np.float64)

        q_tar = np.asarray(q_target, dtype=np.float64).reshape(-1)
        if q_tar.size != 6:
            self.get_logger().error(f"[CHUNK] q_target sai số chiều: {q_tar}")
            return

        points = []
        totalT = float(self.step_seconds)  # 1.2s
        for i in range(1, int(chunks) + 1):
            alpha = i / float(chunks)
            qi = q_now + alpha * (q_tar - q_now)
            pt = JointTrajectoryPoint()
            pt.positions = [float(v) for v in qi]
            pt.velocities = [0.0] * 6
            pt.accelerations = [0.0] * 6
            pt.time_from_start = Duration(seconds=totalT * alpha).to_msg()
            points.append(pt)

        self._send_trajectory_points(points)

    # ---------- Khoảng cách (chỉ link6) ----------
    def compute_min_distance_to_obstacle(self):
        if not self.obstacle_available:
            return float('inf'), float('inf'), 'link6'

        ee = self._lookup_tf_point('link6')
        if ee is None:
            return float('inf'), float('inf'), 'link6'

        p = np.array(ee, dtype=np.float32)
        c = np.array((self.obs_x, self.obs_y, self.obs_z), dtype=np.float32)
        h = OBS_HALF + SAFETY_MARGIN
        dist_outside, signed = point_to_aabb_distance(p, c, h)
        return dist_outside, signed, 'link6'

    # ---------- Reward ----------
    def calculate_reward_funct(self):
        try:
            ee = np.array((self.robot_x, self.robot_y, self.robot_z))
            tgt = np.array((self.pos_sphere_x, self.pos_sphere_y, self.pos_sphere_z))
        except Exception:
            self.get_logger().info('could not calculate the reward yet, trying again...')
            return

        dist_goal = np.linalg.norm(ee - tgt)
        dist_out, signed, _ = self.compute_min_distance_to_obstacle()

        reward = -6.0 * dist_goal
        if np.isfinite(dist_out):
            reward += -5.0 * np.exp(-10.0 * dist_out)

        done = False
        if self.obstacle_available and np.isfinite(signed):
            if signed < -HARD_EPS_SURF:
                print(f"[RL/OBS] HARD: inside safety AABB at link6 (signed={signed:.3f}) -> end episode")
                reward += -50.0
                done = True
            elif dist_out < WARN_EPS_SURF:
                print(f"[RL/OBS] WARN: grazing safety surface near link6 (dist_out={dist_out:.3f})")

        if dist_goal <= 0.05:
            reward += 30.0
            done = True

        return float(reward), bool(done)

    # ---------- State ----------
    def state_space_funct(self):
        try:
            return [
                self.robot_x, self.robot_y, self.robot_z,
                self.joint_1_pos, self.joint_2_pos, self.joint_3_pos, self.joint_4_pos, self.joint_5_pos, self.joint_6_pos,
                self.pos_sphere_x, self.pos_sphere_y, self.pos_sphere_z,
                self.obs_x, self.obs_y, self.obs_z
            ]
        except Exception:
            self.get_logger().info('-------node not ready yet, Still getting values------------------')
            return

