# -*- coding: utf-8 -*-
"""
Gym wrapper cho ROS2 Doosan Arm + Gazebo, dùng trực tiếp với DDPGagent ở DDPG.py
API: reset() -> obs, step(a) -> obs, reward, done, info
"""
import os, sys, time
import numpy as np
import gym
from gym import spaces
import rclpy

# Bảo đảm import được my_environment_pkg/* khi chạy file trực tiếp
CURR = os.path.dirname(os.path.abspath(__file__))
PKG_ROOT = os.path.abspath(os.path.join(CURR, ".."))       # my_environment_pkg/
if PKG_ROOT not in sys.path:
    sys.path.insert(0, PKG_ROOT)

from my_environment_pkg.main_rl_environment import MyRLEnvironmentNode

# Giới hạn khớp khớp với generate_action_funct() trong main_rl_environment.py
J_MIN = np.array([-np.pi, -0.57595, -2.51327, -np.pi, -np.pi, -np.pi], dtype=np.float32)
J_MAX = np.array([ +np.pi,  +0.57595,  +2.51327,  +np.pi,  +np.pi,  +np.pi], dtype=np.float32)

def scale_action(u):
    # u ∈ [-1,1]^6 -> joint targets trong [J_MIN, J_MAX]
    u = np.clip(u, -1.0, 1.0)
    return J_MIN + (u + 1.0) * 0.5 * (J_MAX - J_MIN)

class Ros2ArmEnv(gym.Env):
    metadata = {"render.modes": []}

    def __init__(self, step_sleep=0.18, max_steps=150):
        super().__init__()
        self.step_sleep = float(step_sleep)
        self.max_steps  = int(max_steps)
        self._steps     = 0

        # Khởi tạo ROS2 node (1 lần)
        if not rclpy.ok():
            rclpy.init(args=None)
        self.node = MyRLEnvironmentNode()

        # State = 12: [ee_xyz(3) + 6 joint pos + target_xyz(3)]
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(12,), dtype=np.float32)
        # Action = 6 khớp, chuẩn hoá [-1,1] để hợp với DDPG + OU Noise
        self.action_space      = spaces.Box(low=-1.0, high=1.0, shape=(6,), dtype=np.float32)

        self._wait_ready()

    def _wait_ready(self, tries=300):
        for _ in range(tries):
            s = self.node.state_space_funct()
            if s is not None:
                return
            rclpy.spin_once(self.node, timeout_sec=0.02)
            time.sleep(0.02)
        raise RuntimeError("ROS state chưa sẵn sàng (không nhận được joint/TF/sphere).")

    def reset(self):
        self._steps = 0
        self.node.reset_environment_request()
        time.sleep(self.step_sleep)
        rclpy.spin_once(self.node, timeout_sec=0.02)

        s = self.node.state_space_funct()
        tries = 0
        while s is None and tries < 200:
            rclpy.spin_once(self.node, timeout_sec=0.02)
            time.sleep(0.01)
            s = self.node.state_space_funct()
            tries += 1
        if s is None:
            s = [0.0]*12
        return np.array(s, dtype=np.float32)

    def step(self, action):
        self._steps += 1
        q_target = scale_action(action)
        self.node.action_step_service(q_target)

        time.sleep(self.step_sleep)
        rclpy.spin_once(self.node, timeout_sec=0.02)

        r = self.node.calculate_reward_funct()
        s = self.node.state_space_funct()

        tries = 0
        while (r is None or s is None) and tries < 200:
            rclpy.spin_once(self.node, timeout_sec=0.02)
            time.sleep(0.01)
            r = self.node.calculate_reward_funct()
            s = self.node.state_space_funct()
            tries += 1

        if r is None or s is None:
            reward, done = -1.0, False
            s = [0.0]*12
        else:
            reward, done = r

        done = bool(done) or (self._steps >= self.max_steps)
        return np.array(s, dtype=np.float32), float(reward), done, {}

    def close(self):
        # Không shutdown ROS ở đây vì có thể còn dùng tiếp ở tiến trình khác
        try:
            self.node.destroy_node()
        except Exception:
            pass

