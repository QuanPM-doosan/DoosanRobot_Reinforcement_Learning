# -*- coding: utf-8 -*-
"""
Train DDPG trên mô phỏng ROS2 Doosan Arm (Gazebo) dùng wrapper Gym.
Chạy script này ở TERMINAL 2, sau khi đã launch mô phỏng ở TERMINAL 1.
"""

import os, sys, signal
import numpy as np
from RL_algorithms_basic.DDPG import DDPGagent, OUNoise
from RL_algorithms_basic.ros2_arm_gym_env import Ros2ArmEnv

def main():
    # Hyperparameters
    EPISODES    = 2000
    MAX_STEPS   = 150
    BATCH_SIZE  = 128
    LEARN_EVERY = 5           # đồng bộ với DDPGagent.step_training(...)

    # Env + Agent
    env   = Ros2ArmEnv(step_sleep=0.18, max_steps=MAX_STEPS)
    agent = DDPGagent(env,
                      hidden_size=256,
                      actor_learning_rate=1e-4,
                      critic_learning_rate=1e-3,
                      gamma=0.995,
                      tau=5e-3,
                      max_memory_size=200_000)
    noise = OUNoise(env.action_space, max_sigma=0.20, min_sigma=0.05, decay_period=100_000)

    # Ctrl+C để thoát lịch sự
    stop = {"flag": False}
    def _sigint(_a, _b):
        stop["flag"] = True
        print("\n[INFO] Nhận tín hiệu dừng, kết thúc sau episode hiện tại...")
    signal.signal(signal.SIGINT, _sigint)

    returns = []
    for ep in range(1, EPISODES+1):
        s  = env.reset()
        noise.reset()
        ep_ret = 0.0

        for t in range(MAX_STEPS):
            a = agent.get_action(s)           # [-1,1]^6
            a = noise.get_action(a, t)        # thêm noise OU
            ns, r, done, _ = env.step(a)

            agent.memory.push(s, a, r, ns, done)
            agent.step_training(BATCH_SIZE, learn_every=LEARN_EVERY)

            s = ns
            ep_ret += r
            if done:
                break

        returns.append(ep_ret)
        avg10 = float(np.mean(returns[-10:]))
        print(f"episode: {ep:4d} | return: {ep_ret:7.2f} | avg10: {avg10:7.2f}")

        if stop["flag"]:
            break

    print("[INFO] Kết thúc training.")

if __name__ == "__main__":
    main()

