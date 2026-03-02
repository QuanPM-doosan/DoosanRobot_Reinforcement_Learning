#!/usr/bin/env python3
# rl_ddpg_d4pg_pickplace.py (smooth streaming)
# Home -> Pose A -> GRAB -> (RL avoid obstacle_box_1) -> Pose B -> DROP -> Home

import os, sys, time, pathlib
import numpy as np
import rclpy

# ---------- Paths ----------
_THIS_DIR = pathlib.Path(__file__).parent.resolve()
sys.path.insert(0, str(_THIS_DIR))

# ---------- Import Jogger directly ----------
from jog_to_marker import Jogger  # file này phải định nghĩa class Jogger

# ---------- Add RL_algorithms_basic path ----------
def add_rl_path():
    for anc in [_THIS_DIR] + list(_THIS_DIR.parents):
        p1 = anc / "RL_algorithms_basic"
        p2 = anc / "robotic_arm_environment" / "RL_algorithms_basic"
        if p1.is_dir():
            sys.path.insert(0, str(p1)); print(f"[INFO] RL path: {p1}"); return
        if p2.is_dir():
            sys.path.insert(0, str(p2)); print(f"[INFO] RL path: {p2}"); return
    raise ImportError("Không tìm thấy thư mục RL_algorithms_basic.")

add_rl_path()

# ---------- (STUB) gym để import DDPG/D4PG không cần cài gym ----------
try:
    import gym  # type: ignore
except Exception:
    import types
    sys.modules['gym'] = types.SimpleNamespace()

# ---------- Chọn thuật toán ----------
ALG_NAME = os.environ.get("ALG", "D4PG").upper()
if ALG_NAME == "DDPG":
    from DDPG import DDPGagent as Algo
    from DDPG import OUNoise
    print("[INFO] Using DDPG + OU-Noise")
else:
    from D4PG import D4PGAgent as Algo
    OUNoise = None
    print("[INFO] Using D4PG (Gaussian noise built-in)")

# ---------- Tham số nhiệm vụ ----------
OBSTACLE_NAME = "obstacle_box_1"
Q_HOME = np.array([0,0,0,0,0,0], dtype=float)
Q_A    = np.array([0,0,-1.5,0,-1.57,0], dtype=float)
Q_B    = np.array([ 3.9988, -0.5012, -0.9967,  0.0103, -1.599 , -0.0956], dtype=float)

# --- THAM SỐ LÀM MƯỢT/AN TOÀN ---
STEP_SIZE   = 0.03   # rad; giảm bước để êm (trước là 0.05)
STEP_TIME   = 0.35   # s; tăng thời gian để controller kịp nội suy (trước 0.18)
MAX_JOINT_SPEED = 0.8  # rad/s; giới hạn vận tốc mong muốn khi tính thời gian
ACTION_SMOOTHING_ALPHA = 0.35  # 0..1; càng lớn càng theo hành động mới nhiều
EXTRA_WAIT = 0.05  # giãn cách thêm giữa 2 goal để controller ổn

MAX_STEPS   = 240
GOAL_TOL_Q  = 0.03
SAFE_R      = 0.22
COLLISION_R = 0.12

W_GOAL   = 2.0
W_OBS    = 3.0
W_SMOOTH = 0.2
W_TIME   = 0.005
R_DONE   = 5.0
R_CRASH  = -5.0

MODEL_PATH = str(_THIS_DIR / f"model_{ALG_NAME}.pt")
BATCH_SIZE = 64

# ---------- Fake Gym spaces ----------
class _Space:
    def __init__(self, shape, low=None, high=None):
        self.shape = tuple(shape)
        self.low  = np.array(low)  if low  is not None else None
        self.high = np.array(high) if high is not None else None

class _FakeEnv:
    def __init__(self, obs_dim, act_dim):
        self.observation_space = _Space((obs_dim,))
        self.action_space      = _Space((act_dim,), low=-np.ones(act_dim), high=np.ones(act_dim))

# ---------- ROS helpers ----------
def _spin(node, sec=0.02):
    t_end = time.time() + sec
    while rclpy.ok() and time.time() < t_end:
        rclpy.spin_once(node, timeout_sec=0.01)

def current_q(node: Jogger):
    q = node.current_positions()
    return np.array(q, dtype=float) if q is not None else None

def ee_pos(node: Jogger):
    p, _ = node.ee_pose()
    return np.array(p, dtype=float) if p is not None else None

def model_pos(node: Jogger, name: str):
    p, _ = node.get_model_pose(name)
    return np.array(p, dtype=float) if p is not None else None

# ---------- State (19D) & Reward ----------
def build_state(node: Jogger, q_goal: np.ndarray):
    q_cur = current_q(node)
    if q_cur is None:
        q_cur = np.zeros(6, dtype=float)

    ee = ee_pos(node)
    if ee is None:
        ee = np.zeros(3, dtype=float)

    obs = model_pos(node, OBSTACLE_NAME)
    if obs is None:
        obs = np.zeros(3, dtype=float)

    d_q = q_goal - q_cur
    d_obs = float(np.linalg.norm(ee - obs)) if (ee.any() or obs.any()) else 10.0

    s = np.concatenate([q_cur, d_q, ee, obs, [d_obs]]).astype(np.float32)
    return s, q_cur, ee, obs, d_obs

def compute_reward(q_prev, q_now, q_goal, ee, obs, d_obs_prev, d_obs_now):
    d_goal_prev = np.linalg.norm(q_goal - q_prev)
    d_goal_now  = np.linalg.norm(q_goal - q_now)
    progress = d_goal_prev - d_goal_now
    dq = q_now - q_prev
    r = W_GOAL*progress - W_SMOOTH*np.sum(dq*dq) - W_TIME
    if d_obs_now < SAFE_R:
        r -= W_OBS * (SAFE_R - d_obs_now)**2
    done = False
    if d_goal_now < GOAL_TOL_Q:
        r += R_DONE; done = True
    if d_obs_now < COLLISION_R:
        r += R_CRASH; done = True
    return float(r), done

def clamp_action_to_dq(a):
    a = np.clip(a, -1.0, 1.0)
    return (a * STEP_SIZE).astype(float)

def computed_step_time_for_dq(dq):
    # Tính thời gian đủ để không vượt quá MAX_JOINT_SPEED
    needed = np.max(np.abs(dq)) / max(1e-6, MAX_JOINT_SPEED)
    return max(STEP_TIME, float(needed) + 0.05)

# ---------- Main ----------
def main():
    rclpy.init()
    node = Jogger()
    try:
        node.get_logger().info(f"[{ALG_NAME}] RL pick-place with obstacle avoidance (smooth)")

        # Home -> A -> Grab
        node.go_to_pose(Q_HOME.tolist(), 2.0); _spin(node, 0.25)
        t_pose_a = getattr(node, "pose_a_time", 2.0) or 2.0
        node.go_to_pose(Q_A.tolist(), max(1.8, t_pose_a)); _spin(node, 0.3)
        if hasattr(node, "do_grab"):
            node.do_grab(); _spin(node, 0.35)

        # Init agent
        s0, q0, ee0, obs0, d_obs0 = build_state(node, Q_B)
        obs_dim, act_dim = s0.shape[0], 6
        fake_env = _FakeEnv(obs_dim, act_dim)
        if ALG_NAME == "DDPG":
            agent = Algo(fake_env)
            noise = OUNoise(fake_env.action_space)
        else:
            agent = Algo(fake_env)
            noise = None

        # (optional) load
        if hasattr(agent, "load") and os.path.exists(MODEL_PATH):
            try:
                agent.load(MODEL_PATH)
                node.get_logger().info(f"Loaded model from {MODEL_PATH}")
            except Exception as e:
                node.get_logger().warn(f"Load model failed: {e}")

        # A -> B (learn + act) with smoothing
        total_r = 0.0
        s = s0; q_prev = q0; d_obs_prev = d_obs0
        prev_dq = np.zeros(6, dtype=float)

        for t in range(1, MAX_STEPS+1):
            # chọn action
            if hasattr(agent, "get_action"):
                a = agent.get_action(s)
            elif hasattr(agent, "select_action"):
                a = agent.select_action(s)
            else:
                a = np.random.uniform(-1,1,act_dim).astype(np.float32)

            if ALG_NAME == "DDPG":
                a = noise.get_action(a, t)

            dq_raw = clamp_action_to_dq(a)
            # EMA smoothing
            dq = ACTION_SMOOTHING_ALPHA * dq_raw + (1.0 - ACTION_SMOOTHING_ALPHA) * prev_dq
            prev_dq = dq

            q_now = current_q(node)
            if q_now is None:
                node.get_logger().warn("No joint state; skip step.")
                _spin(node, 0.06); continue

            q_next = (q_now + dq).tolist()
            tsec = computed_step_time_for_dq(dq)

            node.go_to_pose(q_next, tsec)
            _spin(node, EXTRA_WAIT)  # giãn cách giữa hai goal

            s2, q2, ee2, obs2, d_obs2 = build_state(node, Q_B)
            r, done = compute_reward(q_prev, q2, Q_B, ee2, obs2, d_obs_prev, d_obs2)
            total_r += r

            if ALG_NAME == "DDPG":
                agent.memory.push(s, a, r, s2, done)
                agent.step_training(BATCH_SIZE)
            else:
                agent.step_training(s, a, r, s2, done, BATCH_SIZE, per_memory_status=False)

            s = s2; q_prev = q2; d_obs_prev = d_obs2
            if done:
                node.get_logger().info(f"[{ALG_NAME}] stop at step {t} | reward_sum={round(total_r,3)}")
                break

        # Snap B -> drop -> home
        node.go_to_pose(Q_B.tolist(), 2.0); _spin(node, 0.3)
        if hasattr(node, "do_drop"):
            node.do_drop(); _spin(node, 0.35)
        node.go_to_pose(Q_HOME.tolist(), 2.2)

        if hasattr(agent, "save"):
            try:
                agent.save(MODEL_PATH)
                node.get_logger().info(f"Saved model -> {MODEL_PATH}")
            except Exception as e:
                node.get_logger().warn(f"Save model failed: {e}")

        node.get_logger().info(f"Done. Total reward: {round(total_r,3)}")

    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()

