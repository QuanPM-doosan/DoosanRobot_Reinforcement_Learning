

import os
import time
import numpy as np
import rclpy
from rclpy.executors import SingleThreadedExecutor

from .main_rl_environment import (
    MyRLEnvironmentNode, scale_action_unit_to_joint,
    J_MIN, J_MAX
)
from my_environment_pkg.RL_algorithms_basic.DDPG import DDPGagent, OUNoise
from .metrics_logger import EpisodeCSVLogger  # <— GHI LOG CSV

# ================== CHẾ ĐỘ ==================
MODE               = "demo"      
NUM_EPISODES_TRAIN = 30           
NUM_EPISODES_DEMO  = 9
EPISODE_HORIZON    = 4

# =============== RL PARAMS ==================
BATCH_SIZE  = 32
LEARN_EVERY = 1

# ================== LOG =====================
VERBOSE_RL      = True         # in log gọn cho mỗi bước
LOG_EVERY_STEP  = 1

# ====== ĐƯỜNG DẪN CHECKPOINT & LOG =========
CKPT_PREFIX = os.path.expanduser(
    "~/QuanPM_robotic_arm_ws/ros2_ws/rl_checkpoints/ddpg_doosan"
)
os.makedirs(os.path.dirname(CKPT_PREFIX), exist_ok=True)

LOG_DIR  = os.path.expanduser(
    "~/QuanPM_robotic_arm_ws/ros2_ws/rl_logs"
)
os.makedirs(LOG_DIR, exist_ok=True)
RUN_TAG_TRAIN = "ddpg_train"
RUN_TAG_DEMO  = "ddpg_demo"

PHASE_TRAIN = "train"
PHASE_DEMO  = "demo"

# ====== NGƯỠNG KHOẢNG CÁCH EE–AABB ========
THRESH_NEAR = 0.06
THRESH_MID  = 0.12

# ====== CẤU HÌNH CHUYỂN ĐỘNG ===============
FAR_CFG = {
    "STEP_SLEEP":      0.12,   # đợi sau mỗi lệnh
    "CHUNKS_PER_STEP": 1,
    "SMOOTH_ALPHA":    0.90,   # FAR ít mượt để đi nhanh
    "MAX_DELTA":       np.array([1.60, 1.00, 1.60, 2.00, 2.00, 2.00]),
    "MIN_STEP":        np.array([0.000, 0.000, 0.000, 0.000, 0.000, 0.000]),
    "ACTION_REPEAT":   1,
}

MID_CFG = {
    "STEP_SLEEP":      0.18,
    "CHUNKS_PER_STEP": 1,
    "SMOOTH_ALPHA":    0.50,
    "MAX_DELTA":       np.array([0.70, 0.50, 0.70, 0.90, 0.90, 0.90]),
    "MIN_STEP":        np.array([0.006, 0.006, 0.006, 0.008, 0.008, 0.008]),
    "ACTION_REPEAT":   1,
}

NEAR_CFG = {
    "STEP_SLEEP":      0.24,
    "CHUNKS_PER_STEP": 2,
    "SMOOTH_ALPHA":    0.35,
    "MAX_DELTA":       np.array([0.30, 0.22, 0.30, 0.38, 0.38, 0.38]),
    "MIN_STEP":        np.array([0.010, 0.010, 0.010, 0.012, 0.012, 0.012]),
    "ACTION_REPEAT":   1,
}

# =========== smoothing state ===========
_last_q_cmd = None
_cur_cfg    = FAR_CFG.copy()

# =========== CHỐNG KẸT (STALL) =========
STALL_EE_EPS     = 1e-3    # m
STALL_JOINT_EPS  = 3e-3    # rad
STALL_MAX_STEPS  = 6
STALL_JUMP       = np.array([0.25, 0.18, -0.18, 0.0, 0.0, 0.35])  # thoát bẫy

def _ee_from_state(s_vec):
    # state: [ee_x,ee_y,ee_z, j1..j6, sphere_x,y,z]
    return np.array(s_vec[0:3], dtype=np.float64)

# ====== Fake env cho DDPG (shape) ======
class _SimpleSpace:
    def __init__(self, low, high):
        self.low   = np.array(low,  dtype=np.float32)
        self.high  = np.array(high, dtype=np.float32)
        self.shape = self.low.shape

class _FakeEnvForDDPG:
    def __init__(self):
        obs_dim = 15; act_dim = 6
        self.observation_space = _SimpleSpace(low=[-np.inf]*obs_dim, high=[np.inf]*obs_dim)
        self.action_space      = _SimpleSpace(low=[-1.0]*act_dim,   high=[+1.0]*act_dim)

def set_mode_by_distance(dist_out_prev: float):
    global _cur_cfg
    if not np.isfinite(dist_out_prev):
        _cur_cfg = FAR_CFG.copy(); return "FAR"
    if dist_out_prev < THRESH_NEAR:
        _cur_cfg = NEAR_CFG.copy(); return "NEAR"
    elif dist_out_prev < THRESH_MID:
        _cur_cfg = MID_CFG.copy();  return "MID"
    else:
        _cur_cfg = FAR_CFG.copy();  return "FAR"

def smooth_and_limit(q_new, q_prev):
    q_new = np.asarray(q_new, dtype=np.float64).reshape(6)
    if q_prev is None:
        return q_new
    alpha     = float(_cur_cfg["SMOOTH_ALPHA"])
    max_delta = _cur_cfg["MAX_DELTA"]
    deadband  = _cur_cfg["MIN_STEP"]
    q_blend = (1.0 - alpha) * q_prev + alpha * q_new
    dq = q_blend - q_prev
    dq[np.abs(dq) < deadband] = 0.0
    dq = np.clip(dq, -max_delta, max_delta)
    return (q_prev + dq)

def hard_clamp_action(q_target, factor=0.35):
    arr = np.asarray(q_target, dtype=np.float64).reshape(-1) * float(factor)
    if arr.size != 6:
        raise ValueError(f"q_target phải có 6 phần tử (sau clamp): {arr}")
    return [float(v) for v in arr]

def emergency_lift(node: MyRLEnvironmentNode, current_q=None):
    if current_q is None:
        try:
            current_q = np.array([
                node.joint_1_pos, node.joint_2_pos, node.joint_3_pos,
                node.joint_4_pos, node.joint_5_pos, node.joint_6_pos
            ], dtype=np.float32)
        except Exception:
            current_q = np.zeros(6, dtype=np.float32)
    q_safe = current_q.copy()
    q_safe[1] = np.clip(q_safe[1] + 0.30, J_MIN[1], J_MAX[1])
    q_safe[2] = np.clip(q_safe[2] - 0.30, J_MIN[2], J_MAX[2])
    print("[RL/OBS] EMERGENCY LIFT: lifting arm to avoid obstacle")
    node.action_step_service_chunked(q_safe, chunks=2)

def _wait_state_ready(node: MyRLEnvironmentNode, tries=300):
    s = node.state_space_funct(); k = 0
    while s is None and k < tries:
        rclpy.spin_once(node, timeout_sec=0.02)
        time.sleep(0.02)
        s = node.state_space_funct(); k += 1
    return s

# ================== ONE EPISODE ==================
def _run_one_episode(
    node, agent, noise, do_learn: bool, episode_idx: int, horizon: int,
    logger: EpisodeCSVLogger = None, phase: str = "train"
):
    global _last_q_cmd, _cur_cfg
    _last_q_cmd = None

    node.reset_environment_request()
    time.sleep(0.2); rclpy.spin_once(node, timeout_sec=0.05)

    s = _wait_state_ready(node) or [0.0]*15
    s = np.array(s, dtype=np.float32)

    ep_ret = 0.0
    if VERBOSE_RL:
        tag = "TRAIN" if do_learn else "DEMO"
        print(f"\n========== [RL] {tag} EPISODE {episode_idx} START ==========")

    stall_k = 0
    last_ee = _ee_from_state(s)
    global_step_local = 0

    for t in range(horizon):
        # khoảng cách hiện tại
        dist_out_prev, signed_prev, closest_prev = node.compute_min_distance_to_obstacle()
        mode = set_mode_by_distance(dist_out_prev)

        STEP_SLEEP      = float(_cur_cfg["STEP_SLEEP"])
        CHUNKS_PER_STEP = int(_cur_cfg["CHUNKS_PER_STEP"])

        # ===== action từ policy =====
        a = agent.get_action(s)

        # ===== noise scheduling theo khoảng cách =====
        if do_learn:
            base_max = 0.06
            if np.isfinite(dist_out_prev):
                scale = np.clip(dist_out_prev / THRESH_MID, 0.0, 1.0)  # 0..1
            else:
                scale = 1.0
            noise.max_sigma = base_max * (0.20 + 0.80*scale)
            noise.min_sigma = min(noise.min_sigma, noise.max_sigma*0.5)
            a_noisy = noise.get_action(a, t + global_step_local)
        else:
            a_noisy = a  # DEMO: không noise

        q_target = scale_action_unit_to_joint(a_noisy)

        # ===== shield cứng =====
        danger_inside = (np.isfinite(signed_prev) and signed_prev < 0.0)
        danger_graze  = (np.isfinite(dist_out_prev) and dist_out_prev < 0.04)
        danger_near   = (np.isfinite(dist_out_prev) and dist_out_prev < 0.08)

        # Terminate-on-contact (TRAIN)
        if do_learn and danger_inside:
            reward, done = -20.0, True
            agent.memory.push(s, a_noisy, reward, s, done)
            agent.step_training(BATCH_SIZE, learn_every=LEARN_EVERY)
            # log cuối
            if logger is not None:
                dist_out_now, signed_now, closest_now = node.compute_min_distance_to_obstacle()
                mem_len = len(agent.memory) if hasattr(agent, "memory") else 0
                did_learn = getattr(agent, "did_learn_last_step", False)
                logger.log_step(mode=mode, phase=phase, ep=episode_idx, step=t,
                                reward=reward, done=done, mem_len=mem_len, learn_flag=did_learn,
                                state_vec=s, dist_out=dist_out_now, signed=signed_now,
                                closest_link=closest_now)
            print(f"[RL/OBS] TERMINATE: inside safety near {closest_prev} (signed={signed_prev:.3f}) → end EP")
            ep_ret += reward
            break

        if danger_graze:
            print(f"[RL/OBS] WARN: grazing safety surface near {closest_prev} (dist_out={dist_out_prev:.3f}) -> hard clamp")
            q_target = hard_clamp_action(q_target, factor=0.35)
            if do_learn: noise.reset()
        elif danger_near:
            print(f"[RL/OBS] CAUTION: near safety surface near {closest_prev} (dist_out={dist_out_prev:.3f}) -> clamp")
            q_target = hard_clamp_action(q_target, factor=0.55)
            if do_learn: noise.reset()
        else:
            q_target = np.asarray(q_target, dtype=np.float64)

        # FAR: bỏ smoothing để “nhảy” nhanh; MID/NEAR: giữ smoothing
        if mode == "FAR":
            q_cmd = np.asarray(q_target, dtype=np.float64).reshape(6)
            _last_q_cmd = None
        else:
            q_cmd = smooth_and_limit(q_target, _last_q_cmd)
            _last_q_cmd = q_cmd.copy()

        # gửi lệnh
        node.action_step_service_chunked(q_cmd, chunks=CHUNKS_PER_STEP)

        # chờ
        time.sleep(STEP_SLEEP)
        rclpy.spin_once(node, timeout_sec=0.02)

        # ===== reward & next state =====
        r = node.calculate_reward_funct()
        s_next = node.state_space_funct()
        w = 0
        while (r is None or s_next is None) and w < 120:
            rclpy.spin_once(node, timeout_sec=0.02)
            time.sleep(0.01)
            r = node.calculate_reward_funct()
            s_next = node.state_space_funct()
            w += 1

        if r is None or s_next is None:
            reward, done = -1.0, False
            s_next = s
        else:
            reward, done = r

        s_next = np.array(s_next, dtype=np.float32)

        # ===== chống kẹt (stall) =====
        ee_now = _ee_from_state(s_next)
        try:
            dq_now = np.array([
                s_next[3]-s[3], s_next[4]-s[4], s_next[5]-s[5],
                s_next[6]-s[6], s_next[7]-s[7], s_next[8]-s[8]
            ], dtype=np.float64)
        except Exception:
            dq_now = np.zeros(6, dtype=np.float64)

        ee_move = np.linalg.norm(ee_now - last_ee)
        q_move  = float(np.linalg.norm(dq_now))

        if (ee_move < STALL_EE_EPS) and (q_move < STALL_JOINT_EPS) and (not danger_inside):
            stall_k += 1
        else:
            stall_k = 0

        if stall_k >= STALL_MAX_STEPS:
            try:
                cur_q = np.array([
                    node.joint_1_pos, node.joint_2_pos, node.joint_3_pos,
                    node.joint_4_pos, node.joint_5_pos, node.joint_6_pos
                ], dtype=np.float64)
            except Exception:
                cur_q = np.zeros(6, dtype=np.float64)

            scale = 0.6 if mode == "NEAR" else (0.85 if mode == "MID" else 1.0)
            q_escape = cur_q + STALL_JUMP * scale
            q_escape = np.clip(q_escape, J_MIN, J_MAX)

            print(f"[RL/STALLESC] stall={stall_k} (ee_move={ee_move:.4f}, q_move={q_move:.4f}) → ESCAPE")
            node.action_step_service_chunked(q_escape, chunks=max(1, _cur_cfg["CHUNKS_PER_STEP"]))
            time.sleep(_cur_cfg["STEP_SLEEP"])
            rclpy.spin_once(node, timeout_sec=0.02)

            s_next2 = node.state_space_funct() or s_next
            s_next  = np.array(s_next2, dtype=np.float32)
            stall_k = 0
            if do_learn:
                noise.reset()

        # ===== học / log =====
        if do_learn:
            agent.memory.push(s, a_noisy, reward, s_next, done)
            agent.step_training(BATCH_SIZE, learn_every=LEARN_EVERY)

        # LOG CSV mỗi bước (ghi state mới — s_next)
        if logger is not None:
            dist_out_now, signed_now, closest_now = node.compute_min_distance_to_obstacle()
            mem_len = len(agent.memory) if hasattr(agent, "memory") else 0
            did_learn = getattr(agent, "did_learn_last_step", False) if do_learn else False
            logger.log_step(mode=mode, phase=phase, ep=episode_idx, step=t,
                            reward=reward, done=done, mem_len=mem_len, learn_flag=did_learn,
                            state_vec=s_next, dist_out=dist_out_now, signed=signed_now,
                            closest_link=closest_now)

        if VERBOSE_RL and (t % LOG_EVERY_STEP == 0):
            dist_out_now, signed_now, closest_now = node.compute_min_distance_to_obstacle()
            la = getattr(agent, "last_actor_loss", None)
            lc = getattr(agent, "last_critic_loss", None)
            did = getattr(agent, "did_learn_last_step", None) if do_learn else False
            la_s = f"{la:.4f}" if isinstance(la, (float, np.floating)) else "None"
            lc_s = f"{lc:.4f}" if isinstance(lc, (float, np.floating)) else "None"
            prev_tag = closest_prev or "N/A"
            now_tag  = closest_now  or "N/A"
            tag = "TRAIN" if do_learn else "DEMO"
            print(f"[RL-{tag}] mode={mode} | d_surf(prev)={dist_out_prev:.3f}@{prev_tag} "
                  f"-> d_surf(now)={dist_out_now:.3f}@{now_tag} signed(now)={signed_now:.3f} "
                  f"| r={reward:.2f} done={done} mem={len(agent.memory)} learn={did} "
                  f"| Lc={lc_s} La={la_s}")

        ep_ret += reward
        s      = s_next
        last_ee = ee_now
        global_step_local += 1
        if done:
            break

    return ep_ret

# ================== MAIN ==================
def main(args=None):
    rclpy.init(args=args)
    node = MyRLEnvironmentNode()
    executor = SingleThreadedExecutor(); executor.add_node(node)

    fake_env = _FakeEnvForDDPG()
    agent = DDPGagent(env=fake_env, hidden_size=256,
                      actor_learning_rate=1e-4, critic_learning_rate=1e-3,
                      gamma=0.995, tau=5e-3, max_memory_size=200_000)

    noise = OUNoise(fake_env.action_space, max_sigma=0.06, min_sigma=0.01, decay_period=150_000)

    try:
        if MODE.lower() == "train":
            logger = EpisodeCSVLogger(LOG_DIR, run_tag=RUN_TAG_TRAIN)
            returns = []
            for ep in range(1, NUM_EPISODES_TRAIN + 1):
                ep_ret = _run_one_episode(node, agent, noise, do_learn=True,
                                          episode_idx=ep, horizon=EPISODE_HORIZON,
                                          logger=logger, phase=PHASE_TRAIN)
                returns.append(ep_ret)
                avg10 = float(np.mean(returns[-10:])) if len(returns) >= 10 else float(np.mean(returns))
                print(f"[DDPG-TRAIN] episode: {ep:4d} | return: {ep_ret:8.2f} | avg10: {avg10:8.2f}")
                # Lưu checkpoint sau mỗi EP
                try:
                    agent.save(CKPT_PREFIX)
                    print(f"[CKPT] saved to: {CKPT_PREFIX}_*.pt")
                except Exception as e:
                    print(f"[CKPT] save() not available or failed: {e}")
            logger.close()
            print("[TRAIN] Done. Exit.")

        else:
            # DEMO: nạp policy đã train, không học, không noise
            try:
                agent.load(CKPT_PREFIX)
                print(f"[CKPT] loaded from: {CKPT_PREFIX}_*.pt")
            except Exception as e:
                print(f"[CKPT] load() not available or failed: {e} — chạy với init weights.")

            logger = EpisodeCSVLogger(LOG_DIR, run_tag=RUN_TAG_DEMO)
            for ep in range(1, NUM_EPISODES_DEMO + 1):
                _ = _run_one_episode(node, agent, noise=None, do_learn=False,
                                     episode_idx=ep, horizon=EPISODE_HORIZON,
                                     logger=logger, phase=PHASE_DEMO)
            logger.close()

    except KeyboardInterrupt:
        print("\n[INFO] Dừng bởi người dùng.")
    finally:
        executor.shutdown()
        try:
            node.destroy_node()
        except Exception:
            pass
        rclpy.shutdown()

if __name__ == '__main__':
    main()

