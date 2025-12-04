'''
Author: David Valencia (kept) + logging by ChatGPT
Date     : 12 / 10 /2021
Modified : 07 /04  /2022
Patched  : 07 /11 /2025 (add CSV logging + TF warm-up)

Describer:
    Same random runner as original, but now logs CSV (baseline "no RL")
    and warms up TF before starting to avoid NoneType unpack.
'''

import os
import csv
import time
import math
import rclpy
from datetime import datetime
from .main_NOrl_environment import MyRLEnvironmentNode

# ====== cấu hình ghi log ======
LOG_DIR = os.path.expanduser('~/QuanPM_robotic_arm_ws/ros2_ws/rl_logs')
RUN_TAG = 'no_rl_random'      # để plot_metrics nhận biết baseline
PHASE   = 'random'            # phase= "random"
MODE    = 'RANDOM'            # mode  = "RANDOM"
os.makedirs(LOG_DIR, exist_ok=True)

def _new_csv_writer():
    ts = datetime.now().strftime('%Y%m%d-%H%M%S')
    fname = f'{ts}_{RUN_TAG}.csv'
    fpath = os.path.join(LOG_DIR, fname)
    fobj  = open(fpath, 'w', newline='')
    writer = csv.writer(fobj)
    header = [
        'mode','phase','ep','step',
        'reward','done','mem_len','learn_flag',
        'ee_x','ee_y','ee_z',
        'dist_out','signed','closest_link'
    ]
    writer.writerow(header)
    return fobj, writer, fpath

def _safe_reward(node, tries=120):
    for _ in range(tries):
        out = node.calculate_reward_funct()
        if out is not None:
            return out
        rclpy.spin_once(node, timeout_sec=0.02)
        time.sleep(0.01)
    return (-1.0, False)

def _safe_state(node, tries=120):
    for _ in range(tries):
        s = node.state_space_funct()
        if s is not None:
            return s
        rclpy.spin_once(node, timeout_sec=0.02)
        time.sleep(0.01)
    return None

def _min_dist(node):
    try:
        res = node.compute_min_distance_to_obstacle()
        if isinstance(res, (list, tuple)) and len(res) >= 3:
            return float(res[0]), float(res[1]), str(res[2])
    except Exception:
        pass
    return (math.nan, math.nan, "")

def _warmup_tf(node, seconds=1.0):
    """Quay nhẹ executor để TF buffer có dữ liệu (tránh NoneType)."""
    t_end = time.time() + seconds
    while time.time() < t_end:
        rclpy.spin_once(node, timeout_sec=0.02)
        time.sleep(0.01)

def main(args=None):

    rclpy.init(args=args)
    run_env_node = MyRLEnvironmentNode()
    rclpy.spin_once(run_env_node)

    # ====== cấu hình chạy gốc (GIỮ NGUYÊN) ======
    num_episodes = 9
    episonde_horizont = 35

    # ====== chuẩn bị CSV ======
    fobj, logw, fpath = _new_csv_writer()
    print(f'[LOG] writing baseline CSV: {fpath}')

    try:
        # TF warm-up 1 chút trước khi reset/move
        _warmup_tf(run_env_node, seconds=1.0)

        for episode in range(num_episodes):

            run_env_node.reset_environment_request()
            time.sleep(2.0)
            step = 0

            for step in range(episonde_horizont):
                print (f'----------------Episode:{episode+1} Step:{step+1}--------------------')

                action = run_env_node.generate_action_funct()  # random action
                run_env_node.action_step_service(action)        # take the action

                reward, done  = _safe_reward(run_env_node)
                state         = _safe_state(run_env_node)

                if state is not None and len(state) >= 3:
                    ee_x, ee_y, ee_z = float(state[0]), float(state[1]), float(state[2])
                else:
                    ee_x = ee_y = ee_z = math.nan

                dist_out, signed, closest = _min_dist(run_env_node)

                logw.writerow([
                    MODE, PHASE,
                    episode+1, step+1,
                    float(reward), int(bool(done)), 0, 0,
                    ee_x, ee_y, ee_z,
                    dist_out, signed, closest
                ])

                if done == True:
                    print (f'Goal Reach, Episode ends after {step+1} steps')
                    break

                time.sleep(1.0)

            print (f'Episode {episode+1} Ended')

        print ("Total num of episode completed, Exiting ....")

    finally:
        try:
            fobj.flush()
            fobj.close()
        except Exception:
            pass
        rclpy.shutdown()


if __name__ == '__main__':
    main()

