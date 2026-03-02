import csv, os, time
from datetime import datetime

class EpisodeCSVLogger:
    """
    Ghi log từng bước vào CSV để hậu kiểm:
    cột: ts, run_tag, mode, phase, ep, step, reward, done, mem_len, learn_flag,
         ee_x, ee_y, ee_z, j1..j6, sphere_x, sphere_y, sphere_z,
         dist_out, signed, closest_link
    """
    def __init__(self, log_dir, run_tag="ddpg_train"):
        self.log_dir = os.path.abspath(log_dir)
        os.makedirs(self.log_dir, exist_ok=True)
        self.run_tag = run_tag
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.fpath = os.path.join(self.log_dir, f"{stamp}_{run_tag}.csv")
        self._file = open(self.fpath, "w", newline="")
        self._csv  = csv.writer(self._file)
        header = ["ts","run_tag","mode","phase","ep","step","reward","done",
                  "mem_len","learn_flag",
                  "ee_x","ee_y","ee_z",
                  "j1","j2","j3","j4","j5","j6",
                  "sphere_x","sphere_y","sphere_z",
                  "dist_out","signed","closest_link"]
        self._csv.writerow(header)
        self._file.flush()

    def log_step(self, mode, phase, ep, step, reward, done, mem_len, learn_flag,
                 state_vec, dist_out, signed, closest_link):
        t = time.time()
        # state: [ee_x,ee_y,ee_z, j1..j6, sphere_x,y,z] = 15 phần tử
        row = [t, self.run_tag, mode, phase, ep, step, reward, int(bool(done)),
               mem_len, int(bool(learn_flag))]
        if state_vec is None or len(state_vec) < 15:
            row += [None]*12 + [None, None, None]
        else:
            ee_x, ee_y, ee_z = state_vec[0], state_vec[1], state_vec[2]
            j1,j2,j3,j4,j5,j6 = state_vec[3:9]
            spx,spy,spz        = state_vec[9:12]
            row += [ee_x,ee_y,ee_z,j1,j2,j3,j4,j5,j6, spx,spy,spz]
        row += [dist_out, signed, closest_link or ""]
        self._csv.writerow(row)
        self._file.flush()

    def close(self):
        try:
            self._file.close()
        except Exception:
            pass

