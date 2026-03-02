# =========================================================
# 🚀 ROS 2 Humble – Robot Environment & RL Algorithms
# =========================================================
# 📌 Hướng dẫn chạy mô phỏng, thuật toán học tăng cường
# 📌 Mỗi terminal có vai trò riêng – KHÔNG chạy lẫn
# =========================================================


# =========================================================
# 🖥️ TERMINAL 1 – Khởi chạy môi trường mô phỏng
# 👉 Gazebo + RViz + Robot + World
# =========================================================
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 launch my_environment_pkg my_environment.launch.py


# =========================================================
# 🧠 TERMINAL 2 – Chạy thuật toán (chọn 1)
# =========================================================

# 🔹 No-RL Algorithm (Baseline – không học)
ros2 run my_environment_pkg run_Norl_environment

# 🔹 DDPG Algorithm (Deep Deterministic Policy Gradient)
ros2 run my_environment_pkg run_environment

# 🔹 D4PG Algorithm (Distributed Distributional DDPG)
ros2 run my_environment_pkg run_environment_D4PG


# =========================================================
# 📊 VẼ BIỂU ĐỒ ĐÁNH GIÁ HIỆU SUẤT
# 👉 Chạy sau khi thuật toán kết thúc
# =========================================================
ros2 run my_environment_pkg plot_metrics
