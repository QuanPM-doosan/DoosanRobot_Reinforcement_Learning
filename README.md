# 🚀 ROS 2 Humble – Robot Environment & RL Algorithms

Hướng dẫn chạy mô phỏng và các thuật toán học tăng cường cho robot trong ROS 2 Humble.

---

## 🖥️ Terminal 1 – Khởi chạy môi trường mô phỏng
🚀 Gazebo + RViz + Robot + World

```bash
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 launch my_environment_pkg my_environment.launch.py
```

---

## 🧠 Terminal 2 – Thuật toán No-RL

```bash
ros2 run my_environment_pkg run_Norl_environment
```

---

## 🧠 Terminal 2 – Thuật toán DDPG

```bash
ros2 run my_environment_pkg run_environment
```

---

## 🧠 Terminal 2 – Thuật toán D4PG

```bash
ros2 run my_environment_pkg run_environment_D4PG
```

---

## 📊 Vẽ biểu đồ đánh giá (Metrics)

```bash
ros2 run my_environment_pkg plot_metrics
```
