# 🤖 Reinforcement Learning Environment for Robot Control (ROS2 Foxy)

Hướng dẫn cài đặt, build và chạy môi trường mô phỏng + thuật toán DDPG dựa trên ROS2 Foxy & Gazebo.

---

## 🧩 1. Cài đặt các package phụ thuộc (thực hiện trong ros2_ws)

```bash
sudo apt install python3-vcstool
sudo apt install ros-foxy-test-msgs
sudo apt install ros-foxy-control-toolbox
sudo apt install ros-foxy-gazebo-ros-pkgs
sudo apt install ros-foxy-xacro
sudo apt install ros-foxy-joint-state-publisher-gui
sudo apt update
sudo apt install python3-colcon-common-extensions
```

---

## ⚙️ 2. Build workspace (thực hiện trong ros2_ws)

```bash
source /opt/ros/foxy/setup.bash
colcon build 
```

---

## 🔧 3. Thiết lập môi trường (Environment Setup)

### 📄 Sửa file `~/.bashrc`

```bash
gedit ~/.bashrc
```

### ✏️ Thêm vào cuối file (nhớ thay `<Tên workspace của bạn>` bằng tên thật):

```bash
# ROS2 Foxy
source /opt/ros/foxy/setup.bash

# Colcon completion
source /usr/share/colcon_argcomplete/hook/colcon-argcomplete.bash

# Workspace
source /home/quan/<Tên workspace của bạn>/ros2_ws/install/setup.bash
```

### 🔄 Reload lại bashrc

```bash
source ~/.bashrc
```

---

# 🚀 4. Chạy mô phỏng Gazebo + RViz

### 🖥️ Terminal 1

```bash
ros2 launch my_environment_pkg my_environment.launch.py
```

---

# 🧱 5. Chạy môi trường NoRL (không Reinforcement Learning)

### 🧠 Terminal 2

```bash
ros2 run my_environment_pkg run_Norl_environment
```

---

# 🤖 6. Chạy Reinforcement Learning (DDPG)

### 🎯 Terminal 3

```bash
ros2 run my_environment_pkg run_environment
```

---

## 📝 Ghi chú quan trọng

- Luôn chạy **Terminal 1** trước (Gazebo + RViz).  
- Sau đó chọn 1 trong 2:
  - Chạy **NoRL**
  - Hoặc chạy **RL (DDPG)**
- Nếu sửa code, build lại workspace:

```bash
colcon build
source install/setup.bash
```

---

## 📌 Lời cảm ơn 

Ngoài ra, xin cảm ơn tác giả của các kho lưu trử này và các hướng dẫn của họ, nơi tôi đã lấy ra ý tưởng từ đó.

* https://github.com/dvalenciar/robotic_arm_environment
* https://github.com/aws-robotics/aws-robomaker-small-warehouse-world

Tôi muốn cảm ơn Doossan Rbobotics vì các kho lưu trữ và các gói mà họ đã sử dụng để tạo nên mã nguồn này 

* https://github.com/doosan-robotics/doosan-robot2
* https://github.com/doosan-robotics/doosan-robot

