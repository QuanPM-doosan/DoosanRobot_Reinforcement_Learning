import os
import xml.etree.ElementTree as ET

import rclpy
from rclpy.node import Node
from visualization_msgs.msg import Marker, MarkerArray
from builtin_interfaces.msg import Duration


class EnvMarkersNode(Node):
    def __init__(self):
        super().__init__('env_markers_node')

        # Publish một MarkerArray lên /env_markers
        self.publisher_ = self.create_publisher(MarkerArray, 'env_markers', 10)
        self.timer = self.create_timer(0.5, self.timer_callback)  # 2 Hz

        # Frame tham chiếu – trùng với frame Gazebo/robot đang dùng
        self.frame_id = 'world'

        # ----- Pose obstacle_box_1 (trùng với spawn trong launch) -----
        # Trong launch: -x 0.70 -y -0.20 -z 0.6
        self.obstacle_pose = {
            'x': 0.70,
            'y': -0.20,
            'z': 0.6,
        }

        # ----- Pose Bucket (trùng với spawn_bucket trong launch) -----
        # Trong launch: -x 0.40 -y 0.20 -z 0.1
        self.bucket_pose = {
            'x': 1.40,
            'y': 1.20,
            'z': 0.1,
        }
        # ----- Pose Bucket (trùng với spawn_bucket trong launch) -----
        # Trong launch: -x 0.40 -y 0.20 -z 0.1
        self.conveyor_belt_pose = {
            'x': -1.390076,
            'y': -0.174037,
            'z': 0.0,
        }
        # ----- Pose Bucket (trùng với spawn_bucket trong launch) -----
        # Trong launch: -x 0.40 -y 0.20 -z 0.1
        self.ClutteringD_01_pose = {
            'x': -2.718640,
            'y': -0.752731,
            'z': 0.0,
        }
        # Trong launch: -x 0.40 -y 0.20 -z 0.1
        self.DeskC_01_pose = {
            'x': -2.745320,
            'y': -3.283404,
            'z': 0.0,
        }
        # Trong launch: -x 0.40 -y 0.20 -z 0.1
        self.PalletJackB_01_pose = {
            'x': -1.293370,
            'y': -1.939590,
            'z': 0.0,
        }
        # Trong launch: -x 0.40 -y 0.20 -z 0.1
        self.ShelfD_01_pose = {
            'x': 3.311350,
            'y': 3.079530,
            'z': 0.0,
        }
        # Đường dẫn tuyệt đối tới mesh visual của Bucket
        self.bucket_mesh_resource = (
            'file:///home/quan/'
            'QuanPM_robotic_arm_ws/ros2_ws/src/'
            'robotic_arm_environment/my_environment_pkg/models/'
            'aws_robomaker_warehouse_Bucket_01/meshes/'
            'aws_robomaker_warehouse_Bucket_01_visual.DAE'
        )
        self.conveyor_belt_mesh_resource = (
            'file:///home/quan/'
            'QuanPM_robotic_arm_ws/ros2_ws/src/'
            'robotic_arm_environment/my_environment_pkg/models/'
            'conveyor_belt/meshes/'
            'conveyor_belt.dae'
        )
        self.ClutteringD_01_mesh_resource = (
            'file:///home/quan/'
            'QuanPM_robotic_arm_ws/ros2_ws/src/'
            'robotic_arm_environment/my_environment_pkg/models/'
            'aws_robomaker_warehouse_ClutteringD_01/meshes/'
            'aws_robomaker_warehouse_ClutteringD_01_visual.DAE'
        )
        self.DeskC_01_mesh_resource = (
            'file:///home/quan/'
            'QuanPM_robotic_arm_ws/ros2_ws/src/'
            'robotic_arm_environment/my_environment_pkg/models/'
            'aws_robomaker_warehouse_DeskC_01/meshes/'
            'aws_robomaker_warehouse_DeskC_01_visual.DAE'
        )
        self.PalletJackB_01_mesh_resource = (
            'file:///home/quan/'
            'QuanPM_robotic_arm_ws/ros2_ws/src/'
            'robotic_arm_environment/my_environment_pkg/models/'
            'aws_robomaker_warehouse_PalletJackB_01/meshes/'
            'aws_robomaker_warehouse_PalletJackB_01_visual.DAE'
        )
        self.ShelfD_01_mesh_resource = (
            'file:///home/quan/'
            'QuanPM_robotic_arm_ws/ros2_ws/src/'
            'robotic_arm_environment/my_environment_pkg/models/'
            'aws_robomaker_warehouse_ShelfD_01/meshes/'
            'aws_robomaker_warehouse_ShelfD_01_visual.DAE'
        )

        # Đọc kích thước box từ model.sdf của obstacle_box_1
        self.obstacle_scale = self._read_obstacle_box_size_from_sdf()

        self.get_logger().info(
            f"Obstacle box size dùng cho RViz: "
            f"{self.obstacle_scale['x']} x "
            f"{self.obstacle_scale['y']} x "
            f"{self.obstacle_scale['z']} (m)"
        )

    # ================================================================
    # Đọc <box><size> từ file model.sdf của obstacle_box_1
    # ================================================================
    def _read_obstacle_box_size_from_sdf(self):
        # Đường dẫn tới model.sdf của obstacle_box_1
        sdf_path = os.path.expanduser(
            '~/QuanPM_robotic_arm_ws/ros2_ws/src/'
            'robotic_arm_environment/my_environment_pkg/models/'
            'obstacle_box_1/model.sdf'
        )

        default_scale = {'x': 0.4, 'y': 0.4, 'z': 0.4}

        try:
            tree = ET.parse(sdf_path)
            root = tree.getroot()

            # Tìm phần tử <size> bên trong <box>
            size_el = root.find('.//box/size')
            if size_el is None or size_el.text is None:
                self.get_logger().warn(
                    f"Không tìm thấy <box><size> trong {sdf_path}, "
                    f"dùng scale mặc định {default_scale}"
                )
                return default_scale

            parts = size_el.text.strip().split()
            if len(parts) != 3:
                self.get_logger().warn(
                    f"Giá trị <size> không hợp lệ '{size_el.text}', "
                    f"dùng scale mặc định {default_scale}"
                )
                return default_scale

            sx, sy, sz = map(float, parts)
            return {'x': sx, 'y': sy, 'z': sz}

        except Exception as e:
            self.get_logger().warn(
                f"Lỗi đọc SDF obstacle_box_1 ({sdf_path}): {e}. "
                f"Dùng scale mặc định {default_scale}"
            )
            return default_scale

    # ================== MARKER OBSTACLE (BOX ĐỎ) ==================
    def make_obstacle_marker(self, marker_id):
        m = Marker()
        m.header.frame_id = self.frame_id
        m.header.stamp = self.get_clock().now().to_msg()
        m.ns = 'obstacle_box_1'
        m.id = marker_id
        m.type = Marker.CUBE
        m.action = Marker.ADD

        m.pose.position.x = self.obstacle_pose['x']
        m.pose.position.y = self.obstacle_pose['y']
        m.pose.position.z = self.obstacle_pose['z']

        m.pose.orientation.x = 0.0
        m.pose.orientation.y = 0.0
        m.pose.orientation.z = 0.0
        m.pose.orientation.w = 1.0

        # Dùng đúng kích thước <size> đọc từ SDF (m)
        m.scale.x = self.obstacle_scale['x']
        m.scale.y = self.obstacle_scale['y']
        m.scale.z = self.obstacle_scale['z']

        m.color.r = 1.0
        m.color.g = 0.0
        m.color.b = 0.0
        m.color.a = 0.8

        m.lifetime = Duration(sec=0, nanosec=0)
        return m

    # ================== MARKER BUCKET (MESH 3D) ==================
    def make_bucket_marker(self, marker_id):
        m = Marker()
        m.header.frame_id = self.frame_id
        m.header.stamp = self.get_clock().now().to_msg()
        m.ns = 'aws_robomaker_warehouse_Bucket_01'
        m.id = marker_id
        m.type = Marker.MESH_RESOURCE
        m.action = Marker.ADD

        # Mesh thật của bucket
        m.mesh_resource = self.bucket_mesh_resource

        # Không dùng material trong DAE nữa, tự tô màu để tránh bị đen
        m.mesh_use_embedded_materials = False

        m.pose.position.x = self.bucket_pose['x']
        m.pose.position.y = self.bucket_pose['y']
        m.pose.position.z = self.bucket_pose['z']

        m.pose.orientation.x = 0.0
        m.pose.orientation.y = 0.0
        m.pose.orientation.z = 0.0
        m.pose.orientation.w = 1.0

        # scale = 1.0: kích thước mesh gốc; chỉnh nếu thấy quá to/nhỏ
        m.scale.x = 1.0
        m.scale.y = 1.0
        m.scale.z = 1.0

        # Màu xanh dương dễ nhìn
        m.color.r = 0.0
        m.color.g = 0.2
        m.color.b = 1.0
        m.color.a = 1.0

        m.lifetime = Duration(sec=0, nanosec=0)
        return m
    # ================== MARKER belt (MESH 3D) ==================
    def make_conveyor_belt_maker(self, marker_id):
        m = Marker()
        m.header.frame_id = self.frame_id
        m.header.stamp = self.get_clock().now().to_msg()
        m.ns = 'conveyor_belt'
        m.id = marker_id
        m.type = Marker.MESH_RESOURCE
        m.action = Marker.ADD

        # Mesh thật của bucket
        m.mesh_resource = self.conveyor_belt_mesh_resource

        # Không dùng material trong DAE nữa, tự tô màu để tránh bị đen
        m.mesh_use_embedded_materials = False

        m.pose.position.x = self.conveyor_belt_pose['x']
        m.pose.position.y = self.conveyor_belt_pose['y']
        m.pose.position.z = self.conveyor_belt_pose['z']

        m.pose.orientation.x = 0.0
        m.pose.orientation.y = 0.0
        m.pose.orientation.z = 0.0
        m.pose.orientation.w = 1.0

        # scale = 1.0: kích thước mesh gốc; chỉnh nếu thấy quá to/nhỏ
        m.scale.x = 1.0
        m.scale.y = 1.0
        m.scale.z = 1.0

        # Màu xanh dương dễ nhìn
        m.color.r = 1.0
        m.color.g = 0.0
        m.color.b = 0.0
        m.color.a = 1.0

        m.lifetime = Duration(sec=0, nanosec=0)
        return m
    # ================== MARKER belt (MESH 3D) ==================
    def make_ClutteringD_01_maker(self, marker_id):
        m = Marker()
        m.header.frame_id = self.frame_id
        m.header.stamp = self.get_clock().now().to_msg()
        m.ns = 'aws_robomaker_warehouse_ClutteringD_01'
        m.id = marker_id
        m.type = Marker.MESH_RESOURCE
        m.action = Marker.ADD

        # Mesh thật của bucket
        m.mesh_resource = self.ClutteringD_01_mesh_resource

        # Không dùng material trong DAE nữa, tự tô màu để tránh bị đen
        m.mesh_use_embedded_materials = False

        m.pose.position.x = self.ClutteringD_01_pose['x']
        m.pose.position.y = self.ClutteringD_01_pose['y']
        m.pose.position.z = self.ClutteringD_01_pose['z']

        m.pose.orientation.x = 0.0
        m.pose.orientation.y = 0.0
        m.pose.orientation.z = 0.0
        m.pose.orientation.w = 1.0

        # scale = 1.0: kích thước mesh gốc; chỉnh nếu thấy quá to/nhỏ
        m.scale.x = 1.0
        m.scale.y = 1.0
        m.scale.z = 1.0

        # Màu xanh dương dễ nhìn
        m.color.r = 0.0
        m.color.g = 1.0
        m.color.b = 0.0
        m.color.a = 1.0

        m.lifetime = Duration(sec=0, nanosec=0)
        return m
        
    # ================== MARKER belt (MESH 3D) ==================
    def make_DeskC_01_maker(self, marker_id):
        m = Marker()
        m.header.frame_id = self.frame_id
        m.header.stamp = self.get_clock().now().to_msg()
        m.ns = 'aws_robomaker_warehouse_DeskC_01'
        m.id = marker_id
        m.type = Marker.MESH_RESOURCE
        m.action = Marker.ADD

        # Mesh thật của bucket
        m.mesh_resource = self.DeskC_01_mesh_resource

        # Không dùng material trong DAE nữa, tự tô màu để tránh bị đen
        m.mesh_use_embedded_materials = False

        m.pose.position.x = self.DeskC_01_pose['x']
        m.pose.position.y = self.DeskC_01_pose['y']
        m.pose.position.z = self.DeskC_01_pose['z']

        m.pose.orientation.x = 0.0
        m.pose.orientation.y = 0.0
        m.pose.orientation.z = 0.0
        m.pose.orientation.w = 1.0

        # scale = 1.0: kích thước mesh gốc; chỉnh nếu thấy quá to/nhỏ
        m.scale.x = 1.0
        m.scale.y = 1.0
        m.scale.z = 1.0

        # Màu xanh dương dễ nhìn
        m.color.r = 1.0
        m.color.g = 1.0
        m.color.b = 0.0
        m.color.a = 1.0

        m.lifetime = Duration(sec=0, nanosec=0)
        return m
    def make_PalletJackB_01_maker(self, marker_id):
        m = Marker()
        m.header.frame_id = self.frame_id
        m.header.stamp = self.get_clock().now().to_msg()
        m.ns = 'aws_robomaker_warehouse_PalletJackB_01'
        m.id = marker_id
        m.type = Marker.MESH_RESOURCE
        m.action = Marker.ADD

        # Mesh thật của bucket
        m.mesh_resource = self.PalletJackB_01_mesh_resource

        # Không dùng material trong DAE nữa, tự tô màu để tránh bị đen
        m.mesh_use_embedded_materials = False

        m.pose.position.x = self.PalletJackB_01_pose['x']
        m.pose.position.y = self.PalletJackB_01_pose['y']
        m.pose.position.z = self.PalletJackB_01_pose['z']

        m.pose.orientation.x = 0.0
        m.pose.orientation.y = 0.0
        m.pose.orientation.z = 0.0
        m.pose.orientation.w = 1.0

        # scale = 1.0: kích thước mesh gốc; chỉnh nếu thấy quá to/nhỏ
        m.scale.x = 1.0
        m.scale.y = 1.0
        m.scale.z = 1.0

        # Màu xanh dương dễ nhìn
        m.color.r = 1.0
        m.color.g = 0.5
        m.color.b = 0.0
        m.color.a = 1.0

        m.lifetime = Duration(sec=0, nanosec=0)
        return m
    def make_ShelfD_01_maker(self, marker_id):
        m = Marker()
        m.header.frame_id = self.frame_id
        m.header.stamp = self.get_clock().now().to_msg()
        m.ns = 'aws_robomaker_warehouse_ShelfD_01'
        m.id = marker_id
        m.type = Marker.MESH_RESOURCE
        m.action = Marker.ADD

        # Mesh thật của bucket
        m.mesh_resource = self.ShelfD_01_mesh_resource

        # Không dùng material trong DAE nữa, tự tô màu để tránh bị đen
        m.mesh_use_embedded_materials = False

        m.pose.position.x = self.ShelfD_01_pose['x']
        m.pose.position.y = self.ShelfD_01_pose['y']
        m.pose.position.z = self.ShelfD_01_pose['z']

        m.pose.orientation.x = 0.0
        m.pose.orientation.y = 0.0
        m.pose.orientation.z = 0.0
        m.pose.orientation.w = 1.0

        # scale = 1.0: kích thước mesh gốc; chỉnh nếu thấy quá to/nhỏ
        m.scale.x = 1.0
        m.scale.y = 1.0
        m.scale.z = 1.0

        # Màu xanh dương dễ nhìn
        m.color.r = 0.6
        m.color.g = 0.0
        m.color.b = 0.8
        m.color.a = 1.0

        m.lifetime = Duration(sec=0, nanosec=0)
        return m
    # ================== TIMER CALLBACK ==================
    def timer_callback(self):
        marker_array = MarkerArray()

        # Obstacle box
        obstacle_marker = self.make_obstacle_marker(marker_id=1)
        marker_array.markers.append(obstacle_marker)

        # Bucket mesh
        bucket_marker = self.make_bucket_marker(marker_id=2)
        marker_array.markers.append(bucket_marker)
        # Bucket mesh
        conveyor_belt_maker = self.make_conveyor_belt_maker(marker_id=3)
        marker_array.markers.append(conveyor_belt_maker)
        # Bucket mesh
        ClutteringD_01_maker = self.make_ClutteringD_01_maker(marker_id=4)
        marker_array.markers.append(ClutteringD_01_maker)
        # Bucket mesh
        DeskC_01_maker = self.make_DeskC_01_maker(marker_id=5)
        marker_array.markers.append(DeskC_01_maker)
        # Bucket mesh
        PalletJackB_01_maker = self.make_PalletJackB_01_maker(marker_id=6)
        marker_array.markers.append(PalletJackB_01_maker)
        # Bucket mesh
        ShelfD_01_maker = self.make_ShelfD_01_maker(marker_id=7)
        marker_array.markers.append(ShelfD_01_maker)

        self.publisher_.publish(marker_array)


def main(args=None):
    rclpy.init(args=args)
    node = EnvMarkersNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

