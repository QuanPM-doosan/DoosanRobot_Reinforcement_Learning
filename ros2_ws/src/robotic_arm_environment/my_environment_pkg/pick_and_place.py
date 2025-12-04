# my_environment_pkg/pick_and_place.py
import math, time
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from rclpy.action import ActionClient
from control_msgs.action import FollowJointTrajectory
from moveit_msgs.srv import GetPositionIK
from std_srvs.srv import Empty
from builtin_interfaces.msg import Duration

class PickPlaceNode(Node):
    def __init__(self):
        super().__init__('pick_place_node')

        # TODO: sửa lại cho đúng robot của bạn
        self.joint_names = ['joint1','joint2','joint3','joint4','joint5','joint6']   # tên khớp y như controller
        self.base_frame  = 'base_link'   # frame gốc của robot trong URDF/TF
        self.ee_link     = 'ee_link'     # link cuối
        self.group_name  = 'manipulator' # planning group trong MoveIt

        # Action client tới joint_trajectory_controller
        self.traj_ac = ActionClient(self, FollowJointTrajectory, '/joint_trajectory_controller/follow_joint_trajectory')

        # IK service của MoveIt
        self.ik_cli = self.create_client(GetPositionIK, '/compute_ik')

        # Link attacher services (nếu plugin dùng namespace khác, chỉnh tên)
        from gazebo_ros_link_attacher.srv import Attach
        self.attach_cli = self.create_client(Attach, '/link_attacher/attach')
        self.detach_cli = self.create_client(Attach, '/link_attacher/detach')

    def wait(self):
        rclpy.spin_once(self, timeout_sec=0.1)

    def pose(self, x,y,z, qx=0,qy=0,qz=0,qw=1.0):
        p = PoseStamped()
        p.header.frame_id = self.base_frame
        p.pose.position.x = float(x); p.pose.position.y = float(y); p.pose.position.z = float(z)
        p.pose.orientation.x = qx; p.pose.orientation.y = qy; p.pose.orientation.z = qz; p.pose.orientation.w = qw
        return p

    def compute_ik(self, p: PoseStamped):
        # chờ service
        if not self.ik_cli.wait_for_service(timeout_sec=5.0):
            self.get_logger().error('MoveIt /compute_ik not available')
            return None
        req = GetPositionIK.Request()
        req.ik_request.group_name = self.group_name
        req.ik_request.ik_link_name = self.ee_link
        req.ik_request.pose_stamped = p
        req.ik_request.timeout = Duration(sec=1, nanosec=0)
        req.ik_request.attempts = 10
        res = self.ik_cli.call(req)
        if res.error_code.val != res.error_code.SUCCESS:
            self.get_logger().warn(f'IK failed with code {res.error_code.val}')
            return None
        joints = []
        for name in self.joint_names:
            # lấy theo thứ tự tên khớp
            idx = res.solution.joint_state.name.index(name)
            joints.append(res.solution.joint_state.position[idx])
        return joints

    def move_joints(self, q_target, sec=2.0):
        # gửi 1 điểm trajectory đến controller
        if not self.traj_ac.wait_for_server(timeout_sec=5.0):
            self.get_logger().error('FollowJointTrajectory action not available')
            return False
        goal = FollowJointTrajectory.Goal()
        goal.trajectory = JointTrajectory()
        goal.trajectory.joint_names = self.joint_names
        pt = JointTrajectoryPoint()
        pt.positions = q_target
        pt.time_from_start = Duration(sec=int(sec), nanosec=int((sec-int(sec))*1e9))
        goal.trajectory.points = [pt]
        future = self.traj_ac.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, future)
        result_future = future.result().get_result_async()
        rclpy.spin_until_future_complete(self, result_future)
        ok = result_future.result().error_code == 0
        if ok: self.get_logger().info('Trajectory exec OK')
        else:  self.get_logger().warn(f'Trajectory error code {result_future.result().error_code}')
        return ok

    def attach(self, model1, link1, model2, link2):
        from gazebo_ros_link_attacher.srv import Attach
        if not self.attach_cli.wait_for_service(timeout_sec=2.0):
            self.get_logger().error('Attach service not available'); return False
        req = Attach.Request(); req.model_name_1=model1; req.link_name_1=link1; req.model_name_2=model2; req.link_name_2=link2
        res = self.attach_cli.call(req)
        self.get_logger().info(f'Attach: {res.ok}')
        return res.ok

    def detach(self, model1, link1, model2, link2):
        from gazebo_ros_link_attacher.srv import Attach
        if not self.detach_cli.wait_for_service(timeout_sec=2.0):
            self.get_logger().error('Detach service not available'); return False
        req = Attach.Request(); req.model_name_1=model1; req.link_name_1=link1; req.model_name_2=model2; req.link_name_2=link2
        res = self.detach_cli.call(req)
        self.get_logger().info(f'Detach: {res.ok}')
        return res.ok

def main():
    rclpy.init()
    node = PickPlaceNode()

    # === NHẬP toạ độ A/B từ launch env (đã đặt sẵn trong launch) hoặc hardcode ở đây ===
    # Ví dụ dùng env (bạn có thể set trong launch bằng SetEnvironmentVariable):
    import os
    Axyz = (float(os.getenv('A_X', '-0.5')), float(os.getenv('A_Y', '-0.4')), float(os.getenv('A_Z', '0.90')))
    Bxyz = (float(os.getenv('B_X', '0.2')),  float(os.getenv('B_Y', '0.55')), float(os.getenv('B_Z', '0.3')))

    # 1) Tới pre-grasp A (cao hơn A 10cm)
    preA = node.pose(Axyz[0], Axyz[1], Axyz[2] + 0.10)
    q_preA = node.compute_ik(preA); assert q_preA is not None, "IK preA failed"
    node.move_joints(q_preA, sec=2.5)

    # 2) Hạ xuống grasp tại A
    graspA = node.pose(Axyz[0], Axyz[1], Axyz[2])
    q_graspA = node.compute_ik(graspA); assert q_graspA is not None, "IK graspA failed"
    node.move_joints(q_graspA, sec=1.5)

    # 3) Attach pick_object::obj vào ee_link
    node.attach('pick_object','obj','doosan','ee_link')  # TODO: đổi 'doosan' và 'ee_link' đúng tên

    # 4) Nhấc lên preA
    node.move_joints(q_preA, sec=1.5)

    # 5) Sang pre-place B
    preB = node.pose(Bxyz[0], Bxyz[1], Bxyz[2] + 0.10)
    q_preB = node.compute_ik(preB); assert q_preB is not None, "IK preB failed"
    node.move_joints(q_preB, sec=3.0)

    # 6) Hạ xuống place tại B
    placeB = node.pose(Bxyz[0], Bxyz[1], Bxyz[2])
    q_placeB = node.compute_ik(placeB); assert q_placeB is not None, "IK placeB failed"
    node.move_joints(q_placeB, sec=1.5)

    # 7) Detach
    node.detach('pick_object','obj','doosan','ee_link')  # TODO: đổi tên đúng

    # 8) Rút lên preB
    node.move_joints(q_preB, sec=1.5)

    node.get_logger().info('Pick-&-Place done.')
    rclpy.shutdown()

if __name__ == '__main__':
    main()

