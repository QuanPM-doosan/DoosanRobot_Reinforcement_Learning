import rclpy
from rclpy.duration import Duration
from rclpy.action import ActionClient
from rclpy.node import Node
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint
import time


class TrajectoryActionClient(Node):

    def __init__(self):
        super().__init__('points_publisher_node_action_client')
        self.action_client = ActionClient(
            self, FollowJointTrajectory, '/joint_trajectory_controller/follow_joint_trajectory'
        )

    def send_single_goal(self, points):
        joint_names = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']
        goal_msg = FollowJointTrajectory.Goal()
        goal_msg.goal_time_tolerance = Duration(seconds=1).to_msg()
        goal_msg.trajectory.joint_names = joint_names
        goal_msg.trajectory.points = points

        self.action_client.wait_for_server()
        self.send_goal_future = self.action_client.send_goal_async(
            goal_msg, feedback_callback=self.feedback_callback
        )
        self.send_goal_future.add_done_callback(self.goal_response_callback)

        # Wait until the goal is complete
        while not hasattr(self, 'goal_done') or not self.goal_done:
            rclpy.spin_once(self)

        self.goal_done = False  # Reset for next goal

    def send_goal(self):
        def make_point(positions, seconds):
            pt = JointTrajectoryPoint()
            pt.positions = positions
            pt.time_from_start = Duration(seconds=seconds).to_msg()
            return pt

        while rclpy.ok():
            self.get_logger().info('Sending point 1 → 2')
            self.send_single_goal([
                make_point([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 2),
                make_point([0.0, 0.39, 1.57, 0.0, 1.6, 1.6], 6)
            ])
            self.get_logger().info('⏸️ Dừng tại điểm 2 - 3s')
            time.sleep(1)

            self.get_logger().info('Sending point 2 → 3')
            self.send_single_goal([
                make_point([0.0, 0.39, 1.57, 0.0, 1.6, 1.6], 2),
                make_point([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 4)
            ])

            self.get_logger().info('Sending point 3 → 4')
            self.send_single_goal([
                make_point([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 2),
                make_point([0.0, -0.39, -1.57, 0.0, -1.6, 1.6], 6)
            ])
            self.get_logger().info('⏸️ Dừng tại điểm 4 - 3s')
            time.sleep(1)

            self.get_logger().info('Sending point 4 → 5 (reset)')
            self.send_single_goal([
                make_point([0.0, -0.39, -1.57, 0.0, -1.6, 1.6], 2),
                make_point([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 4)
            ])

            time.sleep(1)  # optional pause before loop repeats

    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().info('❌ Goal rejected')
            self.goal_done = True
            return

        self.get_logger().info('✅ Goal accepted')
        self.get_result_future = goal_handle.get_result_async()
        self.get_result_future.add_done_callback(self.get_result_callback)

    def get_result_callback(self, future):
        result = future.result().result
        self.get_logger().info('✔️ Result received')
        self.goal_done = True

    def feedback_callback(self, feedback_msg):
        # Optional: handle feedback if needed
        pass


def main(args=None):
    rclpy.init()
    action_client = TrajectoryActionClient()
    action_client.send_goal()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

