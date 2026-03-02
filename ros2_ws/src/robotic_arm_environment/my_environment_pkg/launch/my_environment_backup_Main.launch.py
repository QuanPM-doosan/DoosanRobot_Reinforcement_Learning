import os
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
from launch.actions import IncludeLaunchDescription, ExecuteProcess, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource

def generate_launch_description():
    my_sphere_files       = get_package_share_directory('my_sphere_pkg')
    my_doosan_robot_files = get_package_share_directory('my_doosan_pkg')
    my_env_files          = get_package_share_directory('my_environment_pkg')

    # Robot + controller
    doosan_robot = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(my_doosan_robot_files, 'launch', 'my_doosan_controller.launch.py')
        )
    )

    # Sphere marker
    sphere_mark = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(my_sphere_files, 'launch', 'my_sphere.launch.py')
        )
    )

    # RViz
    rviz_file = os.path.join(my_env_files, 'rviz', 'my_rviz_env.rviz')
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',                    # dễ thấy log lỗi nếu có
        arguments=['-d', rviz_file],
        parameters=[{'use_sim_time': True}]
    )

    # Gazebo (Classic)
    world = os.path.join(my_env_files, 'worlds', 'my_world.world')
    gazebo_node = ExecuteProcess(
        cmd=['gazebo', '--verbose', world, '-s', 'libgazebo_ros_factory.so'],
        output='screen'
    )

    # Trả về launch desc — có thể delay RViz 2–3 giây cho TF/topic sẵn sàng
    return LaunchDescription([
        doosan_robot,
        sphere_mark,
        gazebo_node,
        TimerAction(period=3.0, actions=[rviz_node]),  # hoặc bỏ TimerAction và thêm rviz_node trực tiếp
    ])

