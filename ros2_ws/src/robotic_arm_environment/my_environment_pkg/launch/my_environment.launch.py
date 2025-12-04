import os
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, ExecuteProcess, TimerAction, RegisterEventHandler
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.event_handlers import OnProcessStart
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def _model_sdf_path(pkg_share, model_name):
    """
    Chọn đường dẫn SDF cho model:
    - Ưu tiên: share/my_environment_pkg/models/<model_name>/model.sdf
    - Fallback: src/robotic_arm_environment/my_environment_pkg/models/<model_name>/model.sdf
    """
    # Đường dẫn trong share/ của package (sau khi install)
    share_path = os.path.join(pkg_share, 'models', model_name, 'model.sdf')
    if os.path.exists(share_path):
        return share_path

    # Fallback: đường dẫn thư mục src mà bạn đang dùng
    src_base = os.path.expanduser(
        '~/TiepDB_robotic_arm_ws_7_5_trieu/ros2_ws/src/robotic_arm_environment/my_environment_pkg/models'
    )
    src_path = os.path.join(src_base, model_name, 'model.sdf')
    return src_path


def generate_launch_description():
    # Lấy đường dẫn share của các package
    #my_sphere_files       = get_package_share_directory('my_sphere_pkg')
    my_doosan_robot_files = get_package_share_directory('my_doosan_pkg')
    my_env_files          = get_package_share_directory('my_environment_pkg')

    # ---------------- GAZEBO CLASSIC ----------------
    world = os.path.join(my_env_files, 'worlds', 'my_world.world')
    gazebo_node = ExecuteProcess(
        cmd=[
            'gazebo', '--verbose', world,
            '-s', 'libgazebo_ros_init.so',
            '-s', 'libgazebo_ros_factory.so'
        ],
        output='screen'
    )

    # ---------------- ROBOT + CONTROLLER ----------------
    doosan_robot = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(my_doosan_robot_files, 'launch', 'my_doosan_controller.launch.py')
        ),
        # THÊM ĐOẠN NÀY ĐỂ BẬT GRIPPER:
        launch_arguments={
            'gripper': 'robotiq_2f',   # Bật gripper Robotiq 2F
            'color': 'blue'            # (Tùy chọn) Đảm bảo màu robot đúng
        }.items(),
    )

    # ---------------- RVIZ2 ----------------
    rviz_file = os.path.join(my_env_files, 'rviz', 'my_rviz_env.rviz')
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        arguments=['-d', rviz_file],
        parameters=[{'use_sim_time': True}]
    )

    # ---------------- NODE VẼ MARKER CHO MÔI TRƯỜNG ----------------
    env_markers_node = Node(
        package='my_environment_pkg',
        executable='env_markers_node',
        name='env_markers_node',
        output='screen',
        parameters=[{'use_sim_time': True}]
    )

    # ---------------- (OPTIONAL) CÁC MODEL KHÁC – ĐỂ COMMENT ----------------
    # sphere_mark = IncludeLaunchDescription(
    #     PythonLaunchDescriptionSource(
    #         os.path.join(my_sphere_files, 'launch', 'my_sphere.launch.py')
    #     )
    # )

    # marker_a_sdf = _model_sdf_path(my_env_files, 'marker_A')
    # marker_b_sdf = _model_sdf_path(my_env_files, 'marker_B')
    # pick_sdf     = _model_sdf_path(my_env_files, 'pick_object')

    # ---------------- ĐƯỜNG DẪN SDF CÁC OBJECT ĐANG DÙNG ----------------
    obs1_sdf   = _model_sdf_path(my_env_files, 'obstacle_box_1')
    bucket_sdf = _model_sdf_path(my_env_files, 'aws_robomaker_warehouse_Bucket_01')
    GroundB_01 = _model_sdf_path(my_env_files, 'aws_robomaker_warehouse_GroundB_01')
    conveyor_belt = _model_sdf_path(my_env_files, 'conveyor_belt')
    ClutteringD_01 = _model_sdf_path(my_env_files, 'aws_robomaker_warehouse_ClutteringD_01')
    DeskC_01 = _model_sdf_path(my_env_files, 'aws_robomaker_warehouse_DeskC_01')
    PalletJackB_01 = _model_sdf_path(my_env_files, 'aws_robomaker_warehouse_PalletJackB_01')
    ShelfD_01 = _model_sdf_path(my_env_files, 'aws_robomaker_warehouse_ShelfD_01')

    # ---------------- SPAWN NODES ----------------

    # Obstacle box
    spawn_obstacle_1 = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        name='spawn_obstacle_box_1',
        output='screen',
        respawn=True, respawn_delay=2.0,
        arguments=[
            '-entity', 'obstacle_box_1',
            '-file', obs1_sdf,
            '-x', '0.70', '-y', '-0.20', '-z', '0.6'
        ]
    )

    # Bucket model từ aws_robomaker_warehouse_Bucket_01
    spawn_bucket = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        name='spawn_aws_bucket_01',
        output='screen',
        respawn=True, respawn_delay=2.0,
        arguments=[
            '-entity', 'aws_robomaker_warehouse_Bucket_01',
            '-file', bucket_sdf,
            '-x', '1.747962', '-y', '1.20', '-z', '0.1'
        ]
    )
    # Bucket model từ aws_robomaker_warehouse_Bucket_01
    spawn_GroundB_01 = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        name='spawn_aws_GroundB_01',
        output='screen',
        respawn=True, respawn_delay=2.0,
        arguments=[
            '-entity', 'aws_robomaker_warehouse_GroundB_01',
            '-file', GroundB_01,
            '-x', '-1.00', '-y', '0.00', '-z', '0.00'
        ]
    )
    #  Quan_Haui
    spawn_conveyor_belt = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        name='spawn_aws_conveyor_belt',
        output='screen',
        respawn=True, respawn_delay=2.0,
        arguments=[
            '-entity', 'conveyor_belt',
            '-file', conveyor_belt,
            '-x', '-1.390076', '-y', '-0.174037', '-z', '0.00'
        ]
    )
    # *Quan_haui
    spawn_ClutteringD_01 = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        name='spawn_aws_ClutteringD_01',
        output='screen',
        respawn=True, respawn_delay=2.0,
        arguments=[
            '-entity', 'aws_robomaker_warehouse_ClutteringD_01',
            '-file', ClutteringD_01,
            '-x', '-2.718640', '-y', '-0.752731', '-z', '0.00'
        ]
    )
    # *** Quan_haui3
    spawn_DeskC_01 = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        name='spawn_aws_DeskC_01',
        output='screen',
        respawn=True, respawn_delay=2.0,
        arguments=[
            '-entity', 'aws_robomaker_warehouse_DeskC_01',
            '-file', DeskC_01,
            '-x', '-2.745320', '-y', '-3.283404', '-z', '0.00'
        ]
    )
    # *** Quan_haui4
    spawn_PalletJackB_01 = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        name='spawn_aws_PalletJackB_01',
        output='screen',
        respawn=True, respawn_delay=2.0,
        arguments=[
            '-entity', 'aws_robomaker_warehouse_PalletJackB_01',
            '-file', PalletJackB_01,
            '-x', '-1.293370', '-y', '-1.939590', '-z', '0.00'
        ]
    )
    # *** Quan_haui5
    spawn_ShelfD_01 = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        name='spawn_aws_ShelfD_01',
        output='screen',
        respawn=True, respawn_delay=2.0,
        arguments=[
            '-entity', 'aws_robomaker_warehouse_ShelfD_01',
            '-file', ShelfD_01,
            '-x', '3.311350', '-y', '3.079530', '-z', '0.00'
        ]
    )

    # ---------------- SPAWN SAU KHI GAZEBO START ----------------
    spawn_after_gazebo = RegisterEventHandler(
        OnProcessStart(
            target_action=gazebo_node,
            on_start=[
                # Sau 6s kể từ lúc Gazebo start → spawn entity
                TimerAction(
                    period=6.0,
                    actions=[
                        spawn_obstacle_1,
                        spawn_bucket,
                        spawn_GroundB_01,
                        spawn_conveyor_belt,
                        spawn_ClutteringD_01,
                        spawn_DeskC_01,
                        spawn_PalletJackB_01,
                        spawn_ShelfD_01,
                    ]
                ),
                # Sau 8s kể từ lúc Gazebo start → mở RViz + node marker
                TimerAction(
                    period=8.0,
                    actions=[
                        rviz_node,
                        env_markers_node,
                    ]
                ),
            ]
        )
    )

    return LaunchDescription([
        gazebo_node,
        doosan_robot,
        spawn_after_gazebo,
    ])

