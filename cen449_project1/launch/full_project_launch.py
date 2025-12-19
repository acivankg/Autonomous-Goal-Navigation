from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    SetEnvironmentVariable,
    TimerAction,
    RegisterEventHandler,
    ExecuteProcess,
)
from launch.conditions import IfCondition
from launch.event_handlers import OnShutdown
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, PythonExpression
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    # Launch arguments (can be overridden from the terminal)
    model = LaunchConfiguration('model')
    run_goal_client = LaunchConfiguration('run_goal_client')

    # Set TURTLEBOT3_MODEL for all processes started by this launch file
    set_tb3_model = SetEnvironmentVariable('TURTLEBOT3_MODEL', model)

    # 1) Start Gazebo with TurtleBot3 world
    gazebo_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                FindPackageShare('turtlebot3_gazebo'),
                'launch',
                'turtlebot3_world.launch.py',
            ])
        )
    )

    # 2) Select correct Nav2 params file (depends on model name)
    nav2_params_file = PathJoinSubstitution([
        FindPackageShare('turtlebot3_navigation2'),
        'param',
        'humble',
        PythonExpression(["'", model, "'", " + '.yaml'"]),
    ])

    # Start Navigation2 with SLAM enabled (no pre-saved map)
    nav2_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                FindPackageShare('turtlebot3_navigation2'),
                'launch',
                'navigation2.launch.py',
            ])
        ),
        launch_arguments={
            'slam': 'True',
            'use_sim_time': 'True',
            'params_file': nav2_params_file,
        }.items(),
    )

    # 3) Start your custom goal client node (optional)
    goal_client_node = Node(
        package='cen449_project1',
        executable='goal_pose_client',
        output='screen',
        condition=IfCondition(run_goal_client),
    )

    # Delay Nav2 and your node so Gazebo has time to fully start
    start_nav2 = TimerAction(period=15.0, actions=[nav2_launch])
    start_goal_client = TimerAction(period=30.0, actions=[goal_client_node])

    # 4) Best-effort cleanup when user stops the launch (Ctrl+C)
    cleanup_on_shutdown = RegisterEventHandler(
        OnShutdown(
            on_shutdown=[
                ExecuteProcess(
                    cmd=[
                        'bash', '-c',
                        'killall -q gzserver gzclient rviz2 component_container_isolated || true'
                    ],
                    output='screen',
                )
            ]
        )
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            'model',
            default_value='burger',
            description='TurtleBot3 model (burger / waffle / waffle_pi)',
        ),
        DeclareLaunchArgument(
            'run_goal_client',
            default_value='true',
            description='Whether to run the goal_pose_client node',
        ),

        set_tb3_model,
        gazebo_launch,
        start_nav2,
        start_goal_client,
        cleanup_on_shutdown,
    ])

