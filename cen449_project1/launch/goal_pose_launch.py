from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    SetEnvironmentVariable,
    TimerAction,
)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():

    model = LaunchConfiguration('model')
    use_sim_time = LaunchConfiguration('use_sim_time')
    slam = LaunchConfiguration('slam')
    run_goal_client = LaunchConfiguration('run_goal_client')


    set_model = SetEnvironmentVariable('TURTLEBOT3_MODEL', model)


    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                FindPackageShare('turtlebot3_gazebo'),
                'launch',
                'turtlebot3_world.launch.py',
            ])
        )
    )


    nav2 = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                FindPackageShare('turtlebot3_navigation2'),
                'launch',
                'navigation2.launch.py',
            ])
        ),

        launch_arguments={
            'slam': slam,
            'use_sim_time': use_sim_time,
        }.items(),
    )


    goal_client_node = Node(
        package='cen449_project1',
        executable='goal_pose_client',
        output='screen',
        condition=IfCondition(run_goal_client),
    )

    start_goal_client = TimerAction(
        period=10.0,         
        actions=[goal_client_node],
    )

    return LaunchDescription([
        DeclareLaunchArgument('model', default_value='burger'),
        DeclareLaunchArgument('use_sim_time', default_value='True'),
        DeclareLaunchArgument('slam', default_value='True'),
        DeclareLaunchArgument('run_goal_client', default_value='True'),

        set_model,
        gazebo,
        nav2,
        start_goal_client,
    ])

