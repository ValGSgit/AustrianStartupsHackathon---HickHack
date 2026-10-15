"""Launch the SO-101 sim with one control source.

    ros2 launch so101_sim_ros sim.launch.py mode:=scripted              # expert picks in a loop
    ros2 launch so101_sim_ros sim.launch.py mode:=teleop                # leader arm via host/leader_to_sim.py
    ros2 launch so101_sim_ros sim.launch.py mode:=policy policy_path:=/ws/outputs/train/<name>/checkpoints/last/pretrained_model
    ros2 launch so101_sim_ros sim.launch.py mode:=idle                  # just the sim; publish your own commands

Foxglove Studio -> "Open connection" -> ws://localhost:8765
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node


def generate_launch_description():
    mode = LaunchConfiguration("mode")
    is_mode = lambda m: IfCondition(PythonExpression(["'", mode, f"' == '{m}'"]))  # noqa: E731

    return LaunchDescription([
        DeclareLaunchArgument("mode", default_value="scripted", description="idle | scripted | teleop | policy"),
        DeclareLaunchArgument("obj", default_value="pen", description="pen | cube"),
        DeclareLaunchArgument("camera_hz", default_value="5.0", description="CPU rendering in Docker: ~100 ms/frame"),
        DeclareLaunchArgument("policy_path", default_value=""),
        DeclareLaunchArgument("task", default_value="Pick up the pen"),
        DeclareLaunchArgument("foxglove", default_value="true"),
        Node(
            package="so101_sim_ros", executable="sim_node", output="screen",
            parameters=[{
                "obj": LaunchConfiguration("obj"),
                "camera_hz": LaunchConfiguration("camera_hz"),
                "autoplay": PythonExpression(["'", mode, "' == 'scripted'"]),
            }],
        ),
        Node(package="so101_sim_ros", executable="teleop_udp_node", output="screen", condition=is_mode("teleop")),
        Node(
            package="so101_sim_ros", executable="policy_node", output="screen", condition=is_mode("policy"),
            parameters=[{"policy_path": LaunchConfiguration("policy_path"), "task": LaunchConfiguration("task")}],
        ),
        Node(
            package="foxglove_bridge", executable="foxglove_bridge", output="log",
            condition=IfCondition(LaunchConfiguration("foxglove")),
            parameters=[{"port": 8765, "address": "0.0.0.0"}],
        ),
    ])
