#!/usr/bin/env python3
"""Spawn N namespaced robots into a generated agri-field world.

Multi-robot namespacing is where most of week 1 goes if you improvise it, so
the three rules that matter are fixed here:

  1. One namespace per robot, and TF frames carry the SAME prefix. A shared
     'base_link' across robots produces a TF tree that looks fine in RViz for
     about ninety seconds and then silently poisons every transform lookup.
  2. /tf is remapped INTO the namespace. Twelve robots publishing odom->base_link
     onto one global /tf is the classic multi-robot failure and it is a
     one-line remap to avoid.
  3. The world is passed by PATH, generated beforehand by generate_field.py.
     The launch file never randomizes anything.

Example:
    python3 -m agri_swarm_core.generate_field --seed 0 --out /tmp/worlds
    ros2 launch agri_swarm_bringup swarm.launch.py \
        n_robots:=8 seed:=0 world_dir:=/tmp/worlds bid_mode:=confidence_energy
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

import xacro


ROW_SPACING = 0.75          # keep in sync with generate_field.py defaults
START_X = -1.2              # headland, behind the crop rows


def _robot_group(idx, seed, world_dir, bid_mode, use_sim_time):
    name = f"robot_{idx}"
    prefix = f"{name}/"

    xacro_file = os.path.join(
        get_package_share_directory("agri_swarm_description"),
        "urdf", "agri_bot.urdf.xacro")
    urdf = xacro.process_file(
        xacro_file, mappings={"prefix": prefix, "use_lidar": "false"}).toxml()

    # Robots line up on the headland, one per lane.
    y = -ROW_SPACING / 2.0 + idx * ROW_SPACING

    common = {"use_sim_time": use_sim_time}

    rsp = Node(
        package="robot_state_publisher", executable="robot_state_publisher",
        namespace=name, output="log",
        parameters=[{"robot_description": urdf,
                     "frame_prefix": prefix, **common}],
        remappings=[("/tf", "tf"), ("/tf_static", "tf_static")],
    )

    spawn = Node(
        package="ros_gz_sim", executable="create", output="log",
        arguments=["-name", name, "-string", urdf,
                   "-x", str(START_X), "-y", str(y), "-z", "0.08"],
    )

    bridge = Node(
        package="ros_gz_bridge", executable="parameter_bridge",
        namespace=name, output="log",
        arguments=[
            f"/model/{name}/cmd_vel@geometry_msgs/msg/Twist]gz.msgs.Twist",
            f"/model/{name}/odometry@nav_msgs/msg/Odometry[gz.msgs.Odometry",
        ],
        remappings=[
            (f"/model/{name}/cmd_vel", "cmd_vel"),
            (f"/model/{name}/odometry", "odom"),
        ],
        parameters=[common],
    )

    detector = Node(
        package="agri_swarm_core", executable="detector_node",
        namespace=name, output="log",
        parameters=[{
            "robot_id": name,
            "seed": seed,
            "ground_truth_csv": os.path.join(world_dir, f"ground_truth_{seed}.csv"),
            **common,
        }],
    )

    allocator = Node(
        package="agri_swarm_allocation", executable="allocator_node",
        namespace=name, output="screen",   # keep split-brain warnings visible
        parameters=[{"robot_id": name, "bid_mode": bid_mode, **common}],
    )

    return [rsp, spawn, bridge, detector, allocator]


def _setup(context, *args, **kwargs):
    n = int(LaunchConfiguration("n_robots").perform(context))
    seed = int(LaunchConfiguration("seed").perform(context))
    world_dir = LaunchConfiguration("world_dir").perform(context)
    bid_mode = LaunchConfiguration("bid_mode").perform(context)
    headless = LaunchConfiguration("headless").perform(context).lower() == "true"

    world = os.path.join(world_dir, f"agri_field_{seed}.sdf")
    if not os.path.isfile(world):
        raise RuntimeError(
            f"{world} not found. Generate it first:\n"
            f"  python3 -m agri_swarm_core.generate_field "
            f"--seed {seed} --out {world_dir}")

    gz_args = f"-r -s {world}" if headless else f"-r {world}"
    gz = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(
            get_package_share_directory("ros_gz_sim"),
            "launch", "gz_sim.launch.py")),
        launch_arguments={"gz_args": gz_args}.items(),
    )

    clock_bridge = Node(
        package="ros_gz_bridge", executable="parameter_bridge", output="log",
        arguments=["/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock"],
    )

    actions = [gz, clock_bridge]
    for i in range(n):
        actions += _robot_group(i, seed, world_dir, bid_mode, True)
    return actions


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("n_robots", default_value="4"),
        DeclareLaunchArgument("seed", default_value="0"),
        DeclareLaunchArgument("world_dir", default_value="/tmp/worlds"),
        DeclareLaunchArgument("bid_mode", default_value="confidence_energy",
                              description="distance | confidence_energy"),
        DeclareLaunchArgument("headless", default_value="true",
                              description="true for batch runs; false to watch"),
        OpaqueFunction(function=_setup),
    ])
