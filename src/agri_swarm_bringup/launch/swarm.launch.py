import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument, EmitEvent, IncludeLaunchDescription, OpaqueFunction,
    RegisterEventHandler,
)
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

import xacro

from agri_swarm_core.pure_pursuit import assign_lanes, load_lanes


START_X = -1.2              # headland, behind the crop rows


def _robot_group(idx, n, lanes, cfg):
    name = f"robot_{idx}"
    prefix = f"{name}/"

    xacro_file = os.path.join(
        get_package_share_directory("agri_swarm_description"),
        "urdf", "agri_bot.urdf.xacro")
    urdf = xacro.process_file(
        xacro_file, mappings={"prefix": prefix, "use_lidar": "false"}).toxml()

    # Spawn each robot on the headland in front of the FIRST lane it is
    # actually assigned. Do not use idx * row_spacing here: assign_lanes()
    # hands out contiguous blocks, so robot 3 of 4 owns lanes 9-10, not lane 3.
    # Spawning by index would make every robot but the first cross several crop
    # rows diagonally before reaching its own work -- which fails the week-1
    # criterion on the very first control tick.
    my_lanes = assign_lanes(len(lanes), n, idx)
    if not my_lanes:
        raise RuntimeError(
            f"robot {idx} has no lanes: {len(lanes)} lanes over {n} robots")
    y = lanes[my_lanes[0]].y

    common = {"use_sim_time": True}
    lanes_csv = os.path.join(cfg["world_dir"], f"lanes_{cfg['seed']}.csv")

    # gz DiffDrive odometry is spawn-relative. Every node that compares a pose
    # against a world coordinate takes the spawn pose as origin_x/origin_y and
    # lifts the pose with world = origin + odom. Passing the wrong value here
    # is silent: robots drive across crop rows and report success.
    frame = {"origin_x": float(START_X), "origin_y": float(y)}

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
            "seed": cfg["seed"],
            "ground_truth_csv": os.path.join(
                cfg["world_dir"], f"ground_truth_{cfg['seed']}.csv"),
            **frame,
            **common,
        }],
    )

    # Publishes the RobotState heartbeat. Without it every peer looks dead to
    # allocator_node's isSilent(), so every award is re-announced and every
    # task is abandoned after max_rounds. The energy figures and the liveness
    # signal come from the same node because they share the same odometry.
    energy_monitor = Node(
        package="agri_swarm_core", executable="energy_monitor_node",
        namespace=name, output="log",
        parameters=[{
            "robot_id": name,
            "energy_capacity_j": cfg["energy_capacity_j"],
            "fail_at_s": cfg["fail_at_s"] if idx == cfg["fail_robot"] else -1.0,
            **frame,
            **common,
        }],
    )

    lane_follower = Node(
        package="agri_swarm_core", executable="lane_follower_node",
        namespace=name, output="log",
        parameters=[{
            "robot_index": idx,
            "n_robots": n,
            "lanes_csv": lanes_csv,
            **frame,
            **common,
        }],
    )

    # Drives the lanes and interrupts them to service awarded tasks. Replaces
    # lane_follower_node rather than joining it: two nodes publishing cmd_vel
    # to one robot is a race, and the resulting motion looks like a controller
    # tuning problem rather than a launch configuration one.
    task_executor = Node(
        package="agri_swarm_core", executable="task_executor_node",
        namespace=name, output="log",
        parameters=[{
            "robot_id": name,
            "robot_index": idx,
            "n_robots": n,
            "lanes_csv": lanes_csv,
            "row_spacing_m": cfg["row_spacing"],
            "n_rows": len(lanes) - 1,
            "spray_reach_m": cfg["spray_reach_m"],
            "treat_duration_s": cfg["treat_duration_s"],
            "headland_margin_m": cfg["headland_margin_m"],
            "fail_at_s": cfg["fail_at_s"] if idx == cfg["fail_robot"] else -1.0,
            **frame,
            **common,
        }],
    )

    allocator = Node(
        package="agri_swarm_allocation", executable="allocator_node",
        namespace=name, output="screen",   # keep split-brain warnings visible
        parameters=[{
            "robot_id": name,
            "bid_mode": cfg["bid_mode"],
            "treat_confidence_threshold": cfg["treat_conf"],
            "energy_capacity_j": cfg["energy_capacity_j"],
            **frame,
            **common,
        }],
    )

    nodes = [rsp, spawn, bridge, detector]

    # use_allocator selects WHICH driver runs, not whether a second one is
    # added. use_lane_follower stays the switch for driving at all.
    if cfg["use_lane_follower"]:
        nodes.append(task_executor if cfg["use_allocator"] else lane_follower)
    if cfg["use_allocator"]:
        nodes += [allocator, energy_monitor]
    return nodes


def _setup(context, *args, **kwargs):
    def arg(k):
        return LaunchConfiguration(k).perform(context)

    def flag(k):
        return arg(k).lower() == "true"

    n = int(arg("n_robots"))
    seed = int(arg("seed"))
    world_dir = arg("world_dir")

    lanes_csv = os.path.join(world_dir, f"lanes_{seed}.csv")
    if not os.path.isfile(lanes_csv):
        raise RuntimeError(f"{lanes_csv} not found; regenerate the field")
    lanes = load_lanes(lanes_csv)

    world = os.path.join(world_dir, f"agri_field_{seed}.sdf")
    if not os.path.isfile(world):
        raise RuntimeError(
            f"{world} not found. Generate it first:\n"
            f"  python3 -m agri_swarm_core.generate_field "
            f"--seed {seed} --out {world_dir}")

    # Read the field geometry from the lanes file rather than restating the
    # generator's defaults, which would drift the moment either changes.
    if len(lanes) < 2:
        raise RuntimeError(f"{lanes_csv} has fewer than two lanes")
    row_spacing = lanes[1].y - lanes[0].y

    spray_reach = float(arg("spray_reach_m"))
    if spray_reach < row_spacing / 2.0:
        raise RuntimeError(
            f"spray_reach_m {spray_reach} is below row_spacing/2 "
            f"{row_spacing / 2.0}. Intra-row weeds sit on the crop row and "
            "would be unreachable from any lane.")

    cfg = {
        "seed": seed,
        "world_dir": world_dir,
        "bid_mode": arg("bid_mode"),
        # Must equal min(--thresholds) in score_run.py, or the low end of the
        # recovered sweep is fabricated: the run never performed the
        # treatments a more permissive threshold would have.
        "treat_conf": float(arg("treat_confidence_threshold")),
        "energy_capacity_j": float(arg("energy_capacity_j")),
        "row_spacing": row_spacing,
        "spray_reach_m": spray_reach,
        "treat_duration_s": float(arg("treat_duration_s")),
        # Lanes span the crop rows only, so lateral movement between them is
        # legal only outside that span. Set to 0.0 if a future lanes file
        # already includes the headland.
        "headland_margin_m": float(arg("headland_margin_m")),
        "fail_robot": int(arg("fail_robot")),
        "fail_at_s": float(arg("fail_at_s")),
        "use_allocator": flag("use_allocator"),
        "use_lane_follower": flag("use_lane_follower"),
    }

    gz_args = f"-r -s {world}" if flag("headless") else f"-r {world}"
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

    # One logger for the whole run, not one per robot: a single writer avoids
    # interleaved partial rows in treatments.csv. It also supervises the run:
    # when every executor has held mission_idle for quiet_period_s it exits,
    # and the event handler below turns that into a shutdown of the graph.
    # Set auto_shutdown:=false to keep a run alive for interactive inspection.
    if cfg["use_allocator"]:
        supervise = flag("auto_shutdown")
        logger = Node(
            package="agri_swarm_core", executable="treatment_logger_node",
            output="screen",
            parameters=[{
                "out_csv": arg("treatments_csv"),
                "n_robots": n if supervise else 0,
                "quiet_period_s": float(arg("quiet_period_s")),
                "max_mission_s": float(arg("max_mission_s")),
                "use_sim_time": True,
            }],
        )
        actions.append(logger)
        if supervise:
            actions.append(RegisterEventHandler(OnProcessExit(
                target_action=logger,
                on_exit=[EmitEvent(event=Shutdown(
                    reason="mission complete"))],
            )))

    for i in range(n):
        actions += _robot_group(i, n, lanes, cfg)
    return actions


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("n_robots", default_value="4"),
        DeclareLaunchArgument("seed", default_value="0"),
        DeclareLaunchArgument("world_dir", default_value="/tmp/worlds"),
        DeclareLaunchArgument("bid_mode", default_value="confidence_energy",
                              description="distance | confidence_energy"),
        DeclareLaunchArgument("treat_confidence_threshold", default_value="0.5",
                              description="detections below this never become "
                                          "tasks; must match the lowest "
                                          "threshold swept offline"),
        DeclareLaunchArgument("energy_capacity_j", default_value="16000.0",
                              description="Robot battery capacity in joules. "
                                          "Must match the energy monitor. A "
                                          "full mission costs 11-19 kJ for the "
                                          "busiest robot; 16000 makes the "
                                          "reserve gate bite near the end."),
        DeclareLaunchArgument("headless", default_value="true",
                              description="true for batch runs; false to watch"),
        DeclareLaunchArgument("use_allocator", default_value="true",
                              description="false for the week-1 milestone; "
                                          "selects task_executor over "
                                          "lane_follower"),
        DeclareLaunchArgument("use_lane_follower", default_value="true",
                              description="whether anything drives at all"),
        DeclareLaunchArgument("treatments_csv", default_value="/tmp/treatments.csv"),
        DeclareLaunchArgument("spray_reach_m", default_value="0.45",
                              description="lateral boom reach; must be at "
                                          "least row_spacing/2"),
        DeclareLaunchArgument("treat_duration_s", default_value="2.0"),
        DeclareLaunchArgument("headland_margin_m", default_value="1.5",
                              description="pushed beyond the lane span to find "
                                          "a legal lateral crossing; 0.0 if "
                                          "the lanes file includes the headland"),
        DeclareLaunchArgument("fail_robot", default_value="-1",
                              description="index to freeze, or -1 for none"),
        DeclareLaunchArgument("fail_at_s", default_value="-1.0",
                              description="seconds after start to freeze it"),
        DeclareLaunchArgument("auto_shutdown", default_value="true",
                              description="end the run once every robot has "
                                          "swept its lanes and emptied its "
                                          "queue; false to keep it alive"),
        DeclareLaunchArgument("quiet_period_s", default_value="30.0",
                              description="how long every robot must stay idle "
                                          "before the mission is declared over"),
        DeclareLaunchArgument("max_mission_s", default_value="20000.0",
                              description="backstop for a wedged run, in "
                                          "simulated seconds; 0 disables"),
        OpaqueFunction(function=_setup),
    ])