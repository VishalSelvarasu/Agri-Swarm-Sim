import os
 
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
 
import xacro
 
from agri_swarm_core.pure_pursuit import assign_lanes, load_lanes
 
 
START_X = -1.2              # headland, behind the crop rows
 
 
def _robot_group(idx, n, seed, world_dir, bid_mode, lanes, energy_capacity_j,
                 use_sim_time, use_allocator, use_lane_follower):
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
 
    lane_follower = Node(
        package="agri_swarm_core", executable="lane_follower_node",
        namespace=name, output="log",
        parameters=[{
            "robot_index": idx,
            "n_robots": n,
            "lanes_csv": os.path.join(world_dir, f"lanes_{seed}.csv"),
            # Must match the spawn pose exactly: gz odometry is spawn-relative.
            "origin_x": float(START_X),
            "origin_y": float(y),
            **common,
        }],
    )
 
    allocator = Node(
        package="agri_swarm_allocation", executable="allocator_node",
        namespace=name, output="screen",   # keep split-brain warnings visible
        parameters=[{
            "robot_id": name,
            "bid_mode": bid_mode,
            # Must match the spawn pose exactly: gz odometry is spawn-relative.
            "origin_x": float(START_X),
            "origin_y": float(y),
            "energy_capacity_j": energy_capacity_j,
            **common,
        }],
    )
 
    nodes = [rsp, spawn, bridge, detector]
    if use_lane_follower:
        nodes.append(lane_follower)
    # Week 1 is explicitly no auction. With no energy monitor publishing
    # RobotState, every award looks like a silent winner and the log fills with
    # abandonment warnings that drown out everything else.
    if use_allocator:
        nodes.append(allocator)
    return nodes
 
 
def _setup(context, *args, **kwargs):
    n = int(LaunchConfiguration("n_robots").perform(context))
    seed = int(LaunchConfiguration("seed").perform(context))
    world_dir = LaunchConfiguration("world_dir").perform(context)
    bid_mode = LaunchConfiguration("bid_mode").perform(context)
    energy_capacity_j = float(LaunchConfiguration("energy_capacity_j").perform(context))
    headless = LaunchConfiguration("headless").perform(context).lower() == "true"
    use_allocator = LaunchConfiguration("use_allocator").perform(context).lower() == "true"
    use_lane_follower = (
        LaunchConfiguration("use_lane_follower").perform(context).lower() == "true")
 
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
        actions += _robot_group(i, n, seed, world_dir, bid_mode, lanes,
                                energy_capacity_j, True, use_allocator,
                                use_lane_follower)
    return actions
 
 
def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("n_robots", default_value="4"),
        DeclareLaunchArgument("seed", default_value="0"),
        DeclareLaunchArgument("world_dir", default_value="/tmp/worlds"),
        DeclareLaunchArgument("bid_mode", default_value="confidence_energy",
                              description="distance | confidence_energy"),
        DeclareLaunchArgument("energy_capacity_j", default_value="40000.0",
                              description="Robot battery capacity in joules"),
        DeclareLaunchArgument("headless", default_value="true",
                              description="true for batch runs; false to watch"),
        DeclareLaunchArgument("use_allocator", default_value="true",
                              description="false for the week-1 milestone"),
        DeclareLaunchArgument("use_lane_follower", default_value="true"),
        OpaqueFunction(function=_setup),
    ])