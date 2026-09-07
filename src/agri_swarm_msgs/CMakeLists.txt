cmake_minimum_required(VERSION 3.8)
project(agri_swarm_msgs)

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(builtin_interfaces REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/WeedDetection.msg"
  "msg/TaskAnnouncement.msg"
  "msg/Bid.msg"
  "msg/TaskAward.msg"
  "msg/RobotState.msg"
  "msg/Treatment.msg"
  DEPENDENCIES std_msgs geometry_msgs builtin_interfaces
)

ament_package()
