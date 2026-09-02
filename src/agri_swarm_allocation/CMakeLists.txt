cmake_minimum_required(VERSION 3.8)
project(agri_swarm_allocation)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()
add_compile_options(-Wall -Wextra -Wpedantic)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(agri_swarm_msgs REQUIRED)

include_directories(include)

add_executable(allocator_node src/allocator_node.cpp)
ament_target_dependencies(allocator_node rclcpp nav_msgs agri_swarm_msgs)

install(TARGETS allocator_node DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY include/ DESTINATION include)

# utility.hpp has no ROS dependency on purpose, so this test also builds
# standalone in CI without a ROS 2 installation:
#   g++ -std=c++17 -Iinclude test/test_utility.cpp -o /tmp/test_utility
if(BUILD_TESTING)
  add_executable(test_utility test/test_utility.cpp)
  target_include_directories(test_utility PRIVATE include)
  add_test(NAME test_utility COMMAND test_utility)
endif()

ament_package()
