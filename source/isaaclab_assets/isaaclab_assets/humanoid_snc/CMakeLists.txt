cmake_minimum_required(VERSION 3.5)
project(humanoid_snc)

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(yaml-cpp REQUIRED)

# Include directories
include_directories(include)

# Add executable
add_executable(humanoid_state_publisher src/humanoid_state_publisher.cpp)

# Link libraries
ament_target_dependencies(humanoid_state_publisher
  rclcpp
  sensor_msgs
  std_msgs
  geometry_msgs
)

# Link yaml-cpp library
target_link_libraries(humanoid_state_publisher yaml-cpp)

# Install targets
install(TARGETS
  humanoid_state_publisher
  DESTINATION lib/${PROJECT_NAME}
)

# Install launch files
install(DIRECTORY launch/
  DESTINATION share/${PROJECT_NAME}/launch
)

# Install config files
install(DIRECTORY config/
  DESTINATION share/${PROJECT_NAME}/config
)

# Install meshes
install(DIRECTORY meshes/
  DESTINATION share/${PROJECT_NAME}/meshes
)

# Install worlds
install(DIRECTORY worlds/
  DESTINATION share/${PROJECT_NAME}/worlds
)

ament_package()