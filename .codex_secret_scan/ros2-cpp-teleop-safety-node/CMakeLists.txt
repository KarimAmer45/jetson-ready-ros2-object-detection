cmake_minimum_required(VERSION 3.8)
project(ros2_cpp_teleop_safety_node)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)

add_library(teleop_safety_node_lib
  src/teleop_safety_node.cpp
)
target_include_directories(teleop_safety_node_lib
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_compile_features(teleop_safety_node_lib PUBLIC cxx_std_17)
ament_target_dependencies(teleop_safety_node_lib
  geometry_msgs
  rclcpp
  sensor_msgs
  std_msgs
)

add_executable(teleop_safety_node
  src/main.cpp
)
target_link_libraries(teleop_safety_node
  teleop_safety_node_lib
)

install(
  TARGETS teleop_safety_node_lib teleop_safety_node
  EXPORT export_teleop_safety_node_lib
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION lib/${PROJECT_NAME}
)

install(
  DIRECTORY include/
  DESTINATION include
)

install(
  DIRECTORY config launch sim
  DESTINATION share/${PROJECT_NAME}
)

ament_export_include_directories(include)
ament_export_libraries(teleop_safety_node_lib)
ament_export_targets(export_teleop_safety_node_lib HAS_LIBRARY_TARGET)
ament_export_dependencies(geometry_msgs rclcpp sensor_msgs std_msgs)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
