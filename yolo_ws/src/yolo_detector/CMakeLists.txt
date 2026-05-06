cmake_minimum_required(VERSION 3.8)
project(yolo_detector)

find_package(ament_cmake REQUIRED)
find_package(rclpy REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(rosidl_default_generators REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/ToolDetection.msg"
  "msg/ToolDetectionArray.msg"
)

ament_export_dependencies(rosidl_default_runtime)

ament_python_install_package(${PROJECT_NAME})

install(PROGRAMS
  yolo_detector/detector_node.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()

