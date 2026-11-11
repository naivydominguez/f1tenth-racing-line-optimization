cmake_minimum_required(VERSION 3.5)
project(raceline_follower)

find_package(ament_cmake REQUIRED)

install(PROGRAMS
  scripts/raceline_follower_node.py
  scripts/circle_test_logger.py
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY launch config waypoints
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
