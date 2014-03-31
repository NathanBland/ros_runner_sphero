cmake_minimum_required(VERSION 2.8.3)
project(runner_sphero)

find_package(catkin REQUIRED COMPONENTS
  joy
  cv_bridge
  image_transport
  roscpp
  rospy
)
find_package(OpenCV)


catkin_package()
include_directories(
  ${catkin_INCLUDE_DIRS}
)

 add_executable(runner_sphero_node src/sphero.cpp)

 target_link_libraries(runner_sphero_node
   ${catkin_LIBRARIES}
 )

