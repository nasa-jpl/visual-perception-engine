cmake_minimum_required(VERSION 3.8)
project(vp_engine_ros)

# if(POLICY CMP0148)
#   cmake_policy(SET CMP0148 OLD)
# endif()
cmake_policy(SET CMP0167 NEW)
cmake_policy(SET CMP0146 OLD)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(ament_index_cpp REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(OpenCV REQUIRED)  
find_package(cv_bridge REQUIRED)
find_package(Python3 3.10 COMPONENTS Interpreter Development REQUIRED)
find_package(Boost COMPONENTS python3 numpy3 REQUIRED)

include_directories(
  include
  ${OpenCV_INCLUDE_DIRS}   
  )

find_package(rosidl_default_generators REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/ObjectDetectionOutput.msg"
  "srv/ChangeModelRate.srv"
  "srv/GetModelNames.srv"
  DEPENDENCIES std_msgs
)
  
add_library(engine_node
  ros_node/engine_wrapper.cpp
  ros_node/engine_node.cpp)
target_include_directories(engine_node PRIVATE ${Boost_INCLUDE_DIRS} ${Python3_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS} ${CMAKE_CURRENT_BINARY_DIR}/rosidl_generator_cpp)
ament_target_dependencies(engine_node rclcpp std_msgs sensor_msgs rosidl_default_generators cv_bridge ament_index_cpp)
target_compile_definitions(engine_node PRIVATE BOOST_BIND_GLOBAL_PLACEHOLDERS)

add_executable(engine_main ros_node/engine_main.cpp)
ament_target_dependencies(engine_main rclcpp ament_index_cpp)

add_executable(test_input ros_node/test_input.cpp)
ament_target_dependencies(test_input rclcpp std_msgs sensor_msgs cv_bridge)
  
target_compile_features(engine_main PUBLIC c_std_99 cxx_std_17) 
target_compile_definitions(engine_main PRIVATE BOOST_BIND_GLOBAL_PLACEHOLDERS)
target_include_directories(engine_main PRIVATE ${Boost_INCLUDE_DIRS} ${Python3_INCLUDE_DIRS})
target_include_directories(engine_main PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  ${CMAKE_CURRENT_BINARY_DIR}/rosidl_generator_cpp
)
rosidl_get_typesupport_target(cpp_typesupport_target ${PROJECT_NAME} "rosidl_typesupport_cpp")
target_link_libraries(engine_node ${cpp_typesupport_target})
target_link_libraries(engine_main ${cpp_typesupport_target})

target_link_libraries(engine_main engine_node)
target_link_libraries(engine_main ${Boost_LIBRARIES} ${Python3_LIBRARIES} ${OpenCV_LIBS})

install(TARGETS engine_node
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib)

install(TARGETS engine_main test_input
  RUNTIME DESTINATION lib/${PROJECT_NAME})

install(DIRECTORY launch DESTINATION share/${PROJECT_NAME}/)

ament_package()