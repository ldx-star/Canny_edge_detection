cmake_minimum_required(VERSION 3.25)
project(Canny_edge_detection)

set(CMAKE_CXX_STANDARD 14)

add_library(lib src/Canny.cpp src/util.cpp)
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(Canny_edge_detection main.cpp)
target_link_libraries(Canny_edge_detection lib ${OpenCV_LIBS})
