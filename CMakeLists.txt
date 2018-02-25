 
cmake_minimum_required(VERSION 2.8)

project(NodeDetector)

find_package(OpenCV REQUIRED)

include_directories(
    ${OpenCV_INCLUDE_DIRS}
    source
)

add_executable(${PROJECT_NAME} 
    source/main.cpp 
    source/nod_detector.cpp
)

target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})