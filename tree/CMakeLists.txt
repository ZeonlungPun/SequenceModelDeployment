cmake_minimum_required(VERSION 3.5)

project(incrementalSFM)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_C_STANDARD 99)

# OpenCV
find_package(OpenCV 4 REQUIRED)

# include directories
include_directories(
    ${OpenCV_INCLUDE_DIRS}
    ${CMAKE_SOURCE_DIR}/include
)

# executable
add_executable(
    treeHAR
    src/treeHAR.cpp
    src/HARmodel.c
)

# link libraries
target_link_libraries(
    treeHAR
    ${OpenCV_LIBS}
)

# optimization
if(MSVC)
    target_compile_options(treeHAR PRIVATE /O2 /Oi)
else()
    target_compile_options(treeHAR PRIVATE -O3 -ffast-math)
endif()