cmake_minimum_required(VERSION 3.16)
project(final_project)

set(CMAKE_CXX_STANDARD 17)
# set(CMAKE_CXX_FLAGS "-g -Wall -O2") debug mode
set(CMAKE_CXX_FLAGS "-g -O2")
set(THREADS_PREFER_PTHREAD_FLAG ON)
include_directories(include /usr/local/include/eigen3/)
include_directories(include /usr/local/Cellar/eigen/3.3.9)
find_package(Boost 1.71 COMPONENTS filesystem program_options REQUIRED)
find_package(Threads REQUIRED)

add_executable(main main.cpp)
target_link_libraries(main ${Boost_LIBRARIES} Threads::Threads)