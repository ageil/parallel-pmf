cmake_minimum_required(VERSION 3.16)
project(final_project)

set(CMAKE_CXX_STANDARD 17)
# set(CMAKE_CXX_FLAGS "-g -Wall -O2") debug mode
set(CMAKE_CXX_FLAGS "-g -O2")
set(THREADS_PREFER_PTHREAD_FLAG ON)

find_package(Boost 1.71 COMPONENTS program_options REQUIRED)
find_package(Threads REQUIRED)

add_subdirectory(models)

add_executable(main.tsk main.cpp)
               
target_link_libraries(main.tsk ${Boost_LIBRARIES} models Threads::Threads)

# tests cases
enable_testing()

add_subdirectory(test)
