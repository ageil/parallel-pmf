cmake_minimum_required(VERSION 3.16)
project(final_project)

set(CMAKE_CXX_STANDARD 17)
# set(CMAKE_CXX_FLAGS "-g -Wall -O2") debug mode
set(CMAKE_CXX_FLAGS "-g -O2")
set(THREADS_PREFER_PTHREAD_FLAG ON)

find_package(Boost 1.71 COMPONENTS filesystem program_options REQUIRED)
find_package(Threads REQUIRED)

add_subdirectory(models)

set(SOURCES main.cpp) 

# tests cases
enable_testing()

add_executable(main.tsk ${SOURCES})

add_executable(unit_test.tsk test/utils_test.cpp models/utils.h)
               
target_link_libraries(main.tsk ${Boost_LIBRARIES} models Threads::Threads)

target_link_libraries(unit_test.tsk models )
