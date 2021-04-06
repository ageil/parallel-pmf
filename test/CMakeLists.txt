cmake_minimum_required(VERSION 3.16)
project(final_project)

add_executable(unit_test.tsk utils_test.cpp)

target_link_libraries(unit_test.tsk models)

add_test(NAME unit_test.tsk
WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
COMMAND "$<TARGET_FILE:unit_test.tsk>")