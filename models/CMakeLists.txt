add_library(models PMF.cpp utils.cpp)

find_package (Eigen3 3.3 REQUIRED NO_MODULE)

target_link_libraries (models PUBLIC 
    Eigen3::Eigen)
