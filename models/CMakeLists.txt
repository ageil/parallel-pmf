add_library(models PMF.cpp datamanager.cpp utils.cpp)

find_package (Eigen3 3.3 REQUIRED NO_MODULE)

include(FetchContent)
FetchContent_Declare(GSL
    GIT_REPOSITORY "https://github.com/microsoft/GSL"
    GIT_TAG "v3.1.0"

)

FetchContent_MakeAvailable(GSL)

target_link_libraries(models PUBLIC
    ${Boost_LIBRARIES}
    Eigen3::Eigen
    Microsoft.GSL::GSL
    )
