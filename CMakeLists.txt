cmake_minimum_required(VERSION 3.10)

project(simple_mpc)

find_package(Eigen3 REQUIRED)
find_package(Ceres REQUIRED)
find_package(glog REQUIRED)

include(FetchContent)

# Install sciplot
FetchContent_Declare(sciplot_content
  GIT_REPOSITORY https://github.com/sciplot/sciplot.git
  GIT_TAG master)

FetchContent_GetProperties(sciplot_content)
if(NOT sciplot_content_POPULATED)
  FetchContent_Populate(sciplot_content)
endif()

include_directories(${sciplot_content_SOURCE_DIR})

add_executable(${PROJECT_NAME}
src/simple_mpc.cpp
)

target_link_libraries(${PROJECT_NAME}
  PRIVATE
    Eigen3::Eigen
    Ceres::ceres
    glog::glog
)
