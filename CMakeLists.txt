cmake_minimum_required(VERSION 3.10)

project(simple_mpc)

find_package(Eigen3 REQUIRED)

find_package(Ceres REQUIRED)

find_package(glog REQUIRED)

add_executable(${PROJECT_NAME}
src/simple_mpc.cpp
)

target_link_libraries(${PROJECT_NAME}
  PRIVATE
    Eigen3::Eigen
    Ceres::ceres
    glog::glog
)
