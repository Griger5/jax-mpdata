cmake_minimum_required(VERSION 3.18)
project(mpdata LANGUAGES C CXX Fortran)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(Python 3.11 REQUIRED COMPONENTS Interpreter Development.Module)
find_package(nanobind CONFIG REQUIRED)

add_library(mpdata_fortran STATIC src/mpdata.f src/mpdata_wrapper.F90)
nanobind_add_module(ncar_mpdata NOMINSIZE src/mpdata_bindings.cpp)

target_link_libraries(ncar_mpdata PRIVATE mpdata_fortran)

install(TARGETS ncar_mpdata LIBRARY DESTINATION .)