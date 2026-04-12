cmake_minimum_required(VERSION 3.18)
project(mpdat LANGUAGES C CXX Fortran)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(Python 3.8 REQUIRED COMPONENTS Interpreter Development.Module)

include(FetchContent)

FetchContent_Declare(
    nanobind
    GIT_REPOSITORY https://github.com/wjakob/nanobind.git
    GIT_TAG v1.9.2
)

FetchContent_MakeAvailable(nanobind)

add_library(mpdata_fortran STATIC
    src/mpdata.f
    src/mpdata_wrapper.F90
)

nanobind_add_module(ncar_mpdata
    src/mpdata.cpp
)

target_link_libraries(ncar_mpdata PRIVATE mpdata_fortran)
target_link_libraries(ncar_mpdata PRIVATE gfortran)

install(TARGETS ncar_mpdata LIBRARY DESTINATION .)