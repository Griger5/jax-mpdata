cmake_minimum_required(VERSION 3.18)
project(libmpdataxx CXX)

set(CMAKE_BUILD_TYPE Release)

find_package(Python 3.8 REQUIRED COMPONENTS Interpreter Development.Module)

find_package(libmpdata++ REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${libmpdataxx_CXX_FLAGS_RELEASE}")

include(FetchContent)

FetchContent_Declare(
    nanobind
    GIT_REPOSITORY https://github.com/wjakob/nanobind.git
    GIT_TAG v1.9.2
)

FetchContent_MakeAvailable(nanobind)

nanobind_add_module(libmpdataxx NOMINSIZE
    src/mpdata.cpp
)

target_link_libraries(libmpdataxx PUBLIC ${libmpdataxx_LIBRARIES})
target_include_directories(libmpdataxx PUBLIC ${libmpdataxx_INCLUDE_DIRS})

target_compile_definitions(libmpdataxx PRIVATE RELEASE)
target_compile_options(libmpdataxx PRIVATE -O3 -ffast-math)

install(TARGETS libmpdataxx LIBRARY DESTINATION .)