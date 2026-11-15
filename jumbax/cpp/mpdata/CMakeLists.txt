cmake_minimum_required(VERSION 3.18)
project(libmpdataxx LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Python 3.11 REQUIRED COMPONENTS Interpreter Development.Module)
find_package(nanobind CONFIG REQUIRED)
find_package(libmpdata++ CONFIG REQUIRED)
find_package(Threads REQUIRED)

nanobind_add_module(libmpdataxx NOMINSIZE src/mpdata_bindings.cpp)
target_link_libraries(libmpdataxx PRIVATE ${libmpdataxx_LIBRARIES} Threads::Threads)

if(TARGET OpenMP::OpenMP_CXX)
    target_link_libraries(libmpdataxx PRIVATE OpenMP::OpenMP_CXX)
endif()

target_include_directories(libmpdataxx PRIVATE ${libmpdataxx_INCLUDE_DIRS})
target_link_directories(libmpdataxx PRIVATE ${BLITZ_LIBRARY_DIRS})
target_compile_definitions(libmpdataxx PRIVATE RELEASE)

separate_arguments(mpdata_compile_options UNIX_COMMAND "${libmpdataxx_CXX_FLAGS_RELEASE}")
list(FILTER mpdata_compile_options EXCLUDE REGEX "^-(march|mtune)=native$")
target_compile_options(libmpdataxx PRIVATE ${mpdata_compile_options})

install(TARGETS libmpdataxx LIBRARY DESTINATION .)