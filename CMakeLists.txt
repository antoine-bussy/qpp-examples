cmake_minimum_required(VERSION 3.16)

if(NOT QPP_EXAMPLES_VERSION)
    set(QPP_EXAMPLES_VERSION 0.0.0)
endif()

# Add C for custom provider
project(qpp-examples LANGUAGES C CXX VERSION ${QPP_EXAMPLES_VERSION})

message(STATUS "${CMAKE_PROJECT_NAME} Version: ${CMAKE_PROJECT_VERSION}")

math(EXPR bits "8*${CMAKE_SIZEOF_VOID_P}")
message(STATUS "Target is ${bits} bits")

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build." FORCE)
endif()

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set_property(GLOBAL PROPERTY USE_FOLDERS ON)
set(CMAKE_DEBUG_POSTFIX d)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_FIND_PACKAGE_PREFER_CONFIG TRUE)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(DEBUG_STREAM "Enable debug stream" OFF)

find_package(TBB 2022.2 REQUIRED)
find_package(Eigen3 3.4 REQUIRED)
find_package(qpp REQUIRED)
add_library(libqpp_fix INTERFACE)
# Fix qpp link to eigen3
target_link_libraries(libqpp_fix INTERFACE libqpp Eigen3::Eigen)
target_compile_options(libqpp_fix INTERFACE $<$<CONFIG:Debug>:-O0>)
target_link_options(libqpp_fix INTERFACE $<$<CONFIG:Debug>:-O0>)
add_library(qpp::qpp ALIAS libqpp_fix)


find_package(GTest REQUIRED)
find_package(OpenMP REQUIRED)

enable_testing()

add_subdirectory(qpp-examples)
