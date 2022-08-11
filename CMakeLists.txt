cmake_minimum_required(VERSION 3.16)

if(NOT QPP_EXAMPLES_VERSION)
    set(QPP_EXAMPLES_VERSION 0.0.0)
endif()

project(qpp-examples LANGUAGES CXX VERSION ${QPP_EXAMPLES_VERSION})

message(STATUS "${CMAKE_PROJECT_NAME} Version: ${CMAKE_PROJECT_VERSION}")

math(EXPR bits "8*${CMAKE_SIZEOF_VOID_P}")
message(STATUS "Target is ${bits} bits")

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set_property(GLOBAL PROPERTY USE_FOLDERS ON)
set(CMAKE_DEBUG_POSTFIX d)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_FIND_PACKAGE_PREFER_CONFIG TRUE)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(qpp REQUIRED)
find_package(GTest REQUIRED)
find_package(TBB 2021.3 REQUIRED)
find_package(OpenMP REQUIRED)

enable_testing()

add_subdirectory(qpp-examples)
