message(STATUS "Using custom provider CMake configuration")

include(FetchContent)

set(DEPS_INSTALL_DIR ${CMAKE_BINARY_DIR}/_deps/_install)

function(base_try_compile dependency_name)
    string(TOLOWER "${dependency_name}" dependency_name_lower)
    try_compile(compile_success PROJECT ${dependency_name}
        SOURCE_DIR ${${dependency_name_lower}_SOURCE_DIR}
        BINARY_DIR ${${dependency_name_lower}_BINARY_DIR}
        TARGET install
        CMAKE_FLAGS ${ARGN}
        OUTPUT_VARIABLE compile_output
    )
    if (NOT compile_success)
        message(FATAL_ERROR "Compilation failed for ${dependency_name}. Output: ${compile_output}")
    endif()
endfunction()


function(default_try_compile dependency_name)
    base_try_compile(${dependency_name}
        -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
        -DCMAKE_INSTALL_PREFIX=${DEPS_INSTALL_DIR}
    )
endfunction()

function(qpp_examples_provide_tbb)
    message(STATUS "Custom Provider: Providing TBB dependency")
    FetchContent_Declare(
        TBB
        GIT_REPOSITORY https://github.com/intel/tbb.git
        GIT_TAG v2022.2.0
    )
    FetchContent_Populate(TBB)
    base_try_compile(TBB
        -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
        -DCMAKE_INSTALL_PREFIX=${DEPS_INSTALL_DIR}
        -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON
        -DTBB_TEST=OFF
        -DTBB_STRICT=OFF
    )
    set(TBB_DIR ${DEPS_INSTALL_DIR}/lib/cmake/TBB CACHE PATH "Path to TBB CMake configuration" FORCE)
endfunction()

function(qpp_examples_provide_gtest)
    message(STATUS "Custom Provider: Providing GTest dependency")
    FetchContent_Declare(
        GTest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG v1.17.0
    )
    FetchContent_Populate(GTest)
    default_try_compile(GTest)
    set(GTest_DIR ${DEPS_INSTALL_DIR}/lib/cmake/GTest CACHE PATH "Path to GTest CMake configuration" FORCE)
endfunction(qpp_examples_provide_gtest)

function(qpp_examples_provide_eigen)
    message(STATUS "Custom Provider: Providing Eigen3 dependency")
    FetchContent_Declare(
        Eigen3
        GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
        GIT_TAG 3.4
    )
    FetchContent_Populate(Eigen3)
    default_try_compile(Eigen3)
    set(Eigen3_DIR ${DEPS_INSTALL_DIR}/share/eigen3/cmake CACHE PATH "Path to Eigen3 CMake configuration" FORCE)
endfunction(qpp_examples_provide_eigen)

function(qpp_examples_provide_qpp)
    message(STATUS "Custom Provider: Providing qpp dependency")
    FetchContent_Declare(
        qpp
        # GIT_REPOSITORY https://github.com/softwareQinc/qpp.git
        GIT_REPOSITORY https://github.com/antoine-bussy/qpp.git
        GIT_TAG main
    )
    FetchContent_Populate(qpp)
    default_try_compile(qpp)
    set(qpp_DIR ${DEPS_INSTALL_DIR}/lib/cmake/qpp CACHE PATH "Path to qpp CMake configuration" FORCE)
endfunction()

function(qpp_examples_provide_dependency method package_name)
    if (NOT method STREQUAL "FIND_PACKAGE")
        message(FATAL_ERROR "Unsupported method: ${method}. Only FIND_PACKAGE is supported.")
    endif()

    if (DEFINED ${package_name}_FOUND AND ${package_name}_FOUND)
        message(STATUS "${package_name} already found, skipping provider.")
        return()
    endif()

    if (package_name STREQUAL "TBB")
        qpp_examples_provide_tbb()
    endif()
    if (package_name STREQUAL "Eigen3")
        qpp_examples_provide_eigen()
    endif()
    if (package_name STREQUAL "GTest")
        qpp_examples_provide_gtest()
    endif()
    if (package_name STREQUAL "qpp")
        qpp_examples_provide_qpp()
    endif()

endfunction()

cmake_language(
  SET_DEPENDENCY_PROVIDER qpp_examples_provide_dependency
  SUPPORTED_METHODS FIND_PACKAGE
)
