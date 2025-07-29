# warnings flags
add_library(warnings INTERFACE)
if(CMAKE_COMPILER_IS_GNUCXX)
    target_compile_options(warnings INTERFACE -Werror -Wall -Wextra -pedantic)
endif()

# maths library
add_library(maths INTERFACE)
target_link_libraries(maths INTERFACE Eigen3::Eigen warnings libqpp)

# maths unit test
add_executable(maths.test)
target_link_libraries(maths.test PRIVATE maths warnings GTest::gtest_main GTest::gmock)
add_test(NAME maths COMMAND maths.test)

# maths test tools
add_library(maths.test_tools INTERFACE)
target_link_libraries(maths.test_tools INTERFACE maths GTest::gmock warnings)

include(maths/SourcesLists.cmake)

get_target_property(MATHS_INTERFACE_INCLUDE_DIRS maths INTERFACE_INCLUDE_DIRECTORIES)
message(STATUS "maths INTERFACE_INCLUDE_DIRECTORIES: ${MATHS_INTERFACE_INCLUDE_DIRS}")

# qube library
add_library(qube INTERFACE)
target_link_libraries(qube INTERFACE Eigen3::Eigen maths warnings libqpp)

if(${QUBE_DEBUG_STREAM})
    target_compile_definitions(qube INTERFACE QUBE_DEBUG_STREAM)
endif()

include(qube/SourcesLists.cmake)
