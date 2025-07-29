# warnings flags
add_library(warnings INTERFACE)
if(CMAKE_COMPILER_IS_GNUCXX)
    target_compile_options(warnings INTERFACE -Werror -Wall -Wextra -pedantic)
endif()

# qube library
add_library(qube INTERFACE)
target_link_libraries(qube INTERFACE warnings Eigen3::Eigen libqpp)

if(${QUBE_DEBUG_STREAM})
    target_compile_definitions(qube INTERFACE QUBE_DEBUG_STREAM)
endif()

# qube unit test
add_executable(qube.test)
target_link_libraries(qube.test PRIVATE qube GTest::gtest_main GTest::gmock)
add_test(NAME qube COMMAND qube.test)

include(qube/SourcesLists.cmake)
