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

include(SourcesLists.cmake)
