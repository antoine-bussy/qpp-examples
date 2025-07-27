file(RELATIVE_PATH current_folder ${CMAKE_SOURCE_DIR}/qpp-examples ${CMAKE_CURRENT_SOURCE_DIR})

include(SourcesLists.cmake)

# maths library
add_library(maths INTERFACE)
target_link_libraries(maths INTERFACE Eigen3::Eigen warnings libqpp)
set_target_properties(maths PROPERTIES PUBLIC_HEADER "${maths_headers}")

# maths unit test
add_executable(maths.test ${maths_test_sources} SourcesLists.cmake)
target_link_libraries(maths.test PRIVATE maths warnings GTest::gtest_main GTest::gmock)
add_test(NAME maths COMMAND maths.test)
set_target_properties(maths.test PROPERTIES FOLDER ${current_folder})

# maths test tools
add_library(maths.test_tools INTERFACE)
target_link_libraries(maths.test_tools INTERFACE maths GTest::gmock warnings)
set_target_properties(maths.test_tools PROPERTIES PUBLIC_HEADER "${maths_test_tools_headers}")
