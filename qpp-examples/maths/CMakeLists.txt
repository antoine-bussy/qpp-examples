file(RELATIVE_PATH current_folder ${CMAKE_SOURCE_DIR}/qpp-examples ${CMAKE_CURRENT_SOURCE_DIR})

include(SourcesLists.cmake)

# maths test tools
add_library(maths.test_tools INTERFACE)
target_link_libraries(maths.test_tools INTERFACE Eigen3::Eigen GTest::gmock warnings)
set_target_properties(maths.test_tools PROPERTIES PUBLIC_HEADER "${maths_test_tools_headers}")

# maths test unit test
add_executable(maths.test_tools.test ${maths_test_tools_test_sources} SourcesLists.cmake)
target_link_libraries(maths.test_tools.test PRIVATE maths.test_tools warnings GTest::gtest_main GTest::gmock)
add_test(NAME maths.test_tools COMMAND maths.test_tools.test)
set_target_properties(maths.test_tools.test PROPERTIES FOLDER ${current_folder})
