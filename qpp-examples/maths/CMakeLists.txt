file(RELATIVE_PATH current_folder ${CMAKE_SOURCE_DIR}/qpp-examples ${CMAKE_CURRENT_SOURCE_DIR})

include(SourcesLists.cmake)

# maths test tools
add_library(maths.test_tools INTERFACE)
target_link_libraries(maths.test_tools INTERFACE Eigen3::Eigen GTest::gmock warnings)
set_target_properties(maths.test_tools PROPERTIES PUBLIC_HEADER "${maths_testtools_headers}")
