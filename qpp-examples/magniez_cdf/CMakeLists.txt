file(RELATIVE_PATH current_folder ${CMAKE_SOURCE_DIR}/qpp-examples ${CMAKE_CURRENT_SOURCE_DIR})

include(SourcesLists.cmake)

# magniez_cdf unit test
add_executable(magniez_cdf.test ${magniez_cdf_test_sources} SourcesLists.cmake)
target_link_libraries(magniez_cdf.test PRIVATE warnings GTest::gtest_main GTest::gmock qpp::qpp)
add_test(NAME magniez_cdf COMMAND magniez_cdf.test)
set_target_properties(magniez_cdf.test PROPERTIES FOLDER ${current_folder})
