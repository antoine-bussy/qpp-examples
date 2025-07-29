include(SourcesLists.cmake)

# magniez_cdf unit test
add_executable(magniez_cdf.test ${magniez_cdf_test_sources} SourcesLists.cmake)
target_link_libraries(magniez_cdf.test PRIVATE warnings GTest::gtest_main GTest::gmock libqpp)
add_test(NAME magniez_cdf COMMAND magniez_cdf.test)
