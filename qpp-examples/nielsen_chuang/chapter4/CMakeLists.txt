include(SourcesLists.cmake)

# Chapter 4 unit test
add_executable(chapter4.test ${chapter4_test_sources} SourcesLists.cmake)
target_link_libraries(chapter4.test PRIVATE warnings qube GTest::gtest_main GTest::gmock libqpp OpenMP::OpenMP_CXX TBB::tbb)
add_test(NAME nielsen_chuang/chapter4 COMMAND chapter4.test)
