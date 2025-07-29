include(SourcesLists.cmake)

# Chapter 1 unit test
add_executable(chapter1.test ${chapter1_test_sources} SourcesLists.cmake)
target_link_libraries(chapter1.test PRIVATE warnings GTest::gtest_main GTest::gmock libqpp OpenMP::OpenMP_CXX TBB::tbb)
add_test(NAME nielsen_chuang/chapter1 COMMAND chapter1.test)
