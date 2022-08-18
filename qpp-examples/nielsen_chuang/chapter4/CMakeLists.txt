file(RELATIVE_PATH current_folder ${CMAKE_SOURCE_DIR}/qpp-examples ${CMAKE_CURRENT_SOURCE_DIR})

include(SourcesLists.cmake)

# Chapter 4 unit test
add_executable(chapter4.test ${chapter4_test_sources} SourcesLists.cmake)
target_link_libraries(chapter4.test PRIVATE warnings GTest::gtest_main GTest::gmock qpp::qpp OpenMP::OpenMP_CXX TBB::tbb)
add_test(NAME nielsen_chuang/chapter4 COMMAND chapter4.test)
set_target_properties(chapter4.test PROPERTIES FOLDER ${current_folder})
