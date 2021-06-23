file(RELATIVE_PATH current_folder ${CMAKE_SOURCE_DIR}/qpp-examples ${CMAKE_CURRENT_SOURCE_DIR})

include(SourcesLists.cmake)

# Chapter 2 unit test
add_executable(chapter2.test ${chapter2_test_sources} SourcesLists.cmake)
target_link_libraries(chapter2.test PRIVATE warnings GTest::gtest_main GTest::gmock qpp::qpp OpenMP::OpenMP_CXX TBB::tbb)
add_test(NAME nielsen_chuang/chapter2 COMMAND chapter2.test)
set_target_properties(chapter2.test PROPERTIES FOLDER ${current_folder})
