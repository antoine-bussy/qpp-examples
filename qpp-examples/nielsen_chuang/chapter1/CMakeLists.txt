file(RELATIVE_PATH current_folder ${CMAKE_SOURCE_DIR}/qpp-examples ${CMAKE_CURRENT_SOURCE_DIR})

include(SourcesLists.cmake)

# Chapter 1 unit test
add_executable(chapter1.test ${chapter1_test_sources} SourcesLists.cmake)
target_link_libraries(chapter1.test PRIVATE warnings GTest::gtest_main GTest::gmock libqpp OpenMP::OpenMP_CXX TBB::tbb)
add_test(NAME nielsen_chuang/chapter1 COMMAND chapter1.test)
set_target_properties(chapter1.test PROPERTIES FOLDER ${current_folder})
