file(RELATIVE_PATH current_folder ${CMAKE_SOURCE_DIR}/qpp-examples ${CMAKE_CURRENT_SOURCE_DIR})

include(SourcesLists.cmake)

# Examples unit test
add_executable(examples.test ${examples_test_sources} SourcesLists.cmake)
target_link_libraries(examples.test PRIVATE warnings GTest::gtest_main GTest::gmock qpp::qpp)
add_test(NAME examples COMMAND examples.test)
set_target_properties(examples.test PROPERTIES FOLDER ${current_folder})
