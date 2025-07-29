include(SourcesLists.cmake)

# Chapter 4 unit test
add_executable(chapter4.test ${chapter4_test_sources} SourcesLists.cmake)
target_link_libraries(chapter4.test PRIVATE qpp.examples)
add_test(NAME nielsen_chuang/chapter4 COMMAND chapter4.test)
