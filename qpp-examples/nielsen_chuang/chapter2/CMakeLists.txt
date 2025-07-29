include(SourcesLists.cmake)

# Chapter 2 unit test
add_executable(chapter2.test ${chapter2_test_sources} SourcesLists.cmake)
target_link_libraries(chapter2.test PRIVATE qpp.examples)
add_test(NAME nielsen_chuang/chapter2 COMMAND chapter2.test)
