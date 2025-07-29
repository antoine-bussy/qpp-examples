include(SourcesLists.cmake)

# Chapter 1 unit test
add_executable(chapter1.test ${chapter1_test_sources} SourcesLists.cmake)
target_link_libraries(chapter1.test PRIVATE qpp.examples)
add_test(NAME nielsen_chuang/chapter1 COMMAND chapter1.test)
