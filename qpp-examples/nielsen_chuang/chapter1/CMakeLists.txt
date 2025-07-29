# Chapter 1 unit test
add_executable(chapter1.test)
target_link_libraries(chapter1.test PRIVATE qpp.examples)
add_test(NAME nielsen_chuang/chapter1 COMMAND chapter1.test)

target_sources(chapter1.test
    PRIVATE
        dummy.test.cpp
        chapter1.2.test.cpp
        chapter1.3.test.cpp
        chapter1.4.test.cpp
        chapter1.5.test.cpp
)
