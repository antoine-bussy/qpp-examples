# Chapter 2 unit test$
add_executable(chapter2.test)
target_link_libraries(chapter2.test PRIVATE qpp.examples)
add_test(NAME nielsen_chuang/chapter2 COMMAND chapter2.test)

target_sources(chapter2.test
    PRIVATE
        chapter2.2.test.cpp
        chapter2.3.test.cpp
        chapter2.4.test.cpp
        chapter2.5.test.cpp
        chapter2.6.test.cpp
)
