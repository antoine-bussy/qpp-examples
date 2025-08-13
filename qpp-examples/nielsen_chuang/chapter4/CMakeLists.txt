# Chapter 4 unit test
add_executable(chapter4.test)
target_link_libraries(chapter4.test PRIVATE qpp.examples)
add_test(NAME nielsen_chuang/chapter4 COMMAND chapter4.test)

target_sources(chapter4.test
    PRIVATE
        chapter4.2.test.cpp
        chapter4.3.test.cpp
        chapter4.4.test.cpp

        chapter4.5.1.test.cpp
        chapter4.5.2.test.cpp
        chapter4.5.3.test.cpp

        chapter4.7.1.test.cpp
        chapter4.7.2.test.cpp
)
