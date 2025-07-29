# maths library
target_sources(maths
    INTERFACE
        FILE_SET interface_headers TYPE HEADERS
            arithmetic.hpp
            compare.hpp
            concepts.hpp
            random.hpp
)

# maths unit test
target_sources(maths.test
    PRIVATE
        FILE_SET sources
            arithmetic.test.cpp
            compare.test.cpp
            concepts.test.cpp
            random.test.cpp
)

# maths test tools
target_sources(maths.test_tools
    INTERFACE
        FILE_SET interface_headers TYPE HEADERS
            gtest_macros.hpp
)
