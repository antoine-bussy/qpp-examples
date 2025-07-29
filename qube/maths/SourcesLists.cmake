# maths library
target_sources(maths
    INTERFACE
        FILE_SET interface_headers
        TYPE HEADERS
        FILES
            maths/arithmetic.hpp
            maths/compare.hpp
            maths/concepts.hpp
            maths/random.hpp
)

# maths unit test
target_sources(maths.test
    PRIVATE
        maths/arithmetic.test.cpp
        maths/compare.test.cpp
        maths/concepts.test.cpp
        maths/random.test.cpp
)

# maths test tools
target_sources(maths.test_tools
    INTERFACE
        FILE_SET interface_headers
        TYPE HEADERS
        FILES
            maths/gtest_macros.hpp
)
