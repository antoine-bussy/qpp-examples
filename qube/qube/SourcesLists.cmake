# qube library
target_sources(qube
    INTERFACE
        FILE_SET interface_headers
        TYPE HEADERS
        FILES
            qube/approximations.hpp
            qube/debug.hpp
            qube/decompositions.hpp
            qube/gates.hpp
            qube/introspection.hpp

            qube/maths/arithmetic.hpp
            qube/maths/compare.hpp
            qube/maths/concepts.hpp
            qube/maths/gtest_macros.hpp
            qube/maths/norm.hpp
            qube/maths/random.hpp
)

# qube unit test
target_sources(qube.test
    PRIVATE
        qube/gates.test.cpp

        qube/maths/arithmetic.test.cpp
        qube/maths/compare.test.cpp
        qube/maths/concepts.test.cpp
        qube/maths/random.test.cpp
)
