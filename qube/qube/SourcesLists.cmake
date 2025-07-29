# qube library
target_sources(qube
    INTERFACE
        FILE_SET interface_headers
        TYPE HEADERS
        FILES
            qube/debug.hpp
            qube/decompositions.hpp
            qube/introspection.hpp
)
