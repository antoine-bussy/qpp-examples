include(SourcesLists.cmake)

# qube library
add_library(qube INTERFACE)
target_link_libraries(qube INTERFACE Eigen3::Eigen maths warnings libqpp)
set_target_properties(qube PROPERTIES PUBLIC_HEADER "${qube_headers}")

if(${QUBE_DEBUG_STREAM})
    target_compile_definitions(qube INTERFACE QUBE_DEBUG_STREAM)
endif()
