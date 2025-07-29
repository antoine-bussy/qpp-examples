include(SourcesLists.cmake)

# qube library
add_library(qube INTERFACE)
target_link_libraries(qube INTERFACE Eigen3::Eigen maths warnings libqpp)
set_target_properties(qube PROPERTIES PUBLIC_HEADER "${qube_headers}")

if(${DEBUG_STREAM})
    target_compile_definitions(qube INTERFACE QPP_E_DEBUG_STREAM=1)
endif()
