include(SourcesLists.cmake)

add_library(warnings INTERFACE)
target_include_directories(warnings INTERFACE
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    )
set_target_properties(warnings PROPERTIES PUBLIC_HEADER "${warnings_headers}")

if(CMAKE_COMPILER_IS_GNUCXX)
    target_compile_options(warnings INTERFACE -Werror -Wall -Wextra -pedantic)
endif()
