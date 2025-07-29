add_library(warnings INTERFACE)
if(CMAKE_COMPILER_IS_GNUCXX)
    target_compile_options(warnings INTERFACE -Werror -Wall -Wextra -pedantic)
endif()
