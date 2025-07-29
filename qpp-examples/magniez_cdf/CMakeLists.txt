# magniez_cdf unit test
add_executable(magniez_cdf.test)
target_link_libraries(magniez_cdf.test PRIVATE qpp.examples)
add_test(NAME magniez_cdf COMMAND magniez_cdf.test)

target_sources(magniez_cdf.test PRIVATE
    perdrix.test.cpp
)
