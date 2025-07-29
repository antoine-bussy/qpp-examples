# Convenience library for all examples
add_library(qpp.examples INTERFACE)
target_link_libraries(qpp.examples INTERFACE qube GTest::gtest_main GTest::gmock OpenMP::OpenMP_CXX TBB::tbb)

add_subdirectory(magniez_cdf)
add_subdirectory(nielsen_chuang)
