add_library (matrix matrix.h matrix.cpp)
target_compile_options(matrix PUBLIC -O3 --std=c++17)
target_link_libraries(matrix PUBLIC pthread)

target_include_directories (matrix PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})