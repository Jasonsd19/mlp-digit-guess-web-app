add_library(mlp multilayerPerceptron.cpp)
target_compile_options(mlp PUBLIC -O3 --std=c++17)
target_link_libraries(mlp PUBLIC matrix)

target_include_directories (mlp PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})