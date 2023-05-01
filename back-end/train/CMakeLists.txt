add_executable(train train.cpp)
target_compile_options(train PUBLIC -O3 --std=c++17)

target_link_libraries(train mlp csvParser)