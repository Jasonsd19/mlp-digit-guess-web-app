cmake_minimum_required(VERSION 3.1.4)
project(mlp VERSION 1.0.0 LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 17)

add_subdirectory(external)
add_subdirectory(internal)
add_subdirectory(model)
add_subdirectory(train)

add_executable(main main.cpp)
target_compile_options(main PUBLIC -O3 --std=c++17)

target_link_libraries(main mlp csvParser uWebSockets json)