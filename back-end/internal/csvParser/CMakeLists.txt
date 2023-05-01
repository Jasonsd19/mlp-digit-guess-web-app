add_library(csvParser csvParser.cpp)
target_compile_options(csvParser PUBLIC -O3 --std=c++17)
target_link_libraries(csvParser PUBLIC matrix)

target_include_directories (csvParser PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})