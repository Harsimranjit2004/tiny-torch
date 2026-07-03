cmake_minimum_required(VERSION 3.20)
project(tinytorch LANGUAGES CXX)

enable_testing()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native")
set(CMAKE_CXX_FLAGS_DEBUG "-g -fsanitize=address,undefined -fno-omit-frame-pointer")
add_compile_options(-Wall -Wextra)

find_package(Eigen3 REQUIRED)

add_library(tinytorch INTERFACE)
target_include_directories(tinytorch INTERFACE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/third_party)
target_link_libraries(tinytorch INTERFACE Eigen3::Eigen)

add_executable(test_tensor tests/test_tensor.cpp)
target_link_libraries(test_tensor PRIVATE tinytorch)
add_test(NAME tensor COMMAND test_tensor)

add_executable(test_activations tests/test_activations.cpp)
target_link_libraries(test_activations PRIVATE tinytorch)
add_test(NAME activations COMMAND test_activations)

add_executable(test_layers tests/test_layers.cpp)
target_link_libraries(test_layers PRIVATE tinytorch)
add_test(NAME layers COMMAND test_layers)

add_executable(test_losses tests/test_losses.cpp)
target_link_libraries(test_losses PRIVATE tinytorch)
add_test(NAME losses COMMAND test_losses)

add_executable(test_data tests/test_data.cpp)
target_link_libraries(test_data PRIVATE tinytorch)
add_test(NAME data COMMAND test_data)

add_executable(test_autograd tests/test_autograd.cpp)
target_link_libraries(test_autograd PRIVATE tinytorch)
add_test(NAME autograd COMMAND test_autograd)