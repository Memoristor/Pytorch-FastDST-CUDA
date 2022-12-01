cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(pytorch_dct VERSION 0.1.0)

add_executable(pytorch_dct torch_dct.cpp)

set(CMAKE_PREFIX_PATH /home/dandri/libtorch)

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

target_link_libraries(pytorch_dct "${TORCH_LIBRARIES}")
set_property(TARGET pytorch_dct PROPERTY CXX_STANDARD 14)

# target_link_libraries(pytorch_dct)

