cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(pytorch_dct VERSION 0.1.0)

set(CMAKE_PREFIX_PATH /home/dandri/libtorch)
find_package(Torch REQUIRED)
include_directories("${TORCH_INCLUDE_DIRS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
message(STATUS "TORCH_INCLUDE_DIRS: ${TORCH_INCLUDE_DIRS}$")
message(STATUS "TORCH_LIBRARIES: ${TORCH_LIBRARIES}$")

find_package(CUDA REQUIRED)
include_directories("${CUDA_INCLUDE_DIRS}")
message(STATUS "CUDA_INCLUDE_DIRS: ${CUDA_INCLUDE_DIRS}$")
message(STATUS "CUDA_LIBRARIES: ${CUDA_LIBRARIES}$")

find_package(PythonLibs REQUIRED)
include_directories("${PYTHON_INCLUDE_DIRS}")
message(STATUS "PYTHON_INCLUDE_DIRS: ${PYTHON_INCLUDE_DIRS}$")
message(STATUS "PYTHON_LIBRARIES: ${PYTHON_LIBRARIES}$")

add_executable(pytorch_dct torch_dct.cpp)

target_link_libraries(pytorch_dct "${TORCH_LIBRARIES}" "${CUDA_LIBRARIES}" "${PYTHON_LIBRARIES}")
set_property(TARGET pytorch_dct PROPERTY CXX_STANDARD 14)


