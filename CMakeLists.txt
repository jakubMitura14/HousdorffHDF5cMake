

cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
set(CMAKE_CUDA_ARCHITECTURES 70)
project(hello LANGUAGES CXX CUDA)
enable_language(CUDA)
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -lineinfo -lcudadevrt")
add_executable(hello hello.cu  )
include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
target_link_libraries(${PROJECT_NAME} ${CUDA_LIBRARIES})



