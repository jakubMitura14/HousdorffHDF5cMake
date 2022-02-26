cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(hello LANGUAGES CXX CUDA)
enable_language(CUDA)


find_package(hdf5 CONFIG REQUIRED)
#find_package(GTest CONFIG REQUIRED)

set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -lineinfo  -rdc=true -lcudadevrt")
set(CMAKE_CUDA_ARCHITECTURES 80)
add_executable(hello hello.cu  )
include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

target_link_libraries(${PROJECT_NAME} ${CUDA_LIBRARIES})

target_link_libraries(hello PRIVATE hdf5::hdf5-shared hdf5::hdf5_hl-shared hdf5::hdf5_cpp-shared hdf5::hdf5_hl_cpp-shared)
#target_link_libraries(hello PRIVATE GTest::gmock GTest::gtest GTest::gmock_main GTest::gtest_main)



