#one need to have installed CUDa - here I use CUDA 11.6
#for pytorch needed CUDNN look into https://www.youtube.com/watch?v=2TcnIzJ1RQs
#pytorch installation in C++ https://youtu.be/6eTVqYGIWx0  and https://pytorch.org/cppdocs/installing.html
# pytorch custom function https://pytorch.org/tutorials/advanced/cpp_extension.html
#for hdf5 and python  I used vcpkg 

cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(hello LANGUAGES CXX CUDA)
enable_language(CUDA)


find_package(hdf5 CONFIG REQUIRED)



set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -lineinfo  -rdc=true -lcudadevrt")
set(CMAKE_CUDA_ARCHITECTURES 80)
add_executable(hello hello.cu  )
include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

target_link_libraries(${PROJECT_NAME} ${CUDA_LIBRARIES})


#find_package(Python3 COMPONENTS Development REQUIRED)
#target_link_libraries(hello PRIVATE Python3::Python)


#find_package(Torch REQUIRED)
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")


#target_link_libraries(hello PRIVATE ${TORCH_LIBRARIES})


# from https://pytorch.org/cppdocs/installing.html
#if (MSVC)
#  file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
#  add_custom_command(TARGET hello
#                     POST_BUILD
#                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
#                     ${TORCH_DLLS}
#                     $<TARGET_FILE_DIR:hello>)
#endif (MSVC)
#-DCMAKE_PREFIX_PATH="D:\\projects\\libTorchC\\libtorch-win-shared-with-deps-1.10.2+cu113\\libtorch"

target_link_libraries(hello PRIVATE hdf5::hdf5-shared hdf5::hdf5_hl-shared hdf5::hdf5_cpp-shared hdf5::hdf5_hl_cpp-shared)




