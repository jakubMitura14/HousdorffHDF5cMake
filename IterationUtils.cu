
#include "cuda_runtime.h"
#include "Structs.cu"
#include <cstdint>
#include <assert.h>
/*
given reference to array  it will return reference to tensor row
tensorslice - holds referenco to slice
tensor - reference to struct representing 3d tensor
YLength - max in Y dimension of array we iterate through
y,z - coordinates of row of intrest in tensor
*/
#pragma once
template <typename UIO>
inline __device__ UIO* getTensorRow(char* tensorslice, array3dWithDimsGPU tensor, int YLength, int y, int z) {
    tensorslice = ((char*)tensor.arrPStr.ptr) + z * tensor.arrPStr.pitch * YLength;
    return (UIO*)(tensorslice + y * tensor.arrPStr.pitch);
}

#pragma once
template <typename UHO>
inline __device__ UHO* getTensorRowSimple(char* tensorslice, cudaPitchedPtr tensor, int YLength, int y, int z) {
    tensorslice = ((char*)tensor.ptr) + z * tensor.pitch * YLength;
    return (UHO*)(tensorslice + y * tensor.pitch);
}



