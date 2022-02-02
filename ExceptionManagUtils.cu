#include "cuda_runtime.h"
#include <cmath>
#include <iostream>
#include <cmath>
#include <cstdint>
#include <assert.h>
#include "device_launch_parameters.h"

#pragma once
inline cudaError_t checkCuda(cudaError_t result, std::string description)
{
    if (result != cudaSuccess) {
        printf("%d", description);
        fprintf(stderr, "CUDA Runtime Error in %d : %s\n", description, cudaGetErrorString(result));
        assert(result == cudaSuccess);
    }
    return result;
}
