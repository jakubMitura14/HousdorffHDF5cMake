#include "cuda_runtime.h"
#include <cmath>
#include "device_launch_parameters.h"
#include <cstdint>
#include <assert.h>
#include <numeric>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cooperative_groups/memcpy_async.h>
#include <cuda/pipeline>
using namespace cooperative_groups;





__global__ void testPipeline(uint32_t* global_out, uint32_t* global_inA, uint32_t* globalOutGPUB, float* globalDummyGPU) {


}
