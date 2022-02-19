#include "cuda_runtime.h"
#include <cmath>

#include "device_launch_parameters.h"

// includes, system
#include <iostream>     // std::cout
#include <algorithm>    // std::min
//#include <helper_cuda.h>
#include <cmath>
//#include "Structs.cu"
#include <math.h>
//#include "MemoryTransfers.cu"
#include <cstdint>
#include <assert.h>
#include <numeric>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cooperative_groups/memcpy_async.h>
#include <cuda/pipeline>

//#include "BoolKernelTests.cu"
#include "testAll.cu"
#include "forBench/hdf5Manag.cu"
#include "CooperativeGroupsUtils.cu"
using namespace cooperative_groups;

#include <iostream>
#include <string>
#include <vector>
#include <H5Cpp.h>

#include <H5Cpp.h>
using namespace H5;






int main(void){

  //  const int WIDTH = atoi(argv[1]), HEIGHT = WIDTH, DEPTH = 1;
 //   Volume img1 = Volume(WIDTH, HEIGHT, DEPTH), img2 = Volume(WIDTH, HEIGHT, DEPTH);

    loadHDF();
  


    return 0;  // successfully terminated
}