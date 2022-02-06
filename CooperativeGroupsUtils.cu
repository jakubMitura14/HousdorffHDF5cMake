#include "cuda_runtime.h"
#include <cstdint>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
using namespace cooperative_groups;

/*
We will  here return true if the thread is at the moment active - Hovewer sometimes we have couple things to do so we will add additional numb
untlin numb will not execute number of available threads in a group it will be executed on this thread in other case we will use some already used thread ...
*/

#pragma once
inline __device__ bool isToBeExecutedOnActive(coalesced_group group, int numb, int metaNumb=0) {
    return ((threadIdx.x == numb) && (threadIdx.y == metaNumb));
    //if ((numb < group.num_threads()) && (threadIdx.x == numb) && (threadIdx.y == metaNumb) ) {
    //    return true;
    //}
    //else {// defoult is first thread in group
    //    if (threadIdx.y == 0 && (threadIdx.y == metaNumb)) {
    //        return true;
    //    }
    //};
    //return false;


}


/*
copy asynchronously into shared memopry using pipeline interface - works only for uint32_t
pipeline: pipeline object
block: thread block in cooperative groups definition
mainShmem : shared memory to which we load data
globalIn : global memory from which we take data
alignSize: defines what is the smallest aligned byte lenghth - for best performance should be 128 so for example 32 uint32_t
shmemStart : where in shared memory we have starting point for our load
globalStart : where in global memory we start load
length : how many uint32_t we want to copy from global to shmem
*/
//#pragma once
//inline __device__ void  loadIntoShmem(cuda::pipeline<cuda::thread_scope_thread> pipeline, thread_block block,
//                                        uint32_t* mainShmem, uint32_t* globalIn 
//                                   ,int shmemStart, int globalStart, int length ) {
//    
//    pipeline.producer_acquire();
//    cuda::memcpy_async(block, &mainShmem[shmemStart], &globalIn[globalStart], cuda::aligned_size_t<4>(sizeof(uint32_t) * length), pipeline);
//    pipeline.producer_commit();
//  
//}
