#include "CPUAllocations.cu"
#include "MetaData.cu"
#include "Structs.cu"
#include "IterationUtils.cu"
#include "ExceptionManagUtils.cu"
#include "CooperativeGroupsUtils.cu"
#include "ForBoolKernel.cu"
#include "FirstMetaPass.cu"
#include "MainPassFunctions.cu"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include "UnitTestUtils.cu"

using namespace cooperative_groups;

/*
  First we need to be sure that we start from global workqueue to be 0 but it is hard to  get without additional grid sync 
  so we will use 2 counters for odd and even iteration number and here below we will zero the old one on one thread of first thread block
  
  We need to populate the worqueue
   We need to get count of the total FP, FN so we will know wheather we should start loop anew 
*/



/*
as we have limited space in work queue we will use also the resShmem and source shmem in order to keep calculations easy 
we will divide all shared memory in  blocks of 32 length what will enable us using fast shift operators 
- then on the basis of the result spot we will deide to which shared memory array to put the data locally
0) we are supplied with a spot obtain from atomic addition to local counter where we want to put our data
1) we divide by shifting 5 times so we will know to which shared memory space to put our data we will need to use if operators
2) using sutractions and getting remainder will give us spot in 32 subblock where to put the data 
https://stackoverflow.com/questions/13548172/bitshifts-to-obtain-remainder
*/




#pragma once
template <typename TKKI>
inline __device__ void metadataPass(ForBoolKernelArgs<TKKI> fbArgs, bool isPaddingPass
    , uint8_t predicateAa, uint8_t predicateAb, uint8_t predicateAc
    , uint8_t predicateBa, uint8_t predicateBb, uint8_t predicateBc
    ,uint32_t mainShmem[], unsigned int globalWorkQueueOffset[1], unsigned int globalWorkQueueCounter[1]
    , unsigned int localWorkQueueCounter[1], unsigned int localTotalLenthOfWorkQueue[1], unsigned int localMinMaxes[5]
    , unsigned int fpFnLocCounter[1], bool isGoldPassToContinue[1], bool isSegmPassToContinue[1], thread_block cta, thread_block_tile<32> tile
    , MetaDataGPU metaData
    , unsigned int* minMaxes, uint32_t* workQueue, uint16_t* metaDataArr

) {
  // preparation loads
if (tile.thread_rank() == 0 && tile.meta_group_rank() == 0) {
    fpFnLocCounter[0] = 0;
}
if (tile.thread_rank() == 1 && tile.meta_group_rank() == 0) {
    localWorkQueueCounter[0] = 0;
}
if (tile.thread_rank() == 2 && tile.meta_group_rank() == 0) {
    localWorkQueueCounter[0] = 0;
}
if (tile.thread_rank() == 3 && tile.meta_group_rank() == 0) {
    localWorkQueueCounter[0] = 0;
}
if (tile.meta_group_rank() == 1) {
    cooperative_groups::memcpy_async(tile, (&localMinMaxes[0]), (&minMaxes[7]), cuda::aligned_size_t<4>(sizeof(unsigned int) * 5));
}
tile.sync();
/*
0 : global FP count;
1 : global FN count;
2 : workQueueCounter
3 : resultFP globalCounter
4 : resultFn globalCounter
     */
if (tile.thread_rank() == 0 && tile.meta_group_rank() == 1) { isGoldPassToContinue[0] = ((localMinMaxes[0] * fbArgs.robustnessPercent) > localMinMaxes[3]); };
if (tile.thread_rank() == 0 && tile.meta_group_rank() == 1) { isGoldPassToContinue[0] = ((localMinMaxes[1] * fbArgs.robustnessPercent) > localMinMaxes[4]); };
sync(cta);

//iterations 
for (uint16_t linIdexMeta = blockIdx.x * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x; linIdexMeta < metaData.totalMetaLength; linIdexMeta += blockDim.x * blockDim.y * gridDim.x) {
    //goldpass


    if (isGoldPassToContinue[0] && metaDataArr[linIdexMeta * metaData.metaDataSectionLength + predicateAa]
        && !metaDataArr[linIdexMeta * metaData.metaDataSectionLength + predicateAb]
        && (isPaddingPass &&  !metaDataArr[linIdexMeta * metaData.metaDataSectionLength + predicateAc])) {

        auto old = atomicAdd_block(&localWorkQueueCounter[0], 1) - 1;
        if (old < lengthOfMainShmem) {
            mainShmem[old] = uint32_t(linIdexMeta + (isGoldOffset) );
        }
        else {
            old = atomicAdd(&(minMaxes[9]), 1);
            workQueue[old] = uint32_t(linIdexMeta + (isGoldOffset) );
        }
        if (isPaddingPass) {
            //setting to be activated to 0 
            metaDataArr[linIdexMeta * metaData.metaDataSectionLength + 11] = 0;
            //setting active to 1
            metaDataArr[linIdexMeta * metaData.metaDataSectionLength + 7] = 1;
        }
    }
    //segm pass
    if (isSegmPassToContinue[0] && metaDataArr[linIdexMeta * metaData.metaDataSectionLength + predicateBa]
        && !metaDataArr[linIdexMeta * metaData.metaDataSectionLength + predicateBb]
        && (isPaddingPass &&  !metaDataArr[linIdexMeta * metaData.metaDataSectionLength + predicateBc]) ) {

        auto old = atomicAdd_block(&localWorkQueueCounter[0], 1) - 1;
        if (old < lengthOfMainShmem) {
            mainShmem[old] = uint32_t(linIdexMeta);
        }
        else {
            old = atomicAdd(&(minMaxes[9]), 1);
            workQueue[old] = uint32_t(linIdexMeta);
        }
        if (isPaddingPass) {
            //setting to be activated to 0 
            metaDataArr[linIdexMeta * metaData.metaDataSectionLength + 12] = 0;
            //setting active to 1
            metaDataArr[linIdexMeta * metaData.metaDataSectionLength + 9] = 1;
        }
    }

}
//getting begining where we would copy local queue to global one 
sync(cta);
if (tile.thread_rank() == 0 && tile.meta_group_rank() == 0) {
    if (localWorkQueueCounter[0] > 0) {
        globalWorkQueueCounter[0] = atomicAdd(&(minMaxes[9]), (localWorkQueueCounter[0]));
    }
}
sync(cta);
for (uint16_t linI =threadIdx.y * blockDim.x + threadIdx.x; linI < localWorkQueueCounter[0]; linI += blockDim.x * blockDim.y ) {
  workQueue[globalWorkQueueCounter[0]+linI]=mainShmem[linI];
}

//cooperative_groups::memcpy_async(cta, (&workQueue[globalWorkQueueCounter[0]]), (&mainArr[0]), (sizeof(uint32_t) * localWorkQueueCounter[0]));
}




