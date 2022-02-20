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




inline __device__ bool getPredGoldPass(const bool isPaddingPass
    , bool(&isGoldPassToContinue)[1], bool(&isSegmPassToContinue)[1]
    , MetaDataGPU& metaData
   , uint32_t*& metaDataArr, uint32_t& linIdexMeta

){
    if (isPaddingPass) {


        return (isGoldPassToContinue[0] && metaDataArr[linIdexMeta * metaData.metaDataSectionLength + 11]
            && !metaDataArr[linIdexMeta * metaData.metaDataSectionLength + 7]
            && !metaDataArr[linIdexMeta * metaData.metaDataSectionLength + 8]);


    }
    else {
        return (isGoldPassToContinue[0] && metaDataArr[linIdexMeta * metaData.metaDataSectionLength + 7]
            && !metaDataArr[linIdexMeta * metaData.metaDataSectionLength + 8]);


    }

}


inline __device__ bool getPredSegmPass(const bool isPaddingPass
    , bool(&isGoldPassToContinue)[1], bool(&isSegmPassToContinue)[1]
    , MetaDataGPU& metaData
   , uint32_t*& metaDataArr, uint32_t& linIdexMeta

) {
    if (isPaddingPass) {
        return (isSegmPassToContinue[0] && metaDataArr[linIdexMeta * metaData.metaDataSectionLength + 12]
            && !metaDataArr[linIdexMeta * metaData.metaDataSectionLength + 9]
            && !metaDataArr[linIdexMeta * metaData.metaDataSectionLength + 10]);

    }
    else {
        return (isSegmPassToContinue[0] && metaDataArr[linIdexMeta * metaData.metaDataSectionLength + 9]
            && !metaDataArr[linIdexMeta * metaData.metaDataSectionLength + 10]);
    }


}



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
inline __device__ void metadataPass(ForBoolKernelArgs<TKKI> fbArgs, const bool isPaddingPass
    , const uint8_t predicateAa, const uint8_t predicateAb, const uint8_t predicateAc
    , const uint8_t predicateBa, const uint8_t predicateBb, const uint8_t predicateBc
    ,uint32_t (&mainShmem)[lengthOfMainShmem], unsigned int(&globalWorkQueueOffset)[1], unsigned int(&globalWorkQueueCounter)[1]
    , unsigned int(&localWorkQueueCounter)[1], unsigned int(&localTotalLenthOfWorkQueue)[1], unsigned int(&localMinMaxes)[5]
    , unsigned int(&fpFnLocCounter)[1], bool(&isGoldPassToContinue)[1], bool(&isSegmPassToContinue)[1]
    , thread_block& cta, thread_block_tile<32>& tile
    , MetaDataGPU& metaData
    , unsigned int*& minMaxes, uint32_t*& workQueue, uint32_t*& metaDataArr

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
    //printf(" workCounter at start %d ", minMaxes[9] );

}

/*
0 : global FP count;
1 : global FN count;
2 : workQueueCounter
3 : resultFP globalCounter
4 : resultFn globalCounter
     */
if (tile.thread_rank() == 0 && tile.meta_group_rank() == 1) { 

  
    isGoldPassToContinue[0] 
= (  (minMaxes[7] * fbArgs.robustnessPercent) > minMaxes[10]); 



    //if (blockIdx.x == 0) {

    //    printf("in meta pass fp count %d  ceiled %f fp counter %d isTo be continued %d \n "
    //        , minMaxes[7]
    //        , minMaxes[7] * fbArgs.robustnessPercent
    //        , minMaxes[10]
    //        , isGoldPassToContinue[0]
    //    );
    //}

};

if (tile.thread_rank() == 0 && tile.meta_group_rank() == 1) { 

    isSegmPassToContinue[0] 
        = ((minMaxes[8] * fbArgs.robustnessPercent) > minMaxes[11]); 
   
    //if (blockIdx.x == 0) {

    //    printf("in meta pass fn count %d  ceiled %f fn counter %d isTo be continued %d \n "
    //        , minMaxes[8]
    //        , minMaxes[8] * fbArgs.robustnessPercent
    //        , minMaxes[11]
    //        , isSegmPassToContinue[0]
    //    );
    //}

};




sync(cta);

//iterations 
for (uint32_t linIdexMeta = blockIdx.x * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x; linIdexMeta <= metaData.totalMetaLength; linIdexMeta += blockDim.x * blockDim.y * gridDim.x) {
    
    //if (metaDataArr[linIdexMeta * metaData.metaDataSectionLength + 11]) {
    //    printf("in meta pass gold  linIdexMeta %d to be activated  1  isActiveGold %d  isFullGold %d \n"
    //        , linIdexMeta
    //        , metaDataArr[linIdexMeta * metaData.metaDataSectionLength + 7]
    //    , metaDataArr[linIdexMeta * metaData.metaDataSectionLength + 8]);

    //
    //}
    //
    //
    //if (metaDataArr[linIdexMeta * metaData.metaDataSectionLength + 12] ) {
    //    printf("in meta pass segm  linIdexMeta %d to be activated  1  isActive %d  isFull %d \n"
    //        , linIdexMeta
    //        , metaDataArr[linIdexMeta * metaData.metaDataSectionLength + 9]
    //        , metaDataArr[linIdexMeta * metaData.metaDataSectionLength + 10]);

    //}


    //goldpass
    if (getPredGoldPass(isPaddingPass, isGoldPassToContinue, isSegmPassToContinue    , metaData, metaDataArr, linIdexMeta)) {

        //printf("in meta pass gold linIdexMeta %d isPaddingPass %d \n", linIdexMeta, isPaddingPass);

        auto old = atomicAdd_block(&localWorkQueueCounter[0], 1) ;
        if (old < lengthOfMainShmem) {
            mainShmem[old] = linIdexMeta + (isGoldOffset);
        }
        else {
            old = atomicAdd(&(minMaxes[9]), 1);
            workQueue[old] = linIdexMeta + (isGoldOffset) ;
        }
        if (isPaddingPass) {
            //setting to be activated to 0 
            metaDataArr[linIdexMeta * metaData.metaDataSectionLength + 11] = 0;
            //setting active to 1
            metaDataArr[linIdexMeta * metaData.metaDataSectionLength + 7] = 1;
        }
    }
    //segm pass
    if (getPredSegmPass(isPaddingPass, isGoldPassToContinue, isSegmPassToContinue  , metaData, metaDataArr, linIdexMeta)) {

        //printf("in meta pass segm linIdexMeta %d isPaddingPass %d \n", linIdexMeta, isPaddingPass);

        auto old = atomicAdd_block(&localWorkQueueCounter[0], 1);
        if (old < lengthOfMainShmem) {
            mainShmem[old] = linIdexMeta;
        }
        else {
            old = atomicAdd(&(minMaxes[9]), 1);
            workQueue[old] = linIdexMeta;
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
        //printf("local work Counter in meta pass %d  \n"
        //    , localWorkQueueCounter[0]
        //    );
        globalWorkQueueCounter[0] = atomicAdd(&(minMaxes[9]), (localWorkQueueCounter[0]));
    }
}
sync(cta);
for (uint32_t linI =threadIdx.y * blockDim.x + threadIdx.x; linI < localWorkQueueCounter[0]; linI += blockDim.x * blockDim.y ) {
  workQueue[globalWorkQueueCounter[0]+linI]=mainShmem[linI];
}

//cooperative_groups::memcpy_async(cta, (&workQueue[globalWorkQueueCounter[0]]), (&mainArr[0]), (sizeof(uint32_t) * localWorkQueueCounter[0]));
}




