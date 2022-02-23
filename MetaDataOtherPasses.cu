#include "CPUAllocations.cu"
#include "MetaData.cu"
#include "Structs.cu"
 
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


inline __device__ void modifyMetaDataInPaddingPass(const bool isPaddingPass, uint32_t*& metaDataArr, uint32_t& linIdexMeta, MetaDataGPU& metaData, const uint8_t toBeActivated, const uint8_t isActiveNumb) {
    if (isPaddingPass) {
        //setting to be activated to 0 
        metaDataArr[linIdexMeta * metaData.metaDataSectionLength + toBeActivated] = 0;
        //setting active to 1
        metaDataArr[linIdexMeta * metaData.metaDataSectionLength + isActiveNumb] = 1;
    }
}


inline __device__ void saveWorkQueueToGlobal(thread_block& cta, thread_block_tile<32>& tile, unsigned int(&localWorkQueueCounter)[1]
    , unsigned int*& minMaxes, uint32_t*& workQueue, unsigned int(&globalWorkQueueCounter)[1]
    , uint32_t(&mainShmem)[lengthOfMainShmem]) {
    if (tile.thread_rank() == 0 && tile.meta_group_rank() == 0) {
        if (localWorkQueueCounter[0] > 0) {
            //printf("local work Counter in meta pass %d  \n"
            //    , localWorkQueueCounter[0]
            //    );
            globalWorkQueueCounter[0] = atomicAdd(&(minMaxes[9]), (localWorkQueueCounter[0]));
        }
    }
    __syncthreads();
    for (uint32_t linI = threadIdx.y * blockDim.x + threadIdx.x; linI < localWorkQueueCounter[0]; linI += blockDim.x * blockDim.y) {
        workQueue[globalWorkQueueCounter[0] + linI] = mainShmem[linI];
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



};

if (tile.thread_rank() == 0 && tile.meta_group_rank() == 1) { 

    isSegmPassToContinue[0] 
        = ((minMaxes[8] * fbArgs.robustnessPercent) > minMaxes[11]); 


};






//for (uint32_t linIdexMeta = blockIdx.x * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x
//        ; linIdexMeta <= fbArgs.metaData.totalMetaLength  // we add in order to 
//        ; linIdexMeta += blockDim.x * blockDim.y * gridDim.x) {
//    //TODO() consider doing it warp centric way  
//    //  we need to be sure that amount of blocks in local work queue do not exceed lengthOfMainShmem probably the most optimal would be to divide work queue to sections where each warp would be responsible for 
//    //  then if number in warp queue will exceed the size of available shared memory it will write it to global memory ... this way we will avoid  thread divergence and keep local work queue in available shared memory space
//    //if (localWorkQueueCounter[0] < (lengthOfMainShmem - (blockDim.x * blockDim.y))) {
//      //goldpass
//    if (getPredGoldPass(isPaddingPass, isGoldPassToContinue, isSegmPassToContinue, metaData, metaDataArr, linIdexMeta)) {
//        //    printf("in meta pass gold linIdexMeta %d isPaddingPass %d  total meta %d \n", linIdexMeta, isPaddingPass, fbArgs.metaData.totalMetaLength);
//        mainShmem[atomicAdd_block(&localWorkQueueCounter[0], 1)] = linIdexMeta + (isGoldOffset);
//        modifyMetaDataInPaddingPass(isPaddingPass, metaDataArr, linIdexMeta, metaData, 11, 7);
//    }
//    //segm pass
//    if (getPredSegmPass(isPaddingPass, isGoldPassToContinue, isSegmPassToContinue, metaData, metaDataArr, linIdexMeta)) {
//        //  printf("in meta pass segm linIdexMeta %d isPaddingPass %d \n", linIdexMeta, isPaddingPass);
//        mainShmem[atomicAdd_block(&localWorkQueueCounter[0], 1)] = linIdexMeta;
//        modifyMetaDataInPaddingPass(isPaddingPass, metaDataArr, linIdexMeta, metaData, 12, 9);
//    }
//    //sync(cta);
//    //if (localWorkQueueCounter[0] > (lengthOfMainShmem - (blockDim.x * blockDim.y))) {
//    //   saveWorkQueueToGlobal(cta, tile, localWorkQueueCounter, minMaxes, workQueue, globalWorkQueueCounter, mainShmem);
//    //}
//    //sync(cta);
//    //localWorkQueueCounter[0] = 0;
//    //sync(cta);
//
//
//};




__syncthreads();
 



for (uint8_t outer = 0; outer <= ceilf(fbArgs.metaData.totalMetaLength / (blockDim.x * blockDim.y * gridDim.x)); outer++) {
    uint32_t linIdexMeta = blockIdx.x * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x + (blockDim.x * blockDim.y * gridDim.x) * outer;
    bool isResFound = false;
    if (linIdexMeta <= fbArgs.metaData.totalMetaLength) {

        //TODO() consider doing it warp centric way  
    //  we need to be sure that amount of blocks in local work queue do not exceed lengthOfMainShmem probably the most optimal would be to divide work queue to sections where each warp would be responsible for 
    //  then if number in warp queue will exceed the size of available shared memory it will write it to global memory ... this way we will avoid  thread divergence and keep local work queue in available shared memory space
    //if (localWorkQueueCounter[0] < (lengthOfMainShmem - (blockDim.x * blockDim.y))) {
      //goldpass
        if (getPredGoldPass(isPaddingPass, isGoldPassToContinue, isSegmPassToContinue, metaData, metaDataArr, linIdexMeta)) {
            //    printf("in meta pass gold linIdexMeta %d isPaddingPass %d  total meta %d \n", linIdexMeta, isPaddingPass, fbArgs.metaData.totalMetaLength);
            // localWorkQueueCounter[0] += 1;
            isResFound = true;
            mainShmem[atomicAdd_block(&localWorkQueueCounter[0], 1)] = linIdexMeta + (isGoldOffset);
           // mainShmem[atomicAdd(localWorkQueueCounter, 1)] = linIdexMeta + (isGoldOffset);

            modifyMetaDataInPaddingPass(isPaddingPass, metaDataArr, linIdexMeta, metaData, 11, 7);
        }

    }




}


__syncthreads();
if (localWorkQueueCounter[0]>0) {
    saveWorkQueueToGlobal(cta, tile, localWorkQueueCounter, minMaxes, workQueue, globalWorkQueueCounter, mainShmem);
}
__syncthreads();
localWorkQueueCounter[0] = 0;
__syncthreads();

for (uint8_t outer = 0; outer <= ceilf(fbArgs.metaData.totalMetaLength / (blockDim.x * blockDim.y * gridDim.x)); outer++) {
    uint32_t linIdexMeta = blockIdx.x * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x + (blockDim.x * blockDim.y * gridDim.x) * outer;

    if (linIdexMeta <= fbArgs.metaData.totalMetaLength) {

        //segm pass
        if (getPredSegmPass(isPaddingPass, isGoldPassToContinue, isSegmPassToContinue, metaData, metaDataArr, linIdexMeta)) {
            //  printf("in meta pass segm linIdexMeta %d isPaddingPass %d \n", linIdexMeta, isPaddingPass);
           // localWorkQueueCounter[0] += 1;
            mainShmem[atomicAdd_block(&localWorkQueueCounter[0], 1)] = linIdexMeta;
            //mainShmem[atomicAdd_block(&localWorkQueueCounter[0], 1)] = linIdexMeta;
           // mainShmem[atomicAdd(localWorkQueueCounter, 1)] = linIdexMeta;


            modifyMetaDataInPaddingPass(isPaddingPass, metaDataArr, linIdexMeta, metaData, 12, 9);
        }
    }
 
}


sync(cta);

if (localWorkQueueCounter[0] > 0) {
    saveWorkQueueToGlobal(cta, tile, localWorkQueueCounter, minMaxes, workQueue, globalWorkQueueCounter, mainShmem);
}





//
//
////getting begining where we would copy local queue to global one 
//sync(cta);
//if (tile.thread_rank() == 0 && tile.meta_group_rank() == 0) {
//    if (localWorkQueueCounter[0] > 0) {
//        //printf("local work Counter in meta pass %d  \n"
//        //    , localWorkQueueCounter[0]
//        //    );
//        globalWorkQueueCounter[0] = atomicAdd(&(minMaxes[9]), (localWorkQueueCounter[0]));
//    }
//}
//sync(cta);
//for (uint32_t linI =threadIdx.y * blockDim.x + threadIdx.x; linI < localWorkQueueCounter[0]; linI += blockDim.x * blockDim.y ) {
//  workQueue[globalWorkQueueCounter[0]+linI]=mainShmem[linI];
//}

//cooperative_groups::memcpy_async(cta, (&workQueue[globalWorkQueueCounter[0]]), (&mainArr[0]), (sizeof(uint32_t) * localWorkQueueCounter[0]));
}




//
//if (tile.thread_rank() == 0 && tile.meta_group_rank() == 0) {
//    if (localWorkQueueCounter[0] > 0) {
//        //printf("local work Counter in meta pass %d  \n"
//        //    , localWorkQueueCounter[0]
//        //    );
//        globalWorkQueueCounter[0] = atomicAdd(&(minMaxes[9]), (localWorkQueueCounter[0]));
//    }
//}
//sync(cta);
//for (uint32_t linI = threadIdx.y * blockDim.x + threadIdx.x; linI < localWorkQueueCounter[0]; linI += blockDim.x * blockDim.y) {
//    workQueue[globalWorkQueueCounter[0] + linI] = mainShmem[linI];
//}






//uint32_t linIdexMeta = blockIdx.x * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;
//bool isLoopToContinue = (linIdexMeta <= fbArgs.metaData.totalMetaLength);
//isLoopToContinue = __syncthreads_or(isLoopToContinue);
//
//while (isLoopToContinue) {
//        //goldpass
//        if (getPredGoldPass(isPaddingPass, isGoldPassToContinue, isSegmPassToContinue, metaData, metaDataArr, linIdexMeta)) {
//            //    printf("in meta pass gold linIdexMeta %d isPaddingPass %d  total meta %d \n", linIdexMeta, isPaddingPass, fbArgs.metaData.totalMetaLength);
//            mainShmem[atomicAdd_block(&localWorkQueueCounter[0], 1)] = linIdexMeta + (isGoldOffset);
//            modifyMetaDataInPaddingPass(isPaddingPass, metaDataArr, linIdexMeta, metaData, 11, 7);
//        }
//        //segm pass
//        if (getPredSegmPass(isPaddingPass, isGoldPassToContinue, isSegmPassToContinue, metaData, metaDataArr, linIdexMeta)) {
//            //  printf("in meta pass segm linIdexMeta %d isPaddingPass %d \n", linIdexMeta, isPaddingPass);
//            mainShmem[atomicAdd_block(&localWorkQueueCounter[0], 1)] = linIdexMeta;
//            modifyMetaDataInPaddingPass(isPaddingPass, metaDataArr, linIdexMeta, metaData, 12, 9);
//        }
//        // this sway of stopping a loop avoids thread diverging and stalling during synchronization ...
//        isLoopToContinue = (linIdexMeta <= fbArgs.metaData.totalMetaLength );
//        isLoopToContinue = __syncthreads_or(isLoopToContinue);
//
//
//        if (localWorkQueueCounter[0] > (lengthOfMainShmem - (blockDim.x * blockDim.y))) {
//            saveWorkQueueToGlobal(cta, tile, localWorkQueueCounter, minMaxes, workQueue, globalWorkQueueCounter, mainShmem);
//           sync(cta);
//           localWorkQueueCounter[0] = 0;
//           sync(cta);
//             }
//      //  sync(cta);
//       // localWorkQueueCounter[0] = 0;
//        //sync(cta);
//        linIdexMeta += (blockDim.x * blockDim.y * gridDim.x);
//
//}



//for (uint32_t linIdexMeta = blockIdx.x * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x
//        ; linIdexMeta <= fbArgs.metaData.totalMetaLength  // we add in order to 
//        ; linIdexMeta += blockDim.x * blockDim.y * gridDim.x) {
//    //TODO() consider doing it warp centric way  
//    //  we need to be sure that amount of blocks in local work queue do not exceed lengthOfMainShmem probably the most optimal would be to divide work queue to sections where each warp would be responsible for 
//    //  then if number in warp queue will exceed the size of available shared memory it will write it to global memory ... this way we will avoid  thread divergence and keep local work queue in available shared memory space
//    //if (localWorkQueueCounter[0] < (lengthOfMainShmem - (blockDim.x * blockDim.y))) {
//      //goldpass
//    if (getPredGoldPass(isPaddingPass, isGoldPassToContinue, isSegmPassToContinue, metaData, metaDataArr, linIdexMeta)) {
//        //    printf("in meta pass gold linIdexMeta %d isPaddingPass %d  total meta %d \n", linIdexMeta, isPaddingPass, fbArgs.metaData.totalMetaLength);
//        mainShmem[atomicAdd_block(&localWorkQueueCounter[0], 1)] = linIdexMeta + (isGoldOffset);
//        modifyMetaDataInPaddingPass(isPaddingPass, metaDataArr, linIdexMeta, metaData, 11, 7);
//    }
//    //segm pass
//    if (getPredSegmPass(isPaddingPass, isGoldPassToContinue, isSegmPassToContinue, metaData, metaDataArr, linIdexMeta)) {
//        //  printf("in meta pass segm linIdexMeta %d isPaddingPass %d \n", linIdexMeta, isPaddingPass);
//        mainShmem[atomicAdd_block(&localWorkQueueCounter[0], 1)] = linIdexMeta;
//        modifyMetaDataInPaddingPass(isPaddingPass, metaDataArr, linIdexMeta, metaData, 12, 9);
//    }
//    //sync(cta);
//    //if (localWorkQueueCounter[0] > (lengthOfMainShmem - (blockDim.x * blockDim.y))) {
//    //   saveWorkQueueToGlobal(cta, tile, localWorkQueueCounter, minMaxes, workQueue, globalWorkQueueCounter, mainShmem);
//    //}
//    //sync(cta);
//    //localWorkQueueCounter[0] = 0;
//    //sync(cta);
//
//
//};


//for (uint8_t outer = 0; outer < 1; outer++) {
//    uint32_t linIdexMeta = blockIdx.x * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x + (blockDim.x * blockDim.y * gridDim.x) * outer;
//    if (linIdexMeta <= fbArgs.metaData.totalMetaLength) {
//
//        //TODO() consider doing it warp centric way  
//    //  we need to be sure that amount of blocks in local work queue do not exceed lengthOfMainShmem probably the most optimal would be to divide work queue to sections where each warp would be responsible for 
//    //  then if number in warp queue will exceed the size of available shared memory it will write it to global memory ... this way we will avoid  thread divergence and keep local work queue in available shared memory space
//    //if (localWorkQueueCounter[0] < (lengthOfMainShmem - (blockDim.x * blockDim.y))) {
//      //goldpass
//        if (getPredGoldPass(isPaddingPass, isGoldPassToContinue, isSegmPassToContinue, metaData, metaDataArr, linIdexMeta)) {
//            //    printf("in meta pass gold linIdexMeta %d isPaddingPass %d  total meta %d \n", linIdexMeta, isPaddingPass, fbArgs.metaData.totalMetaLength);
//            mainShmem[atomicAdd_block(&localWorkQueueCounter[0], 1)] = linIdexMeta + (isGoldOffset);
//            modifyMetaDataInPaddingPass(isPaddingPass, metaDataArr, linIdexMeta, metaData, 11, 7);
//        }
//        //segm pass
//        if (getPredSegmPass(isPaddingPass, isGoldPassToContinue, isSegmPassToContinue, metaData, metaDataArr, linIdexMeta)) {
//            //  printf("in meta pass segm linIdexMeta %d isPaddingPass %d \n", linIdexMeta, isPaddingPass);
//            mainShmem[atomicAdd_block(&localWorkQueueCounter[0], 1)] = linIdexMeta;
//            modifyMetaDataInPaddingPass(isPaddingPass, metaDataArr, linIdexMeta, metaData, 12, 9);
//        }
//    }
//}
//
//


//
//if (linIdexMeta <= fbArgs.metaData.totalMetaLength) {
//
//    //TODO() consider doing it warp centric way  
////  we need to be sure that amount of blocks in local work queue do not exceed lengthOfMainShmem probably the most optimal would be to divide work queue to sections where each warp would be responsible for 
////  then if number in warp queue will exceed the size of available shared memory it will write it to global memory ... this way we will avoid  thread divergence and keep local work queue in available shared memory space
////if (localWorkQueueCounter[0] < (lengthOfMainShmem - (blockDim.x * blockDim.y))) {
//  //goldpass
//    if (getPredGoldPass(isPaddingPass, isGoldPassToContinue, isSegmPassToContinue, metaData, metaDataArr, linIdexMeta)) {
//        //    printf("in meta pass gold linIdexMeta %d isPaddingPass %d  total meta %d \n", linIdexMeta, isPaddingPass, fbArgs.metaData.totalMetaLength);
//        mainShmem[atomicAdd_block(&localWorkQueueCounter[0], 1)] = linIdexMeta + (isGoldOffset);
//        modifyMetaDataInPaddingPass(isPaddingPass, metaDataArr, linIdexMeta, metaData, 11, 7);
//    }
//    //segm pass
//    if (getPredSegmPass(isPaddingPass, isGoldPassToContinue, isSegmPassToContinue, metaData, metaDataArr, linIdexMeta)) {
//        //  printf("in meta pass segm linIdexMeta %d isPaddingPass %d \n", linIdexMeta, isPaddingPass);
//        mainShmem[atomicAdd_block(&localWorkQueueCounter[0], 1)] = linIdexMeta;
//        modifyMetaDataInPaddingPass(isPaddingPass, metaDataArr, linIdexMeta, metaData, 12, 9);
//    }
//}
////    sync(cta);
////if (localWorkQueueCounter[0] > (lengthOfMainShmem - (blockDim.x * blockDim.y))) {
////   //saveWorkQueueToGlobal(cta, tile, localWorkQueueCounter, minMaxes, workQueue, globalWorkQueueCounter, mainShmem);
////    sync(cta);
////    if (tile.thread_rank() == 0 && tile.meta_group_rank() == 1) {
////    //    localWorkQueueCounter[0] = 0;
////    }
////    sync(cta);
////}
//
//
//}
