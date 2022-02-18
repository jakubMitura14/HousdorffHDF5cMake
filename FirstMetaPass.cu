#include <cuda_runtime.h>
#include "CPUAllocations.cu"
#include "MetaData.cu"
#include "IterationUtils.cu"
#include "ExceptionManagUtils.cu"
#include "CooperativeGroupsUtils.cu"
#include "ForBoolKernel.cu"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
using namespace cooperative_groups;


/*
    a) we define offsets in the result list to have the results organized and avoid overwiting
    b) if metadata block is active we add it in the work queue
*/


/*
we add here to appropriate queue data  about metadata of blocks of intrest
minMaxesPos- marks in minmaxes the postion of global offset counter -12) global FP offset 13) global FnOffset
offsetMetadataArr- arrays from metadata holding data about result list offsets it can be either fbArgs.metaData.fpOffset or fbArgs.metaData.fnOffset
*/


#pragma once
__device__ inline void addToQueue( uint32_t linIdexMeta, uint8_t isGold
    , unsigned int fpFnLocCounter[1], uint32_t localWorkQueue[1600], uint32_t localOffsetQueue[1600], unsigned int localWorkQueueCounter[1]
    , uint8_t countIndexNumb, uint8_t isActiveIndexNumb, uint8_t offsetIndexNumb
    , uint32_t* metaDataArr, MetaDataGPU metaData, unsigned int* minMaxes,uint32_t* workQueue) {

    unsigned int count = metaDataArr[linIdexMeta * metaData.metaDataSectionLength + countIndexNumb];
        //given fp is non zero we need to  add this to local queue
        if (metaDataArr[linIdexMeta * metaData.metaDataSectionLength + isActiveIndexNumb]==1) {

           // printf("adding to local in first meta pass linIdexMeta %d isGold %d isActiveIndexNumb %d \n  ", linIdexMeta, isGold, isActiveIndexNumb);

            count = atomicAdd_block(&fpFnLocCounter[0], count);
            unsigned int  old = atomicAdd_block(&localWorkQueueCounter[0], 1);
            //we check weather we still have space in shared memory
            if (old < 1590) {// so we still have space in shared memory
                // will be equal or above isGoldOffset  if it is gold pass
                localWorkQueue[old] = linIdexMeta+(isGoldOffset * isGold);
                localOffsetQueue[old] = uint32_t(count);
                     }
            else {// so we do not have any space more in the sared memory  - it is unlikely so we will just in this case save immidiately to global memory
                old = atomicAdd(&(minMaxes[9]), old);
                //workQueue
                workQueue[old] = linIdexMeta + (isGoldOffset * isGold);
                //and offset 
                metaDataArr[linIdexMeta * metaData.metaDataSectionLength + offsetIndexNumb] = atomicAdd(&(minMaxes[12]), count);
            };
     }
}


#pragma once
template <typename PYO>
__global__ void firstMetaPrepareKernel(ForBoolKernelArgs<PYO> fbArgs
    , MetaDataGPU metaData, unsigned int* minMaxes, uint32_t* workQueue
    , uint32_t* origArrs, uint32_t* metaDataArr) {

    //////initializations
    thread_block cta = this_thread_block();
     char* tensorslice;// needed for iterations over 3d arrays
    //local offset counters  for fp and fn's
    __shared__ unsigned int fpFnLocCounter[1];
    // used to store the start position in global memory for whole block
    __shared__ unsigned int globalOffsetForBlock[1];
    __shared__ unsigned int globalWorkQueueCounter[1];
    //used as local work queue counter
    __shared__ unsigned int localWorkQueueCounter[1];     
    //according to https://forums.developer.nvidia.com/t/find-the-limit-of-shared-memory-that-can-be-used-per-block/48556 it is good to keep shared memory below 16kb kilo bytes so it will give us 1600 length of shared memory
    //so here we will store locally the calculated offsets and coordinates of meta data block of intrest marking also wheather we are  talking about gold or segmentation pass (fp or fn )
    __shared__ uint32_t localWorkQueue[1600];
    __shared__ uint32_t localOffsetQueue[1600];
    if ((threadIdx.x == 0)) {
        fpFnLocCounter[0] = 0;
    }
    if ((threadIdx.x == 1)) {
        localWorkQueueCounter[0] = 0;
    }
    if ((threadIdx.x == 2)) {
        globalWorkQueueCounter[0] = 0;
    }
    if ((threadIdx.x == 3)) {
        globalOffsetForBlock[0] = 0;
    }
    sync(cta);


    // classical grid stride loop - in case of unlikely event we will run out of space we will empty it prematurly
    //main metadata iteration
    for (uint32_t linIdexMeta = blockIdx.x * blockDim.x + threadIdx.x; linIdexMeta < metaData.totalMetaLength; linIdexMeta += blockDim.x * gridDim.x) {
         
       // if ((threadIdx.x == 0) && (threadIdx.y == 0)) {
          //  printf("in first meta pass linIdexMeta %d blockIdx.x %d blockDim.x %d metaData.totalMetaLength %d threadIdx.x %d \n  ", linIdexMeta, blockIdx.x, blockDim.x, metaData.totalMetaLength, threadIdx.x );
        //}
        
        //goldpass
        addToQueue( linIdexMeta, 0
            , fpFnLocCounter, localWorkQueue, localOffsetQueue, localWorkQueueCounter
            , 1, 9, 6
            , metaDataArr, metaData, minMaxes, workQueue);
          //segmPass  
        addToQueue( linIdexMeta, 1
            , fpFnLocCounter, localWorkQueue, localOffsetQueue, localWorkQueueCounter
            , 2, 7, 5
            , metaDataArr, metaData, minMaxes, workQueue);
    
        
        
 /*       addToQueue(fbArgs, old, count, tensorslice, xMeta, yMeta, zMeta, fbArgs.metaData.fpOffset, fbArgs.metaData.fpCount, 0, fbArgs.metaData.isActiveSegm, fpFnLocCounter, localWorkAndOffsetQueue, localWorkQueueCounter);
        addToQueue(fbArgs, old, count, tensorslice, xMeta, yMeta, zMeta, fbArgs.metaData.fnOffset, fbArgs.metaData.fnCount, 1, fbArgs.metaData.isActiveGold, fpFnLocCounter, localWorkAndOffsetQueue, localWorkQueueCounter);*/
        }
    sync(cta);
    if ((threadIdx.x == 0) ) {
        globalOffsetForBlock[0] = atomicAdd(&(minMaxes[12]), (fpFnLocCounter[0]));

       /* if (fpFnLocCounter[0]>0) {
            printf("\n in meta first pass global offset %d  locCounter %d \n  ", globalOffsetForBlock[0], fpFnLocCounter[0]);
        }*/
    };
    if ((threadIdx.x == 1) ) {
        if (localWorkQueueCounter[0]>0) {
            globalWorkQueueCounter[0] = atomicAdd(&(minMaxes[9]), (localWorkQueueCounter[0]));
         }
    }
    sync(cta);

    //exporting to global work queue
    //cooperative_groups::memcpy_async(cta, (&workQueue[globalWorkQueueCounter[0]]), (localWorkQueue), (sizeof(uint32_t) * localWorkQueueCounter[0]));

    
    //setting offsets
    for (uint32_t i = threadIdx.x; i < localWorkQueueCounter[0]; i += blockDim.x) {
        workQueue[globalWorkQueueCounter[0] +i]=localWorkQueue[i]; 

   /*     printf("FFIrst meta pass lin meta to Work Q %d is gold %d to spot %d  \n "
    , localWorkQueue[i] - isGoldOffset*(localWorkQueue[i] >= isGoldOffset)
        , (localWorkQueue[i] >= isGoldOffset), globalWorkQueueCounter[0] + i);*/

        //FP pass
        if (localWorkQueue[i]>= isGoldOffset) {
            metaDataArr[(localWorkQueue[i] - isGoldOffset) * metaData.metaDataSectionLength + 5] = localOffsetQueue[i] + globalOffsetForBlock[0];
            //printf("fp offset lin meta %d total offset  %d  global part %d local part %d \n "
            //    , localWorkQueue[i] - isGoldOffset
            //    , localOffsetQueue[i] + globalOffsetForBlock[0] 
            //, globalOffsetForBlock[0]
            //, localOffsetQueue[i]);

        }
        //FN pass
        else {
            metaDataArr[(localWorkQueue[i]) * metaData.metaDataSectionLength + 6] = localOffsetQueue[i] + globalOffsetForBlock[0];
            //printf("fn offset lin meta %d total offset  %d  global part %d local part %d \n "
            //    , localWorkQueue[i] 
            //    , localOffsetQueue[i] + globalOffsetForBlock[0]
            //    , globalOffsetForBlock[0]
            //    , localOffsetQueue[i]);
        
        };

        //sync(cta);

        
    }

           

    };







