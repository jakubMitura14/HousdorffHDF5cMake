#include "CPUAllocations.cu"
#include "MetaData.cu"
#include "IterationUtils.cu"
#include "ExceptionManagUtils.cu"
#include "CooperativeGroupsUtils.cu"
#include "ForBoolKernel.cu"
#include "FirstMetaPass.cu"
#include "MainPassFunctions.cu"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include "UnitTestUtils.cu"
#include "MetaDataOtherPasses.cu"

using namespace cooperative_groups;

template <typename TKKI>
inline __global__ void mainDilatation(ForBoolKernelArgs<TKKI> fbArgs) {


    thread_block cta = this_thread_block();
    thread_block_tile<32> tile = tiled_partition<32>(cta);

    char* tensorslice;
    bool isBlockFull = true;// usefull to establish do we have block completely filled and no more dilatations possible
    unsigned int old = 0;

    // some references using as aliases
    unsigned int& oldRef = old;



    // main shared memory spaces 
    __shared__ uint32_t sourceShared[32][32];
    __shared__ uint32_t resShared[32][32];
    // holding data about paddings 


    // holding data weather we have anything in padding 0)top  1)bottom, 2)left 3)right, 4)anterior, 5)posterior,
    __shared__ bool isAnythingInPadding[6];
    //variables needed for all threads
    __shared__ unsigned int iterationNumb[1];
    __shared__ unsigned int globalWorkQueueOffset[1];
    __shared__ unsigned int globalWorkQueueCounter[1];
    __shared__ unsigned int localWorkQueueCounter[1];
    __shared__ bool isBlockToBeValidated[1];
    // keeping data wheather gold or segmentation pass should continue - on the basis of global counters

    __shared__ unsigned int localTotalLenthOfWorkQueue[1];
    //counters for per block number of results added in this iteration
    __shared__ unsigned int localFpConter[1];
    __shared__ unsigned int localFnConter[1];

    __shared__ unsigned int blockFpConter[1];
    __shared__ unsigned int blockFnConter[1];

    //result list offset - needed to know where to write a result in a result list
    __shared__ unsigned int resultfpOffset[1];
    __shared__ unsigned int resultfnOffset[1];

    __shared__ unsigned int worQueueStep[1];

    // we will load here multiple entries from workqueue
    __shared__ uint16_t localWorkQueue[localWorkQueLength][4];
    //initializations and loading    
    auto active = coalesced_threads();
    if (isToBeExecutedOnActive(active, 0)) { iterationNumb[0] = getTensorRow<unsigned int>(tensorslice, fbArgs.metaData.minMaxes, 1, 0, 0)[13]; };
    //here we caclulate the offset for given block depending on length of the workqueue and number of the  available blocks in a grid
    // - this will give us number of work queue items per block - we will calculate offset on the basis of the block number

    if (isToBeExecutedOnActive(active, 3)) {
        localWorkQueueCounter[0] = 0;
    };

    if (isToBeExecutedOnActive(active, 4)) {
        blockFpConter[0] = 0;
    };
    if (isToBeExecutedOnActive(active, 5)) {
        blockFnConter[0] = 0;
    };
    if (isToBeExecutedOnActive(active, 6)) {
        localFpConter[0] = 0;
    };
    if (isToBeExecutedOnActive(active, 7)) {
        localFnConter[0] = 0;
    };




    if (isToBeExecutedOnActive(active, 1)) {
        localTotalLenthOfWorkQueue[0] = getTensorRow<unsigned int>(tensorslice, fbArgs.metaData.minMaxes, 1, 0, 0)[9];
        globalWorkQueueOffset[0] = floor((float)(localTotalLenthOfWorkQueue[0] / gridDim.x)) + 1;
        worQueueStep[0] = min(localWorkQueLength, globalWorkQueueOffset[0]);
    };
    sync(cta);
    // TODO - use pipelines as described at 201 in https://docs.nvidia.com/cuda/pdf/CUDA_C_Programming_Guide.pdf
    /// load work QueueData into shared memory 

    //TODO change looping so it will access contigous memory
    for (uint8_t bigloop = blockIdx.x * globalWorkQueueOffset[0]; bigloop < ((blockIdx.x + 1) * globalWorkQueueOffset[0]); bigloop += worQueueStep[0]) {
        // grid stride loop - sadly most of threads will be idle 
        ///////////// loading to work queue
        loadFromGlobalToLocalWorkQueue(fbArgs, tensorslice, localWorkQueue, bigloop, globalWorkQueueOffset, localTotalLenthOfWorkQueue, worQueueStep);

        sync(cta);// now local work queue is populated 

            //now all of the threads in the block needs to have the same i value so we will increment by 1
        for (uint8_t i = 0; i < worQueueStep[0]; i += 1) {
            if (((bigloop + i) < localTotalLenthOfWorkQueue[0]) && ((bigloop + i) < ((blockIdx.x + 1) * globalWorkQueueOffset[0]))) {

                // now we have metadata coordinates we need to start go over associated data block - in order to make it as efficient as possible data block size is set to be the same as datablock size
                // so we do not need iteration loop 

                loadAndDilatateAndSave(fbArgs, tensorslice, localWorkQueue, bigloop, sourceShared, resShared, isAnythingInPadding, iterationNumb, isBlockFull, cta, i,
                    isBlockToBeValidated, localTotalLenthOfWorkQueue, localFpConter, localFnConter, resultfpOffset, resultfnOffset, worQueueStep);

                /////////////////////// validation if it is to be validated, also we checked for bing full before dilatations - if it was full at the begining - no point in validation
                validateAndUpMetaCounter(fbArgs, tensorslice, localWorkQueue, bigloop, sourceShared, resShared, isAnythingInPadding, iterationNumb, isBlockFull, cta, i,
                    isBlockToBeValidated, localTotalLenthOfWorkQueue, localFpConter, localFnConter, resultfpOffset, resultfnOffset, worQueueStep,  oldRef, blockFpConter, blockFnConter);

                ////on the basis of isAnythingInPadding we will mark  the neighbouring block as to be activated if there is and if such neighbouring block exists
                auto activeC = coalesced_threads();

                if (localWorkQueue[i][3] == 1) {//gold
                    setNextBlocksActivity(tensorslice, localWorkQueue, i, fbArgs.metaData.isToBeActivatedGold, isAnythingInPadding, activeC);
                };
                if (localWorkQueue[i][3] == 0) {//segm
                    setNextBlocksActivity(tensorslice, localWorkQueue, i, fbArgs.metaData.isToBeActivatedSegm, isAnythingInPadding, activeC);
                };
                // marking blocks as full 

                if (localWorkQueue[i][3] == 1) {//gold
                    markIsBlockFull(tensorslice, localWorkQueue, i, isBlockFull, fbArgs.metaData.isFullGold, activeC);
                };
                if (localWorkQueue[i][3] == 0) {//segm
                    markIsBlockFull(tensorslice, localWorkQueue, i, isBlockFull, fbArgs.metaData.isFullSegm, activeC);
                };
                sync(cta);// all results that should be saved to result list are saved                        

                //we need to clear isAnythingInPadding to 0
                clearisAnythingInPadding(isAnythingInPadding);
            }
        }
    }
    sync(cta);
    //     updating global counters
    updateGlobalCountersAndClear(fbArgs, tensorslice, blockFpConter, blockFnConter, localWorkQueueCounter, localFpConter, localFnConter);


}




template <typename TKKI>
inline __global__ void paddingDilatation(ForBoolKernelArgs<TKKI> fbArgs) {



    thread_block cta = this_thread_block();
    thread_block_tile<32> tile = tiled_partition<32>(cta);

    char* tensorslice;
    bool isBlockFull = true;// usefull to establish do we have block completely filled and no more dilatations possible
    unsigned int old = 0;

    // some references using as aliases
    unsigned int& oldRef = old;



    // main shared memory spaces 
    __shared__ uint32_t sourceShared[32][32];
    __shared__ uint32_t resShared[32][32];
    // holding data about paddings 


    // holding data weather we have anything in padding 0)top  1)bottom, 2)left 3)right, 4)anterior, 5)posterior,
    __shared__ bool isAnythingInPadding[6];
    //variables needed for all threads
    __shared__ unsigned int iterationNumb[1];
    __shared__ unsigned int globalWorkQueueOffset[1];
    __shared__ unsigned int globalWorkQueueCounter[1];
    __shared__ unsigned int localWorkQueueCounter[1];
    __shared__ bool isBlockToBeValidated[1];
    // keeping data wheather gold or segmentation pass should continue - on the basis of global counters

    __shared__ unsigned int localTotalLenthOfWorkQueue[1];
    //counters for per block number of results added in this iteration
    __shared__ unsigned int localFpConter[1];
    __shared__ unsigned int localFnConter[1];

    __shared__ unsigned int blockFpConter[1];
    __shared__ unsigned int blockFnConter[1];

    //result list offset - needed to know where to write a result in a result list
    __shared__ unsigned int resultfpOffset[1];
    __shared__ unsigned int resultfnOffset[1];

    __shared__ unsigned int worQueueStep[1];

    // we will load here multiple entries from workqueue
    __shared__ uint16_t localWorkQueue[localWorkQueLength][4];
    //initializations and loading    
    auto active = coalesced_threads();
    if (isToBeExecutedOnActive(active, 0)) { iterationNumb[0] = getTensorRow<unsigned int>(tensorslice, fbArgs.metaData.minMaxes, 1, 0, 0)[13]; };
    //here we caclulate the offset for given block depending on length of the workqueue and number of the  available blocks in a grid
    // - this will give us number of work queue items per block - we will calculate offset on the basis of the block number

    if (isToBeExecutedOnActive(active, 3)) {
        localWorkQueueCounter[0] = 0;
    };

    if (isToBeExecutedOnActive(active, 4)) {
        blockFpConter[0] = 0;
    };
    if (isToBeExecutedOnActive(active, 5)) {
        blockFnConter[0] = 0;
    };

    if (isToBeExecutedOnActive(active, 6)) {
        localFpConter[0] = 0;
    };
    if (isToBeExecutedOnActive(active, 7)) {
        localFnConter[0] = 0;
    };



    if (isToBeExecutedOnActive(active, 1)) {
        localTotalLenthOfWorkQueue[0] = getTensorRow<unsigned int>(tensorslice, fbArgs.metaData.minMaxes, 1, 0, 0)[9];
        globalWorkQueueOffset[0] = floor((float)(localTotalLenthOfWorkQueue[0] / gridDim.x)) + 1;
        worQueueStep[0] = min(localWorkQueLength, globalWorkQueueOffset[0]);
    };
    sync(cta);
    // TODO - use pipelines as described at 201 in https://docs.nvidia.com/cuda/pdf/CUDA_C_Programming_Guide.pdf
    /// load work QueueData into shared memory 

    //TODO change looping so it will access contigous memory
    for (uint8_t bigloop = blockIdx.x * globalWorkQueueOffset[0]; bigloop < ((blockIdx.x + 1) * globalWorkQueueOffset[0]); bigloop += worQueueStep[0]) {
        // grid stride loop - sadly most of threads will be idle 
        ///////////// loading to work queue
        loadFromGlobalToLocalWorkQueue(fbArgs, tensorslice, localWorkQueue, bigloop, globalWorkQueueOffset, localTotalLenthOfWorkQueue, worQueueStep);

        sync(cta);// now local work queue is populated 

            //now all of the threads in the block needs to have the same i value so we will increment by 1
        for (uint8_t i = 0; i < worQueueStep[0]; i += 1) {
            if (((bigloop + i) < localTotalLenthOfWorkQueue[0]) && ((bigloop + i) < ((blockIdx.x + 1) * globalWorkQueueOffset[0]))) {



                //TODO() remove
       /*         auto activee = coalesced_threads();
                if (isToBeExecutedOnActive(activee, 3)) {
                    printf("\n in padding looping  xMeta %d yMeta %d zMeta %d isGold %d \n"
                        , localWorkQueue[i][0], localWorkQueue[i][1], localWorkQueue[i][2], localWorkQueue[i][3]);
                };*/



                // now we have metadata coordinates we need to start go over associated data block - in order to make it as efficient as possible data block size is set to be the same as datablock size
                // so we do not need iteration loop 

                loadAndDilatateAndSave(fbArgs, tensorslice, localWorkQueue, bigloop, sourceShared, resShared, isAnythingInPadding, iterationNumb, isBlockFull, cta, i,
                    isBlockToBeValidated, localTotalLenthOfWorkQueue, localFpConter, localFnConter, resultfpOffset, resultfnOffset, worQueueStep);

                ///////////////////////// validation if it is to be validated, also we checked for bing full before dilatations - if it was full at the begining - no point in validation
                validateAndUpMetaCounter(fbArgs, tensorslice, localWorkQueue, bigloop, sourceShared, resShared, isAnythingInPadding, iterationNumb, isBlockFull, cta, i,
                    isBlockToBeValidated, localTotalLenthOfWorkQueue, localFpConter, localFnConter, resultfpOffset, resultfnOffset, worQueueStep, oldRef, blockFpConter, blockFnConter);

                sync(cta);
            }
        }
    }
    sync(cta);
    //     updating global counters
    updateGlobalCountersAndClear(fbArgs, tensorslice, blockFpConter, blockFnConter, localWorkQueueCounter, localFpConter, localFnConter);



}