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
#include <cooperative_groups/memcpy_async.h>
#include <cuda/pipeline>
using namespace cooperative_groups;

constexpr auto startOfLocalWorkQ = 372;


template <typename TKKI>
inline __global__ void mainDilatation(ForBoolKernelArgs<TKKI> fbArgs, uint32_t* mainArr, MetaDataGPU metaData
    , unsigned int* minMaxes, uint32_t* workQueue
    , uint32_t* resultListPointerMeta, uint16_t* resultListPointerLocal, uint16_t* resultListPointerIterNumb) {


    thread_block cta = this_thread_block();
    thread_block_tile<32> tile = tiled_partition<32>(cta);
    __shared__ cuda::barrier<cuda::thread_scope::thread_scope_block> barrier;

    bool isBlockFull = true;// usefull to establish do we have block completely filled and no more dilatations possible
    /*
    * according to https://forums.developer.nvidia.com/t/find-the-limit-of-shared-memory-that-can-be-used-per-block/48556 it is good to keep shared memory below 16kb kilo bytes
    main shared memory spaces 
    0-1023 : sourceShmem
    1024-2047 : resShmem
    2048-3071 : first register space
    3072-4095 : second register space
    4096-4468 (372 length) : place for local work queue in dilatation kernels
    */ 
    __shared__ uint32_t mainShmem[4468];
    // holding data about paddings 
        

    // holding data weather we have anything in padding 0)top  1)bottom, 2)left 3)right, 4)anterior, 5)posterior,
    __shared__ bool isAnythingInPadding[6];
    //variables needed for all threads
    __shared__ uint32_t iterationNumb[1];
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
    __shared__ uint32_t isGold[1];

    /* will be used to store all of the minMaxes varibles from global memory (from 7 to 11)
    0 : global FP count; 
    1 : global FN count;
    2 : workQueueCounter
    3 : resultFP globalCounter
    4 : resultFn globalCounter
    */
    __shared__ unsigned int localMinMaxes[5];

    /* will be used to store all of block metadata
  nothing at  0 index 
 1 :fpCount
 2 :fnCount
 3 :fpCounter
 4 :fnCounter
 5 :fpOffset
 6 :fnOffset
 7 :isActiveGold
 8 :isFullGold
 9 :isActiveSegm
 10 :isFullSegm
 11 :isToBeActivatedGold
 12 :isToBeActivatedSegm
 12 :isToBeActivatedSegm
//now linear indexes of the blocks in all sides - if there is no block in given direction it will equal UINT32_MAX
 13 : top
 14 : bottom
 15 : left
 16 : right
 17 : anterior
 18 : posterior
    */
    __shared__ unsigned int localBlockMetaData[19];
    
    //initializations and loading    
    if (tile.thread_rank() == 5 && tile.meta_group_rank() == 0) { iterationNumb[0] = 0; };
    //here we caclulate the offset for given block depending on length of the workqueue and number of the  available blocks in a grid
    // - this will give us number of work queue items per block - we will calculate offset on the basis of the block number

    if (tile.thread_rank() == 6 && tile.meta_group_rank() == 0) {
        localWorkQueueCounter[0] = 0;
    };

    if (tile.thread_rank() == 1 && tile.meta_group_rank() == 0) {
        blockFpConter[0] = 0;
    };
    if (tile.thread_rank() == 2 && tile.meta_group_rank() == 0) {
        blockFnConter[0] = 0;
    };
    if (tile.thread_rank() == 3 && tile.meta_group_rank() == 0) {
        localFpConter[0] = 0;
    };
    if (tile.thread_rank() == 4 && tile.meta_group_rank() == 0) {
        localFnConter[0] = 0;
    };
    if (tile.thread_rank() == 5 && tile.meta_group_rank() == 0) {
        init(&barrier, blockDim.x * blockDim.y);
    };







     if (tile.thread_rank() == 0 && tile.meta_group_rank() == 0) {
        localTotalLenthOfWorkQueue[0] = minMaxes[9];
        globalWorkQueueOffset[0] = floor((float)(localTotalLenthOfWorkQueue[0] / gridDim.x)) + 1;
        worQueueStep[0] = min(localWorkQueLength, globalWorkQueueOffset[0]);
    };
     /* will be used to store all of the minMaxes varibles from global memory (from 7 to 11)
0 : global FP count;
1 : global FN count;
2 : workQueueCounter
3 : resultFP globalCounter
4 : resultFn globalCounter
*/
     if (tile.meta_group_rank() == 1) {
         cooperative_groups::memcpy_async(tile, (&localMinMaxes[0]), (&minMaxes[7]), cuda::aligned_size_t<4>(sizeof(unsigned int) * 5));
     }

    sync(cta);
    // TODO - use pipelines as described at 201 in https://docs.nvidia.com/cuda/pdf/CUDA_C_Programming_Guide.pdf
    /// load work QueueData into shared memory 

    //TODO change looping so it will access contigous memory
    for (uint8_t bigloop = blockIdx.x * globalWorkQueueOffset[0]; bigloop < ((blockIdx.x + 1) * globalWorkQueueOffset[0]); bigloop += worQueueStep[0]) {
        // grid stride loop - sadly most of threads will be idle 
        ///////////// loading to work queue
        cooperative_groups::memcpy_async(cta, (&mainShmem[4096]), (&workQueue[bigloop]), cuda::aligned_size_t<4>(sizeof(uint32_t) * worQueueStep[0]));
        sync(cta);

            //now all of the threads in the block needs to have the same i value so we will increment by 1
        for (uint8_t i = 0; i < worQueueStep[0]; i += 1) {
            if (((bigloop + i) < localTotalLenthOfWorkQueue[0]) && ((bigloop + i) < ((blockIdx.x + 1) * globalWorkQueueOffset[0]))) {
                //preparations fo block 
                if (tile.thread_rank() == 0 && tile.meta_group_rank() == 0) {// this is how it is encoded wheather it is gold or segm block
                    isGold[0] = uint32_t(mainShmem[startOfLocalWorkQ+i] >= UINT16_MAX);
                    if (mainShmem[startOfLocalWorkQ + i] >= UINT16_MAX) {
                        //removing info about wheather it is gold or not pass so we will be able to use it as linear metadata index
                        mainShmem[startOfLocalWorkQ + i] = mainShmem[startOfLocalWorkQ + i] - UINT16_MAX;
                    }
                };
                tile.sync();
                //load meta data of a block to shared memory
                if (tile.meta_group_rank() == 1) {
                    //mainShmem[startOfLocalWorkQ + i] is linindexMeta - linear index pointing out to what metadata block it is
                    cooperative_groups::memcpy_async(tile, (&localBlockMetaData[0]), (&mainArr[  mainShmem[startOfLocalWorkQ + i]*metaData.mainArrSectionLength + metaData.metaDataOffset])
                        , cuda::aligned_size_t<4>(sizeof(uint32_t) *18));
                }


                sync(cta);
                // now we have metadata linear coordinate and information is it gold or segm pass ...

                //loadAndDilatateAndSave(fbArgs, tensorslice, localWorkQueue, bigloop, sourceShared, resShared, isAnythingInPadding, iterationNumb, isBlockFull, cta, i,
                //    isBlockToBeValidated, localTotalLenthOfWorkQueue, localFpConter, localFnConter, resultfpOffset, resultfnOffset, worQueueStep);

                ///////////////////////// validation if it is to be validated, also we checked for bing full before dilatations - if it was full at the begining - no point in validation
                //validateAndUpMetaCounter(fbArgs, tensorslice, localWorkQueue, bigloop, sourceShared, resShared, isAnythingInPadding, iterationNumb, isBlockFull, cta, i,
                //    isBlockToBeValidated, localTotalLenthOfWorkQueue, localFpConter, localFnConter, resultfpOffset, resultfnOffset, worQueueStep,  oldRef, blockFpConter, blockFnConter);

                //////on the basis of isAnythingInPadding we will mark  the neighbouring block as to be activated if there is and if such neighbouring block exists
                //auto activeC = coalesced_threads();

                //if (localWorkQueue[i][3] == 1) {//gold
                //    setNextBlocksActivity(tensorslice, localWorkQueue, i, fbArgs.metaData.isToBeActivatedGold, isAnythingInPadding, activeC);
                //};
                //if (localWorkQueue[i][3] == 0) {//segm
                //    setNextBlocksActivity(tensorslice, localWorkQueue, i, fbArgs.metaData.isToBeActivatedSegm, isAnythingInPadding, activeC);
                //};
                //// marking blocks as full 

                //if (localWorkQueue[i][3] == 1) {//gold
                //    markIsBlockFull(tensorslice, localWorkQueue, i, isBlockFull, fbArgs.metaData.isFullGold, activeC);
                //};
                //if (localWorkQueue[i][3] == 0) {//segm
                //    markIsBlockFull(tensorslice, localWorkQueue, i, isBlockFull, fbArgs.metaData.isFullSegm, activeC);
                //};
                //sync(cta);// all results that should be saved to result list are saved                        

                ////we need to clear isAnythingInPadding to 0
                //clearisAnythingInPadding(isAnythingInPadding);
            }
        }
    }
    sync(cta);
    //     updating global counters
//    updateGlobalCountersAndClear(fbArgs, tensorslice, blockFpConter, blockFnConter, localWorkQueueCounter, localFpConter, localFnConter);


}


//
//
//template <typename TKKI>
//inline __global__ void paddingDilatation(ForBoolKernelArgs<TKKI> fbArgs) {
//
//
//
//    thread_block cta = this_thread_block();
//    thread_block_tile<32> tile = tiled_partition<32>(cta);
//
//    char* tensorslice;
//    bool isBlockFull = true;// usefull to establish do we have block completely filled and no more dilatations possible
//    unsigned int old = 0;
//
//    // some references using as aliases
//    unsigned int& oldRef = old;
//
//
//
//    // main shared memory spaces 
//    __shared__ uint32_t sourceShared[32][32];
//    __shared__ uint32_t resShared[32][32];
//    // holding data about paddings 
//
//
//    // holding data weather we have anything in padding 0)top  1)bottom, 2)left 3)right, 4)anterior, 5)posterior,
//    __shared__ bool isAnythingInPadding[6];
//    //variables needed for all threads
//    __shared__ unsigned int iterationNumb[1];
//    __shared__ unsigned int globalWorkQueueOffset[1];
//    __shared__ unsigned int globalWorkQueueCounter[1];
//    __shared__ unsigned int localWorkQueueCounter[1];
//    __shared__ bool isBlockToBeValidated[1];
//    // keeping data wheather gold or segmentation pass should continue - on the basis of global counters
//
//    __shared__ unsigned int localTotalLenthOfWorkQueue[1];
//    //counters for per block number of results added in this iteration
//    __shared__ unsigned int localFpConter[1];
//    __shared__ unsigned int localFnConter[1];
//
//    __shared__ unsigned int blockFpConter[1];
//    __shared__ unsigned int blockFnConter[1];
//
//    //result list offset - needed to know where to write a result in a result list
//    __shared__ unsigned int resultfpOffset[1];
//    __shared__ unsigned int resultfnOffset[1];
//
//    __shared__ unsigned int worQueueStep[1];
//
//    // we will load here multiple entries from workqueue
//    __shared__ uint16_t localWorkQueue[localWorkQueLength][4];
//    //initializations and loading    
//    auto active = coalesced_threads();
//    if (isToBeExecutedOnActive(active, 0)) { iterationNumb[0] = getTensorRow<unsigned int>(tensorslice, fbArgs.metaData.minMaxes, 1, 0, 0)[13]; };
//    //here we caclulate the offset for given block depending on length of the workqueue and number of the  available blocks in a grid
//    // - this will give us number of work queue items per block - we will calculate offset on the basis of the block number
//
//    if (isToBeExecutedOnActive(active, 3)) {
//        localWorkQueueCounter[0] = 0;
//    };
//
//    if (isToBeExecutedOnActive(active, 4)) {
//        blockFpConter[0] = 0;
//    };
//    if (isToBeExecutedOnActive(active, 5)) {
//        blockFnConter[0] = 0;
//    };
//
//    if (isToBeExecutedOnActive(active, 6)) {
//        localFpConter[0] = 0;
//    };
//    if (isToBeExecutedOnActive(active, 7)) {
//        localFnConter[0] = 0;
//    };
//
//
//
//    if (isToBeExecutedOnActive(active, 1)) {
//        localTotalLenthOfWorkQueue[0] = getTensorRow<unsigned int>(tensorslice, fbArgs.metaData.minMaxes, 1, 0, 0)[9];
//        globalWorkQueueOffset[0] = floor((float)(localTotalLenthOfWorkQueue[0] / gridDim.x)) + 1;
//        worQueueStep[0] = min(localWorkQueLength, globalWorkQueueOffset[0]);
//    };
//    sync(cta);
//    // TODO - use pipelines as described at 201 in https://docs.nvidia.com/cuda/pdf/CUDA_C_Programming_Guide.pdf
//    /// load work QueueData into shared memory 
//
//    //TODO change looping so it will access contigous memory
//    for (uint8_t bigloop = blockIdx.x * globalWorkQueueOffset[0]; bigloop < ((blockIdx.x + 1) * globalWorkQueueOffset[0]); bigloop += worQueueStep[0]) {
//        // grid stride loop - sadly most of threads will be idle 
//        ///////////// loading to work queue
//        loadFromGlobalToLocalWorkQueue(fbArgs, tensorslice, localWorkQueue, bigloop, globalWorkQueueOffset, localTotalLenthOfWorkQueue, worQueueStep);
//
//        sync(cta);// now local work queue is populated 
//
//            //now all of the threads in the block needs to have the same i value so we will increment by 1
//        for (uint8_t i = 0; i < worQueueStep[0]; i += 1) {
//            if (((bigloop + i) < localTotalLenthOfWorkQueue[0]) && ((bigloop + i) < ((blockIdx.x + 1) * globalWorkQueueOffset[0]))) {
//
//
//
//                //TODO() remove
//       /*         auto activee = coalesced_threads();
//                if (isToBeExecutedOnActive(activee, 3)) {
//                    printf("\n in padding looping  xMeta %d yMeta %d zMeta %d isGold %d \n"
//                        , localWorkQueue[i][0], localWorkQueue[i][1], localWorkQueue[i][2], localWorkQueue[i][3]);
//                };*/
//
//
//
//                // now we have metadata coordinates we need to start go over associated data block - in order to make it as efficient as possible data block size is set to be the same as datablock size
//                // so we do not need iteration loop 
//
//                loadAndDilatateAndSave(fbArgs, tensorslice, localWorkQueue, bigloop, sourceShared, resShared, isAnythingInPadding, iterationNumb, isBlockFull, cta, i,
//                    isBlockToBeValidated, localTotalLenthOfWorkQueue, localFpConter, localFnConter, resultfpOffset, resultfnOffset, worQueueStep);
//
//                ///////////////////////// validation if it is to be validated, also we checked for bing full before dilatations - if it was full at the begining - no point in validation
//                validateAndUpMetaCounter(fbArgs, tensorslice, localWorkQueue, bigloop, sourceShared, resShared, isAnythingInPadding, iterationNumb, isBlockFull, cta, i,
//                    isBlockToBeValidated, localTotalLenthOfWorkQueue, localFpConter, localFnConter, resultfpOffset, resultfnOffset, worQueueStep, oldRef, blockFpConter, blockFnConter);
//
//                sync(cta);
//            }
//        }
//    }
//    sync(cta);
//    //     updating global counters
//    updateGlobalCountersAndClear(fbArgs, tensorslice, blockFpConter, blockFnConter, localWorkQueueCounter, localFpConter, localFnConter);
//
//
//    //KROWA!!!
//    //remember to zero out the global work queue counter
//    //and inccrement iterationNumb[1]
//}