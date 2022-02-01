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








//
//
//#pragma once
//template <typename POYO>
//__device__ void addToQueueOtherPasses(ForBoolKernelArgs<POYO> fbArgs, 
//    char* tensorslice, uint8_t xMeta, uint8_t yMeta, uint8_t zMeta, uint8_t isGold, unsigned int fpFnLocCounter[1]
//    , uint8_t localWorkAndOffsetQueue[3000][4], unsigned int localWorkQueueCounter[1], bool metaDataPredicate
//) {
//
//    //given fp is non zero we need to  add this to local queue
//    if (metaDataPredicate) {
//        //we need to establish where to put the entry in the local queue
//        old = atomicAdd(&localWorkQueueCounter[0], 1);
//        //we check weather we still have space in shared memory
//        if (old < 2990) {// so we still have space in shared memory
//            localWorkAndOffsetQueue[old][0] = xMeta;
//            localWorkAndOffsetQueue[old][1] = yMeta;
//            localWorkAndOffsetQueue[old][2] = zMeta;
//            localWorkAndOffsetQueue[old][3] = isGold;// marking it is about gold pass - FP
//        }
//        else {// so we do not have any space more in the sared memory  - it is unlikely so we will just in this case save immidiately to global memory
//            unsigned int old = atomicAdd(&(getTensorRow<unsigned int>(tensorslice, fbArgs.metaData.minMaxes, 1, 0, 0)[9]), old);
//            getTensorRow<uint16_t>(tensorslice, fbArgs.metaData.workQueue, fbArgs.metaData.workQueue.Ny, 0, 0)[old] = xMeta;
//            getTensorRow<uint16_t>(tensorslice, fbArgs.metaData.workQueue, fbArgs.metaData.workQueue.Ny, 1, 0)[old] = yMeta;
//            getTensorRow<uint16_t>(tensorslice, fbArgs.metaData.workQueue, fbArgs.metaData.workQueue.Ny, 2, 0)[old] = zMeta;
//            getTensorRow<uint16_t>(tensorslice, fbArgs.metaData.workQueue, fbArgs.metaData.workQueue.Ny, 3, 0)[old] = isGold;
//        };
//
//        if (isGold == 1) {
//            //so we check is counter smaller than total count
//            getTensorRow<bool>(tensorslice, fbArgs.metaData.isToBeValidatedFp, fbArgs.metaData.isToBeValidatedFp.Ny, yMeta, zMeta)[xMeta]
//                = (getTensorRow<unsigned int>(tensorslice, fbArgs.metaData.fpCounter, fbArgs.metaData.fpCounter.Ny, yMeta, zMeta)[xMeta]
//                    < getTensorRow<unsigned int>(tensorslice, fbArgs.metaData.fpCount, fbArgs.metaData.fpCount.Ny, yMeta, zMeta)[xMeta]);
//
//        }
//        //FN pass
//        else {
//            //so we check is counter smaller than total count
//            getTensorRow<bool>(tensorslice, fbArgs.metaData.isToBeValidatedFn, fbArgs.metaData.isToBeValidatedFn.Ny, yMeta, zMeta)[xMeta]
//                = (getTensorRow<unsigned int>(tensorslice, fbArgs.metaData.fnCounter, fbArgs.metaData.fnCounter.Ny, yMeta, zMeta)[xMeta]
//                    < getTensorRow<unsigned int>(tensorslice, fbArgs.metaData.fnCount, fbArgs.metaData.fnCount.Ny, yMeta, zMeta)[xMeta]);
//        };
//
//
//    }
//}







#pragma once
template <typename PYO>
inline __global__ void getWorkQueeueFromIsToBeActivated(ForBoolKernelArgs<PYO> fbArgs) {

    //////initializations
    thread_block cta = this_thread_block();
    char* tensorslice;// needed for iterations over 3d arrays
    unsigned int count = 0;// local variable

    __shared__ bool isGoldPassToContinue[1];
    __shared__ bool isSegmPassToContinue[1];
    //local offset counters  for fp and fn's
    __shared__ unsigned int fpFnLocCounter[1];
    // used to store the start position in global memory for whole block
    __shared__ unsigned int globalOffsetForBlock[1];
    __shared__ unsigned int globalWorkQueueCounter[1];
    //used as local work queue counter
    __shared__ unsigned int localWorkQueueCounter[1];
    //according to https://forums.developer.nvidia.com/t/find-the-limit-of-shared-memory-that-can-be-used-per-block/48556 it is good to keep shared memory below 16kb kilo bytes so it will give us 1600 length of shared memory
    //so here we will store locally the calculated offsets and coordinates of meta data block of intrest marking also wheather we are  talking about gold or segmentation pass (fp or fn )
    __shared__ uint8_t localWorkAndOffsetQueue[2000][4];
    if ((threadIdx.x == 0) && (threadIdx.y == 0)) {
        fpFnLocCounter[0] = 0;
    }
    if ((threadIdx.x == 0) && (threadIdx.y == 0)) {
        localWorkQueueCounter[0] = 0;
    }

        checkIsToBeDilatated(fbArgs, tensorslice, isGoldPassToContinue, isSegmPassToContinue);
    sync(cta);


    ///////// now we need to look through blocks that we just  activated 
    for (uint16_t linIdexMeta = blockIdx.x * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x; linIdexMeta < fbArgs.metaData.totalMetaLength; linIdexMeta += blockDim.x * blockDim.y * gridDim.x) {
        //we get from linear index  the coordinates of the metadata block of intrest
        uint8_t xMeta = linIdexMeta % fbArgs.metaData.metaXLength;
        uint8_t zMeta = floor((float)(linIdexMeta / (fbArgs.metaData.metaXLength * fbArgs.metaData.MetaYLength)));
        uint8_t yMeta = floor((float)((linIdexMeta - ((zMeta * fbArgs.metaData.metaXLength * fbArgs.metaData.MetaYLength) + xMeta)) / fbArgs.metaData.metaXLength));
        //gold pass

        bool isToBeActivated = isGoldPassToContinue[0] && (getTensorRow<bool>(tensorslice, fbArgs.metaData.isToBeActivatedGold, fbArgs.metaData.isToBeActivatedGold.Ny, yMeta, zMeta)[xMeta]
            && !getTensorRow<bool>(tensorslice, fbArgs.metaData.isActiveGold, fbArgs.metaData.isActiveGold.Ny, yMeta, zMeta)[xMeta]
            && !getTensorRow<bool>(tensorslice, fbArgs.metaData.isFullGold, fbArgs.metaData.isFullGold.Ny, yMeta, zMeta)[xMeta]);


        //given fp is non zero we need to  add this to local queue
        if (isToBeActivated) {
                   //     printf("to be activated pass putting to work queue xMeta %d yMeta %d zMeta %d isGold %d \n", xMeta,yMeta, zMeta, 0 );

            //we need to establish where to put the entry in the local queue
            unsigned int old = atomicAdd(&localWorkQueueCounter[0], 1);
            //we check weather we still have space in shared memory
            if (old < 1990) {// so we still have space in shared memory
      /*          printf( "\naaaa adding to shmem to be activated xMeta %d yMeta %d zMeta %d localWorkQueueCounter %d     \n"
                , xMeta, yMeta, zMeta, localWorkQueueCounter[0] );
*/

                localWorkAndOffsetQueue[old][0] = xMeta;
                localWorkAndOffsetQueue[old][1] = yMeta;
                localWorkAndOffsetQueue[old][2] = zMeta;
                localWorkAndOffsetQueue[old][3] = 1;// marking it is about gold pass - FP
            }
            else {// so we do not have any space more in the sared memory  - it is unlikely so we will just in this case save immidiately to global memory
                old = atomicAdd(&(getTensorRow<unsigned int>(tensorslice, fbArgs.metaData.minMaxes, 1, 0, 0)[9]), old);
                getTensorRow<uint16_t>(tensorslice, fbArgs.metaData.workQueue, fbArgs.metaData.workQueue.Ny, 0, 0)[old] = xMeta;
                getTensorRow<uint16_t>(tensorslice, fbArgs.metaData.workQueue, fbArgs.metaData.workQueue.Ny, 1, 0)[old] = yMeta;
                getTensorRow<uint16_t>(tensorslice, fbArgs.metaData.workQueue, fbArgs.metaData.workQueue.Ny, 2, 0)[old] = zMeta;
                getTensorRow<uint16_t>(tensorslice, fbArgs.metaData.workQueue, fbArgs.metaData.workQueue.Ny, 3, 0)[old] = 1;
            };
            //printf("\n isToBeValidated Fn  %d count %d counter %d     xMeta %d yMeta %d zMeta %d   \n  ",
            //    getTensorRow<unsigned int>(tensorslice, fbArgs.metaData.fnCounter, fbArgs.metaData.fnCounter.Ny, yMeta, zMeta)[xMeta]
            //    < getTensorRow<unsigned int>(tensorslice, fbArgs.metaData.fnCount, fbArgs.metaData.fnCount.Ny, yMeta, zMeta)[xMeta]
            //    , getTensorRow<unsigned int>(tensorslice, fbArgs.metaData.fnCount, fbArgs.metaData.fnCount.Ny, yMeta, zMeta)[xMeta]
            //    , getTensorRow<unsigned int>(tensorslice, fbArgs.metaData.fnCounter, fbArgs.metaData.fnCounter.Ny, yMeta, zMeta)[xMeta]
            //    , xMeta, yMeta, zMeta);

            //printf("\n isToBeValidatedFp %d count %d counter %d     %d xMeta %d yMeta %d zMeta %d \n  ",
            //    getTensorRow<unsigned int>(tensorslice, fbArgs.metaData.fpCounter, fbArgs.metaData.fpCounter.Ny, yMeta, zMeta)[xMeta]
            //    < getTensorRow<unsigned int>(tensorslice, fbArgs.metaData.fpCount, fbArgs.metaData.fpCount.Ny, yMeta, zMeta)[xMeta]
            //    , getTensorRow<unsigned int>(tensorslice, fbArgs.metaData.fpCount, fbArgs.metaData.fpCount.Ny, yMeta, zMeta)[xMeta]
            //    , getTensorRow<unsigned int>(tensorslice, fbArgs.metaData.fpCounter, fbArgs.metaData.fpCounter.Ny, yMeta, zMeta)[xMeta]
            //    , xMeta, yMeta, zMeta);

        }
        //sync(cta);
        //if (isToBeActivated) {


        //         //so we check is counter smaller than total count
        //    getTensorRow<bool>(tensorslice, fbArgs.metaData.isToBeValidatedFn, fbArgs.metaData.isToBeValidatedFn.Ny, yMeta, zMeta)[xMeta]
        //        = (getTensorRow<unsigned int>(tensorslice, fbArgs.metaData.fnCounter, fbArgs.metaData.fnCounter.Ny, yMeta, zMeta)[xMeta]
        //            < getTensorRow<unsigned int>(tensorslice, fbArgs.metaData.fnCount, fbArgs.metaData.fnCount.Ny, yMeta, zMeta)[xMeta]);

        //    getTensorRow<bool>(tensorslice, fbArgs.metaData.isToBeValidatedFp, fbArgs.metaData.isToBeValidatedFp.Ny, yMeta, zMeta)[xMeta]
        //        = (getTensorRow<unsigned int>(tensorslice, fbArgs.metaData.fpCounter, fbArgs.metaData.fpCounter.Ny, yMeta, zMeta)[xMeta]
        //            < getTensorRow<unsigned int>(tensorslice, fbArgs.metaData.fpCount, fbArgs.metaData.fpCount.Ny, yMeta, zMeta)[xMeta]);


        //}



            
        if (isToBeActivated) {
            getTensorRow<bool>(tensorslice, fbArgs.metaData.isToBeActivatedGold, fbArgs.metaData.isToBeActivatedGold.Ny, yMeta, zMeta)[xMeta] = false;
            getTensorRow<bool>(tensorslice, fbArgs.metaData.isActiveGold, fbArgs.metaData.isActiveGold.Ny, yMeta, zMeta)[xMeta] = true;

        }
        //segmPass
        isToBeActivated = isSegmPassToContinue[0] && (getTensorRow<bool>(tensorslice, fbArgs.metaData.isToBeActivatedSegm, fbArgs.metaData.isToBeActivatedSegm.Ny, yMeta, zMeta)[xMeta]
            && !getTensorRow<bool>(tensorslice, fbArgs.metaData.isActiveSegm, fbArgs.metaData.isActiveSegm.Ny, yMeta, zMeta)[xMeta]
            && !getTensorRow<bool>(tensorslice, fbArgs.metaData.isFullSegm, fbArgs.metaData.isFullSegm.Ny, yMeta, zMeta)[xMeta]);
       
        //given fp is non zero we need to  add this to local queue
        if (isToBeActivated) {
          //  printf("to be activated pass putting to work queue xMeta %d yMeta %d zMeta %d isGold %d \n", xMeta, yMeta, zMeta, 0);

            //we need to establish where to put the entry in the local queue
            unsigned int old = atomicAdd(&localWorkQueueCounter[0], 1);
            //we check weather we still have space in shared memory
            if (old < 1990) {// so we still have space in shared memory
                localWorkAndOffsetQueue[old][0] = xMeta;
                localWorkAndOffsetQueue[old][1] = yMeta;
                localWorkAndOffsetQueue[old][2] = zMeta;
                localWorkAndOffsetQueue[old][3] = 0;

               // printf("\naaaa adding to shmem to be activated xMeta %d yMeta %d zMeta %d localWorkQueueCounter %d     \n"
            //        , xMeta, yMeta, zMeta, localWorkQueueCounter[0]);

            }
            else {// so we do not have any space more in the sared memory  - it is unlikely so we will just in this case save immidiately to global memory
               old = atomicAdd(&(getTensorRow<unsigned int>(tensorslice, fbArgs.metaData.minMaxes, 1, 0, 0)[9]), old);
                getTensorRow<uint16_t>(tensorslice, fbArgs.metaData.workQueue, fbArgs.metaData.workQueue.Ny, 0, 0)[old] = xMeta;
                getTensorRow<uint16_t>(tensorslice, fbArgs.metaData.workQueue, fbArgs.metaData.workQueue.Ny, 1, 0)[old] = yMeta;
                getTensorRow<uint16_t>(tensorslice, fbArgs.metaData.workQueue, fbArgs.metaData.workQueue.Ny, 2, 0)[old] = zMeta;
                getTensorRow<uint16_t>(tensorslice, fbArgs.metaData.workQueue, fbArgs.metaData.workQueue.Ny, 3, 0)[old] = 0;
            };

            //printf("\n isToBeValidated Fn  %d count %d counter %d     xMeta %d yMeta %d zMeta %d   \n  ",
            //    getTensorRow<unsigned int>(tensorslice, fbArgs.metaData.fnCounter, fbArgs.metaData.fnCounter.Ny, yMeta, zMeta)[xMeta]
            //    < getTensorRow<unsigned int>(tensorslice, fbArgs.metaData.fnCount, fbArgs.metaData.fnCount.Ny, yMeta, zMeta)[xMeta]
            //    , getTensorRow<unsigned int>(tensorslice, fbArgs.metaData.fnCount, fbArgs.metaData.fnCount.Ny, yMeta, zMeta)[xMeta]
            //    , getTensorRow<unsigned int>(tensorslice, fbArgs.metaData.fnCounter, fbArgs.metaData.fnCounter.Ny, yMeta, zMeta)[xMeta]
            //, xMeta, yMeta, zMeta);

            //printf("\n isToBeValidatedFp %d count %d counter %d     %d xMeta %d yMeta %d zMeta %d \n  ",
            //    getTensorRow<unsigned int>(tensorslice, fbArgs.metaData.fpCounter, fbArgs.metaData.fpCounter.Ny, yMeta, zMeta)[xMeta]
            //    < getTensorRow<unsigned int>(tensorslice, fbArgs.metaData.fpCount, fbArgs.metaData.fpCount.Ny, yMeta, zMeta)[xMeta]
            //    , getTensorRow<unsigned int>(tensorslice, fbArgs.metaData.fpCount, fbArgs.metaData.fpCount.Ny, yMeta, zMeta)[xMeta]
            //    , getTensorRow<unsigned int>(tensorslice, fbArgs.metaData.fpCounter, fbArgs.metaData.fpCounter.Ny, yMeta, zMeta)[xMeta]
            //    , xMeta, yMeta, zMeta);
        }
        //sync(cta);
        //if (isToBeActivated) {

        //        //so we check is counter smaller than total count
        //        getTensorRow<bool>(tensorslice, fbArgs.metaData.isToBeValidatedFn, fbArgs.metaData.isToBeValidatedFn.Ny, yMeta, zMeta)[xMeta]
        //            = (getTensorRow<unsigned int>(tensorslice, fbArgs.metaData.fnCounter, fbArgs.metaData.fnCounter.Ny, yMeta, zMeta)[xMeta]
        //                < getTensorRow<unsigned int>(tensorslice, fbArgs.metaData.fnCount, fbArgs.metaData.fnCount.Ny, yMeta, zMeta)[xMeta]);

        //        getTensorRow<bool>(tensorslice, fbArgs.metaData.isToBeValidatedFp, fbArgs.metaData.isToBeValidatedFp.Ny, yMeta, zMeta)[xMeta]
        //            = (getTensorRow<unsigned int>(tensorslice, fbArgs.metaData.fpCounter, fbArgs.metaData.fpCounter.Ny, yMeta, zMeta)[xMeta]
        //                < getTensorRow<unsigned int>(tensorslice, fbArgs.metaData.fpCount, fbArgs.metaData.fpCount.Ny, yMeta, zMeta)[xMeta]);

        //}



        if (isToBeActivated) {
            getTensorRow<bool>(tensorslice, fbArgs.metaData.isToBeActivatedSegm, fbArgs.metaData.isToBeActivatedSegm.Ny, yMeta, zMeta)[xMeta] = false;
            getTensorRow<bool>(tensorslice, fbArgs.metaData.isActiveSegm, fbArgs.metaData.isActiveSegm.Ny, yMeta, zMeta)[xMeta] = true;

            //printf("\n found to be actvated xMeta %d yMeta %d zMeta %d isGold  %d isSegmPassToContinue[0] %d  isActive %d isFull %d \n ", xMeta, yMeta, zMeta, 0, isSegmPassToContinue[0], getTensorRow<bool>(tensorslice, fbArgs.metaData.isActiveSegm
            //    , fbArgs.metaData.isActiveSegm.Ny, yMeta, zMeta)[xMeta], getTensorRow<bool>(tensorslice, fbArgs.metaData.isFullSegm, fbArgs.metaData.isFullSegm.Ny, yMeta, zMeta)[xMeta]);
        }
    }
    sync(cta);
    if ((threadIdx.x == 1) && (threadIdx.y == 0)) {
        if (localWorkQueueCounter[0] > 0) {
            globalWorkQueueCounter[0] = atomicAdd(&(getTensorRow<unsigned int>(tensorslice, fbArgs.metaData.minMaxes, 1, 0, 0)[9]), (localWorkQueueCounter[0]));
           // printf(" \n globalWorkQueueCounter in looking for padding blocks %d   \n ", globalWorkQueueCounter[0]);
        }
    }
    sync(cta);
    //grid stride loop for pushing value from local memory to global 


    for (uint16_t i = threadIdx.x; i < localWorkQueueCounter[0]; i += blockDim.x) {

        // printf("addTo %d global Queue xMeta [%d] yMeta [%d] zMeta [%d] isGold %d \n", globalWorkQueueCounter[0] + i, localWorkAndOffsetQueue[i][0], localWorkAndOffsetQueue[i][1], localWorkAndOffsetQueue[i][2], localWorkAndOffsetQueue[i][3]);
         //TODO() instead of copying memory manually better would be to use mempcyasync ...
        // printf("\n saving to local work queue xMeta %d  yMeta %d  zMeta %d  isGold %d   ", localWorkAndOffsetQueue[i][0], localWorkAndOffsetQueue[i][1], localWorkAndOffsetQueue[i][2], localWorkAndOffsetQueue[i][3]);

        getTensorRow<uint16_t>(tensorslice, fbArgs.metaData.workQueue, fbArgs.metaData.workQueue.Ny, 0, 0)[globalWorkQueueCounter[0] + i] = localWorkAndOffsetQueue[i][0];
        getTensorRow<uint16_t>(tensorslice, fbArgs.metaData.workQueue, fbArgs.metaData.workQueue.Ny, 1, 0)[globalWorkQueueCounter[0] + i] = localWorkAndOffsetQueue[i][1];
        getTensorRow<uint16_t>(tensorslice, fbArgs.metaData.workQueue, fbArgs.metaData.workQueue.Ny, 2, 0)[globalWorkQueueCounter[0] + i] = localWorkAndOffsetQueue[i][2];
        getTensorRow<uint16_t>(tensorslice, fbArgs.metaData.workQueue, fbArgs.metaData.workQueue.Ny, 3, 0)[globalWorkQueueCounter[0] + i] = localWorkAndOffsetQueue[i][3];
        //and offset 

    }


}

#pragma once
template <typename PYO>
inline __global__ void getWorkQueeueFromActive_mainPass(ForBoolKernelArgs<PYO> fbArgs) {
    //////initializations
    thread_block cta = this_thread_block();
    char* tensorslice;// needed for iterations over 3d arrays
    unsigned int count = 0;// local variable

    __shared__ bool isGoldPassToContinue[1];
    __shared__ bool isSegmPassToContinue[1];
    //local offset counters  for fp and fn's
    __shared__ unsigned int fpFnLocCounter[1];
    // used to store the start position in global memory for whole block
    __shared__ unsigned int globalOffsetForBlock[1];
    __shared__ unsigned int globalWorkQueueCounter[1];
    //used as local work queue counter
    __shared__ unsigned int localWorkQueueCounter[1];
    //according to https://forums.developer.nvidia.com/t/find-the-limit-of-shared-memory-that-can-be-used-per-block/48556 it is good to keep shared memory below 16kb kilo bytes so it will give us 1600 length of shared memory
    //so here we will store locally the calculated offsets and coordinates of meta data block of intrest marking also wheather we are  talking about gold or segmentation pass (fp or fn )
    __shared__ uint8_t localWorkAndOffsetQueue[2000][4];
    if ((threadIdx.x == 0) && (threadIdx.y == 0)) {
        fpFnLocCounter[0] = 0;
    }
    if ((threadIdx.x == 0) && (threadIdx.y == 0)) {
        localWorkQueueCounter[0] = 0;
    }

    checkIsToBeDilatated(fbArgs, tensorslice, isGoldPassToContinue, isSegmPassToContinue);
    sync(cta);


    ///////// now we need to look through blocks that we just  activated 
    for (uint16_t linIdexMeta = blockIdx.x * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x; linIdexMeta < fbArgs.metaData.totalMetaLength; linIdexMeta += blockDim.x * blockDim.y * gridDim.x) {
        //we get from linear index  the coordinates of the metadata block of intrest
        uint8_t xMeta = linIdexMeta % fbArgs.metaData.metaXLength;
        uint8_t zMeta = floor((float)(linIdexMeta / (fbArgs.metaData.metaXLength * fbArgs.metaData.MetaYLength)));
        uint8_t yMeta = floor((float)((linIdexMeta - ((zMeta * fbArgs.metaData.metaXLength * fbArgs.metaData.MetaYLength) + xMeta)) / fbArgs.metaData.metaXLength));
        //gold pass

        bool isToBeActivated = isGoldPassToContinue[0] && (getTensorRow<bool>(tensorslice, fbArgs.metaData.isActiveGold, fbArgs.metaData.isActiveGold.Ny, yMeta, zMeta)[xMeta]
            && !getTensorRow<bool>(tensorslice, fbArgs.metaData.isFullGold, fbArgs.metaData.isFullGold.Ny, yMeta, zMeta)[xMeta]);




        //given fp is non zero we need to  add this to local queue
        if (isToBeActivated) {
            //we need to establish where to put the entry in the local queue
            unsigned int old = atomicAdd(&localWorkQueueCounter[0], 1);
            //printf("main pass putting to work queue xMeta %d yMeta %d zMeta %d isGold %d \n", xMeta, yMeta, zMeta, 1);
            //we check weather we still have space in shared memory
            if (old < 1990) {// so we still have space in shared memory
                localWorkAndOffsetQueue[old][0] = xMeta;
                localWorkAndOffsetQueue[old][1] = yMeta;
                localWorkAndOffsetQueue[old][2] = zMeta;
                localWorkAndOffsetQueue[old][3] = 1;// marking it is about gold pass - FP
            }
            else {// so we do not have any space more in the sared memory  - it is unlikely so we will just in this case save immidiately to global memory
                old = atomicAdd(&(getTensorRow<unsigned int>(tensorslice, fbArgs.metaData.minMaxes, 1, 0, 0)[9]), old);
                getTensorRow<uint16_t>(tensorslice, fbArgs.metaData.workQueue, fbArgs.metaData.workQueue.Ny, 0, 0)[old] = xMeta;
                getTensorRow<uint16_t>(tensorslice, fbArgs.metaData.workQueue, fbArgs.metaData.workQueue.Ny, 1, 0)[old] = yMeta;
                getTensorRow<uint16_t>(tensorslice, fbArgs.metaData.workQueue, fbArgs.metaData.workQueue.Ny, 2, 0)[old] = zMeta;
                getTensorRow<uint16_t>(tensorslice, fbArgs.metaData.workQueue, fbArgs.metaData.workQueue.Ny, 3, 0)[old] = 1;
            };
            //printf("\n isToBeValidated Fn  %d count %d counter %d     xMeta %d yMeta %d zMeta %d   \n  ",
            //    getTensorRow<unsigned int>(tensorslice, fbArgs.metaData.fnCounter, fbArgs.metaData.fnCounter.Ny, yMeta, zMeta)[xMeta]
            //    < getTensorRow<unsigned int>(tensorslice, fbArgs.metaData.fnCount, fbArgs.metaData.fnCount.Ny, yMeta, zMeta)[xMeta]
            //    , getTensorRow<unsigned int>(tensorslice, fbArgs.metaData.fnCount, fbArgs.metaData.fnCount.Ny, yMeta, zMeta)[xMeta]
            //    , getTensorRow<unsigned int>(tensorslice, fbArgs.metaData.fnCounter, fbArgs.metaData.fnCounter.Ny, yMeta, zMeta)[xMeta]
            //    , xMeta, yMeta, zMeta);

            //printf("\n isToBeValidatedFp %d count %d counter %d     %d xMeta %d yMeta %d zMeta %d \n  ",
            //    getTensorRow<unsigned int>(tensorslice, fbArgs.metaData.fpCounter, fbArgs.metaData.fpCounter.Ny, yMeta, zMeta)[xMeta]
            //    < getTensorRow<unsigned int>(tensorslice, fbArgs.metaData.fpCount, fbArgs.metaData.fpCount.Ny, yMeta, zMeta)[xMeta]
            //    , getTensorRow<unsigned int>(tensorslice, fbArgs.metaData.fpCount, fbArgs.metaData.fpCount.Ny, yMeta, zMeta)[xMeta]
            //    , getTensorRow<unsigned int>(tensorslice, fbArgs.metaData.fpCounter, fbArgs.metaData.fpCounter.Ny, yMeta, zMeta)[xMeta]
            //    , xMeta, yMeta, zMeta);
        }
        //sync(cta);
        //if (isToBeActivated) {

        //    //so we check is counter smaller than total count
        //    getTensorRow<bool>(tensorslice, fbArgs.metaData.isToBeValidatedFn, fbArgs.metaData.isToBeValidatedFn.Ny, yMeta, zMeta)[xMeta]
        //        = (getTensorRow<unsigned int>(tensorslice, fbArgs.metaData.fnCounter, fbArgs.metaData.fnCounter.Ny, yMeta, zMeta)[xMeta]
        //            < getTensorRow<unsigned int>(tensorslice, fbArgs.metaData.fnCount, fbArgs.metaData.fnCount.Ny, yMeta, zMeta)[xMeta]);

        //    getTensorRow<bool>(tensorslice, fbArgs.metaData.isToBeValidatedFp, fbArgs.metaData.isToBeValidatedFp.Ny, yMeta, zMeta)[xMeta]
        //        = (getTensorRow<unsigned int>(tensorslice, fbArgs.metaData.fpCounter, fbArgs.metaData.fpCounter.Ny, yMeta, zMeta)[xMeta]
        //            < getTensorRow<unsigned int>(tensorslice, fbArgs.metaData.fpCount, fbArgs.metaData.fpCount.Ny, yMeta, zMeta)[xMeta]);


        //}

        //segmPass
        isToBeActivated = isSegmPassToContinue[0] && (getTensorRow<bool>(tensorslice, fbArgs.metaData.isActiveSegm, fbArgs.metaData.isActiveSegm.Ny, yMeta, zMeta)[xMeta]
            && !getTensorRow<bool>(tensorslice, fbArgs.metaData.isFullSegm, fbArgs.metaData.isFullSegm.Ny, yMeta, zMeta)[xMeta]);

        //given fp is non zero we need to  add this to local queue
        if (isToBeActivated) {
            //we need to establish where to put the entry in the local queue
            unsigned int old = atomicAdd(&localWorkQueueCounter[0], 1);
           // printf("main pass putting to work queue xMeta %d yMeta %d zMeta %d isGold %d \n", xMeta,yMeta, zMeta, 0 );
            //we check weather we still have space in shared memory
            if (old < 1990) {// so we still have space in shared memory
                localWorkAndOffsetQueue[old][0] = xMeta;
                localWorkAndOffsetQueue[old][1] = yMeta;
                localWorkAndOffsetQueue[old][2] = zMeta;
                localWorkAndOffsetQueue[old][3] = 0;
            }
            else {// so we do not have any space more in the sared memory  - it is unlikely so we will just in this case save immidiately to global memory
                old = atomicAdd(&(getTensorRow<unsigned int>(tensorslice, fbArgs.metaData.minMaxes, 1, 0, 0)[9]), old);
                getTensorRow<uint16_t>(tensorslice, fbArgs.metaData.workQueue, fbArgs.metaData.workQueue.Ny, 0, 0)[old] = xMeta;
                getTensorRow<uint16_t>(tensorslice, fbArgs.metaData.workQueue, fbArgs.metaData.workQueue.Ny, 1, 0)[old] = yMeta;
                getTensorRow<uint16_t>(tensorslice, fbArgs.metaData.workQueue, fbArgs.metaData.workQueue.Ny, 2, 0)[old] = zMeta;
                getTensorRow<uint16_t>(tensorslice, fbArgs.metaData.workQueue, fbArgs.metaData.workQueue.Ny, 3, 0)[old] = 0;
            };

            //printf("\n isToBeValidated Fn  %d count %d counter %d     xMeta %d yMeta %d zMeta %d   \n  ",
            //    getTensorRow<unsigned int>(tensorslice, fbArgs.metaData.fnCounter, fbArgs.metaData.fnCounter.Ny, yMeta, zMeta)[xMeta]
            //    < getTensorRow<unsigned int>(tensorslice, fbArgs.metaData.fnCount, fbArgs.metaData.fnCount.Ny, yMeta, zMeta)[xMeta]
            //    , getTensorRow<unsigned int>(tensorslice, fbArgs.metaData.fnCount, fbArgs.metaData.fnCount.Ny, yMeta, zMeta)[xMeta]
            //    , getTensorRow<unsigned int>(tensorslice, fbArgs.metaData.fnCounter, fbArgs.metaData.fnCounter.Ny, yMeta, zMeta)[xMeta]
            //, xMeta, yMeta, zMeta);

            //printf("\n isToBeValidatedFp %d count %d counter %d     %d xMeta %d yMeta %d zMeta %d \n  ",
            //    getTensorRow<unsigned int>(tensorslice, fbArgs.metaData.fpCounter, fbArgs.metaData.fpCounter.Ny, yMeta, zMeta)[xMeta]
            //    < getTensorRow<unsigned int>(tensorslice, fbArgs.metaData.fpCount, fbArgs.metaData.fpCount.Ny, yMeta, zMeta)[xMeta]
            //    , getTensorRow<unsigned int>(tensorslice, fbArgs.metaData.fpCount, fbArgs.metaData.fpCount.Ny, yMeta, zMeta)[xMeta]
            //    , getTensorRow<unsigned int>(tensorslice, fbArgs.metaData.fpCounter, fbArgs.metaData.fpCounter.Ny, yMeta, zMeta)[xMeta]
            //    , xMeta, yMeta, zMeta);

        }
        //sync(cta);
        //if (isToBeActivated) {

        //    //so we check is counter smaller than total count
        //    getTensorRow<bool>(tensorslice, fbArgs.metaData.isToBeValidatedFn, fbArgs.metaData.isToBeValidatedFn.Ny, yMeta, zMeta)[xMeta]
        //        = (getTensorRow<unsigned int>(tensorslice, fbArgs.metaData.fnCounter, fbArgs.metaData.fnCounter.Ny, yMeta, zMeta)[xMeta]
        //            < getTensorRow<unsigned int>(tensorslice, fbArgs.metaData.fnCount, fbArgs.metaData.fnCount.Ny, yMeta, zMeta)[xMeta]);

        //    getTensorRow<bool>(tensorslice, fbArgs.metaData.isToBeValidatedFp, fbArgs.metaData.isToBeValidatedFp.Ny, yMeta, zMeta)[xMeta]
        //        = (getTensorRow<unsigned int>(tensorslice, fbArgs.metaData.fpCounter, fbArgs.metaData.fpCounter.Ny, yMeta, zMeta)[xMeta]
        //            < getTensorRow<unsigned int>(tensorslice, fbArgs.metaData.fpCount, fbArgs.metaData.fpCount.Ny, yMeta, zMeta)[xMeta]);

        //}


    }
    sync(cta);
    if ((threadIdx.x == 1) && (threadIdx.y == 0)) {
        if (localWorkQueueCounter[0] > 0) {
            globalWorkQueueCounter[0] = atomicAdd(&(getTensorRow<unsigned int>(tensorslice, fbArgs.metaData.minMaxes, 1, 0, 0)[9]), (localWorkQueueCounter[0]));

        }
    }
    sync(cta);
    //grid stride loop for pushing value from local memory to global 


    for (uint16_t i = threadIdx.x; i < localWorkQueueCounter[0]; i += blockDim.x) {

        // printf("addTo %d global Queue xMeta [%d] yMeta [%d] zMeta [%d] isGold %d \n", globalWorkQueueCounter[0] + i, localWorkAndOffsetQueue[i][0], localWorkAndOffsetQueue[i][1], localWorkAndOffsetQueue[i][2], localWorkAndOffsetQueue[i][3]);
         //TODO() instead of copying memory manually better would be to use mempcyasync ...
        // printf("\n saving to local work queue xMeta %d  yMeta %d  zMeta %d  isGold %d   ", localWorkAndOffsetQueue[i][0], localWorkAndOffsetQueue[i][1], localWorkAndOffsetQueue[i][2], localWorkAndOffsetQueue[i][3]);

        getTensorRow<uint16_t>(tensorslice, fbArgs.metaData.workQueue, fbArgs.metaData.workQueue.Ny, 0, 0)[globalWorkQueueCounter[0] + i] = localWorkAndOffsetQueue[i][0];
        getTensorRow<uint16_t>(tensorslice, fbArgs.metaData.workQueue, fbArgs.metaData.workQueue.Ny, 1, 0)[globalWorkQueueCounter[0] + i] = localWorkAndOffsetQueue[i][1];
        getTensorRow<uint16_t>(tensorslice, fbArgs.metaData.workQueue, fbArgs.metaData.workQueue.Ny, 2, 0)[globalWorkQueueCounter[0] + i] = localWorkAndOffsetQueue[i][2];
        getTensorRow<uint16_t>(tensorslice, fbArgs.metaData.workQueue, fbArgs.metaData.workQueue.Ny, 3, 0)[globalWorkQueueCounter[0] + i] = localWorkAndOffsetQueue[i][3];
        //and offset 

    }






}
























/*

#pragma once
template <typename PYOPP>
inline __device__ void getValueOfLocalWorQ(ForBoolKernelArgs<PYOPP> fbArgs, uint8_t  subSpot,  uint32_t sourceShared[32][32], uint32_t resShared[32][32]
    , uint16_t localWorkQueue[localWorkQueLength][4], uint16_t& i, unsigned int globalWorkQueueCounter[1], char* tensorslice){
  if( (i>>5)==0){ // using local work queue
     // remainder div 16 + is oddd times 16 ...
      getTensorRow<uint16_t>(tensorslice, fbArgs.metaData.workQueue, fbArgs.metaData.workQueue.Ny, subSpot, 0)[globalWorkQueueCounter[0] + i]= localWorkQueue[((i & (15)) + 16 * (((i >> 5) & 1)))][subSpot];
  }else if((i>>5)< 5){// using source shmem
      getTensorRow<uint16_t>(tensorslice, fbArgs.metaData.workQueue, fbArgs.metaData.workQueue.Ny, subSpot, 0)[globalWorkQueueCounter[0] + i]=  sourceShared[((i & (15)) + 16 * (((i >> 5) & 1)))][subSpot + ((i >> 5) - 1) * 4];
  }else{// using resshmem
      getTensorRow<uint16_t>(tensorslice, fbArgs.metaData.workQueue, fbArgs.metaData.workQueue.Ny, subSpot, 0)[globalWorkQueueCounter[0] + i]= resShared[((i & (15)) + 16 * (((i >> 5) & 1)))][subSpot + ((i >> 5) - 5) * 4] ;
  }

};
#pragma once
inline __device__ void setValueOfLocalWorQ(unsigned int spot, uint8_t  subSpot, uint8_t value, uint32_t sourceShared[32][32], uint32_t resShared[32][32]
    , uint16_t localWorkQueue[localWorkQueLength][4]  ) {
    if ((spot >> 5) == 0) { // using local work queue
       // remainder div 16 + is oddd times 16 ...
        localWorkQueue[((spot & (15)) + 16 * (((spot >> 5) & 1)))][subSpot] = value;
    }
    else if ((spot >> 5) < 5) {// using source shmem
        sourceShared[((spot & (15)) + 16 * (((spot >> 5) & 1)))][subSpot + ((spot >> 5) - 1) * 4] = value;
    }
    else {// using resshmem
        resShared[((spot & (15)) + 16 * (((spot >> 5) & 1)))][subSpot + ((spot >> 5) - 5) * 4] = value;
    }

};


*/




/*#pragma once
template <typename PYO>
inline __device__ void addToQueueOtherPasses(ForBoolKernelArgs<PYO> fbArgs
, unsigned int& old, char* tensorslice
    , uint8_t& xMeta, uint8_t& yMeta, uint8_t& zMeta    , uint8_t isGold
      , uint16_t localWorkQueue[30][4], unsigned int localWorkQueueCounter[1], uint32_t sourceShared[32][32], uint32_t resShared[32][32],
    bool metaDataPredicate
) {

        if (metaDataPredicate) {
            //we need to establish where to put the entry in the local queue
            old = atomicAdd(&localWorkQueueCounter[0], 1);
           // printf("\n saving to shmem xMeta %d yMeta %d zMeta %d  isGold %d \n" , xMeta, yMeta, zMeta, isGold);
            //we check weather we still have space in shared memory
              if (old < totalCombinedShmemWorkQueue) {// so we still have space in shared memory
                  setValueOfLocalWorQ(old, 0, xMeta, sourceShared, resShared, localWorkQueue);
                  setValueOfLocalWorQ(old, 1, yMeta, sourceShared, resShared, localWorkQueue);
                  setValueOfLocalWorQ(old, 2, zMeta, sourceShared, resShared, localWorkQueue);
                  setValueOfLocalWorQ(old, 3, isGold, sourceShared, resShared, localWorkQueue);
                }
                else {// so we do not have any space more in the shared memory  -
                old = atomicAdd(&(getTensorRow<unsigned int>(tensorslice, fbArgs.metaData.minMaxes, 1, 0, 0)[9]), old);
                getTensorRow<uint16_t>(tensorslice, fbArgs.metaData.workQueue, fbArgs.metaData.workQueue.Ny, 0, 0)[old] = xMeta;
                getTensorRow<uint16_t>(tensorslice, fbArgs.metaData.workQueue, fbArgs.metaData.workQueue.Ny, 1, 0)[old] = yMeta;
                getTensorRow<uint16_t>(tensorslice, fbArgs.metaData.workQueue, fbArgs.metaData.workQueue.Ny, 2, 0)[old] = zMeta;
                getTensorRow<uint16_t>(tensorslice, fbArgs.metaData.workQueue, fbArgs.metaData.workQueue.Ny, 3, 0)[old] = isGold;
            };

       if (isGold == 1) {
           //so we check is counter smaller than total count
            getTensorRow<bool>(tensorslice, fbArgs.metaData.isToBeValidatedFp, fbArgs.metaData.isToBeValidatedFp.Ny, yMeta, zMeta)[xMeta]
                = (getTensorRow<unsigned int>(tensorslice, fbArgs.metaData.fpCounter, fbArgs.metaData.fpCounter.Ny, yMeta, zMeta)[xMeta] 
                    < getTensorRow<unsigned int>(tensorslice, fbArgs.metaData.fpCount, fbArgs.metaData.fpCount.Ny, yMeta, zMeta)[xMeta]);
            
        }
        //FN pass
        else {
           //so we check is counter smaller than total count
           getTensorRow<bool>(tensorslice, fbArgs.metaData.isToBeValidatedFn, fbArgs.metaData.isToBeValidatedFn.Ny, yMeta, zMeta)[xMeta]
               = (getTensorRow<unsigned int>(tensorslice, fbArgs.metaData.fnCounter, fbArgs.metaData.fnCounter.Ny, yMeta, zMeta)[xMeta]
                   < getTensorRow<unsigned int>(tensorslice, fbArgs.metaData.fnCount, fbArgs.metaData.fnCount.Ny, yMeta, zMeta)[xMeta]);
        };




        }
}







#pragma once
template <typename PKKYO>
inline __device__ void fromShmemToGlobalWorkQueue(ForBoolKernelArgs<PKKYO> fbArgs , unsigned int& old, uint16_t& i, uint32_t sourceShared[32][32], uint32_t resShared[32][32]
    , uint16_t localWorkQueue[localWorkQueLength][4], unsigned int globalWorkQueueCounter[1], char* tensorslice, unsigned int localWorkQueueCounter[1]) {
    for (i = threadIdx.x; i < localWorkQueueCounter[0]; i += blockDim.x) {
       // printf("\n loading from shmem xMeta %d yMeta %d zMeta %d  isGold %d \n", localWorkQueue[i][0], localWorkQueue[i][1], localWorkQueue[i][2], localWorkQueue[i][3]);
        getValueOfLocalWorQ(fbArgs, 0, sourceShared, resShared, localWorkQueue, i, globalWorkQueueCounter, tensorslice);
        getValueOfLocalWorQ(fbArgs, 1, sourceShared, resShared, localWorkQueue, i, globalWorkQueueCounter, tensorslice);
        getValueOfLocalWorQ(fbArgs, 2, sourceShared, resShared, localWorkQueue, i, globalWorkQueueCounter, tensorslice);
        getValueOfLocalWorQ(fbArgs, 3, sourceShared, resShared, localWorkQueue, i, globalWorkQueueCounter, tensorslice);

    }
}




*/