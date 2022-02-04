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
template <typename PYO>
__device__ void addToQueue(ForBoolKernelArgs<PYO> fbArgs, unsigned int& old, unsigned int& count, char* tensorslice
    , uint16_t xMeta, uint16_t yMeta, uint16_t zMeta, array3dWithDimsGPU& offsetMetadataArr, array3dWithDimsGPU& countMetadataArr
    , uint16_t isGold, array3dWithDimsGPU& isActiveArr, unsigned int fpFnLocCounter[1], uint16_t localWorkAndOffsetQueue[1600][5], unsigned int localWorkQueueCounter[1]
) {

    count = getTensorRow<unsigned int>(tensorslice, countMetadataArr, countMetadataArr.Ny, yMeta, zMeta)[xMeta];
        //given fp is non zero we need to  add this to local queue
        if (getTensorRow<bool>(tensorslice, isActiveArr, isActiveArr.Ny, yMeta, zMeta)[xMeta]) {
            //we need to establish where to put the entry in the local queue
            //if (count>0) {
            //    printf("\n in add queue count %d xMeta %d yMeta %d zMeta %d \n", count, xMeta, yMeta, zMeta);
            //}
            count = atomicAdd(&fpFnLocCounter[0], count);
            //printf("\n in add queue fpFnLocCounter %d xMeta %d yMeta %d zMeta %d \n", fpFnLocCounter[0], xMeta, yMeta, zMeta);

            old = atomicAdd(&localWorkQueueCounter[0], 1);
            //we check weather we still have space in shared memory
            if (old < 1590) {// so we still have space in shared memory
                localWorkAndOffsetQueue[old][0] = xMeta;
                localWorkAndOffsetQueue[old][1] = yMeta;
                localWorkAndOffsetQueue[old][2] = zMeta;
                localWorkAndOffsetQueue[old][3] = isGold;// marking it is about gold pass - FP
                localWorkAndOffsetQueue[old][4] = count;// marking local offset - this will need to be incremented later by global and local value
            }
            else {// so we do not have any space more in the sared memory  - it is unlikely so we will just in this case save immidiately to global memory
                old = atomicAdd(&(getTensorRow<unsigned int>(tensorslice, fbArgs.metaData.minMaxes, 1, 0, 0)[9]), old);
                getTensorRow<uint16_t>(tensorslice, fbArgs.metaData.workQueue, fbArgs.metaData.workQueue.Ny, 0, 0)[old] = xMeta;
                getTensorRow<uint16_t>(tensorslice, fbArgs.metaData.workQueue, fbArgs.metaData.workQueue.Ny, 1, 0)[old] = yMeta;
                getTensorRow<uint16_t>(tensorslice, fbArgs.metaData.workQueue, fbArgs.metaData.workQueue.Ny, 2, 0)[old] = zMeta;
                getTensorRow<uint16_t>(tensorslice, fbArgs.metaData.workQueue, fbArgs.metaData.workQueue.Ny, 3, 0)[old] = isGold;
                //and offset 
                getTensorRow<unsigned int>(tensorslice, offsetMetadataArr, offsetMetadataArr.Ny, yMeta, zMeta)[xMeta]
                    = atomicAdd(&(getTensorRow<unsigned int>(tensorslice, fbArgs.metaData.minMaxes, 1, 0, 0)[12]), count);
            };


        }
}







#pragma once
template <typename PYO>
__global__ void firstMetaPrepareKernel(ForBoolKernelArgs<PYO> fbArgs) {

    //////initializations
    thread_block cta = this_thread_block();
     char* tensorslice;// needed for iterations over 3d arrays
     unsigned int old = 0;// local variable
     unsigned int count = 0;// local variable
     uint16_t xMeta=0;
     uint16_t yMeta=0;
     uint16_t zMeta=0;
    //local offset counters  for fp and fn's
    __shared__ unsigned int fpFnLocCounter[1];
    // used to store the start position in global memory for whole block
    __shared__ unsigned int globalOffsetForBlock[1];
    __shared__ unsigned int globalWorkQueueCounter[1];
    //used as local work queue counter
    __shared__ unsigned int localWorkQueueCounter[1];     
    //according to https://forums.developer.nvidia.com/t/find-the-limit-of-shared-memory-that-can-be-used-per-block/48556 it is good to keep shared memory below 16kb kilo bytes so it will give us 1600 length of shared memory
    //so here we will store locally the calculated offsets and coordinates of meta data block of intrest marking also wheather we are  talking about gold or segmentation pass (fp or fn )
    __shared__ uint16_t localWorkAndOffsetQueue[1600][5];
    if ((threadIdx.x == 0) && (threadIdx.y == 0)) {
        fpFnLocCounter[0] = 0;
    }
    sync(cta);


    // classical grid stride loop - in case of unlikely event we will run out of space we will empty it prematurly
    //main metadata iteration
    for (uint16_t linIdexMeta = blockIdx.x * blockDim.x + threadIdx.x; linIdexMeta < fbArgs.metaData.totalMetaLength; linIdexMeta += blockDim.x * gridDim.x) {
        //we get from linear index  the coordinates of the metadata block of intrest
        xMeta = linIdexMeta % fbArgs.metaData.metaXLength;
        zMeta = floor((float)(linIdexMeta / (fbArgs.metaData.metaXLength * fbArgs.metaData.MetaYLength)));
        yMeta = floor((float)((linIdexMeta - ((zMeta * fbArgs.metaData.metaXLength * fbArgs.metaData.MetaYLength) + xMeta)) / fbArgs.metaData.metaXLength));
        //we define offsets in the result list to have the results organizedand avoid overwiting

        //TODO remove only debugging    
        //getTensorRow<unsigned int>(tensorslice, fbArgs.forDebugArr, fbArgs.forDebugArr.Ny, yMeta, zMeta)[xMeta] += 1;

        ////gold pass
        //addToQueue(fbArgs, old, count, tensorslice, xMeta, yMeta, zMeta, fbArgs.metaData.fpOffset, fbArgs.metaData.fpCount, 1,fbArgs.metaData.isActiveGold,  fpFnLocCounter, localWorkAndOffsetQueue, localWorkQueueCounter);
        ////segmPass
        //addToQueue(fbArgs, old, count, tensorslice, xMeta, yMeta, zMeta, fbArgs.metaData.fnOffset, fbArgs.metaData.fnCount, 0,fbArgs.metaData.isActiveSegm,  fpFnLocCounter, localWorkAndOffsetQueue, localWorkQueueCounter);
        
        addToQueue(fbArgs, old, count, tensorslice, xMeta, yMeta, zMeta, fbArgs.metaData.fpOffset, fbArgs.metaData.fpCount, 0, fbArgs.metaData.isActiveSegm, fpFnLocCounter, localWorkAndOffsetQueue, localWorkQueueCounter);
        addToQueue(fbArgs, old, count, tensorslice, xMeta, yMeta, zMeta, fbArgs.metaData.fnOffset, fbArgs.metaData.fnCount, 1, fbArgs.metaData.isActiveGold, fpFnLocCounter, localWorkAndOffsetQueue, localWorkQueueCounter);


        }
    sync(cta);
    if ((threadIdx.x == 0) && (threadIdx.y == 0)) {
        globalOffsetForBlock[0] = atomicAdd(&(getTensorRow<unsigned int>(tensorslice, fbArgs.metaData.minMaxes, 1, 0, 0)[12]), (fpFnLocCounter[0]));
       /* if (fpFnLocCounter[0]>0) {
            printf("\n in meta first pass global offset %d  locCounter %d \n  ", globalOffsetForBlock[0], fpFnLocCounter[0]);
        }*/
    };
    if ((threadIdx.x == 1) && (threadIdx.y == 0)) {
        if (localWorkQueueCounter[0]>0) {
            globalWorkQueueCounter[0] = atomicAdd(&(getTensorRow<unsigned int>(tensorslice, fbArgs.metaData.minMaxes, 1, 0, 0)[9]), (localWorkQueueCounter[0]));

         }
    }
    sync(cta);
    //grid stride loop for pushing value from local memory to global 


    for (uint16_t i = threadIdx.x; i < localWorkQueueCounter[0]; i += blockDim.x) {
        
       // printf("addTo %d global Queue xMeta [%d] yMeta [%d] zMeta [%d] isGold %d \n", globalWorkQueueCounter[0] + i, localWorkAndOffsetQueue[i][0], localWorkAndOffsetQueue[i][1], localWorkAndOffsetQueue[i][2], localWorkAndOffsetQueue[i][3]);
        //TODO() instead of copying memory manually better would be to use mempcyasync ...
       // printf("\n saving to local work queue xMeta %d  yMeta %d  zMeta %d  isGold %d   ", localWorkAndOffsetQueue[i][0], localWorkAndOffsetQueue[i][1], localWorkAndOffsetQueue[i][2], localWorkAndOffsetQueue[i][3]);

        getTensorRow<uint16_t>(tensorslice, fbArgs.metaData.workQueue, fbArgs.metaData.workQueue.Ny, 0, 0)[globalWorkQueueCounter[0]+i] = localWorkAndOffsetQueue[i][0];
        getTensorRow<uint16_t>(tensorslice, fbArgs.metaData.workQueue, fbArgs.metaData.workQueue.Ny, 1, 0)[globalWorkQueueCounter[0] + i] = localWorkAndOffsetQueue[i][1];
        getTensorRow<uint16_t>(tensorslice, fbArgs.metaData.workQueue, fbArgs.metaData.workQueue.Ny, 2, 0)[globalWorkQueueCounter[0] + i] = localWorkAndOffsetQueue[i][2];
        getTensorRow<uint16_t>(tensorslice, fbArgs.metaData.workQueue, fbArgs.metaData.workQueue.Ny, 3, 0)[globalWorkQueueCounter[0] + i] = localWorkAndOffsetQueue[i][3];
        //and offset 
        
        //FP pass
        if (localWorkAndOffsetQueue[i][3] == 1) {

          /*  printf("\n in meta first pass saving  offset %d  locCounter  %d xMeta %d yMeta %d zMeta %d \n  ", globalOffsetForBlock[0], fpFnLocCounter[0]
                , localWorkAndOffsetQueue[i][0], localWorkAndOffsetQueue[i][1], localWorkAndOffsetQueue[i][3]);*/

            getTensorRow<unsigned int>(tensorslice, fbArgs.metaData.fpOffset, fbArgs.metaData.fpOffset.Ny, localWorkAndOffsetQueue[i][1], localWorkAndOffsetQueue[i][2])[localWorkAndOffsetQueue[i][0]]
                = localWorkAndOffsetQueue[i][4] + globalOffsetForBlock[0];

        }
        //FN pass
        else {
            getTensorRow<unsigned int>(tensorslice, fbArgs.metaData.fnOffset, fbArgs.metaData.fnOffset.Ny, localWorkAndOffsetQueue[i][1], localWorkAndOffsetQueue[i][2])[localWorkAndOffsetQueue[i][0]]
                = localWorkAndOffsetQueue[i][4] + globalOffsetForBlock[0];
        };




    }

           

    };



    //for (uint8_t xMeta = threadIdx.x; xMeta < krowa; xMeta += blockDim.x) {
    //    for (uint8_t yMeta = threadIdx.y; yMeta < krowa; yMeta += blockDim.y) {
    //            for (uint8_t zMeta = 0; zMeta < krowa; zMeta++) {






    //            }
    //         
    //            sync(cta); // just to reduce the warp divergence

    //                       
    //    }

    //}









//
//
//
//#pragma once
//extern "C" inline bool firstMetaAndBoolRun (ForFullBoolPrepArgs<int> fFArgs) {
//
//
//    cudaError_t syncErr;
//    cudaError_t asyncErr;
//
//
//
//
//
//    //for debugging
//    array3dWithDimsGPU forDebug = allocate3dInGPU(fFArgs.forDebugArr);
//    //main arrays allocations
//    array3dWithDimsGPU goldArr = allocate3dInGPU(fFArgs.goldArr);
//
//    array3dWithDimsGPU segmArr = allocate3dInGPU(fFArgs.segmArr);
//    ////reduced arrays
//    array3dWithDimsGPU reducedGold = allocate3dInGPU(fFArgs.reducedGold);
//    array3dWithDimsGPU reducedSegm = allocate3dInGPU(fFArgs.reducedSegm);
//
//
//
//    array3dWithDimsGPU paddingsStore = allocate3dInGPU(fFArgs.paddingsStore);
//
//
//
//
//
//
//    ForBoolKernelArgs<int> fbArgs = getArgsForKernel<int>(fFArgs, forDebug, goldArr, segmArr, reducedGold, reducedSegm, paddingsStore);
//
//    //preparation kernel
//    boolPrepareKernel << <fFArgs.blocks, fFArgs.threads >> > (fbArgs);
//    //sync
//    checkCuda(cudaDeviceSynchronize(), "just after boolPrepareKernel");
//
//    
//    //here threads one dimensionsonal !!
//    //TODO() reallocate memory - make reduced arrs and metadata smaller - allocate work queue, padding store, result list ...
//
//
//    firstMetaPrepareKernel << <fFArgs.blocksFirstMetaDataPass, fFArgs.threadsFirstMetaDataPass >> > (fbArgs);
//    //sync
//    checkCuda(cudaDeviceSynchronize(), "just after boolPrepareKernel");
//
//
//
//
//    //deviceTohost
//
//    copyDeviceToHost3d(forDebug, fFArgs.forDebugArr);
//
//
//    copyDeviceToHost3d(goldArr, fFArgs.goldArr);
//    copyDeviceToHost3d(segmArr, fFArgs.segmArr);
//
//    copyDeviceToHost3d(reducedGold, fFArgs.reducedGold);
//    copyDeviceToHost3d(reducedSegm, fFArgs.reducedSegm);
//
//
//    copyMetaDataToCPU(fFArgs.metaData, fbArgs.metaData);
//
//
//
//    checkCuda(cudaDeviceSynchronize(), "just after copy device to host");
//    //cudaGetLastError();
//
//    cudaFree(forDebug.arrPStr.ptr);
//    cudaFree(goldArr.arrPStr.ptr);
//    cudaFree(segmArr.arrPStr.ptr);
//    cudaFree(reducedGold.arrPStr.ptr);
//    cudaFree(reducedSegm.arrPStr.ptr);
//
//
//    freeMetaDataGPU(fbArgs.metaData);
//
//
//    /*
// * Catch errors for both the kernel launch above and any
// * errors that occur during the asynchronous `doubleElements`
// * kernel execution.
// */
//
//    syncErr = cudaGetLastError();
//    asyncErr = cudaDeviceSynchronize();
//
//    /*
//     * Print errors should they exist.
//     */
//
//    if (syncErr != cudaSuccess) printf("Error in syncErr: %s\n", cudaGetErrorString(syncErr));
//    if (asyncErr != cudaSuccess) printf("Error in asyncErr: %s\n", cudaGetErrorString(asyncErr));
//
//
//
//    return true;
//}
