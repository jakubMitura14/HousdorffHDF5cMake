#pragma once


#include "cuda_runtime.h"
#include "CPUAllocations.cu"
#include "MetaData.cu"
#include "IterationUtils.cu"
#include "ExceptionManagUtils.cu"
#include "CooperativeGroupsUtils.cu"
#include "MainPassFunctions.cu"
#include <cstdint>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cooperative_groups/memcpy_async.h>
#include <cuda/pipeline>

using namespace cooperative_groups;

/*
load and dilatates the entries in gold or segm ...
all operations are on the single data block represented by single entry in metadata 

1) load data into pipeline head source shmem from either gold or segmentation array - and either ref or prev depending on wheather iteration number is odd or not ...
2) a) compute dilatations of souce and save to rs shmem; 
   b) also mark up and bottom of is anythink in padding
   d) simultaneously pipeline should load the  data from the block above (if it exist) to register shmem  one
3) dilatate from block above save to resshmem and simultaneously load data from block below and save to register shmem two
commit so we will have register shmem one free
4) dilatate from below  and simulatenously using 4 tiles we load into register one the padding info required for dilatations - anterior, posterior, left, right
5) we dilatate anterior, posterior, left, right we need to have registers 2 cleared  to use and load to it the data from reduced gold or segm (originals) if it is to be validated - if not we skip this
   b) mark is block full - if it is all of the resshmem entries are equal UINT32_MAX
6) if it was to be validated we compare resshmem to loaded data and write down results
7) save data from resshmem to global memory
8) a) in case of non padding pass we use the data from is anything in padding  to activate neighbouring blocks
   b) save the updated values of block metadata back to global memory
*/
#pragma once
template <typename TXTOIO>
inline __device__ void loadAndDilatateAndSave(ForBoolKernelArgs<TXTOIO> fbArgs, char* tensorslice,
    uint16_t localWorkQueue[localWorkQueLength][4], uint8_t bigloop,
    uint32_t sourceShared[32][32], uint32_t resShared[32][32]
    , bool isAnythingInPadding[6], unsigned int iterationNumb[1], bool& isBlockFull, thread_block cta, uint16_t i
    , bool isBlockToBeValidated[1], unsigned int localTotalLenthOfWorkQueue[1], unsigned int localFpConter[1], unsigned int localFnConter[1]
    , unsigned int resultfpOffset[1], unsigned int resultfnOffset[1], unsigned int worQueueStep[1]
    , uint32_t* mainArr, MetaDataGPU metaData , unsigned int* minMaxes, uint32_t* workQueue, unsigned int localMinMaxes[5], unsigned int localBlockMetaData[19]
    , uint32_t mainShmem[4468], uint32_t isGold[1]
    , cuda::barrier<cuda::thread_scope::thread_scope_block> barrier
) {

    /*
     main shared memory spaces reference 
    0-1023 : sourceShmem
    1024-2047 : resShmem
    2048-3071 : first register space
    3072-4095 : second register space
    4096-4468 (372 length) : place for local work queue in dilatation kernels
    */
    //we use isGold[0] and iteration number to establish what we need to load
    //(iterationNumb[0] & 1) will evaluate to 1 for odd iteration rest of calculation will lead to correct list for given combination
    
    //loading sourceshmem
    cooperative_groups::memcpy_async(cta, (&mainShmem[0]), (&mainArr[ metaData.mainArrXLength*( 1+ (1-isGold[0]) +  ((1+ (iterationNumb[0] & 1))*2 ) ) ])
    , cuda::aligned_size_t<128>(sizeof(uint32_t) * (metaData.mainArrXLength) ));
    //now to registers we load also 
    


    /// ///////////////// dilatations
    // first we perform up and down dilatations
    resShared[threadIdx.x][threadIdx.y] = bitDilatate(sourceShared[threadIdx.x][threadIdx.y]);

    //we also need to set shmem paddings on the basis of first and last bits ...

    //top            0)top  1)bottom, 2)left 3)right, 4)anterior, 5)posterior, 
    if (isBitAt(sourceShared[threadIdx.x][threadIdx.y], 0)) {
        // printf("setting padding top val %d \n ", isAnythingInPadding[0]);
        isAnythingInPadding[0] = true;
    };
    //shmemPaddingsTopBottom[threadIdx.x][threadIdx.y][0]=true; };
//bottom
    if (isBitAt(sourceShared[threadIdx.x][threadIdx.y], (fbArgs.dbZLength - 1))) {
        //shmemPaddingsTopBottom[threadIdx.x][threadIdx.y][1] = true;
        isAnythingInPadding[1] = true;
    };
    //now we will  additionally get bottom bit of block above and top of block below given they exist 
    checkBlockToUpAndBottom(fbArgs, tensorslice, localWorkQueue, i, getSourceReduced(fbArgs, localWorkQueue, i, iterationNumb), resShared);

    //we also need to save data into shared memory weather this block is marked to be validated (are there any voxels that can be potentially saved into result queue)
    auto activeC = coalesced_threads();

    loadSmallVars(fbArgs, tensorslice, resultfpOffset, resultfnOffset, isBlockToBeValidated, localWorkQueue[i][0], localWorkQueue[i][1], localWorkQueue[i][2], localWorkQueue[i][3]
        , activeC, localFpConter, localFnConter);




    sync(cta);//we loaded and  dilatated up and down - we need also to dilatate anterior, posterior, Hovewer in those cases we need also to check boundary conditions ...


              //TODO() 4 corner threads has too much work and probably couse warp divergence ...- so those that for example have both threadidx and y=0 or max ...
    //we will also immidiately send data to    

    //krowa so we can use 8 tiles 4 will check for the is anythink in padding and 4 will load from neighbours ...

    //#left
    dilatateHelper((threadIdx.x == 0), 2, threadIdx.y, (-1), (0), sourceShared, resShared, isAnythingInPadding, localWorkQueue[i][0] > 0,
        tensorslice, fbArgs, localWorkQueue, i, iterationNumb,
        threadIdx.y, (fbArgs.dbXLength - 1));
    ////right
    dilatateHelper((threadIdx.x == (fbArgs.dbXLength - 1)), 3, threadIdx.y, (1), (0), sourceShared, resShared, isAnythingInPadding
        , (localWorkQueue[i][0] < (fbArgs.metaData.metaXLength - 1)), tensorslice, fbArgs, localWorkQueue, i, iterationNumb, threadIdx.y, 0);
    sync(cta);// we are synchronizing just becouse of corners TODO() rethink corners                
    //posterior
    dilatateHelper((threadIdx.y == 0), 5, threadIdx.x, (0), (-1), sourceShared, resShared, isAnythingInPadding, localWorkQueue[i][1] > 0,
        tensorslice, fbArgs, localWorkQueue, i, iterationNumb, (fbArgs.dbYLength - 1), threadIdx.x);
    //anterior
    dilatateHelper((threadIdx.y == (fbArgs.dbYLength - 1)), 4, threadIdx.x, (0), (1), sourceShared, resShared, isAnythingInPadding
        , localWorkQueue[i][1] < (fbArgs.dbYLength - 1), tensorslice, fbArgs, localWorkQueue, i, iterationNumb, 0, threadIdx.x);



    //syncing we now check is block full
    //marking that we have no more space for dilatations
    isBlockFull = (resShared[threadIdx.x][threadIdx.y] == UINT32_MAX);

    isBlockFull = __syncthreads_and(isBlockFull); ;// all dilatations completed 


   //now we need to move the data into global memory - so dilatated arrays to dilatation reduced arrays and paddings to paddings store
    saveToDilatationArr(fbArgs, tensorslice, resShared, getTargetReduced(fbArgs, localWorkQueue, i, iterationNumb), localWorkQueue, i);

}







/*
load and dilatates the entries in gold or segm ...
*/
#pragma once
template <typename TXTOIO>
inline __device__ void validateAndUpMetaCounter(ForBoolKernelArgs<TXTOIO> fbArgs, char* tensorslice,
    uint16_t localWorkQueue[localWorkQueLength][4], uint8_t bigloop,
    uint32_t sourceShared[32][32], uint32_t resShared[32][32]
    , bool isAnythingInPadding[6], unsigned int iterationNumb[1], bool isBlockFull, thread_block cta, uint16_t i
    , bool isBlockToBeValidated[1], unsigned int localTotalLenthOfWorkQueue[1], unsigned int localFpConter[1], unsigned int localFnConter[1]
    , unsigned int resultfpOffset[1], unsigned int resultfnOffset[1], unsigned int worQueueStep[1], unsigned int& old
    , unsigned int blockFpConter[1], unsigned int blockFnConter[1]
) {
    if ((isBlockToBeValidated[0] || iterationNumb[0] == 0) && !isBlockFull) {
        //now first we need to check for bits that are true now after dilatation but were not in source we will save it in res shmem becouse we will no longer need it
        resShared[threadIdx.x][threadIdx.y] = ((~sourceShared[threadIdx.x][threadIdx.y]) & resShared[threadIdx.x][threadIdx.y]);
        //now we load appropriate reference array (opposite to source)

        if (localWorkQueue[i][3] == 0) { loadDataToShmem(fbArgs, tensorslice, sourceShared, fbArgs.reducedGoldRef, localWorkQueue, i); };
        if (localWorkQueue[i][3] == 1) { loadDataToShmem(fbArgs, tensorslice, sourceShared, fbArgs.reducedSegmRef, localWorkQueue, i); };

        //we now look for bits prasent in both reference arrays and current one
        resShared[threadIdx.x][threadIdx.y] = ((sourceShared[threadIdx.x][threadIdx.y]) & resShared[threadIdx.x][threadIdx.y]);
        for (uint8_t bitPos = 0; bitPos < 32; bitPos++) {
            //if any bit here is set it means it should be added to result list 
            if (isBitAt(resShared[threadIdx.x][threadIdx.y], bitPos)) {
                //first we add to the resList
                //TODO consider first passing it into shared memory and then async mempcy ...
                //we use offset plus number of results already added (we got earlier count from global memory now we just atomically add locally)


                ////// IMPORTANT for some reason in order to make it work resultfnOffset and resultfnOffset swith places
                if (localWorkQueue[i][3] == 1) { old = atomicAdd(&(localFpConter[0]), 1) + resultfnOffset[0]; };
                if (localWorkQueue[i][3] == 0) { old = atomicAdd(&(localFnConter[0]), 1) + resultfpOffset[0]; };




                fbArgs.metaData.resultList[old * 5] = (localWorkQueue[i][0] * fbArgs.dbXLength + threadIdx.x);
                fbArgs.metaData.resultList[old * 5 + 1] = (localWorkQueue[i][1] * fbArgs.dbYLength + threadIdx.y);
                fbArgs.metaData.resultList[old * 5 + 2] = (localWorkQueue[i][2] * fbArgs.dbZLength + bitPos);
                fbArgs.metaData.resultList[old * 5 + 3] = (localWorkQueue[i][3]);
                fbArgs.metaData.resultList[old * 5 + 4] = (iterationNumb[0]);

                //getTensorRow<int>(tensorslice, fbArgs.metaData.resultList, fbArgs.metaData.resultList.Ny, 0, 0)[old] = int(localWorkQueue[i][0] * fbArgs.dbXLength + threadIdx.x);
                //getTensorRow<int>(tensorslice, fbArgs.metaData.resultList, fbArgs.metaData.resultList.Ny, 1, 0)[old] = int(localWorkQueue[i][1] * fbArgs.dbYLength + threadIdx.y);
                //getTensorRow<int>(tensorslice, fbArgs.metaData.resultList, fbArgs.metaData.resultList.Ny, 2, 0)[old] = int(localWorkQueue[i][2] * fbArgs.dbZLength + bitPos);
                //getTensorRow<int>(tensorslice, fbArgs.metaData.resultList, fbArgs.metaData.resultList.Ny, 3, 0)[old] = int(localWorkQueue[i][3]);
                //getTensorRow<int>(tensorslice, fbArgs.metaData.resultList, fbArgs.metaData.resultList.Ny, 4, 0)[old] = int(iterationNumb[0]);






    //            if (getTensorRow<int>(tensorslice, fbArgs.metaData.resultList, fbArgs.metaData.resultList.Ny, 4, 0)[old] !=9) {
    //    printf("\n in kernel saving result x %d y %d z %d isGold %d iteration %d spotToUpdate %d  fpLocCounter %d  fnLocCounter %d   resultfpOffset %d  resultfnOffset %d  xMeta %d yMeta %d zMeta %d isGold %d \n ",

    //        getTensorRow<int>(tensorslice, fbArgs.metaData.resultList, fbArgs.metaData.resultList.Ny, 0, 0)[old],
    //        getTensorRow<int>(tensorslice, fbArgs.metaData.resultList, fbArgs.metaData.resultList.Ny, 1, 0)[old],
    //        getTensorRow<int>(tensorslice, fbArgs.metaData.resultList, fbArgs.metaData.resultList.Ny, 2, 0)[old],
    //        getTensorRow<int>(tensorslice, fbArgs.metaData.resultList, fbArgs.metaData.resultList.Ny, 3, 0)[old],
    //        getTensorRow<int>(tensorslice, fbArgs.metaData.resultList, fbArgs.metaData.resultList.Ny, 4, 0)[old]
    //        , old
    //        , localFpConter[0]
    //        , localFnConter[0]
    //        , resultfnOffset[0]
    //        , resultfpOffset[0]
    //        , localWorkQueue[i][0]
    //        , localWorkQueue[i][1]
    //        , localWorkQueue[i][2]
    //        , localWorkQueue[i][3]

    //    );
    //}
    //else {
    //    printf(" *** ");
    //}

            }
        }
        sync(cta);


        coalesced_group activeE = coalesced_threads();
        //update metadata  fp, fn conters
        if (localWorkQueue[i][3] == 1) {//gold
            updateMetaCounters(tensorslice, localWorkQueue[i][0], localWorkQueue[i][1], localWorkQueue[i][2], localWorkQueue[i][3], fbArgs.metaData.fpCounter, localFpConter[0], activeE);
        };
        if (localWorkQueue[i][3] == 0) {//segm
            updateMetaCounters(tensorslice, localWorkQueue[i][0], localWorkQueue[i][1], localWorkQueue[i][2], localWorkQueue[i][3], fbArgs.metaData.fnCounter, localFnConter[0], activeE);
        };
        if (isToBeExecutedOnActive(activeE, 4)) {
            blockFpConter[0] += localFpConter[0];
            localFpConter[0] = 0;
        };
        if (isToBeExecutedOnActive(activeE, 5)) {
            blockFnConter[0] += localFnConter[0];
            localFnConter[0] = 0;

        };

    }
}


