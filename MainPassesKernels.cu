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
#include "DilatationKernels.cu"
#include "MinMaxesKernel.cu"
#include "MainKernelMetaHelpers.cu"
#include "BiggerMainFunctions.cu"
#include <cooperative_groups/memcpy_async.h>

using namespace cooperative_groups;


/*
5)Main block
    a) we define the work queue iteration - so we divide complete work queue into parts  and each thread block analyzes its own part - one data block at a textLinesFromStrings
    b) we load values of data block into shared memory  and immidiately do the bit wise up and down dilatations, and mark booleans needed to establish is the datablock full
    c) synthreads - left,right, anterior,posterior dilatations...
    d) add the dilatated info into dilatation array and padding info from dilatation to global memory
    e) if block is to be validated we check is there is in the point of currently coverd voxel some voxel in other mas if so we add it to the result list and increment local reult counter
    f) syncgrid()
6)analyze padding
    we iterate over work queue as in 5
    a) we load into shared memory information from padding from blocks all around the block of intrest checking for boundary conditions
    b) we save data of dilatated voxels into dilatation array making sure to synchronize appropriately in the thread block
    c) we analyze the positive entries given the block is to be validated  so we check is such entry is already in dilatation mask if not is it in other mask if first no and second yes we add to the result
    d) also given any positive entry we set block as to be activated simple sum reduction should be sufficient
    e) sync grid
*/





/*
we need to
Data
- shared memory
    -for uploaded data from reduced arrays
    -for dilatation results
    -for result paddings
0) load data about what metadata blocks should be analyzed from work queue
1) load data from given reduced arr into shared memory
2) perform bit  dilatations in 6 directions
    and save to result to result shared memory - additionally dilatations into its own shared memory
3) given the block is to be validated (in case it is first main pass - all needs to be) we check  if
    - if there is set bit (voxel) in res shmem but not in source shmem
        - we establish is there anything of intrest in the primary given array of other type (so for gold we check segm and for segm gold - but original ones)
        - if so we add this to the result list in a spot we established from offsets of metadata
            - we set metadata's fp and fn result counters - so later we will be able to establish wheather block should be validated at all
            - we also increment local counters of fp and fn - those will be used for later
4) we save data from result shmem into reduced arrays and from paddings into padding store (both in global memory)

*/




/**
CPU part of the loop - where we copy data required to know wheather next loop should be executed and to increment the iteration number
*/
template <typename TKKI>
inline bool runAfterOneLoop(ForBoolKernelArgs<TKKI> gpuArgs, ForFullBoolPrepArgs<TKKI> cpuArgs, unsigned int& cpuIterNumb) {
    cpuIterNumb += 1;

    //copy on cpu
    copyDeviceToHost3d(gpuArgs.metaData.minMaxes, cpuArgs.metaData.minMaxes);
    //read an modify
    cpuArgs.metaData.minMaxes.arrP[0][0][13] = cpuIterNumb;
    //copy back on gpu
    copyHostToDevice(gpuArgs.metaData.minMaxes, cpuArgs.metaData.minMaxes);
    // returning true - so signal that we need to loop on only when we did not reach yet the required percent of covered voxels
    return ((ceil(cpuArgs.metaData.minMaxes.arrP[0][0][7] * cpuArgs.robustnessPercent) > cpuArgs.metaData.minMaxes.arrP[0][0][10])
        || (ceil(cpuArgs.metaData.minMaxes.arrP[0][0][8] * cpuArgs.robustnessPercent) > cpuArgs.metaData.minMaxes.arrP[0][0][11]));

}












template <typename TKKI>
inline __global__ void testKernel(ForBoolKernelArgs<TKKI> fbArgs, unsigned int* minMaxes, uint32_t* mainArr, MetaDataGPU metaData, uint32_t* workQueue, uint32_t* origArr) {
    thread_block cta = this_thread_block();

    //work queue !!
    //if ((threadIdx.x == 0) && (threadIdx.y == 0)) {
    //    for (uint32_t ii = blockIdx.x; ii < 7; ii += gridDim.x) {
    //        if (workQueue[ii] > 0) {
    //            if (workQueue[ii] > (isGoldOffset-1)) {
    //                printf("in gold workqueue elment %d  \n", (workQueue[ii] - isGoldOffset));
    //            }
    //            else {
    //                printf("in segm workqueue elment %d  \n", (workQueue[ii]));

    //            }

    //        }

    //    }
    //}
    // 
        //results  !!
    if ((threadIdx.x == 0) && (threadIdx.y == 0)) {
        for (uint32_t ii = blockIdx.x; ii < 10; ii += gridDim.x) {
            if (fbArgs.resultListPointerMeta[ii] > 0) {
                printf("in TEST kernel  result lin meta %d ii  \n", fbArgs.resultListPointerMeta[ii]);

            }

        }
    }



    sync(cta);
    char* tensorslice;


    for (uint32_t linIdexMeta = blockIdx.x; linIdexMeta < metaData.totalMetaLength; linIdexMeta += gridDim.x) {
        //we get from linear index  the coordinates of the metadata block of intrest
        uint8_t xMeta = linIdexMeta % metaData.metaXLength;
        uint8_t zMeta = floor((float)(linIdexMeta / (metaData.metaXLength * metaData.MetaYLength)));
        uint8_t yMeta = floor((float)((linIdexMeta - ((zMeta * metaData.metaXLength * metaData.MetaYLength) + xMeta)) / metaData.metaXLength));

        for (uint8_t xLoc = threadIdx.x; xLoc < fbArgs.dbXLength; xLoc += blockDim.x) {
            uint32_t x = (xMeta + metaData.minX) * fbArgs.dbXLength + xLoc;//absolute position
            for (uint8_t yLoc = threadIdx.y; yLoc < fbArgs.dbYLength; yLoc += blockDim.y) {
                uint32_t  y = (yMeta + metaData.minY) * fbArgs.dbYLength + yLoc;//absolute position
                for (uint8_t zLoc = 0; zLoc < fbArgs.dbZLength; zLoc++) {

                    uint32_t z = (zMeta + metaData.minZ) * fbArgs.dbZLength + zLoc;//absolute position
                    uint8_t ww = 0;
                    //uint32_t column = mainArr[linIdexMeta * metaData.mainArrSectionLength + (threadIdx.x + threadIdx.y * fbArgs.dbXLength) + (metaData.mainArrXLength)*ww];//
                    uint32_t column = mainArr[linIdexMeta * metaData.mainArrSectionLength + (xLoc + yLoc * fbArgs.dbXLength) + (metaData.mainArrXLength) * ww];//
                    //uint32_t column = mainArr[linIdexMeta * metaData.mainArrSectionLength + (threadIdx.x + threadIdx.y * fbArgs.dbXLength)];




                    //rrrrresult meta 1 isGold 1 old 0 localFpConter 1 localFnConter 0 fpOffset 0 fnOffset 0 linIndUpdated 655351  localInd 24544

                    //if (linIdexMeta== 1 ) {
                    //    if (  (fbArgs.dbYLength * 32 * zLoc + yLoc * 32 + xLoc) == 24544) {
                    //            printf("res in TEST kernel x %d y%d z %d linearLocal %d linIdexMeta  \n"
                    //  ,  x, y, z, (xLoc + yLoc * fbArgs.dbXLength), linIdexMeta);

                    //    }
                    //
                    //}
                    ////    rrrrresult meta 2 isGold 1 old 1 localFpConter 1 localFnConter 0 fpOffset 0 fnOffset 0 linIndUpdated 655352  localInd 23839

                    //if (linIdexMeta == 2) {
                    //    if ((fbArgs.dbYLength * 32 * zLoc + yLoc * 32 + xLoc) == 23839) {
                    //        printf( "res in TEST kernel x %d y%d z %d linearLocal %d linIdexMeta  \n"
                    //            , x, y, z, (xLoc + yLoc * fbArgs.dbXLength), linIdexMeta);

                    //    }

                    //}
                    ////    rrrrresult meta 4 isGold 1 old 2 localFpConter 1 localFnConter 0 fpOffset 0 fnOffset 0 linIndUpdated 655354  localInd 767

                    //if (linIdexMeta == 4) {
                    //    if ((fbArgs.dbYLength * 32 * zLoc + yLoc * 32 + xLoc) == 767) {
                    //        printf("res in TEST kernel x %d y%d z %d linearLocal %d linIdexMeta  \n"
                    //            , x, y, z, (xLoc + yLoc * fbArgs.dbXLength), linIdexMeta);

                    //    }

                    //}
                    ////    rrrrresult meta 0 isGold 0 old 3 localFpConter 0 localFnConter 1 fpOffset 3 fnOffset 1 linIndUpdated 0  localInd 24575

                    //if (linIdexMeta == 0) {
                    //    if ((fbArgs.dbYLength * 32 * zLoc + yLoc * 32 + xLoc) == 24575) {
                    //        printf("res in TEST kernel x %d y%d z %d linearLocal %d linIdexMeta  \n"
                    //            , x, y, z, (xLoc + yLoc * fbArgs.dbXLength), linIdexMeta);

                    //    }

                    //}



                 //if (x==33 && y==1 && z==71) {
                 //    printf("in 33 1 71 TEST kernel Metax %d yMeta %d zMeta %d x %d y%d z %d linearLocal %d linIdexMeta %d column %d looking in %d \n"
                 //        , xMeta, yMeta, zMeta, x, y, z, (xLoc + yLoc * fbArgs.dbXLength), linIdexMeta
                 //        , column, linIdexMeta * metaData.mainArrSectionLength + (threadIdx.x + threadIdx.y * fbArgs.dbXLength) + (metaData.mainArrXLength) * ww);
                 //}




                    if (isBitAt(column, zLoc) && column > 0) {


                        printf("in TEST kernel Metax %d yMeta %d zMeta %d x %d y%d z %d linearLocal %d linIdexMeta %d looking in %d    \n"
                                    , xMeta, yMeta, zMeta,x,y,z,  (xLoc + yLoc * fbArgs.dbXLength), linIdexMeta
                                , column , linIdexMeta * metaData.mainArrSectionLength + (xLoc + yLoc * fbArgs.dbXLength) + (metaData.mainArrXLength) * ww, fbArgs.dbYLength);
                    }

                    ww = 1;
                    // uint32_t column = mainArr[linIdexMeta * metaData.mainArrSectionLength + (threadIdx.x + threadIdx.y * fbArgs.dbXLength) + (metaData.mainArrXLength) * ww];//
                    column = mainArr[linIdexMeta * metaData.mainArrSectionLength + (xLoc + yLoc * fbArgs.dbXLength) + (metaData.mainArrXLength) * ww];//


                    //if (x == 33 && y == 1 && z == 71) {
                    //    printf("in 33 1 71 TEST kernel Metax %d yMeta %d zMeta %d x %d y%d z %d linearLocal %d linIdexMeta %d column %d looking in %d \n"
                    //        , xMeta, yMeta, zMeta, x, y, z, (xLoc + yLoc * fbArgs.dbXLength), linIdexMeta
                    //        , column, linIdexMeta * metaData.mainArrSectionLength + (threadIdx.x + threadIdx.y * fbArgs.dbXLength) + (metaData.mainArrXLength) * ww);
                    //}

                    if (isBitAt(column, zLoc) && column > 0) {

                           printf("in TEST kernel Metax %d yMeta %d zMeta %d x %d y%d z %d linearLocal %d linIdexMeta %d looking in %d   \n"
                               , xMeta, yMeta, zMeta, x, y, z, (xLoc + yLoc * fbArgs.dbXLength), linIdexMeta
                               , column, linIdexMeta * metaData.mainArrSectionLength + (xLoc + yLoc * fbArgs.dbXLength) + (metaData.mainArrXLength) * ww, fbArgs.dbYLength);
                    }

                }
            }
        }

        //if ((threadIdx.x == 0) && (threadIdx.y == 0)) {
        //    auto count = fbArgs.metaDataArrPointer[linIdexMeta * metaData.metaDataSectionLength + 1];
        //    if (count > 0) {
        //        printf("in TEST kernel looking fp count  xMeta %d yMeta %d zMeta %d linIdexMeta %d count %d counter %d \n"
        //            , xMeta, yMeta, zMeta, linIdexMeta, count, fbArgs.metaDataArrPointer[linIdexMeta * metaData.metaDataSectionLength + 3]);
        //    }
        //}
        //if ((threadIdx.x == 1) && (threadIdx.y == 0)) {
        //    auto count = fbArgs.metaDataArrPointer[linIdexMeta * metaData.metaDataSectionLength + 2];
        //    if (count > 0) {
        //        printf("in TEST kernel looking fn count   xMeta %d yMeta %d zMeta %d linIdexMeta %d count %d counter %d \n"
        //            , xMeta, yMeta, zMeta, linIdexMeta, count, fbArgs.metaDataArrPointer[linIdexMeta * metaData.metaDataSectionLength + 4]);
        //    }
        //}





        //if ((threadIdx.x == 0) && (threadIdx.y == 0)) {
        //    auto count = mainArr[linIdexMeta * metaData.mainArrSectionLength+ metaData.metaDataOffset + 7];
        //    if (count ==1) {
        //        printf("in TEST kernel looking active gold  xMeta %d yMeta %d zMeta %d linIdexMeta %d count %d \n"
        //            , xMeta, yMeta, zMeta, linIdexMeta, count);
        //    }
        //}
        //if ((threadIdx.x == 0) && (threadIdx.y == 0)) {
        //    auto count = mainArr[linIdexMeta * metaData.mainArrSectionLength + metaData.metaDataOffset + 9];
        //    if (count == 1) {
        //        printf("in TEST kernel looking active segm  xMeta %d yMeta %d zMeta %d linIdexMeta %d count %d \n"
        //            , xMeta, yMeta, zMeta, linIdexMeta, count);
        //    }
        //}
        ///// testing  calculation of surrounding blocks linear indicies
        // block 1,1,1
        //if ((threadIdx.x == 0) && (threadIdx.y == 0)) {
        //    if (xMeta==1 && yMeta==1 && zMeta==1) {
        //        printf("linear indicies from metadata  top %d bottom %d left %d right %d anterior %d posterior %d  linIdexMeta current %d \n    "
        //            ,mainArr[linIdexMeta * metaData.mainArrSectionLength + metaData.metaDataOffset + 13]
        //            , mainArr[linIdexMeta * metaData.mainArrSectionLength + metaData.metaDataOffset + 14]

        //            , mainArr[linIdexMeta * metaData.mainArrSectionLength + metaData.metaDataOffset + 15]
        //            , mainArr[linIdexMeta * metaData.mainArrSectionLength + metaData.metaDataOffset + 16]

        //            , mainArr[linIdexMeta * metaData.mainArrSectionLength + metaData.metaDataOffset + 17]
        //            , mainArr[linIdexMeta * metaData.mainArrSectionLength + metaData.metaDataOffset + 18]
        //            , linIdexMeta
        //        );
        //    }
        //    if (xMeta ==  1&& yMeta == 1 && zMeta == 0) {
        //        printf("linear index top linIdexMeta %d \n    ", linIdexMeta);
        //    
        //    }
        //    if (xMeta ==1 && yMeta == 1 && zMeta == 2) {
        //        printf("linear index bottom linIdexMeta %d \n    ", linIdexMeta);

        //    }
        //    if (xMeta == 1&& yMeta == 2 && zMeta == 1) {
        //        printf("linear index anterior linIdexMeta %d \n    ", linIdexMeta);

        //    }
        //    if (xMeta == 1&& yMeta == 0 && zMeta == 1) {
        //        printf("linear index posterior linIdexMeta %d \n    ", linIdexMeta);

        //    }

        //    if (xMeta ==2 && yMeta == 1 && zMeta == 1) {
        //        printf("linear index right linIdexMeta %d \n    ", linIdexMeta);

        //    }
        //    if (xMeta == 0&& yMeta == 1 && zMeta == 1) {
        //        printf("linear index left linIdexMeta %d \n    ", linIdexMeta);

        //    }

        //}

//// checking weather on edges it shows UINT32_MAX
        //   if ((threadIdx.x == 0) && (threadIdx.y == 0)) {

        //    if (xMeta ==  1&& yMeta == 1 && zMeta == 0) {
        //        printf("linear index top linIdexMeta %d  and max is %d \n    ", mainArr[linIdexMeta * metaData.mainArrSectionLength + metaData.metaDataOffset + 13], UINT32_MAX);
        //    
        //    }
        //    if (xMeta ==1 && yMeta == 1 && zMeta == 3) {
        //        printf("linear index bottom linIdexMeta %d \n    ", mainArr[linIdexMeta * metaData.mainArrSectionLength + metaData.metaDataOffset + 14]);

        //    }
        //    if (xMeta == 1&& yMeta == 5 && zMeta == 1) {
        //        printf("linear index anterior linIdexMeta %d \n    ", mainArr[linIdexMeta * metaData.mainArrSectionLength + metaData.metaDataOffset + 17]);

        //    }
        //    if (xMeta == 1&& yMeta == 0 && zMeta == 1) {
        //        printf("linear index posterior linIdexMeta %d \n    ", mainArr[linIdexMeta * metaData.mainArrSectionLength + metaData.metaDataOffset + 18]);

        //    }

        //    if (xMeta ==2 && yMeta == 1 && zMeta == 1) {
        //        printf("linear index right linIdexMeta %d \n    ", mainArr[linIdexMeta * metaData.mainArrSectionLength + metaData.metaDataOffset + 16]);

        //    }
        //    if (xMeta == 0&& yMeta == 1 && zMeta == 1) {
        //        printf("linear index left linIdexMeta %d \n    ", mainArr[linIdexMeta * metaData.mainArrSectionLength + metaData.metaDataOffset + 15]);

        //    }

        //}

        //if ((threadIdx.x == 0) && (threadIdx.y == 0)) {
        //    auto count = mainArr[linIdexMeta * metaData.mainArrSectionLength + metaData.metaDataOffset + 5];
        //    if (count >0) {
        //        printf("in TEST kernel offset fp  xMeta %d yMeta %d zMeta %d linIdexMeta %d count %d \n"
        //            , xMeta, yMeta, zMeta, linIdexMeta, count);
        //    }
        //}
        //if ((threadIdx.x == 0) && (threadIdx.y == 0)) {
        //    auto count = mainArr[linIdexMeta * metaData.mainArrSectionLength + metaData.metaDataOffset + 6];
        //    if (count > 0) {
        //        printf("in TEST kernel offset fn  xMeta %d yMeta %d zMeta %d linIdexMeta %d count %d \n"
        //            , xMeta, yMeta, zMeta, linIdexMeta, count);
        //    }
        //}

    }




    //for (uint32_t linIdexMeta = blockIdx.x * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x; linIdexMeta < 80; linIdexMeta += blockDim.x * blockDim.y * gridDim.x) {


    // /*   if (fbArgs.metaData.resultList[linIdexMeta * 5 + 4] != 131 && fbArgs.metaData.resultList[linIdexMeta * 5] > 0) {

    //        printf("\n in kernel saving result x %d y %d z %d isGold %d iteration %d spotToUpdate %d \n ",
    //            fbArgs.metaData.resultList[linIdexMeta * 5]
    //            , fbArgs.metaData.resultList[linIdexMeta * 5 + 1]
    //            , fbArgs.metaData.resultList[linIdexMeta * 5 + 2]
    //            , fbArgs.metaData.resultList[linIdexMeta * 5 + 3]
    //            , fbArgs.metaData.resultList[linIdexMeta * 5 + 4]
    //            , linIdexMeta


    //        );
    //    }
    //    else {
    //        printf(" *** ");
    //        atomicAdd(&(getTensorRow<unsigned int>(tensorslice, fbArgs.metaData.minMaxes, 1, 0, 0)[17]), 1);

    //    }*/
    //}
}





template <typename TKKI>
inline __global__ void mainPassKernel(ForBoolKernelArgs<TKKI> fbArgs) {

    //inline __global__ void mainPassKernel(ForBoolKernelArgs<TKKI> fbArgs, uint32_t * mainArr, MetaDataGPU metaData
    //    , unsigned int* minMaxes, uint32_t * workQueue
    //    , uint32_t * resultListPointerMeta, uint32_t * resultListPointerLocal, uint32_t * resultListPointerIterNumb, uint32_t * origArrs, uint32_t * metaDataArr) {



    thread_block cta = this_thread_block();
    thread_block_tile<32> tile = tiled_partition<32>(cta);
    grid_group grid = cooperative_groups::this_grid();


    /*
    * according to https://forums.developer.nvidia.com/t/find-the-limit-of-shared-memory-that-can-be-used-per-block/48556 it is good to keep shared memory below 16kb kilo bytes
    main shared memory spaces
    0-1023 : sourceShmem
    1024-2047 : resShmem
    2048-3071 : first register space
    3072-4095 : second register space
    4096-  4127: small 32 length resgister 3 space
    4128-4500 (372 length) : place for local work queue in dilatation kernels
    */
    __shared__ uint32_t mainShmem[lengthOfMainShmem];
    //usefull for iterating through local work queue
    __shared__ bool isGoldForLocQueue[localWorkQueLength];
    // holding data about paddings 


    // holding data weather we have anything in padding 0)top  1)bottom, 2)left 3)right, 4)anterior, 5)posterior,
    __shared__ bool isAnythingInPadding[6];

    __shared__ bool isBlockFull[1];
    //marks wheather there can be any result of intest there
    __shared__ bool isBlockToBeValidated[1];
    //variables needed for all threads
    __shared__ int iterationNumb[1];
    __shared__ unsigned int globalWorkQueueOffset[1];
    __shared__ unsigned int globalWorkQueueCounter[1];
    __shared__ unsigned int localWorkQueueCounter[1];
    // keeping data wheather gold or segmentation pass should continue - on the basis of global counters

    __shared__ unsigned int localTotalLenthOfWorkQueue[1];
    //counters for per block number of results added in this iteration
    __shared__ unsigned int localFpConter[1];
    __shared__ unsigned int localFnConter[1];

    __shared__ unsigned int blockFpConter[1];
    __shared__ unsigned int blockFnConter[1];

    __shared__ unsigned int fpFnLocCounter[1];

    //result list offset - needed to know where to write a result in a result list
    __shared__ unsigned int resultfpOffset[1];
    __shared__ unsigned int resultfnOffset[1];

    __shared__ unsigned int worQueueStep[1];

    __shared__ uint32_t isGold[1];
    __shared__ uint32_t currLinIndM[1];


    __shared__ uint32_t oldIsGold[1];
    __shared__ uint32_t oldLinIndM[1];

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

    __shared__ uint32_t localBlockMetaData[60];

    /*
 //now linear indexes of the previous block in all sides - if there is no block in given direction it will equal UINT32_MAX

 0 : top
 1 : bottom
 2 : left
 3 : right
 4 : anterior
 5 : posterior

    */


    /////used mainly in meta passes

//    __shared__ unsigned int fpFnLocCounter[1];
    __shared__ bool isGoldPassToContinue[1];
    __shared__ bool isSegmPassToContinue[1];





    //initializations and loading    
    if (tile.thread_rank() == 9 && tile.meta_group_rank() == 0) { iterationNumb[0] = -1; };
    if (tile.thread_rank() == 11 && tile.meta_group_rank() == 0) {
        isGoldPassToContinue[0] = true;
    };
    if (tile.thread_rank() == 12 && tile.meta_group_rank() == 0) {
        isSegmPassToContinue[0] = true;
    };

    //here we caclulate the offset for given block depending on length of the workqueue and number of the  available blocks in a grid
    // - this will give us number of work queue items per block - we will calculate offset on the basis of the block number





    //while (isGoldPassToContinue[0] || isSegmPassToContinue[0]) {



    mainDilatation(false, fbArgs, fbArgs.mainArrAPointer, fbArgs.mainArrBPointer, fbArgs.metaData, fbArgs.minMaxes
        , fbArgs.workQueuePointer
        , fbArgs.resultListPointerMeta, fbArgs.resultListPointerLocal, fbArgs.resultListPointerIterNumb
        , cta, tile, grid, mainShmem
        , isAnythingInPadding, isBlockFull, iterationNumb, globalWorkQueueOffset,
        globalWorkQueueCounter, localWorkQueueCounter, localTotalLenthOfWorkQueue, localFpConter,
        localFnConter, blockFpConter, blockFnConter, resultfpOffset,
        resultfnOffset, worQueueStep, isGold, currLinIndM, localMinMaxes
        , localBlockMetaData, fpFnLocCounter, isGoldPassToContinue, isSegmPassToContinue, fbArgs.origArrsPointer
        , fbArgs.metaDataArrPointer, oldIsGold, oldLinIndM,  isGoldForLocQueue, isBlockToBeValidated);





    // grid.sync();

     //  krowa predicates must be lambdas probablu now they will not compute well as we do not have for example linIdexMeta ...
    /////////////// loading work queue for padding dilatations
    metadataPass(fbArgs, true, 11, 7, 8,
        12, 9, 10
        , mainShmem, globalWorkQueueOffset, globalWorkQueueCounter
        , localWorkQueueCounter, localTotalLenthOfWorkQueue, localMinMaxes
        , fpFnLocCounter, isGoldPassToContinue, isSegmPassToContinue, cta, tile
        , fbArgs.metaData, fbArgs.minMaxes, fbArgs.workQueuePointer, fbArgs.metaDataArrPointer);
    //////////// padding dilatations






//     grid.sync();
     ////////////////////////main metadata pass
        //  krowa predicates must be lambdas probablu now they will not compute well as we do not have for example linIdexMeta ...

     //metadataPass(false,(isGoldPassToContinue[0] &&  mainArr[linIdexMeta * metaData.mainArrSectionLength + metaData.metaDataOffset + 7]
     //         && !mainArr[linIdexMeta * metaData.mainArrSectionLength + metaData.metaDataOffset + 8]),
     //         (isSegmPassToContinue[0] && mainArr[linIdexMeta * metaData.mainArrSectionLength + metaData.metaDataOffset + 9]
     //             && !mainArr[linIdexMeta * metaData.mainArrSectionLength + metaData.metaDataOffset + 10]),
     //         , mainShmem, globalWorkQueueOffset, globalWorkQueueCounter
     //         , localWorkQueueCounter, localTotalLenthOfWorkQueue, localMinMaxes
     //         , fpFnLocCounter, isGoldPassToContinue, isSegmPassToContinue, cta, tile
     //         , mainArr, metaData, minMaxes, workQueue,metaDataArr);
     // 

//  }// end while

  //setting global iteration number to local one 

}



#pragma once
extern "C" inline bool mainKernelsRun(ForFullBoolPrepArgs<int> fFArgs) {

    cudaDeviceReset();
    cudaError_t syncErr;
    cudaError_t asyncErr;
    int device = 0;
    unsigned int cpuIterNumb = -1;
    cudaDeviceProp deviceProp;
    cudaGetDevice(&device);
    cudaGetDeviceProperties(&deviceProp, device);
    int blockSize; // The launch configurator returned block size
    int minGridSize; // The minimum grid size needed to achieve the maximum occupancy for a full device launch
    int gridSize; // The actual grid size needed, based on input size

    // for min maxes kernel 
    cudaOccupancyMaxPotentialBlockSize(
        &minGridSize,
        &blockSize,
        (void*)getMinMaxes<int>,
        0);
    int warpsNumbForMinMax = blockSize / 32;
    int blockSizeForMinMax = minGridSize;

    // for min maxes kernel 
    cudaOccupancyMaxPotentialBlockSize(
        &minGridSize,
        &blockSize,
        (void*)boolPrepareKernel<int>,
        0);
    int warpsNumbForboolPrepareKernel = blockSize / 32;
    int blockSizeFoboolPrepareKernel = minGridSize;
    // for first meta pass kernel
    cudaOccupancyMaxPotentialBlockSize(
        &minGridSize,
        &blockSize,
        (void*)boolPrepareKernel<int>,
        0);
    int theadsForFirstMetaPass = blockSize;
    int blockForFirstMetaPass = minGridSize;
    //for main pass kernel
    cudaOccupancyMaxPotentialBlockSize(
        &minGridSize,
        &blockSize,
        (void*)mainPassKernel<int>,
        0);
    int warpsNumbForMainPass = blockSize / 32;
    int blockForMainPass = minGridSize;

    printf("warpsNumbForMainPass %d blockForMainPass %d  ", warpsNumbForMainPass, blockForMainPass);






    //for debugging
    array3dWithDimsGPU forDebug = allocate3dInGPU(fFArgs.forDebugArr);
    //main arrays allocations
    array3dWithDimsGPU goldArr = allocate3dInGPU(fFArgs.goldArr);

    array3dWithDimsGPU segmArr = allocate3dInGPU(fFArgs.segmArr);
    //pointers ...
    uint32_t* resultListPointerMeta;
    uint32_t* resultListPointerLocal;
    uint32_t* resultListPointerIterNumb;

    uint32_t* origArrsPointer;
    uint32_t* mainArrAPointer;
    uint32_t* mainArrBPointer;
    uint32_t* metaDataArrPointer;

    uint32_t* workQueuePointer;
    unsigned int* minMaxes;
    size_t size = sizeof(unsigned int) * 20;
    cudaMalloc(&minMaxes, size);


    checkCuda(cudaDeviceSynchronize(), "a0");
    ForBoolKernelArgs<int> fbArgs = getArgsForKernel<int>(fFArgs, forDebug, goldArr, segmArr, minMaxes, warpsNumbForMainPass, blockForMainPass);
    MetaDataGPU metaData = fbArgs.metaData;
    fbArgs.metaData.minMaxes = minMaxes;


    //3086


    ////preparation kernel

    // initialize, then launch

    checkCuda(cudaDeviceSynchronize(), "a1");


    getMinMaxes << <blockSizeForMinMax, dim3(32, warpsNumbForMinMax) >> > (fbArgs, minMaxes);

    checkCuda(cudaDeviceSynchronize(), "a1");


    checkCuda(cudaDeviceSynchronize(), "a2");

    metaData = allocateMemoryAfterMinMaxesKernel(fbArgs, fFArgs, workQueuePointer, minMaxes, metaData, origArrsPointer, metaDataArrPointer);

    checkCuda(cudaDeviceSynchronize(), "a2");

    boolPrepareKernel << <blockSizeFoboolPrepareKernel, dim3(32, warpsNumbForboolPrepareKernel) >> > (fbArgs, metaData, origArrsPointer, metaDataArrPointer);
    //uint32_t* origArrs, uint32_t* metaDataArr     metaDataArr[linIdexMeta * metaData.metaDataSectionLength     metaDataOffset

    checkCuda(cudaDeviceSynchronize(), "a3");


    allocateMemoryAfterBoolKernel(fbArgs, fFArgs, resultListPointerMeta, resultListPointerLocal, resultListPointerIterNumb, origArrsPointer, mainArrAPointer, mainArrBPointer, metaData, goldArr, segmArr);

    checkCuda(cudaDeviceSynchronize(), "a4");

    firstMetaPrepareKernel << <blockForFirstMetaPass, theadsForFirstMetaPass >> > (fbArgs, metaData, minMaxes, workQueuePointer, origArrsPointer, metaDataArrPointer);

    checkCuda(cudaDeviceSynchronize(), "a5");
    //void* kernel_args[] = { &fbArgs, mainArrPointer,&metaData,minMaxes, workQueuePointer,resultListPointerMeta,resultListPointerLocal, resultListPointerIterNumb };
    fbArgs.forDebugArr = forDebug;
    fbArgs.goldArr = goldArr;
    fbArgs.segmArr = segmArr;
    fbArgs.metaData = metaData;

    fbArgs.resultListPointerMeta = resultListPointerMeta;
    fbArgs.resultListPointerLocal = resultListPointerLocal;
    fbArgs.resultListPointerIterNumb = resultListPointerIterNumb;

    fbArgs.origArrsPointer = origArrsPointer;
    fbArgs.mainArrAPointer = mainArrAPointer;
    fbArgs.mainArrBPointer = mainArrBPointer;


    fbArgs.metaDataArrPointer = metaDataArrPointer;
    fbArgs.workQueuePointer = workQueuePointer;
    fbArgs.minMaxes = minMaxes;
    void* kernel_args[] = { &fbArgs };


   // cudaLaunchCooperativeKernel((void*)(mainPassKernel<int>), blockForMainPass, dim3(32, warpsNumbForMainPass), kernel_args);
    cudaLaunchCooperativeKernel((void*)(mainPassKernel<int>), 10, dim3(32, warpsNumbForMainPass), kernel_args);



    checkCuda(cudaDeviceSynchronize(), "a6");


    //cudaLaunchCooperativeKernel((void*)mainPassKernel<int>, deviceProp.multiProcessorCount, fFArgs.threadsMainPass, fbArgs);



  //  checkCuda(cudaDeviceSynchronize(), "cc");




  //  ////mainPassKernel << <fFArgs.blocksMainPass, fFArgs.threadsMainPass >> > (fbArgs);

    testKernel << <blockSizeFoboolPrepareKernel, dim3(32, warpsNumbForboolPrepareKernel) >> > (fbArgs, minMaxes, mainArrBPointer, metaData, workQueuePointer, origArrsPointer);

    //  testKernel << <10, 512 >> > (fbArgs, minMaxes);


      ////sync
    checkCuda(cudaDeviceSynchronize(), "cc");




    //deviceTohost



    copyDeviceToHost3d(forDebug, fFArgs.forDebugArr);


    //copyDeviceToHost3d(goldArr, fFArgs.goldArr);
    //copyDeviceToHost3d(segmArr, fFArgs.segmArr);
    // getting arrays allocated on  cpu to 


    copyMetaDataToCPU(fFArgs.metaData, fbArgs.metaData);

    // printForDebug(fbArgs, fFArgs, resultListPointer, mainArrPointer, workQueuePointer, metaData);


    checkCuda(cudaDeviceSynchronize(), "just after copy device to host");
    //cudaGetLastError();

    cudaFreeAsync(forDebug.arrPStr.ptr, 0);
    //cudaFreeAsync(goldArr.arrPStr.ptr, 0);
    //cudaFreeAsync(segmArr.arrPStr.ptr, 0);


    cudaFreeAsync(resultListPointerMeta, 0);
    cudaFreeAsync(resultListPointerLocal, 0);
    cudaFreeAsync(resultListPointerIterNumb, 0);
    cudaFreeAsync(workQueuePointer, 0);
    cudaFreeAsync(origArrsPointer, 0);
    cudaFreeAsync(metaDataArrPointer, 0);

    checkCuda(cudaDeviceSynchronize(), "last ");

    /*   cudaFree(reducedGold.arrPStr.ptr);
       cudaFree(reducedSegm.arrPStr.ptr);
       cudaFree(reducedGoldPrev.arrPStr.ptr);
       cudaFree(reducedSegmPrev.arrPStr.ptr);*/

       //    cudaFreeAsync(resultListPointer, 0);

       //    freeMetaDataGPU(fbArgs.metaData);


           /*
        * Catch errors for both the kernel launch above and any
        * errors that occur during the asynchronous `doubleElements`
        * kernel execution.
        */

    syncErr = cudaGetLastError();
    asyncErr = cudaDeviceSynchronize();
    if (syncErr != cudaSuccess) printf("Error in syncErr: %s\n", cudaGetErrorString(syncErr));
    if (asyncErr != cudaSuccess) printf("Error in asyncErr: %s\n", cudaGetErrorString(asyncErr));


    cudaDeviceReset();

    return true;
}



