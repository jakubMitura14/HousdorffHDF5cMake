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
inline __global__ void testKernel(ForBoolKernelArgs<TKKI> fbArgs, unsigned int* minMaxes, uint32_t* mainArr, MetaDataGPU metaData) {
    char* tensorslice;


    for (uint16_t linIdexMeta = blockIdx.x; linIdexMeta < metaData.totalMetaLength; linIdexMeta += gridDim.x) {
        //we get from linear index  the coordinates of the metadata block of intrest
        uint8_t xMeta = linIdexMeta % metaData.metaXLength;
        uint8_t zMeta = floor((float)(linIdexMeta / (metaData.metaXLength * metaData.MetaYLength)));
        uint8_t yMeta = floor((float)((linIdexMeta - ((zMeta * metaData.metaXLength * metaData.MetaYLength) + xMeta)) / metaData.metaXLength));

        for (uint8_t xLoc = threadIdx.x; xLoc < fbArgs.dbXLength; xLoc += blockDim.x) {
            uint16_t x = (xMeta + metaData.minX) * fbArgs.dbXLength + xLoc;//absolute position
            for (uint8_t yLoc = threadIdx.y; yLoc < fbArgs.dbYLength; yLoc += blockDim.y) {
                uint16_t  y = (yMeta + metaData.minY) * fbArgs.dbYLength + yLoc;//absolute position
                for (uint8_t zLoc = 0; zLoc < fbArgs.dbZLength; zLoc++) {

                    uint16_t z = (zMeta + metaData.minZ) * fbArgs.dbZLength + zLoc;//absolute position
                    uint32_t column = mainArr[linIdexMeta * metaData.mainArrSectionLength + (threadIdx.x + threadIdx.y * fbArgs.dbXLength)];
                    
                   // if (isBitAt(column, zLoc) && column>0) {
                   //// if (x==33 && y==1 && z==71) {
                   //// if (x==75 && y==20 && z==70) {
                   //// if (column>0) {
                   //     //in kernel x 33 y 1 z 71 linearLocal 33 linIdexMeta 0
                   //     //    in kernel x 75 y 20 z 70 linearLocal 267 linIdexMeta 3
                   //     printf("in TEST kernel Metax %d yMeta %d zMeta %d x %d y%d z %d linearLocal %d linIdexMeta %d column %d \n"
                   //                 , xMeta, yMeta, zMeta,x,y,z,  (xLoc + yLoc * fbArgs.dbXLength), linIdexMeta
                   //             , column);
                   // }

                }
            }
        }
        //if ((threadIdx.x == 0) && (threadIdx.y == 0)) {
        //    auto count = mainArr[linIdexMeta * metaData.mainArrSectionLength+ metaData.metaDataOffset + 9];
        //    if (count ==1) {
        //        printf("in TEST kernel looking for xMeta %d yMeta %d zMeta %d linIdexMeta %d count %d \n"
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



    }




    //for (uint16_t linIdexMeta = blockIdx.x * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x; linIdexMeta < 80; linIdexMeta += blockDim.x * blockDim.y * gridDim.x) {
  

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

/*
becouse we need a lot of the additional memory spaces to minimize memory consumption allocations will be postponed after first kernel run enabling
*/
#pragma once
template <typename ZZR>
inline void allocateMemoryAfterBoolKernel(ForBoolKernelArgs<ZZR> gpuArgs, ForFullBoolPrepArgs<ZZR> cpuArgs, void* resultListPointer) {
    //copy on cpu
    copyDeviceToHost3d(gpuArgs.metaData.minMaxes, cpuArgs.metaData.minMaxes);
    //read an modify
    //1)maxX 2)minX 3)maxY 4) minY 5) maxZ 6) minZ
    //7)global FP count; 8)global FN count
    unsigned int fpPlusFn = cpuArgs.metaData.minMaxes.arrP[0][0][7] + cpuArgs.metaData.minMaxes.arrP[0][0][8];

    size_t size = sizeof(uint16_t) * 5 * fpPlusFn + 1;
    cudaMallocAsync(&resultListPointer, size, 0);
    gpuArgs.metaData.resultList = resultListPointer;


    // cudaFreeAsync(gpuArgs.metaData.resultList, 0);

     //cudaFree(resultListPointer);


};


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
    int warpsNumbForMinMax = blockSize/32;
    int blockSizeForMinMax = minGridSize;

    // for min maxes kernel 
    cudaOccupancyMaxPotentialBlockSize(
        &minGridSize,
        &blockSize,
        (void*)boolPrepareKernel<int>,
        0);
    int warpsNumbForboolPrepareKernel = blockSize / 32;
    int blockSizeFoboolPrepareKernel = minGridSize;



    //for debugging
    array3dWithDimsGPU forDebug = allocate3dInGPU(fFArgs.forDebugArr);
    //main arrays allocations
    array3dWithDimsGPU goldArr = allocate3dInGPU(fFArgs.goldArr);

    array3dWithDimsGPU segmArr = allocate3dInGPU(fFArgs.segmArr);
        //pointers ...
    uint32_t* resultListPointer;
    uint32_t* mainArrPointer;
    uint32_t* workQueuePointer;
    unsigned int* minMaxes;
    size_t size = sizeof(unsigned int) * 20;
    cudaMalloc(&minMaxes, size);
    

    //MetaDataGPU metaDataGPU = allocateMetaDataOnGPU(fFArgs.metaData);
    checkCuda(cudaDeviceSynchronize(), "a0");
    ForBoolKernelArgs<int> fbArgs = getArgsForKernel<int>(fFArgs, forDebug, goldArr, segmArr, minMaxes);
    MetaDataGPU metaData= fbArgs.metaData;
    fbArgs.metaData.minMaxes = minMaxes;


    ////preparation kernel

    // initialize, then launch

    checkCuda(cudaDeviceSynchronize(), "a1");

    void* kernel_args[] = { &fbArgs };

        getMinMaxes << <blockSizeForMinMax, dim3(32,warpsNumbForMinMax) >> > (fbArgs, minMaxes);
       
        checkCuda(cudaDeviceSynchronize(), "a1");


    checkCuda(cudaDeviceSynchronize(), "a2");

    metaData = allocateMemoryAfterMinMaxesKernel(fbArgs, fFArgs, mainArrPointer, workQueuePointer,minMaxes, metaData);

    checkCuda(cudaDeviceSynchronize(), "a2");

        boolPrepareKernel << <blockSizeFoboolPrepareKernel, dim3(32, warpsNumbForboolPrepareKernel) >> > (fbArgs, mainArrPointer, metaData);

    checkCuda(cudaDeviceSynchronize(), "a3");

        allocateMemoryAfterBoolKernel(fbArgs, fFArgs, resultListPointer);

    checkCuda(cudaDeviceSynchronize(), "a4");

    
    // cudaLaunchCooperativeKernel((void*)(boolPrepareKernel<int>), deviceProp.multiProcessorCount, fFArgs.threads, kernel_args);


   /* unsigned int fpPlusFn = fFArgs.metaData.minMaxes.arrP[0][0][7] + fFArgs.metaData.minMaxes.arrP[0][0][8];
    uint16_t* resultListPointer;
    size_t size = sizeof(uint16_t) * 5 * fpPlusFn + 1;
    cudaMallocAsync(&resultListPointer, size, 0);
    fbArgs.metaData.resultList = resultListPointer;*/

    //allocateMemoryAfterBoolKernel(fbArgs, fFArgs, resultListPointer);

    //cudaLaunchCooperativeKernel((void*)(firstMetaPrepareKernel<int>), deviceProp.multiProcessorCount, fFArgs.threadsFirstMetaDataPass, kernel_args);


    //cudaLaunchCooperativeKernel((void*)(firstMetaPrepareKernel<int>), deviceProp.multiProcessorCount, fFArgs.threadsFirstMetaDataPass, kernel_args);

    checkCuda(cudaDeviceSynchronize(), "bb");


    //cudaLaunchCooperativeKernel((void*)mainPassKernel<int>, deviceProp.multiProcessorCount, fFArgs.threadsMainPass, fbArgs);

  // // for (int i = 0; i < 205; i++) {
  //  while(runAfterOneLoop(fbArgs, fFArgs, cpuIterNumb)){
  //     // runAfterOneLoop(fbArgs, fFArgs, cpuIterNumb);

  //    /*  checkCuda(cudaDeviceSynchronize(), "bb");
  //      printf("mainDilatation %d  \n", cpuIterNumb);*/

  //      //cudaLaunchCooperativeKernel((void*)(mainDilatation<int>), deviceProp.multiProcessorCount, fFArgs.threadsMainPass, kernel_args);
  //      mainDilatation << <deviceProp.multiProcessorCount, fFArgs.threadsMainPass >> > (fbArgs);

  //    /*  syncErr = cudaGetLastError();
  //      asyncErr = cudaDeviceSynchronize();
  //      if (syncErr != cudaSuccess) printf("Error in syncErr: %s\n", cudaGetErrorString(syncErr));
  //      if (asyncErr != cudaSuccess) printf("Error in asyncErr: %s\n", cudaGetErrorString(asyncErr));*/


  //      //cudaLaunchCooperativeKernel((void*)(getWorkQueeueFromIsToBeActivated<int>), deviceProp.multiProcessorCount, fFArgs.threadsMainPass, kernel_args);
  //      getWorkQueeueFromIsToBeActivated << <deviceProp.multiProcessorCount, fFArgs.threadsMainPass >> > (fbArgs);


  //     /* checkCuda(cudaDeviceSynchronize(), "bb");
  //      printf("getWorkQueeueFromIsToBeActivated %d  \n", cpuIterNumb);
  //      syncErr = cudaGetLastError();
  //      asyncErr = cudaDeviceSynchronize();
  //      if (syncErr != cudaSuccess) printf("Error in syncErr: %s\n", cudaGetErrorString(syncErr));
  //      if (asyncErr != cudaSuccess) printf("Error in asyncErr: %s\n", cudaGetErrorString(asyncErr));*/

  //      paddingDilatation << <deviceProp.multiProcessorCount, fFArgs.threadsMainPass >> > (fbArgs);

  //      //cudaLaunchCooperativeKernel((void*)(paddingDilatation<int>), deviceProp.multiProcessorCount, fFArgs.threadsMainPass, kernel_args);
  //      checkCuda(cudaDeviceSynchronize(), "bb");


  //      /*checkCuda(cudaDeviceSynchronize(), "bb");
  //      printf("paddingDilatation %d  \n", cpuIterNumb);
  //      syncErr = cudaGetLastError();
  //      asyncErr = cudaDeviceSynchronize();
  //      if (syncErr != cudaSuccess) printf("Error in syncErr: %s\n", cudaGetErrorString(syncErr));
  //      if (asyncErr != cudaSuccess) printf("Error in asyncErr: %s\n", cudaGetErrorString(asyncErr));*/

  //      //cudaLaunchCooperativeKernel((void*)(getWorkQueeueFromActive_mainPass<int>), deviceProp.multiProcessorCount, fFArgs.threadsMainPass, kernel_args);
  //      getWorkQueeueFromActive_mainPass << <deviceProp.multiProcessorCount, fFArgs.threadsMainPass >> > (fbArgs);


  ///*      checkCuda(cudaDeviceSynchronize(), "bb");
  //      printf("getWorkQueeueFromActive_mainPass %d  \n", cpuIterNumb);
  //      syncErr = cudaGetLastError();
  //      asyncErr = cudaDeviceSynchronize();
  //      if (syncErr != cudaSuccess) printf("Error in syncErr: %s\n", cudaGetErrorString(syncErr));
  //      if (asyncErr != cudaSuccess) printf("Error in asyncErr: %s\n", cudaGetErrorString(asyncErr));*/
  // }
  //  checkCuda(cudaDeviceSynchronize(), "cc");




  //  ////mainPassKernel << <fFArgs.blocksMainPass, fFArgs.threadsMainPass >> > (fbArgs);

   testKernel << <blockSizeFoboolPrepareKernel, dim3(32, warpsNumbForboolPrepareKernel) >> > (fbArgs, minMaxes, mainArrPointer, metaData);

  //  testKernel << <10, 512 >> > (fbArgs, minMaxes);


    ////sync
    checkCuda(cudaDeviceSynchronize(), "cc");




    //deviceTohost



    copyDeviceToHost3d(forDebug, fFArgs.forDebugArr);


    copyDeviceToHost3d(goldArr, fFArgs.goldArr);
    copyDeviceToHost3d(segmArr, fFArgs.segmArr);
    // getting arrays allocated on  cpu to 


    copyMetaDataToCPU(fFArgs.metaData, fbArgs.metaData);

   // printForDebug(fbArgs, fFArgs, resultListPointer, mainArrPointer, workQueuePointer, metaData);


    checkCuda(cudaDeviceSynchronize(), "just after copy device to host");
    //cudaGetLastError();

    //cudaFreeAsync(forDebug.arrPStr.ptr, 0);
    //cudaFreeAsync(goldArr.arrPStr.ptr, 0);
    //cudaFreeAsync(segmArr.arrPStr.ptr, 0);


    cudaFreeAsync(resultListPointer,0);
    cudaFreeAsync(mainArrPointer, 0);
    cudaFreeAsync(workQueuePointer, 0);




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













/*

template <typename TKKI>
__global__ void mainPassKernel(ForBoolKernelArgs<TKKI> fbArgs) {
    thread_block cta = this_thread_block();
    thread_block_tile<32> tile = tiled_partition<32>(cta);

    char* tensorslice;
    bool isBlockFull = true;// usefull to establish do we have block completely filled and no more dilatations possible
    unsigned int old = 0;
    uint16_t i = 0;
    uint8_t j = 0;
    uint8_t bigloop = 0;
    uint8_t bitPos = 0;
    // some references using as aliases
    unsigned int& oldRef = old;
    uint16_t& linIdexMeta = i;
    uint8_t& xMeta = j;
    uint8_t& yMeta = bigloop;
    uint8_t& zMeta = bitPos;
    bool& isToBeActivated = isBlockFull;


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
    __shared__ bool isGoldPassToContinue[1];
    __shared__ bool isSegmPassToContinue[1];


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
        isGoldPassToContinue[0] = true;
    };
    if (isToBeExecutedOnActive(active, 5)) {
        isSegmPassToContinue[0] = true;
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
    for (bigloop = blockIdx.x * globalWorkQueueOffset[0]; bigloop < ((blockIdx.x + 1) * globalWorkQueueOffset[0]); bigloop += worQueueStep[0]) {
        // grid stride loop - sadly most of threads will be idle
        ///////////// loading to work queue
        loadFromGlobalToLocalWorkQueue(fbArgs, tensorslice, localWorkQueue, bigloop, globalWorkQueueOffset, localTotalLenthOfWorkQueue, worQueueStep,j);

        sync(cta);// now local work queue is populated

            //now all of the threads in the block needs to have the same i value so we will increment by 1
        for (i = 0; i < worQueueStep[0]; i += 1) {
            if (((bigloop + i) < localTotalLenthOfWorkQueue[0]) && ((bigloop + i) < ((blockIdx.x + 1) * globalWorkQueueOffset[0]))) {

                // now we have metadata coordinates we need to start go over associated data block - in order to make it as efficient as possible data block size is set to be the same as datablock size
                // so we do not need iteration loop

                loadAndDilatateAndSave(fbArgs, tensorslice, localWorkQueue, bigloop, sourceShared, resShared, isAnythingInPadding, iterationNumb, isBlockFull, cta, i,
                    isBlockToBeValidated, localTotalLenthOfWorkQueue, localFpConter, localFnConter, resultfpOffset, resultfnOffset, worQueueStep);

                /////////////////////// validation if it is to be validated, also we checked for bing full before dilatations - if it was full at the begining - no point in validation
                validateAndUpMetaCounter(fbArgs, tensorslice, localWorkQueue, bigloop, sourceShared, resShared, isAnythingInPadding, iterationNumb, isBlockFull, cta, i,
                    isBlockToBeValidated, localTotalLenthOfWorkQueue, localFpConter, localFnConter, resultfpOffset, resultfnOffset, worQueueStep, bitPos, oldRef, blockFpConter, blockFnConter);

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


    grid.sync();
    auto activeE = coalesced_threads();
    if (isToBeExecutedOnActive(activeE, 0)) {
        getTensorRow<unsigned int>(tensorslice, fbArgs.metaData.minMaxes, 1, 0, 0)[9] = 0;
    };

    grid.sync();
    // checking global count and counters
    checkIsToBeDilatated(fbArgs, tensorslice, isGoldPassToContinue, isSegmPassToContinue);

    sync(cta);




    auto activeO = coalesced_threads();
    //if (isToBeExecutedOnActive(activeO, 0)) {
    //    printf("\n ****************************** \n");
    //};

    ///////// now we need to look through blocks that we just  activated
    for (linIdexMeta = blockIdx.x * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x; linIdexMeta < fbArgs.metaData.totalMetaLength; linIdexMeta += blockDim.x * blockDim.y * gridDim.x) {
        //we get from linear index  the coordinates of the metadata block of intrest
        xMeta = linIdexMeta % fbArgs.metaData.metaXLength;
        zMeta = floor((float)(linIdexMeta / (fbArgs.metaData.metaXLength * fbArgs.metaData.MetaYLength)));
        yMeta = floor((float)((linIdexMeta - ((zMeta * fbArgs.metaData.metaXLength * fbArgs.metaData.MetaYLength) + xMeta)) / fbArgs.metaData.metaXLength));
        //gold pass

        isToBeActivated = isGoldPassToContinue[0] && (getTensorRow<bool>(tensorslice, fbArgs.metaData.isToBeActivatedGold, fbArgs.metaData.isToBeActivatedGold.Ny, yMeta, zMeta)[xMeta]
            && !getTensorRow<bool>(tensorslice, fbArgs.metaData.isActiveGold, fbArgs.metaData.isActiveGold.Ny, yMeta, zMeta)[xMeta]
            && !getTensorRow<bool>(tensorslice, fbArgs.metaData.isFullGold, fbArgs.metaData.isFullGold.Ny, yMeta, zMeta)[xMeta]);

        addToQueueOtherPasses(fbArgs,oldRef, tensorslice, xMeta, yMeta, zMeta ,1  ,  localWorkQueue, localWorkQueueCounter , sourceShared, resShared, isToBeActivated);
        if (isToBeActivated) {
            getTensorRow<bool>(tensorslice, fbArgs.metaData.isToBeActivatedGold, fbArgs.metaData.isToBeActivatedGold.Ny, yMeta, zMeta)[xMeta] = false;
        }
        //segmPass
        isToBeActivated = isSegmPassToContinue[0] && (getTensorRow<bool>(tensorslice, fbArgs.metaData.isToBeActivatedSegm, fbArgs.metaData.isToBeActivatedSegm.Ny, yMeta, zMeta)[xMeta]
            && !getTensorRow<bool>(tensorslice, fbArgs.metaData.isActiveSegm, fbArgs.metaData.isActiveSegm.Ny, yMeta, zMeta)[xMeta]
            && !getTensorRow<bool>(tensorslice, fbArgs.metaData.isFullSegm, fbArgs.metaData.isFullSegm.Ny, yMeta, zMeta)[xMeta]  );

            addToQueueOtherPasses(fbArgs, oldRef, tensorslice, xMeta, yMeta, zMeta, 0, localWorkQueue, localWorkQueueCounter, sourceShared, resShared, isToBeActivated);
        if (isToBeActivated) {
            getTensorRow<bool>(tensorslice, fbArgs.metaData.isToBeActivatedSegm, fbArgs.metaData.isToBeActivatedSegm.Ny, yMeta, zMeta)[xMeta] = false;

            //printf("\n found to be actvated xMeta %d yMeta %d zMeta %d isGold  %d isSegmPassToContinue[0] %d  isActive %d isFull %d \n ", xMeta, yMeta, zMeta, 0, isSegmPassToContinue[0], getTensorRow<bool>(tensorslice, fbArgs.metaData.isActiveSegm
            //    , fbArgs.metaData.isActiveSegm.Ny, yMeta, zMeta)[xMeta], getTensorRow<bool>(tensorslice, fbArgs.metaData.isFullSegm, fbArgs.metaData.isFullSegm.Ny, yMeta, zMeta)[xMeta]);
        }
    }

    sync(cta);
    auto activeF = coalesced_threads();

     if(isToBeExecutedOnActive(activeF, 0)) {
        globalWorkQueueCounter[0] = atomicAdd(&(getTensorRow<int>(tensorslice, fbArgs.metaData.minMaxes, 1, 0, 0)[9]), (localWorkQueueCounter[0]));
    }

     sync(cta);
     // pushing work queue to global memory
    fromShmemToGlobalWorkQueue(fbArgs, oldRef, i, sourceShared, resShared, localWorkQueue, globalWorkQueueCounter, tensorslice, localWorkQueueCounter);
    grid.sync();



    sync(cta);
    clearShmemBeforeDilatation(fbArgs, tensorslice, blockFpConter, blockFnConter, localWorkQueueCounter, localFpConter, localFnConter);
    if (isToBeExecutedOnActive(active, 1)) {
        localTotalLenthOfWorkQueue[0] = getTensorRow<unsigned int>(tensorslice, fbArgs.metaData.minMaxes, 1, 0, 0)[9];
        globalWorkQueueOffset[0] = floor((float)(localTotalLenthOfWorkQueue[0] / gridDim.x)) + 1;
        worQueueStep[0] = min(localWorkQueLength, globalWorkQueueOffset[0]);
    };

   sync(cta);

    ////// now we do the dilatations and validations of blocks that were just activated

    //TODO change looping so it will access contigous memory
    for (bigloop = blockIdx.x * globalWorkQueueOffset[0]; bigloop < ((blockIdx.x + 1) * globalWorkQueueOffset[0]); bigloop += worQueueStep[0]) {
        // grid stride loop - sadly most of threads will be idle
        ///////////// loading to work queue
        loadFromGlobalToLocalWorkQueue(fbArgs, tensorslice, localWorkQueue, bigloop, globalWorkQueueOffset, localTotalLenthOfWorkQueue, worQueueStep,j);

        sync(cta);// now local work queue is populated

            //now all of the threads in the block needs to have the same i value so we will increment by 1
        for (i = 0; i < worQueueStep[0]; i += 1) {
            if (((bigloop + i) < localTotalLenthOfWorkQueue[0]) && ((bigloop + i) < ((blockIdx.x + 1) * globalWorkQueueOffset[0]))) {



                //if (isToBeExecutedOnActive(activeJF, 0)) {
                //    printf("\n local work queue xMeta %d  yMeta %d  zMeta %d  isGold %d  i %d workQueLength %d workQueueStep %d globalWorkQueueOffset %d bigloop %d blockIdx.x %d"
                //        , localWorkQueue[i][0], localWorkQueue[i][1], localWorkQueue[i][2], localWorkQueue[i][3], i
                //        , localTotalLenthOfWorkQueue[0], worQueueStep[0], globalWorkQueueOffset[0], bigloop, blockIdx.x);
                //}


                // now we have metadata coordinates we need to start go over associated data block - in order to make it as efficient as possible data block size is set to be the same as datablock size
                // so we do not need iteration loop

               // loadAndDilatateAndSave(fbArgs, tensorslice, localWorkQueue, bigloop, sourceShared, resShared, isAnythingInPadding, iterationNumb, isBlockFull, cta, i,
                 //   isBlockToBeValidated, localTotalLenthOfWorkQueue, localFpConter, localFnConter, resultfpOffset, resultfnOffset, worQueueStep);

                /////////////////////// validation if it is to be validated, also we checked for bing full before dilatations - if it was full at the begining - no point in validation
              //  validateAndUpMetaCounter(fbArgs, tensorslice, localWorkQueue, bigloop, sourceShared, resShared, isAnythingInPadding, iterationNumb, isBlockFull, cta, i,
              //      isBlockToBeValidated, localTotalLenthOfWorkQueue, localFpConter, localFnConter, resultfpOffset, resultfnOffset, worQueueStep, bitPos, oldRef, blockFpConter, blockFnConter);


    //first we load data to source shmem
                loadDataToShmem(fbArgs, tensorslice, sourceShared, getSourceReduced(fbArgs, localWorkQueue, i, iterationNumb), localWorkQueue, i);





            }
        }
    }
    //sync(cta);
    ////we need to clear isAnythingInPadding to 0
    //clearisAnythingInPadding(isAnythingInPadding);
    ////     updating global counters
    //updateGlobalCountersAndClear(fbArgs, tensorslice, blockFpConter, blockFnConter, localWorkQueueCounter, localFpConter, localFnConter);

    //grid.sync();
    //// checking global count and counters
    //checkIsToBeDilatated(fbArgs, tensorslice, isGoldPassToContinue, isSegmPassToContinue);

    //sync(cta);







    ///////// now we need to look through all  blocks - for next dilatation pass ...
    for (linIdexMeta = blockIdx.x * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x; linIdexMeta < fbArgs.metaData.totalMetaLength; linIdexMeta += blockDim.x * blockDim.y * gridDim.x) {
        //we get from linear index  the coordinates of the metadata block of intrest
        xMeta = linIdexMeta % fbArgs.metaData.metaXLength;
        zMeta = floor((float)(linIdexMeta / (fbArgs.metaData.metaXLength * fbArgs.metaData.MetaYLength)));
        yMeta = floor((float)((linIdexMeta - ((zMeta * fbArgs.metaData.metaXLength * fbArgs.metaData.MetaYLength) + xMeta)) / fbArgs.metaData.metaXLength));
        //gold pass

        isToBeActivated = isGoldPassToContinue[0] && (getTensorRow<bool>(tensorslice, fbArgs.metaData.isActiveGold, fbArgs.metaData.isActiveGold.Ny, yMeta, zMeta)[xMeta]
            && !getTensorRow<bool>(tensorslice, fbArgs.metaData.isFullGold, fbArgs.metaData.isFullGold.Ny, yMeta, zMeta)[xMeta]);

        addToQueueOtherPasses(fbArgs, oldRef, tensorslice, xMeta, yMeta, zMeta, 1, localWorkQueue, localWorkQueueCounter, sourceShared, resShared, isToBeActivated);

        //segmPass
        isToBeActivated = isSegmPassToContinue[0] && (getTensorRow<bool>(tensorslice, fbArgs.metaData.isActiveSegm, fbArgs.metaData.isActiveSegm.Ny, yMeta, zMeta)[xMeta]
            && !getTensorRow<bool>(tensorslice, fbArgs.metaData.isFullSegm, fbArgs.metaData.isFullSegm.Ny, yMeta, zMeta)[xMeta]);

        addToQueueOtherPasses(fbArgs, oldRef, tensorslice, xMeta, yMeta, zMeta, 0, localWorkQueue, localWorkQueueCounter, sourceShared, resShared, isToBeActivated);

    }

    sync(cta);
    auto activeG = coalesced_threads();

    if (isToBeExecutedOnActive(activeG, 0)) {
        globalWorkQueueCounter[0] = atomicAdd(&(getTensorRow<int>(tensorslice, fbArgs.metaData.minMaxes, 1, 0, 0)[9]), (localWorkQueueCounter[0]));
    }

    sync(cta);
    // pushing work queue to global memory
    fromShmemToGlobalWorkQueue(fbArgs, oldRef, i, sourceShared, resShared, localWorkQueue, globalWorkQueueCounter, tensorslice, localWorkQueueCounter);




    // TODO - use pipelines as described at 201 in https://docs.nvidia.com/cuda/pdf/CUDA_C_Programming_Guide.pdf



}


*/


// runAfterOneLoop(fbArgs, fFArgs, cpuIterNumb);// cpu part



//#pragma once
//extern "C" inline bool mainKernelsTestRun(ForFullBoolPrepArgs<int> fFArgs, forTestPointStruct allPointsA[]
//    , forTestMetaDataStruct allMetas[], int pointsNumber, int metasNumber) {
//
//
//    cudaError_t syncErr;
//    cudaError_t asyncErr;
//
//    unsigned int cpuIterNumb = -1;
//    int device = 0;
//    cudaDeviceProp deviceProp;
//    cudaGetDeviceProperties(&deviceProp, device);
//
//
//    for debugging
//    array3dWithDimsGPU forDebug = allocate3dInGPU(fFArgs.forDebugArr);
//    main arrays allocations
//    array3dWithDimsGPU goldArr = allocate3dInGPU(fFArgs.goldArr);
//
//    array3dWithDimsGPU segmArr = allocate3dInGPU(fFArgs.segmArr);
//    //reduced arrays
//    array3dWithDimsGPU reducedGold = allocate3dInGPU(fFArgs.reducedGold);
//    array3dWithDimsGPU reducedSegm = allocate3dInGPU(fFArgs.reducedSegm);
//
//    array3dWithDimsGPU reducedGoldRef = allocate3dInGPU(fFArgs.reducedGoldRef);
//    array3dWithDimsGPU reducedSegmRef = allocate3dInGPU(fFArgs.reducedSegmRef);
//
//
//    array3dWithDimsGPU reducedGoldPrev = allocate3dInGPU(fFArgs.reducedGoldPrev);
//    array3dWithDimsGPU reducedSegmPrev = allocate3dInGPU(fFArgs.reducedSegmPrev);
//
//
//
//
//
//
//    ForBoolKernelArgs<int> fbArgs = getArgsForKernel<int>(fFArgs, forDebug, goldArr, segmArr, reducedGold, reducedSegm, reducedGoldRef, reducedSegmRef, reducedGoldPrev, reducedSegmPrev);
//    void* kernel_args[] = { &fbArgs };
//
//    //preparation kernel
//    cudaLaunchCooperativeKernel((void*)(boolPrepareKernel<int>), deviceProp.multiProcessorCount, fFArgs.threads, kernel_args);
//    //sync
//    checkCuda(cudaDeviceSynchronize(), "aa");
//
//     bool test
//    copyDeviceToHost3d(forDebug, fFArgs.forDebugArr);
//    copyDeviceToHost3d(goldArr, fFArgs.goldArr);
//    copyDeviceToHost3d(segmArr, fFArgs.segmArr);
//    copyDeviceToHost3d(reducedGold, fFArgs.reducedGold);
//    copyDeviceToHost3d(reducedSegm, fFArgs.reducedSegm);
//    copyDeviceToHost3d(reducedGoldRef, fFArgs.reducedGoldRef);
//    copyDeviceToHost3d(reducedSegmRef, fFArgs.reducedSegmRef);
//    copyDeviceToHost3d(reducedGoldPrev, fFArgs.reducedGoldPrev);
//    copyDeviceToHost3d(reducedSegmPrev, fFArgs.reducedSegmPrev);
//    copyMetaDataToCPU(fFArgs.metaData, fbArgs.metaData);
//
//    checkCuda(cudaDeviceSynchronize(), "aa");
//    forBoolKernelTestUnitTests(fFArgs, allPointsA, allMetas, pointsNumber, metasNumber, fbArgs.dbXLength, fbArgs.dbYLength, fbArgs.dbZLength);
//    checkCuda(cudaDeviceSynchronize(), "aa");
//
//       //here threads one dimensionsonal !!
//       //TODO() reallocate memory - make reduced arrs and metadata smaller - allocate work queue, padding store, result list ...
//
//    cudaLaunchCooperativeKernel((void*)(firstMetaPrepareKernel<int>), deviceProp.multiProcessorCount, fFArgs.threadsFirstMetaDataPass, kernel_args);
//
//    checkCuda(cudaDeviceSynchronize(), "aa");
//
//    copyDeviceToHost3d(forDebug, fFArgs.forDebugArr);
//    copyDeviceToHost3d(goldArr, fFArgs.goldArr);
//    copyDeviceToHost3d(segmArr, fFArgs.segmArr);
//    copyDeviceToHost3d(reducedGold, fFArgs.reducedGold);
//    copyDeviceToHost3d(reducedSegm, fFArgs.reducedSegm);
//    copyDeviceToHost3d(reducedGoldRef, fFArgs.reducedGoldRef);
//    copyDeviceToHost3d(reducedSegmRef, fFArgs.reducedSegmRef);
//    copyDeviceToHost3d(reducedGoldPrev, fFArgs.reducedGoldPrev);
//    copyDeviceToHost3d(reducedSegmPrev, fFArgs.reducedSegmPrev);
//    copyMetaDataToCPU(fFArgs.metaData, fbArgs.metaData);
//
//    firstMetaPassKernelTestUnitTests(fFArgs, allPointsA, allMetas, pointsNumber, metasNumber, fbArgs.dbXLength, fbArgs.dbYLength, fbArgs.dbZLength);
//
//
//
//    runAfterOneLoop(fbArgs, fFArgs, cpuIterNumb);// cpu part
//
//    checkCuda(cudaDeviceSynchronize(), "bb");
//    cudaLaunchCooperativeKernel((void*)(mainDilatation<int>), deviceProp.multiProcessorCount, fFArgs.threadsMainPass, kernel_args);
//    checkCuda(cudaDeviceSynchronize(), "bb");
//    cudaLaunchCooperativeKernel((void*)(getWorkQueeueFromIsToBeActivated<int>), deviceProp.multiProcessorCount, fFArgs.threadsMainPass, kernel_args);
//    checkCuda(cudaDeviceSynchronize(), "bb");
//    cudaLaunchCooperativeKernel((void*)(paddingDilatation<int>), deviceProp.multiProcessorCount, fFArgs.threadsMainPass, kernel_args);
//    checkCuda(cudaDeviceSynchronize(), "bb");
//    cudaLaunchCooperativeKernel((void*)(getWorkQueeueFromActive_mainPass<int>), deviceProp.multiProcessorCount, fFArgs.threadsMainPass, kernel_args);
//
//
//
//    checkCuda(cudaDeviceSynchronize(), "bb");
//
//    deviceTohost
//    copyDeviceToHost3d(forDebug, fFArgs.forDebugArr);
//    copyDeviceToHost3d(goldArr, fFArgs.goldArr);
//    copyDeviceToHost3d(segmArr, fFArgs.segmArr);
//    copyDeviceToHost3d(reducedGold, fFArgs.reducedGold);
//    copyDeviceToHost3d(reducedSegm, fFArgs.reducedSegm);
//
//
//    copyDeviceToHost3d(reducedGold, fFArgs.reducedGold);
//    copyDeviceToHost3d(reducedSegm, fFArgs.reducedSegm);
//    copyDeviceToHost3d(reducedGoldRef, fFArgs.reducedGoldRef);
//    copyDeviceToHost3d(reducedSegmRef, fFArgs.reducedSegmRef);
//    copyDeviceToHost3d(reducedGoldPrev, fFArgs.reducedGoldPrev);
//    copyDeviceToHost3d(reducedSegmPrev, fFArgs.reducedSegmPrev);
//
//    copyMetaDataToCPU(fFArgs.metaData, fbArgs.metaData);
//
//    mainPassKernelTestUnitTests(fFArgs, allPointsA, allMetas, pointsNumber, metasNumber
//        , fbArgs.dbXLength, fbArgs.dbYLength, fbArgs.dbZLength, fFArgs.metaData.MetaZLength, goldArr.Ny, goldArr.Nx);
//
//
//
//    runAfterOneLoop(fbArgs, fFArgs, cpuIterNumb);// cpu part
//    checkCuda(cudaDeviceSynchronize(), "bb");
//    cudaLaunchCooperativeKernel((void*)(mainDilatation<int>), deviceProp.multiProcessorCount, fFArgs.threadsMainPass, kernel_args);
//    checkCuda(cudaDeviceSynchronize(), "bb");
//    cudaLaunchCooperativeKernel((void*)(getWorkQueeueFromIsToBeActivated<int>), deviceProp.multiProcessorCount, fFArgs.threadsMainPass, kernel_args);
//    checkCuda(cudaDeviceSynchronize(), "bb");
//    cudaLaunchCooperativeKernel((void*)(paddingDilatation<int>), deviceProp.multiProcessorCount, fFArgs.threadsMainPass, kernel_args);
//    checkCuda(cudaDeviceSynchronize(), "bb");
//    cudaLaunchCooperativeKernel((void*)(getWorkQueeueFromActive_mainPass<int>), deviceProp.multiProcessorCount, fFArgs.threadsMainPass, kernel_args);
//    checkCuda(cudaDeviceSynchronize(), "bb");
//
//
//
//    deviceTohost
//    copyDeviceToHost3d(forDebug, fFArgs.forDebugArr);
//    copyDeviceToHost3d(goldArr, fFArgs.goldArr);
//    copyDeviceToHost3d(segmArr, fFArgs.segmArr);
//    copyDeviceToHost3d(reducedGold, fFArgs.reducedGold);
//    copyDeviceToHost3d(reducedSegm, fFArgs.reducedSegm);
//
//
//    copyDeviceToHost3d(reducedGold, fFArgs.reducedGold);
//    copyDeviceToHost3d(reducedSegm, fFArgs.reducedSegm);
//    copyDeviceToHost3d(reducedGoldRef, fFArgs.reducedGoldRef);
//    copyDeviceToHost3d(reducedSegmRef, fFArgs.reducedSegmRef);
//    copyDeviceToHost3d(reducedGoldPrev, fFArgs.reducedGoldPrev);
//    copyDeviceToHost3d(reducedSegmPrev, fFArgs.reducedSegmPrev);
//
//    copyMetaDataToCPU(fFArgs.metaData, fbArgs.metaData);
//    checkCuda(cudaDeviceSynchronize(), "bb");
//
//    checkAfterSecondDil(fFArgs, allPointsA, allMetas, pointsNumber, metasNumber, fbArgs.dbXLength, fbArgs.dbYLength, fbArgs.dbZLength);
//
//
//
//    
//
//
//
//    sync
//
//
//    checkCuda(cudaDeviceSynchronize(), "just after copy device to host");
//    cudaGetLastError();
//
//    cudaFree(forDebug.arrPStr.ptr);
//    cudaFree(goldArr.arrPStr.ptr);
//    cudaFree(segmArr.arrPStr.ptr);
//    cudaFree(reducedGold.arrPStr.ptr);
//    cudaFree(reducedSegm.arrPStr.ptr);
//    cudaFree(reducedGoldPrev.arrPStr.ptr);
//    cudaFree(reducedSegmPrev.arrPStr.ptr);
//
//
//    freeMetaDataGPU(fbArgs.metaData);
//
//
//       /*
//    * Catch errors for both the kernel launch above and any
//    * errors that occur during the asynchronous `doubleElements`
//    * kernel execution.
//    */
//
//       syncErr = cudaGetLastError();
//       asyncErr = cudaDeviceSynchronize();
//
//       /*
//        * Print errors should they exist.
//        */
//
//       if (syncErr != cudaSuccess) printf("Error in syncErr: %s\n", cudaGetErrorString(syncErr));
//       if (asyncErr != cudaSuccess) printf("Error in asyncErr: %s\n", cudaGetErrorString(asyncErr));
//
//
//
//    return true;
//}
//

