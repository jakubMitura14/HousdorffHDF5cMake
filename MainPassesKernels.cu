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

    __shared__ uint32_t localBlockMetaData[20];

    /*
 //now linear indexes of the previous block in all sides - if there is no block in given direction it will equal UINT32_MAX

 0 : top
 1 : bottom
 2 : left
 3 : right
 4 : anterior
 5 : posterior

    */

    __shared__ uint32_t localBlockMetaDataOld[20];

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
        , fbArgs.metaDataArrPointer, oldIsGold, oldLinIndM, localBlockMetaDataOld, isGoldForLocQueue, isBlockToBeValidated);





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
ForBoolKernelArgs<int> mainKernelsRun(ForFullBoolPrepArgs<int> fFArgs, uint32_t*& reducedResCPU
    , uint32_t*& resultListPointerMetaCPU
    ,uint32_t*& resultListPointerLocalCPU
    ,uint32_t*& resultListPointerIterNumbCPU
    ,uint32_t*& metaDataArrPointerCPU
    ,uint32_t*& workQueuePointerCPU
    ,uint32_t*& origArrsCPU
) {

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
    //cudaOccupancyMaxPotentialBlockSize(
    //    &minGridSize,
    //    &blockSize,
    //    (void*)mainPassKernel<int>,
    //    0);
    //int warpsNumbForMainPass = blockSize / 32;
    //int blockForMainPass = minGridSize;
    //    printf("warpsNumbForMainPass %d blockForMainPass %d  ", warpsNumbForMainPass, blockForMainPass);


    int warpsNumbForMainPass = 10;
    int blockForMainPass = 4;







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


   int fpPlusFn =  allocateMemoryAfterBoolKernel(fbArgs, fFArgs, resultListPointerMeta, resultListPointerLocal, resultListPointerIterNumb, origArrsPointer, mainArrAPointer, mainArrBPointer, metaData, goldArr, segmArr);

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


    cudaLaunchCooperativeKernel((void*)(mainPassKernel<int>), blockForMainPass, dim3(32, warpsNumbForMainPass), kernel_args);



    checkCuda(cudaDeviceSynchronize(), "a6");

    //copy to CPU
    size_t sizeCPU = metaData.totalMetaLength * metaData.mainArrSectionLength * sizeof(uint32_t);
    reducedResCPU = (uint32_t*)calloc(metaData.totalMetaLength * metaData.mainArrSectionLength, sizeof(uint32_t));
    cudaMemcpy(reducedResCPU, mainArrBPointer, sizeCPU, cudaMemcpyDeviceToHost);


    origArrsCPU = (uint32_t*)calloc(metaData.totalMetaLength * metaData.mainArrSectionLength, sizeof(uint32_t));
    cudaMemcpy(origArrsCPU, origArrsPointer, sizeCPU, cudaMemcpyDeviceToHost);


    size_t sizeRes = sizeof(uint32_t) * (fpPlusFn + 50);


  resultListPointerMetaCPU= (uint32_t*)calloc(fpPlusFn + 50, sizeof(uint32_t));
    resultListPointerLocalCPU= (uint32_t*)calloc(fpPlusFn + 50, sizeof(uint32_t));
   resultListPointerIterNumbCPU= (uint32_t*)calloc(fpPlusFn + 50, sizeof(uint32_t));
   cudaMemcpy(resultListPointerMetaCPU, resultListPointerMeta, sizeRes, cudaMemcpyDeviceToHost);
   cudaMemcpy(resultListPointerLocalCPU, resultListPointerLocal, sizeRes, cudaMemcpyDeviceToHost);
   cudaMemcpy(resultListPointerIterNumbCPU, resultListPointerIterNumb, sizeRes, cudaMemcpyDeviceToHost);

   size_t sizemetaDataArr = metaData.totalMetaLength * (20) * sizeof(uint32_t);
   metaDataArrPointerCPU = (uint32_t*)calloc(metaData.totalMetaLength * (20), sizeof(uint32_t));
   cudaMemcpy(metaDataArrPointerCPU, metaDataArrPointer, sizeRes, cudaMemcpyDeviceToHost);

   size_t sizeC = (metaData.totalMetaLength * sizeof(uint32_t));

   workQueuePointerCPU = (uint32_t*)calloc(metaData.totalMetaLength, sizeof(uint32_t));
   cudaMemcpy(workQueuePointerCPU, workQueuePointer, sizeC, cudaMemcpyDeviceToHost);









    //cudaLaunchCooperativeKernel((void*)mainPassKernel<int>, deviceProp.multiProcessorCount, fFArgs.threadsMainPass, fbArgs);




  //  ////mainPassKernel << <fFArgs.blocksMainPass, fFArgs.threadsMainPass >> > (fbArgs);

    //testKernel << <blockSizeFoboolPrepareKernel, dim3(32, warpsNumbForboolPrepareKernel) >> > (fbArgs, minMaxes, mainArrBPointer, metaData, workQueuePointer, origArrsPointer);

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

    return fbArgs;
}













