#include "CPUAllocations.cu"
#include "MetaData.cu"
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






template <typename TKKI>
inline __global__ void mainPassKernel(ForBoolKernelArgs<TKKI> fbArgs) {

    //inline __global__ void mainPassKernel(ForBoolKernelArgs<TKKI> fbArgs, uint32_t * mainArr, MetaDataGPU metaData
    //    , unsigned int* minMaxes, uint32_t * workQueue
    //    , uint32_t * resultListPointerMeta, uint32_t * resultListPointerLocal, uint32_t * resultListPointerIterNumb, uint32_t * origArrs, uint32_t * metaDataArr) {

    //if (threadIdx.x == 0 && threadIdx.y == 0) {
    //    printf("in metadataPass totalMetaLength  %d   \n", fbArgs.metaData.totalMetaLength);

    //};

    thread_block cta = cooperative_groups::this_thread_block();

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



    constexpr size_t stages_count = 2; // Pipeline stages number

    // Allocate shared storage for a two-stage cuda::pipeline:
    __shared__ cuda::pipeline_shared_state<
        cuda::thread_scope::thread_scope_block,
        stages_count
    > shared_state;

    //cuda::pipeline<cuda::thread_scope_thread>  pipeline = cuda::make_pipeline(cta, &shared_state);
    cuda::pipeline<cuda::thread_scope_block>  pipeline = cuda::make_pipeline(cta, &shared_state);



    //usefull for iterating through local work queue
    __shared__ bool isGoldForLocQueue[localWorkQueLength];
    // holding data about paddings 


    // holding data weather we have anything in padding 0)top  1)bottom, 2)left 3)right, 4)anterior, 5)posterior,
    __shared__ bool isAnythingInPadding[6];

    __shared__ bool isBlockFull[1];

    __shared__ uint32_t lastI[1];


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

    __shared__ uint32_t localBlockMetaData[40];

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




    do{
        if (threadIdx.x == 2 && threadIdx.y == 0) {
    if (blockIdx.x == 0) {
     //   printf("iter nuumb %d \n", iterationNumb[0]);
      //  fbArgs.metaData.minMaxes[13] = iterationNumb[0];
    }
};

        mainDilatation(false, fbArgs, fbArgs.mainArrAPointer, fbArgs.mainArrBPointer, fbArgs.metaData, fbArgs.minMaxes
            , fbArgs.workQueuePointer
            , fbArgs.resultListPointerMeta, fbArgs.resultListPointerLocal, fbArgs.resultListPointerIterNumb
            , cta, tile, grid, mainShmem
            , isAnythingInPadding, isBlockFull, iterationNumb, globalWorkQueueOffset
            , globalWorkQueueCounter
            , localWorkQueueCounter
            , localTotalLenthOfWorkQueue
            , localFpConter
            , localFnConter, blockFpConter
            , blockFnConter
            , resultfpOffset
            , resultfnOffset, worQueueStep, localMinMaxes
            , localBlockMetaData, fpFnLocCounter
            , isGoldPassToContinue, isSegmPassToContinue
            , fbArgs.origArrsPointer
            , fbArgs.metaDataArrPointer, isGoldForLocQueue
            , lastI, pipeline

        );

        grid.sync();

        ///////////// loading work queue for padding dilatations
        metadataPass(fbArgs, true, 11, 7, 8,
            12, 9, 10
            , mainShmem, globalWorkQueueOffset, globalWorkQueueCounter
            , localWorkQueueCounter, localTotalLenthOfWorkQueue, localMinMaxes
            , fpFnLocCounter, isGoldPassToContinue, isSegmPassToContinue, cta, tile
            , fbArgs.metaData, fbArgs.minMaxes, fbArgs.workQueuePointer, fbArgs.metaDataArrPointer);




        //////////// padding dilatations
        grid.sync();
        mainDilatation(true, fbArgs, fbArgs.mainArrAPointer, fbArgs.mainArrBPointer, fbArgs.metaData, fbArgs.minMaxes
            , fbArgs.workQueuePointer
            , fbArgs.resultListPointerMeta, fbArgs.resultListPointerLocal, fbArgs.resultListPointerIterNumb
            , cta, tile, grid, mainShmem
            , isAnythingInPadding, isBlockFull, iterationNumb, globalWorkQueueOffset
            , globalWorkQueueCounter
            , localWorkQueueCounter
            , localTotalLenthOfWorkQueue
            , localFpConter
            , localFnConter, blockFpConter
            , blockFnConter
            , resultfpOffset
            , resultfnOffset, worQueueStep, localMinMaxes
            , localBlockMetaData, fpFnLocCounter
            , isGoldPassToContinue, isSegmPassToContinue
            , fbArgs.origArrsPointer
            , fbArgs.metaDataArrPointer, isGoldForLocQueue
            , lastI, pipeline

        );


        grid.sync();
        ////////////////////////main metadata pass
        metadataPass(fbArgs, false, 7, 8, 8,
            9, 10, 8
            , mainShmem, globalWorkQueueOffset, globalWorkQueueCounter
            , localWorkQueueCounter, localTotalLenthOfWorkQueue, localMinMaxes
            , fpFnLocCounter, isGoldPassToContinue, isSegmPassToContinue, cta, tile
            , fbArgs.metaData, fbArgs.minMaxes, fbArgs.workQueuePointer, fbArgs.metaDataArrPointer);
        grid.sync();
        //if (tile.thread_rank() == 12 && tile.meta_group_rank() == 0) {
        //    printf("  isGoldPassToContinue %d isSegmPassToContinue %d \n ", isGoldPassToContinue[0], isSegmPassToContinue[0]);
        //};
    
    } while (isGoldPassToContinue[0] || isSegmPassToContinue[0]);

    //grid.sync();

    ////for final result
    //if (threadIdx.x == 2 && threadIdx.y == 0) {
    //    if (blockIdx.x == 0) {

    //      //  fbArgs.metaData.minMaxes[13] = iterationNumb[0];
    //    }
    //};
    

    //grid.sync();


    //if (tile.thread_rank() == 12 && tile.meta_group_rank() == 0) {
    //    printf("  isGoldPassToContinue %d isSegmPassToContinue %d \n ", isGoldPassToContinue[0], isSegmPassToContinue[0]);
    //};

//  }// end while

  //setting global iteration number to local one 

}





#pragma once
template <typename T>
ForBoolKernelArgs<T> mainKernelsRun(ForFullBoolPrepArgs<T> fFArgs, uint32_t*& reducedResCPU
    , uint32_t*& resultListPointerMetaCPU
    ,uint32_t*& resultListPointerLocalCPU
    ,uint32_t*& resultListPointerIterNumbCPU
    ,uint32_t*& metaDataArrPointerCPU
    ,uint32_t*& workQueuePointerCPU
    ,uint32_t*& origArrsCPU
    , const int WIDTH, const int HEIGHT, const int DEPTH
) {

    //cudaDeviceReset();
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
        (void*)getMinMaxes<T>,
        0);
    int warpsNumbForMinMax = blockSize / 32;
    int blockSizeForMinMax = minGridSize;

    // for min maxes kernel 
    cudaOccupancyMaxPotentialBlockSize(
        &minGridSize,
        &blockSize,
        (void*)boolPrepareKernel<T>,
        0);
    int warpsNumbForboolPrepareKernel = blockSize / 32;
    int blockSizeFoboolPrepareKernel = minGridSize;
    // for first meta pass kernel
    cudaOccupancyMaxPotentialBlockSize(
        &minGridSize,
        &blockSize,
        (void*)boolPrepareKernel<T>,
        0);
    int theadsForFirstMetaPass = blockSize;
    int blockForFirstMetaPass = minGridSize;
    //for main pass kernel
    cudaOccupancyMaxPotentialBlockSize(
        &minGridSize,
        &blockSize,
        (void*)mainPassKernel<T>,
        0);
    int warpsNumbForMainPass = blockSize / 32;
    int blockForMainPass = minGridSize;
        printf("warpsNumbForMainPass %d blockForMainPass %d  ", warpsNumbForMainPass, blockForMainPass);


   // warpsNumbForMainPass = 5;
  //  blockForMainPass = 1;

        




    //pointers ...
    uint32_t* resultListPointerMeta;
    uint32_t* resultListPointerLocal;
    uint32_t* resultListPointerIterNumb;

    uint32_t* origArrsPointer;
    uint32_t* mainArrAPointer;
    uint32_t* mainArrBPointer;
    uint32_t* metaDataArrPointer;

    uint32_t* workQueuePointer;

    
    
    //main arrays allocations
    T* goldArrPointer;
    T* segmArrPointer;
    //size_t sizeMainArr = (sizeof(T) * WIDTH * HEIGHT * DEPTH);
    size_t sizeMainArr = (sizeof(T) * WIDTH * HEIGHT * DEPTH);

    cudaMallocAsync(&goldArrPointer, sizeMainArr,0);
    cudaMallocAsync(&segmArrPointer, sizeMainArr,0);

    cudaMemcpyAsync(goldArrPointer, fFArgs.goldArr.arrP, sizeMainArr, cudaMemcpyHostToDevice, 0);
    cudaMemcpyAsync(segmArrPointer, fFArgs.segmArr.arrP, sizeMainArr, cudaMemcpyHostToDevice, 0);


    array3dWithDimsGPU<T> goldArr;
    array3dWithDimsGPU<T> segmArr;

    goldArr.arrP = goldArrPointer;
    goldArr.Nx = WIDTH;
    goldArr.Ny = HEIGHT;
    goldArr.Nz = DEPTH;



    segmArr.arrP = segmArrPointer;
    segmArr.Nx = WIDTH;
    segmArr.Ny = HEIGHT;
    segmArr.Nz = DEPTH;
    checkCuda(cudaDeviceSynchronize(), "a0a");

    unsigned int* minMaxes;
    size_t sizeminMaxes = sizeof(unsigned int) * 20;
    cudaMallocAsync(&minMaxes, sizeminMaxes,0);




    checkCuda(cudaDeviceSynchronize(), "a0b");
    ForBoolKernelArgs<T> fbArgs = getArgsForKernel<T>(fFArgs, goldArrPointer, segmArrPointer, minMaxes, warpsNumbForMainPass, blockForMainPass, WIDTH,HEIGHT, DEPTH);
    MetaDataGPU metaData = fbArgs.metaData;
    fbArgs.metaData.minMaxes = minMaxes;
    fbArgs.minMaxes = minMaxes;


    fbArgs.goldArr = goldArr;
    fbArgs.segmArr = segmArr;


    ////preparation kernel

    // initialize, then launch

    checkCuda(cudaDeviceSynchronize(), "a1");


    //getMinMaxes << <blockSizeForMinMax, dim3(32, warpsNumbForMinMax) >> > ( minMaxes);
    getMinMaxes << <blockSizeForMinMax, dim3(32, warpsNumbForMinMax) >> > (fbArgs, minMaxes, goldArrPointer, segmArrPointer);

    checkCuda(cudaDeviceSynchronize(), "a1b");


    checkCuda(cudaDeviceSynchronize(), "a2a");

    metaData = allocateMemoryAfterMinMaxesKernel(fbArgs, fFArgs, workQueuePointer, minMaxes, metaData, origArrsPointer, metaDataArrPointer);

    checkCuda(cudaDeviceSynchronize(), "a2b");

   boolPrepareKernel << <blockSizeFoboolPrepareKernel, dim3(32, warpsNumbForboolPrepareKernel) >> > (fbArgs, metaData, origArrsPointer, metaDataArrPointer, goldArrPointer, segmArrPointer, minMaxes);
  //  //uint32_t* origArrs, uint32_t* metaDataArr     metaDataArr[linIdexMeta * metaData.metaDataSectionLength     metaDataOffset

   checkCuda(cudaDeviceSynchronize(), "a3");



  int fpPlusFn =  allocateMemoryAfterBoolKernel(fbArgs, fFArgs, resultListPointerMeta, resultListPointerLocal, resultListPointerIterNumb, origArrsPointer, mainArrAPointer, mainArrBPointer, metaData,goldArr,segmArr);




    checkCuda(cudaDeviceSynchronize(), "a4");

    //cudaFreeAsync(goldArrPointer, 0);
    //cudaFreeAsync(segmArrPointer, 0);

    firstMetaPrepareKernel << <blockForFirstMetaPass, theadsForFirstMetaPass >> > (fbArgs, metaData, minMaxes, workQueuePointer, origArrsPointer, metaDataArrPointer);

   checkCuda(cudaDeviceSynchronize(), "a5");
    //void* kernel_args[] = { &fbArgs, mainArrPointer,&metaData,minMaxes, workQueuePointer,resultListPointerMeta,resultListPointerLocal, resultListPointerIterNumb };
    
    
    
    //fbArgs.goldArr = goldArr;
    //fbArgs.segmArr = segmArr;
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



    size_t sizeMinnMax  = sizeof(unsigned int) * 20;

    cudaMemcpy(fFArgs.metaData.minMaxes, minMaxes, sizeMinnMax, cudaMemcpyDeviceToHost);

    //copy to CPU
    size_t sizeCPU = metaData.totalMetaLength * metaData.mainArrSectionLength * sizeof(uint32_t);
    reducedResCPU = (uint32_t*)calloc(metaData.totalMetaLength * metaData.mainArrSectionLength, sizeof(uint32_t));
    cudaMemcpy(reducedResCPU, mainArrAPointer, sizeCPU, cudaMemcpyDeviceToHost);

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
   cudaMemcpy(metaDataArrPointerCPU, metaDataArrPointer, sizemetaDataArr, cudaMemcpyDeviceToHost);

   size_t sizeC = (metaData.totalMetaLength * sizeof(uint32_t));

   workQueuePointerCPU = (uint32_t*)calloc(metaData.totalMetaLength, sizeof(uint32_t));
   cudaMemcpy(workQueuePointerCPU, workQueuePointer, sizeC, cudaMemcpyDeviceToHost);



   checkCuda(cudaDeviceSynchronize(), "a7");






  //  //cudaLaunchCooperativeKernel((void*)mainPassKernel<int>, deviceProp.multiProcessorCount, fFArgs.threadsMainPass, fbArgs);




  //  ////copyDeviceToHost3d(goldArr, fFArgs.goldArr);
  //  ////copyDeviceToHost3d(segmArr, fFArgs.segmArr);
  //  //// getting arrays allocated on  cpu to 


  //  //copyMetaDataToCPU(fFArgs.metaData, fbArgs.metaData);

  //  //// printForDebug(fbArgs, fFArgs, resultListPointer, mainArrPointer, workQueuePointer, metaData);


  //  checkCuda(cudaDeviceSynchronize(), "just after copy device to host");
  //  //cudaGetLastError();

  //cudaFreeAsync(goldArrPointer, 0);
  //cudaFreeAsync(segmArrPointer, 0);


    cudaFreeAsync(resultListPointerMeta, 0);
    cudaFreeAsync(resultListPointerLocal, 0);
    cudaFreeAsync(resultListPointerIterNumb, 0);
    cudaFreeAsync(workQueuePointer, 0);
    cudaFreeAsync(origArrsPointer, 0);
    cudaFreeAsync(metaDataArrPointer, 0);
    cudaFreeAsync(mainArrAPointer, 0);
    cudaFreeAsync(mainArrBPointer, 0);



    checkCuda(cudaDeviceSynchronize(), "last ");

/////////// error handling 
    syncErr = cudaGetLastError();
    asyncErr = cudaDeviceSynchronize();
    if (syncErr != cudaSuccess) printf("Error in syncErr: %s\n", cudaGetErrorString(syncErr));
    if (asyncErr != cudaSuccess) printf("Error in asyncErr: %s\n", cudaGetErrorString(asyncErr));


    cudaDeviceReset();

    ForBoolKernelArgs<T> res;
    return res;
   // return fbArgs;
}













