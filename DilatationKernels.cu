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




template <typename TKKI>
inline __device__ void mainDilatation(bool isPaddingPass, ForBoolKernelArgs<TKKI> fbArgs, uint32_t* mainArr, MetaDataGPU metaData
    , unsigned int* minMaxes, uint32_t* workQueue
    , uint32_t* resultListPointerMeta, uint16_t* resultListPointerLocal, uint16_t* resultListPointerIterNumb,
    thread_block cta, thread_block_tile<32> tile, grid_group grid, uint32_t mainShmem[lengthOfMainShmem]
    , bool isAnythingInPadding[6]  , bool isBlockFull[1], uint32_t iterationNumb[1], unsigned int globalWorkQueueOffset[1],
    unsigned int globalWorkQueueCounter[1], unsigned int localWorkQueueCounter[1],
    unsigned int localTotalLenthOfWorkQueue[1], unsigned int localFpConter[1],
    unsigned int localFnConter[1], unsigned int blockFpConter[1],
    unsigned int blockFnConter[1], unsigned int resultfpOffset[1],
    unsigned int resultfnOffset[1], unsigned int worQueueStep[1],
    uint32_t isGold[1], uint32_t currLinIndM[1], unsigned int localMinMaxes[5]
    , uint16_t localBlockMetaData[20], unsigned int fpFnLocCounter[1]
    , bool isGoldPassToContinue[1], bool isSegmPassToContinue[1]
    , uint32_t* origArrs, uint16_t* metaDataArr
) {
    auto pipeline = cuda::make_pipeline();
    auto bigShape = cuda::aligned_size_t<128>(sizeof(uint32_t) * (metaData.mainArrXLength));
    auto thirdRegShape = cuda::aligned_size_t<128>(sizeof(uint32_t) * (32));
    thread_block_tile<1> miniTile = tiled_partition<1>(block);


    if (tile.thread_rank() == 7 && tile.meta_group_rank() == 0  && !isPaddingPass) {
        iterationNumb[0] += 1;
    };

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
    if (tile.thread_rank() == 9 && tile.meta_group_rank() == 0) {
        isBlockFull[0] =true;
    };
    if (tile.thread_rank() == 10 && tile.meta_group_rank() == 0) {
        fpFnLocCounter[0] = 0;
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
    /// load work QueueData into shared memory 
    for (uint16_t bigloop = blockIdx.x * globalWorkQueueOffset[0]; bigloop < ((blockIdx.x + 1) * globalWorkQueueOffset[0]); bigloop += worQueueStep[0]) {
        // grid stride loop - sadly most of threads will be idle 
        /////////// loading to work queue
        
        cooperative_groups::memcpy_async(cta, (&mainShmem[startOfLocalWorkQ]), (&workQueue[bigloop]), cuda::aligned_size_t<4>(sizeof(uint32_t) * worQueueStep[0]));
        sync(cta);
        //now all of the threads in the block needs to have the same i value so we will increment by 1
        // we are preloading to the pipeline block metaData
        ////##### pipeline Step 0
        pipeline.producer_acquire();

        cuda::memcpy_async(cta, (&localBlockMetaData[0]), (&metaDataArr[(mainShmem[startOfLocalWorkQ] - UINT16_MAX * (mainShmem[startOfLocalWorkQ] >= UINT16_MAX)) * metaData.metaDataSectionLength])
            , cuda::aligned_size_t<4>(sizeof(uint16_t) * 20), pipeline);



        pipeline.producer_commit();
        
        for (uint16_t i = 0; i < worQueueStep[0]; i += 1) {
            if (((bigloop + i) < localTotalLenthOfWorkQueue[0]) && ((bigloop + i) < ((blockIdx.x + 1) * globalWorkQueueOffset[0]))) {
                 ///#### pipeline step 1) now we load data for next step (to mainly sourceshmem and left-right if apply) and process data loaded in previous step
                    pipeline.producer_acquire();

                    cuda::memcpy_async(cta, (&mainShmem[(((mainShmem[startOfLocalWorkQ + i] - UINT16_MAX * (mainShmem[startOfLocalWorkQ + i] >= UINT16_MAX)) > 0) * (-32)) + begSourceShmem]), // we check weather there is anything to the left - not on left border if so we need place for left 32 entries
                        &mainArr[((mainShmem[startOfLocalWorkQ + i] - UINT16_MAX * (mainShmem[startOfLocalWorkQ + i] >= UINT16_MAX)) *(-32)) // we check weather there is anything to the left - not on left border if so we load left 32 entries
                        +  getIndexForSourceShmem(metaData, mainShmem, iterationNumb,i )] , 
                        cuda::aligned_size_t<128>(sizeof(uint32_t) * //below we check weather we have block to the left and right if so we increase number of copied entries
                            (metaData.mainArrXLength+32*(((mainShmem[startOfLocalWorkQ + i] - UINT16_MAX * (mainShmem[startOfLocalWorkQ + i] >= UINT16_MAX)) > 0)
                                + ((mainShmem[startOfLocalWorkQ + i] - UINT16_MAX * (mainShmem[startOfLocalWorkQ + i] >= UINT16_MAX)) <(metaData.totalMetaLength -1)))   ))
                        , pipeline);


                    pipeline.producer_commit();

        //        ////compute first we load data about calculated linear index meta and information is it gold iteration ...
                   pipeline.consumer_wait();
                       if (tile.thread_rank() == 0 && tile.meta_group_rank() == 0) {// this is how it is encoded wheather it is gold or segm block
                           isGold[0] = uint32_t(mainShmem[startOfLocalWorkQ + i] >= UINT16_MAX);
                           if (isGold[0]) {
                               //removing info about wheather it is gold or not pass so we will be able to use it as linear metadata index
                               currLinIndM[0] = mainShmem[startOfLocalWorkQ + i] - UINT16_MAX;
                           }
                       };
                      if (tile.thread_rank() <6 && tile.meta_group_rank() == 1) {// this is how it is encoded wheather it is gold or segm block
                            isAnythingInPadding[0] = false;                       
                       };
                   pipeline.consumer_release();


               ////////#### pipeline step 2) 
               //load for next step - so we load posterior of anterior block  and anterior of posterior block given they exist
                   //anterior and posterior
                   if (localBlockMetaData[17] < UINT16_MAX   || (localBlockMetaData[18] < UINT16_MAX) {
                       pipeline.producer_acquire();
                           //posterior of the block to anterior we load it using single threads and multple mempcy async becouse memory is non aligned
                           if (localBlockMetaData[17] < UINT16_MAX  && miniTile.meta_group_rank()< fbArgs.dbYLength) {
                               cooperative_groups::memcpy_async(miniTile, (&mainShmem[begfirstRegShmem+32+ miniTile.meta_group_rank()]),
                                   (&mainArr[getIndexForNeighbourForShmem(metaData, mainShmem, iterationNumb, isGold, currLinIndM, localBlockMetaData, 17)] //basic offset
                                       //we look for indicies 0,32,64... up to metaData.mainArrXLength
                                       + miniTile.meta_group_rank()*32
                                       )
                                   , cuda::aligned_size_t<4>(sizeof(uint32_t)), pipeline);
                           }
                           //anterior of the block to posterior
                           if (localBlockMetaData[18] < UINT16_MAX&& miniTile.meta_group_rank()< fbArgs.dbYLength*2) {
                               cooperative_groups::memcpy_async(miniTile, (&mainShmem[begfirstRegShmem+64+ miniTile.meta_group_rank() ]),
                                   (&mainArr[getIndexForNeighbourForShmem(metaData, mainShmem, iterationNumb, isGold, currLinIndM, localBlockMetaData, 18)
                                       //we look for indicies 31,63... up to metaData.mainArrXLength
                                       + (miniTile.meta_group_rank() * 32)+31
                                   ])
                                   , cuda::aligned_size_t<4>(sizeof(uint32_t)), pipeline);
                           }
                       pipeline.producer_commit();
                   }
                     //compute - now we have data in source shmem about this block and left and right padding
                       pipeline.consumer_wait();
                           // first we perform up and down dilatations inside the block
                           mainShmem[begResShmem+threadIdx.x+threadIdx.y*32] = bitDilatate(mainShmem[begSourceShmem+threadIdx.x + threadIdx.y * 32]);
                           //we also do the left and right dilatations
                           if (localBlockMetaData[17] < UINT16_MAX) {

                           };
                           if (localBlockMetaData[18] < UINT16_MAX) {

                           };

                       pipeline.consumer_release();


        //         ////////#### pipeline step 3) process anterior block data and load posterior
        //        loadNextAndProcessPreviousSides(pipeline,cta//some needed CUDA objects
        //        localBlockMetaData,mainShmem,iterationNumb,isGold, currLinIndM// shared memory arrays used block wide
        //        , metaData,mainArr, //pointers to arrays with data
        //        //now some variables needed to load data  
        //            18 // where is the index describing linear index of the neighbour in direction of intrest
        //            ,begSecRegShmem //offset defined in shared memory used to load data into 
        //            , bigShape // shape and alignment of data in load - inludes length of data
        //        //now variables needed for dilatations we dilatate to anterior
        //            17 // where is the index describing linear index of the neighbour in direction of intrest
        //            ,begfirstRegShmem//offset defined in shared memory used to process  data from 
        //        ,(threadIdx.y == (fbArgs.dbYLength - 1) // defining when our thread is a corner case and need to load data from outside of the block
        //        , 4,// needed to know wheather block in given direction should be marked as to be activated
        //        (0), (1)// x and y changes
        //        , 0, threadIdx.x// coordinates in new block

        //         ////////#### pipeline step 4) process posterior block data and load right
        //        loadNextAndProcessPreviousSides(pipeline,cta//some needed CUDA objects
        //        localBlockMetaData,mainShmem,iterationNumb,isGold, currLinIndM// shared memory arrays used block wide
        //        , metaData,mainArr, //pointers to arrays with data
        //        //now some variables needed to load data  
        //            16 // where is the index describing linear index of the neighbour in direction of intrest
        //            ,begfirstRegShmem //offset defined in shared memory used to load data into 
        //            , thirdRegShape // shape and alignment of data in load - inludes length of data
        //        //now variables needed for dilatations we dilatate to anterior
        //            18 // where is the index describing linear index of the neighbour in direction of intrest
        //            ,begSecRegShmem//offset defined in shared memory used to process  data from 
        //        ,(threadIdx.y == 0) // defining when our thread is a corner case and need to load data from outside of the block
        //        , 5,// needed to know wheather block in given direction should be marked as to be activated
        //        (0), (-1)// x and y changes
        //        , (fbArgs.dbYLength - 1), threadIdx.x)// coordinates in new block


        //         ////////#### pipeline step 5) process right block data and load left
        //        loadNextAndProcessPreviousSides(pipeline,cta//some needed CUDA objects
        //        localBlockMetaData,mainShmem,iterationNumb,isGold, currLinIndM// shared memory arrays used block wide
        //        , metaData,mainArr, //pointers to arrays with data
        //        //now some variables needed to load data  
        //            15 // where is the index describing linear index of the neighbour in direction of intrest
        //            ,begSecRegShmem //offset defined in shared memory used to load data into 
        //            , bigShape // shape and alignment of data in load - inludes length of data
        //        //now variables needed for dilatations we dilatate to anterior
        //            16 // where is the index describing linear index of the neighbour in direction of intrest
        //            ,begfirstRegShmem//offset defined in shared memory used to process  data from 
        //        ,(threadIdx.x == (fbArgs.dbXLength - 1) // defining when our thread is a corner case and need to load data from outside of the block
        //        , 3,// needed to know wheather block in given direction should be marked as to be activated
        //        (1), (0)// x and y changes
        //        , threadIdx.y, 0// coordinates in new block


        //         ////////#### pipeline step 6) process left block data and load top
        //        loadNextAndProcessPreviousSides(pipeline,cta//some needed CUDA objects
        //        localBlockMetaData,mainShmem,iterationNumb,isGold, currLinIndM// shared memory arrays used block wide
        //        , metaData,mainArr, //pointers to arrays with data
        //        //now some variables needed to load data  
        //            13 // where is the index describing linear index of the neighbour in direction of intrest
        //            ,begfirstRegShmem //offset defined in shared memory used to load data into 
        //            , bigShape // shape and alignment of data in load - inludes length of data
        //        //now variables needed for dilatations we dilatate to anterior
        //            15 // where is the index describing linear index of the neighbour in direction of intrest
        //            ,begSecRegShmem //offset defined in shared memory used to process  data from 
        //        ,(threadIdx.x == 0) // defining when our thread is a corner case and need to load data from outside of the block
        //        , 2,// needed to know wheather block in given direction should be marked as to be activated
        //        (-1), (0)// x and y changes
        //        , threadIdx.y, (fbArgs.dbXLength - 1))// coordinates in new block

        //         ////////#### pipeline step 7) process top block data and load bottom
        //            if (localBlockMetaData[14]<UINT16_MAX) {
        //                pipeline.producer_acquire();
        //                   cooperative_groups::memcpy_async(cta, (&mainShmem[begSecRegShmem]),
        //                    (&mainArr[getIndexForNeighbourForShmem(metaData, mainShmem, iterationNumb, isGold, currLinIndM, localBlockMetaData,14 )])
        //                    , bigShape, pipeline);
        //                pipeline.producer_commit();
        //              }
        //        //compute
        //        pipeline.consumer_wait();
        //        dilatateHelperTopDown(0, mainShmem, isAnythingInPadding, pipeline, localBlockMetaData, 13,
        //            , 1// represent a uint32 number that has a bit of intrest in this block set and all others 0 here first bit is set
        //            , 2147483648
        //            , begfirstRegShmem);
        //            pipeline.consumer_release();    
        //         ////////#### pipeline step 8) process bottom block data  - do final operations for a block and load reference data if block is to be validated
        //        // now we need to establish weather this block should be validated so weahter the counter in metadata is smaller than metadata count
        //        //load

        //        if( localBlockMetaData[((1-isGold[0])+1)] //fp for gold and fn count for not gold
        //            > localBlockMetaData[((1-isGold[0])+1)]   ){// so count is bigger than counter so we should validate
        //        //now we load data from referenca arrays 
    

        //        }else{//if we are not validating we immidiately start loading data for next loop
        //            lastLoad(pipeline,cta,worQueueStep, localBlockMetaData, mainArr, mainShmem, i, metaData
        //        )
        //        }

        //        //compute bottom block data
        //        pipeline.consumer_wait();

        //        dilatateHelperTopDown(1, mainShmem, isAnythingInPadding, pipeline,localBlockMetaData,14, 
        //                , 2147483648// represent a uint32 number that has a bit of intrest in this block set and all others 0 here last bit is set
        //                , 1
        //                ,begfirstRegShmem)

        //        
        //         krowa additionally we need to establish and save information is block full and mark neighbouring blocks as to be activated if it is not a padding pass       
        //                we also need to save results of res shmem into dilatation array
        //
        //
        //        pipeline.consumer_release();    
        //         ////////#### pipeline step 9 ) this step exists only  if block is to be validated 
        //        if( localBlockMetaData[((1-isGold[0])+1)] //fp for gold and fn count for not gold
        //            > localBlockMetaData[((1-isGold[0])+1)]   ){// so count is bigger than counter so we should validate
        //            lastLoad(pipeline,cta/
        //        worQueueStep, localBlockMetaData, mainArr, mainShmem, i, metaData)
        //            //here we are establishing weather we have any results if so we save it to global memory
    
    
        //        }












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
    if (tile.thread_rank() == 0 && tile.meta_group_rank() == 0) {
        atomicAdd(&(minMaxes[10]), (blockFpConter[0]));
    };
    if (tile.thread_rank() == 1 && tile.meta_group_rank() == 0) {
         atomicAdd(&(minMaxes[11]), (blockFnConter[0]));
    };
    // in first thread block we zero work queue counter
    if (tile.thread_rank() == 2 && tile.meta_group_rank() == 0) {
        if (blockIdx.x==0) {
            minMaxes[9] = 0;
        }
    };





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
