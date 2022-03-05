
#include "cuda_runtime.h"
#include "MetaData.cu"

#include "ExceptionManagUtils.cu"
#include <cstdint>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cooperative_groups/memcpy_async.h>
#include <cuda/pipeline>


#include "MetaData.cu"
#include "ExceptionManagUtils.cu"
#include "ForBoolKernel.cu"
#include "FirstMetaPass.cu"

#include <cuda/annotated_ptr>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include "MinMaxesKernel.cu"
#include "MainKernelMetaHelpers.cu"
#include <cooperative_groups/memcpy_async.h>
using namespace cooperative_groups;



//#include <torch/extension.h>
//#include <iostream>


//#include "hdf5Manag.cu"
#include <iostream>
#include <string>
#include <vector>
#define H5_BUILT_AS_DYNAMIC_LIB 1
#include <H5Cpp.h>


//#include "xlsxwriter.h
//#include "torch/torch.h"
//#include <torch/extension.h>
//#include <iostream>



/*
gettinng  array for dilatations
basically arrays will alternate between iterations once one will be source other target then they will switch - we will decide upon knowing
wheather the iteration number is odd or even
*/
#pragma once
template <typename TXPI>
inline __device__ uint32_t* getSourceReduced(const ForBoolKernelArgs<TXPI>& fbArgs, const int(&iterationNumb)[1]) {


    if ((iterationNumb[0] & 1) == 0) {
        return fbArgs.mainArrAPointer;

    }
    else {
        return fbArgs.mainArrBPointer;
    }


}


/*
gettinng target array for dilatations
*/
#pragma once
template <typename TXPPI>
inline __device__ uint32_t* getTargetReduced(const ForBoolKernelArgs<TXPPI>& fbArgs, const  int(&iterationNumb)[1]) {

    if ((iterationNumb[0] & 1) == 0) {
        //printf(" BB ");

        return fbArgs.mainArrBPointer;

    }
    else {
        // printf(" AA ");

        return fbArgs.mainArrAPointer;

    }

}


/*
dilatation up and down - using bitwise operators
*/
#pragma once
inline __device__ uint32_t bitDilatate(const uint32_t& x) {
    return ((x) >> 1) | (x) | ((x) << 1);
}

/*
return 1 if at given position of given number bit is set otherwise 0
*/
#pragma once
inline __device__ uint32_t isBitAt(const uint32_t& numb, const int pos) {
    return (numb & (1 << (pos)));
}

#pragma once
inline uint32_t isBitAtCPU(const uint32_t& numb, const int pos) {
    return (numb & (1 << (pos)));
}







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




template <typename TKKI>
inline __global__ void mainPassKernel(ForBoolKernelArgs<TKKI> fbArgs) {



    thread_block cta = cooperative_groups::this_thread_block();

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
    // __shared__ uint32_t mainShmem[lengthOfMainShmem];
    __shared__ uint32_t mainShmem[lengthOfMainShmem];
    cuda::associate_access_property(&mainShmem, cuda::access_property::shared{});



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

    __shared__ bool isBlockFull[2];

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
    if (threadIdx.x == 9 && threadIdx.y == 0) { iterationNumb[0] = -1; };
    if (threadIdx.x == 11 && threadIdx.y == 0) {
        isGoldPassToContinue[0] = true;
    };
    if (threadIdx.x == 12 && threadIdx.y == 0) {
        isSegmPassToContinue[0] = true;

    };


    //here we caclulate the offset for given block depending on length of the workqueue and number of the  available blocks in a grid
    // - this will give us number of work queue items per block - we will calculate offset on the basis of the block number
    sync(cta);

    do {

        for (uint8_t isPaddingPass = 0; isPaddingPass < 2; isPaddingPass++) {


            /////////////////////////****************************************************************************************************************  
            /////////////////////////****************************************************************************************************************  
            /////////////////////////****************************************************************************************************************  
            /////////////////////////****************************************************************************************************************  
            /////////////////////////****************************************************************************************************************  
            /// dilataions

    //initial cleaning  and initializations include loading min maxes
            if (threadIdx.x == 7 && threadIdx.y == 0 && !isPaddingPass) {
                iterationNumb[0] += 1;
            };

            if (threadIdx.x == 6 && threadIdx.y == 0) {
                localWorkQueueCounter[0] = 0;
            };

            if (threadIdx.x == 1 && threadIdx.y == 0) {
                blockFpConter[0] = 0;
            };
            if (threadIdx.x == 2 && threadIdx.y == 0) {
                blockFnConter[0] = 0;
            };
            if (threadIdx.x == 3 && threadIdx.y == 0) {
                localFpConter[0] = 0;
            };
            if (threadIdx.x == 4 && threadIdx.y == 0) {
                localFnConter[0] = 0;
            };
            if (threadIdx.x == 9 && threadIdx.y == 0) {
                isBlockFull[0] = true;
            };
            if (threadIdx.x == 9 && threadIdx.y == 1) {
                isBlockFull[1] = true;
            };

            if (threadIdx.x == 10 && threadIdx.y == 0) {
                fpFnLocCounter[0] = 0;
            };


            if (threadIdx.x == 10 && threadIdx.y == 2) {// this is how it is encoded wheather it is gold or segm block

                lastI[0] = UINT32_MAX;
            };


            if (threadIdx.x == 0 && threadIdx.y == 0) {
                localTotalLenthOfWorkQueue[0] = fbArgs.minMaxes[9];
                globalWorkQueueOffset[0] = floor((float)(localTotalLenthOfWorkQueue[0] / gridDim.x)) + 1;
                worQueueStep[0] = min(localWorkQueLength, globalWorkQueueOffset[0]);
            };

            if (threadIdx.y == 1) {
                cooperative_groups::memcpy_async(cta, (&localMinMaxes[0]), (&fbArgs.minMaxes[7]), cuda::aligned_size_t<4>(sizeof(unsigned int) * 5));
            }

            sync(cta);

            /// load work QueueData into shared memory 
            for (uint32_t bigloop = blockIdx.x * globalWorkQueueOffset[0]; bigloop < ((blockIdx.x + 1) * globalWorkQueueOffset[0]); bigloop += worQueueStep[0]) {

                //grid stride loop - sadly most of threads will be idle 
               ///////// loading to work queue
                if (((bigloop) < localTotalLenthOfWorkQueue[0]) && ((bigloop) < ((blockIdx.x + 1) * globalWorkQueueOffset[0]))) {

                    for (uint16_t ii = cta.thread_rank(); ii < worQueueStep[0]; ii += cta.size()) {

                        mainShmem[startOfLocalWorkQ + ii] = fbArgs.workQueuePointer[bigloop + ii];
                        isGoldForLocQueue[ii] = (mainShmem[startOfLocalWorkQ + ii] >= isGoldOffset);
                        mainShmem[startOfLocalWorkQ + ii] = mainShmem[startOfLocalWorkQ + ii] - isGoldOffset * isGoldForLocQueue[ii];


                    }

                }
                //now all of the threads in the block needs to have the same i value so we will increment by 1 we are preloading to the pipeline block metaData
                ////##### pipeline Step 0

                sync(cta);




                //loading metadata
                pipeline.producer_acquire();
                if (((bigloop) < localTotalLenthOfWorkQueue[0]) && ((bigloop) < ((blockIdx.x + 1) * globalWorkQueueOffset[0]))) {

                    cuda::memcpy_async(cta, (&localBlockMetaData[0]),
                        (&fbArgs.metaDataArrPointer[mainShmem[startOfLocalWorkQ] * fbArgs.metaData.metaDataSectionLength])
                        , cuda::aligned_size_t<4>(sizeof(uint32_t) * 20), pipeline);

                }
                pipeline.producer_commit();


                sync(cta);

                for (uint32_t i = 0; i < worQueueStep[0]; i += 1) {




                    if (((bigloop + i) < localTotalLenthOfWorkQueue[0]) && ((bigloop + i) < ((blockIdx.x + 1) * globalWorkQueueOffset[0]))) {



                        pipeline.producer_acquire();
                        cuda::memcpy_async(cta, &mainShmem[begSourceShmem], &getSourceReduced(fbArgs, iterationNumb)[
                            mainShmem[startOfLocalWorkQ + i] * fbArgs.metaData.mainArrSectionLength + fbArgs.metaData.mainArrXLength * (1 - isGoldForLocQueue[i])],
                            cuda::aligned_size_t<128>(sizeof(uint32_t) * fbArgs.metaData.mainArrXLength), pipeline);
                        pipeline.producer_commit();

                        //just so pipeline will work well
                        pipeline.consumer_wait();



                        pipeline.consumer_release();
                        sync(cta);

                        ///////// step 1 load top and process main data 
                                        //load top 
                        pipeline.producer_acquire();
                        if (localBlockMetaData[(i & 1) * 20 + 13] < isGoldOffset) {
                            cuda::memcpy_async(cta, (&mainShmem[begfirstRegShmem]),
                                &getSourceReduced(fbArgs, iterationNumb)[localBlockMetaData[(i & 1) * 20 + 13]
                                * fbArgs.metaData.mainArrSectionLength + fbArgs.metaData.mainArrXLength * (1 - isGoldForLocQueue[i])],
                                cuda::aligned_size_t<128>(sizeof(uint32_t) * fbArgs.metaData.mainArrXLength)
                                , pipeline);
                        }
                        pipeline.producer_commit();
                        //process main
                        pipeline.consumer_wait();
                        //marking weather block is already full and no more dilatations are possible 
                        if (__popc(mainShmem[begSourceShmem + threadIdx.x + threadIdx.y * 32]) < 32) {
                            isBlockFull[i & 1] = false;
                        }
                        mainShmem[begResShmem + threadIdx.x + threadIdx.y * 32] = bitDilatate(mainShmem[begSourceShmem + threadIdx.x + threadIdx.y * 32]);
                        pipeline.consumer_release();

                        ///////// step 2 load bottom and process top 
                                        //load bottom
                        pipeline.producer_acquire();
                        if (localBlockMetaData[(i & 1) * 20 + 14] < isGoldOffset) {
                            cuda::memcpy_async(cta, (&mainShmem[begSecRegShmem]),
                                &getSourceReduced(fbArgs, iterationNumb)[localBlockMetaData[(i & 1) * 20 + 14]
                                * fbArgs.metaData.mainArrSectionLength + fbArgs.metaData.mainArrXLength * (1 - isGoldForLocQueue[i])],
                                cuda::aligned_size_t<128>(sizeof(uint32_t) * fbArgs.metaData.mainArrXLength)
                                , pipeline);
                        }
                        pipeline.producer_commit();
                        //process top
                        pipeline.consumer_wait();


                        if (localBlockMetaData[(i & 1) * 20 + 13] < isGoldOffset) {
                            if (isBitAt(mainShmem[begSourceShmem + threadIdx.x + threadIdx.y * 32], 0)) {
                                // printf("setting padding top val %d \n ", isAnythingInPadding[0]);
                                isAnythingInPadding[0] = true;
                            };
                            // if in bit of intrest of neighbour block is set
                            mainShmem[begResShmem + threadIdx.x + threadIdx.y * 32] |= ((mainShmem[begfirstRegShmem + threadIdx.x + threadIdx.y * 32] >> 31) & 1) << 0;
                        }

                        pipeline.consumer_release();
                        sync(cta);

                        /////////// step 3 load right  process bottom  
                        pipeline.producer_acquire();
                        if (localBlockMetaData[(i & 1) * 20 + 16] < isGoldOffset) {
                            cuda::memcpy_async(cta, (&mainShmem[begfirstRegShmem]),
                                &getSourceReduced(fbArgs, iterationNumb)[localBlockMetaData[(i & 1) * 20 + 16] * fbArgs.metaData.mainArrSectionLength + fbArgs.metaData.mainArrXLength * (1 - isGoldForLocQueue[i])],
                                cuda::aligned_size_t<128>(sizeof(uint32_t) * fbArgs.metaData.mainArrXLength)
                                , pipeline);
                        }
                        pipeline.producer_commit();
                        //process bototm
                        pipeline.consumer_wait();


                        if (localBlockMetaData[(i & 1) * 20 + 14] < isGoldOffset) {
                            if (isBitAt(mainShmem[begSourceShmem + threadIdx.x + threadIdx.y * 32], 31)) {
                                isAnythingInPadding[1] = true;
                            };
                            // if in bit of intrest of neighbour block is set
                            mainShmem[begResShmem + threadIdx.x + threadIdx.y * 32] |= ((mainShmem[begSecRegShmem + threadIdx.x + threadIdx.y * 32] >> 0) & 1) << 31;
                        }



                        /*  dilatateHelperTopDown(1, mainShmem, isAnythingInPadding, localBlockMetaData, 14
                              , 0, 31
                              , begSecRegShmem, i);*/

                        pipeline.consumer_release();
                        /////////// step 4 load left process right  
                                        //load left 
                        pipeline.producer_acquire();
                        if (mainShmem[startOfLocalWorkQ + i] > 0) {
                            cuda::memcpy_async(cta, (&mainShmem[begSecRegShmem]),
                                &getSourceReduced(fbArgs, iterationNumb)[(mainShmem[startOfLocalWorkQ + i] - 1) * fbArgs.metaData.mainArrSectionLength + fbArgs.metaData.mainArrXLength * (1 - isGoldForLocQueue[i])],
                                cuda::aligned_size_t<128>(sizeof(uint32_t) * fbArgs.metaData.mainArrXLength)
                                , pipeline);
                        }
                        pipeline.producer_commit();
                        //process right
                        pipeline.consumer_wait();

                        if (threadIdx.x == (fbArgs.dbXLength - 1)) {
                            // now we need to load the data from the neigbouring blocks
                            //first checking is there anything to look to 
                            if (localBlockMetaData[(i & 1) * 20 + 16] < isGoldOffset) {
                                //now we load - we already done earlier up and down so now we are considering only anterior, posterior , left , right possibilities
                                if (mainShmem[begSourceShmem + threadIdx.x + threadIdx.y * 32] > 0) {
                                    isAnythingInPadding[3] = true;

                                };
                                mainShmem[begResShmem + threadIdx.x + threadIdx.y * 32] =
                                    mainShmem[begResShmem + threadIdx.x + threadIdx.y * 32]
                                    | mainShmem[begfirstRegShmem + (threadIdx.y * 32)];

                            };
                        }
                        else {//given we are not in corner case we need just to do the dilatation using biwise or with the data inside the block
                            mainShmem[begResShmem + threadIdx.x + threadIdx.y * 32]
                                = mainShmem[begSourceShmem + (threadIdx.x + 1) + (threadIdx.y) * 32]
                                | mainShmem[begResShmem + threadIdx.x + threadIdx.y * 32];

                        }

                        pipeline.consumer_release();
                        sync(cta);
                        /////// step 5 load anterior process left 
                                        //load anterior
                        pipeline.producer_acquire();
                        if (localBlockMetaData[(i & 1) * 20 + 17] < isGoldOffset) {

                            cuda::memcpy_async(cta, (&mainShmem[begfirstRegShmem]),
                                &getSourceReduced(fbArgs, iterationNumb)[localBlockMetaData[(i & 1) * 20 + 17] * fbArgs.metaData.mainArrSectionLength + fbArgs.metaData.mainArrXLength * (1 - isGoldForLocQueue[i])],
                                cuda::aligned_size_t<128>(sizeof(uint32_t) * fbArgs.metaData.mainArrXLength)
                                , pipeline);
                        }
                        pipeline.producer_commit();
                        //process left 
                        pipeline.consumer_wait();

                        // so we first check for corner cases 
                        if (threadIdx.x == 0) {
                            // now we need to load the data from the neigbouring blocks
                            //first checking is there anything to look to 
                            if (localBlockMetaData[(i & 1) * 20 + 15] < isGoldOffset) {
                                //now we load - we already done earlier up and down so now we are considering only anterior, posterior , left , right possibilities
                                if (mainShmem[begSourceShmem + threadIdx.x + threadIdx.y * 32] > 0) {
                                    isAnythingInPadding[2] = true;

                                };
                                mainShmem[begResShmem + threadIdx.x + threadIdx.y * 32] =
                                    mainShmem[begResShmem + threadIdx.x + threadIdx.y * 32]
                                    | mainShmem[begSecRegShmem + 31 + threadIdx.y * 32];

                            };
                        }
                        else {//given we are not in corner case we need just to do the dilatation using biwise or with the data inside the block
                            mainShmem[begResShmem + threadIdx.x + threadIdx.y * 32]
                                = mainShmem[begSourceShmem + (threadIdx.x - 1) + (threadIdx.y) * 32]
                                | mainShmem[begResShmem + threadIdx.x + threadIdx.y * 32];

                        }


                        pipeline.consumer_release();
                        sync(cta);

                        /////// step 6 load posterior process anterior 
                                        //load posterior
                        pipeline.producer_acquire();
                        if (localBlockMetaData[(i & 1) * 20 + 18] < isGoldOffset) {


                            cuda::memcpy_async(cta, (&mainShmem[begSecRegShmem]),
                                &getSourceReduced(fbArgs, iterationNumb)[localBlockMetaData[(i & 1) * 20 + 18] * fbArgs.metaData.mainArrSectionLength + fbArgs.metaData.mainArrXLength * (1 - isGoldForLocQueue[i])],
                                cuda::aligned_size_t<128>(sizeof(uint32_t) * fbArgs.metaData.mainArrXLength)
                                , pipeline);
                        }
                        pipeline.producer_commit();

                        //process anterior
                        pipeline.consumer_wait();

                        // so we first check for corner cases 
                        if (threadIdx.y == (fbArgs.dbYLength - 1)) {
                            // now we need to load the data from the neigbouring blocks
                            //first checking is there anything to look to 
                            if (localBlockMetaData[(i & 1) * 20 + 17] < isGoldOffset) {
                                //now we load - we already done earlier up and down so now we are considering only anterior, posterior , left , right possibilities
                                if (mainShmem[begSourceShmem + threadIdx.x + threadIdx.y * 32] > 0) {
                                    isAnythingInPadding[4] = true;

                                };
                                mainShmem[begResShmem + threadIdx.x + threadIdx.y * 32] =
                                    mainShmem[begResShmem + threadIdx.x + threadIdx.y * 32]
                                    | mainShmem[begfirstRegShmem + threadIdx.x];

                            };
                        }
                        else {//given we are not in corner case we need just to do the dilatation using biwise or with the data inside the block
                            mainShmem[begResShmem + threadIdx.x + threadIdx.y * 32]
                                = mainShmem[begSourceShmem + (threadIdx.x) + (threadIdx.y + 1) * 32]
                                | mainShmem[begResShmem + threadIdx.x + threadIdx.y * 32];

                        }


                        pipeline.consumer_release();
                        sync(cta);

                        /////// step 7 
                                       //load reference if needed or data for next iteration if there is such 
                                        //process posterior, save data from res shmem to global memory also we mark weather block is full
                        pipeline.producer_acquire();

                        //if block should be validated we load data for validation
                        if (localBlockMetaData[(i & 1) * 20 + ((1 - isGoldForLocQueue[i]) + 1)] //fp for gold and fn count for not gold
                        > localBlockMetaData[(i & 1) * 20 + ((1 - isGoldForLocQueue[i]) + 3)]) {// so count is bigger than counter so we should validate
                            cuda::memcpy_async(cta, (&mainShmem[begfirstRegShmem]),
                                &fbArgs.origArrsPointer[mainShmem[startOfLocalWorkQ + i] * fbArgs.metaData.mainArrSectionLength + fbArgs.metaData.mainArrXLength * (isGoldForLocQueue[i])], //we look for 
                                cuda::aligned_size_t<128>(sizeof(uint32_t) * fbArgs.metaData.mainArrXLength)
                                , pipeline);

                        }
                        else {//if we are not validating we immidiately start loading data for next loop
                            if (i + 1 < worQueueStep[0]) {
                                cuda::memcpy_async(cta, (&localBlockMetaData[((i + 1) & 1) * 20]),
                                    (&fbArgs.metaDataArrPointer[(mainShmem[startOfLocalWorkQ + 1 + i])
                                        * fbArgs.metaData.metaDataSectionLength])
                                    , cuda::aligned_size_t<4>(sizeof(uint32_t) * 20), pipeline);


                            }
                        }


                        pipeline.producer_commit();

                        //processPosteriorAndSaveResShmem

                        pipeline.consumer_wait();
                        //dilatate posterior 


                        // so we first check for corner cases 
                        if (threadIdx.y == 0) {
                            // now we need to load the data from the neigbouring blocks
                            //first checking is there anything to look to 
                            if (localBlockMetaData[(i & 1) * 20 + 18] < isGoldOffset) {
                                //now we load - we already done earlier up and down so now we are considering only anterior, posterior , left , right possibilities
                                if (mainShmem[begSourceShmem + threadIdx.x + threadIdx.y * 32] > 0) {
                                    isAnythingInPadding[5] = true;

                                };
                                mainShmem[begResShmem + threadIdx.x + threadIdx.y * 32] =
                                    mainShmem[begResShmem + threadIdx.x + threadIdx.y * 32]
                                    | mainShmem[begSecRegShmem + threadIdx.x + (fbArgs.dbYLength - 1) * 32];

                            };
                        }
                        else {//given we are not in corner case we need just to do the dilatation using biwise or with the data inside the block
                            mainShmem[begResShmem + threadIdx.x + threadIdx.y * 32]
                                = mainShmem[begSourceShmem + (threadIdx.x) + (threadIdx.y - 1) * 32]
                                | mainShmem[begResShmem + threadIdx.x + threadIdx.y * 32];

                        }

                        //now all data should be properly dilatated we save it to global memory
                        //try save target reduced via mempcy async ...


                        //cuda::memcpy_async(cta,
                        //    &getTargetReduced(fbArgs, iterationNumb)[mainShmem[startOfLocalWorkQ + i] * fbArgs.metaData.mainArrSectionLength + fbArgs.metaData.mainArrXLength * (1 - isGoldForLocQueue[i])]
                        //    , (&mainShmem[begResShmem]),
                        //    cuda::aligned_size_t<128>(sizeof(uint32_t) * fbArgs.metaData.mainArrXLength)
                        //    , pipeline);



                        getTargetReduced(fbArgs, iterationNumb)[mainShmem[startOfLocalWorkQ + i] * fbArgs.metaData.mainArrSectionLength + fbArgs.metaData.mainArrXLength * (1 - isGoldForLocQueue[i])
                            + threadIdx.x + threadIdx.y * 32]
                            = mainShmem[begResShmem + threadIdx.x + threadIdx.y * 32];





                        pipeline.consumer_release();

                        sync(cta);

                        //////// step 8 basically in order to complete here anyting the count need to be bigger than counter
                                                      // loading for next block if block is not to be validated it was already done earlier
                        pipeline.producer_acquire();
                        if (localBlockMetaData[(i & 1) * 20 + ((1 - isGoldForLocQueue[i]) + 1)] //fp for gold and fn count for not gold
                            > localBlockMetaData[(i & 1) * 20 + ((1 - isGoldForLocQueue[i]) + 3)]) {// so count is bigger than counter so we should validate
                            if (i + 1 < worQueueStep[0]) {


                                cuda::memcpy_async(cta, (&localBlockMetaData[((i + 1) & 1) * 20]),
                                    (&fbArgs.metaDataArrPointer[(mainShmem[startOfLocalWorkQ + 1 + i])
                                        * fbArgs.metaData.metaDataSectionLength])
                                    , cuda::aligned_size_t<4>(sizeof(uint32_t) * 20), pipeline);

                            }
                        }
                        pipeline.producer_commit();




                        sync(cta);

                        //validation - so looking for newly covered voxel for opposite array so new fps or new fns
                        pipeline.consumer_wait();

                        if (localBlockMetaData[(i & 1) * 20 + ((1 - isGoldForLocQueue[i]) + 1)] //fp for gold and fn count for not gold
                            > localBlockMetaData[(i & 1) * 20 + ((1 - isGoldForLocQueue[i]) + 3)]) {// so count is bigger than counter so we should validate
                                        // now we look through bits and when some is set we call it a result 
#pragma unroll
                            for (uint8_t bitPos = 0; bitPos < 32; bitPos++) {
                                //if any bit here is set it means it should be added to result list 
                                if (isBitAt(mainShmem[begResShmem + threadIdx.x + threadIdx.y * 32], bitPos)
                                    && !isBitAt(mainShmem[begSourceShmem + threadIdx.x + threadIdx.y * 32], bitPos)
                                    && isBitAt(mainShmem[begfirstRegShmem + threadIdx.x + threadIdx.y * 32], bitPos)
                                    ) {

                                    //just re
                                    mainShmem[begSecRegShmem + threadIdx.x + threadIdx.y * 32] = 0;
                                    ////// IMPORTANT for some reason in order to make it work resultfnOffset and resultfnOffset swith places
                                    if (isGoldForLocQueue[i]) {
                                        mainShmem[begSecRegShmem + threadIdx.x + threadIdx.y * 32] = uint32_t(atomicAdd_block(&(localFpConter[0]), 1) + localBlockMetaData[(i & 1) * 20 + 6] + localBlockMetaData[(i & 1) * 20 + 3]);
                                        //TODO remove
                                        //atomicAdd_block(&(blockFpConter[0]), 1);

                                    }
                                    else {

                                        mainShmem[begSecRegShmem + threadIdx.x + threadIdx.y * 32] = uint32_t(atomicAdd_block(&(localFnConter[0]), 1) + localBlockMetaData[(i & 1) * 20 + 5] + localBlockMetaData[(i & 1) * 20 + 4]);

                                        //TODO remove
                                        //atomicAdd_block(&(blockFnConter[0]), 1);

                                        //    printf("local fn counter add \n");

                                    };
                                    //   add results to global memory    
                                    //we add one gere jjust to distinguish it from empty result
                                    fbArgs.resultListPointerMeta[mainShmem[begSecRegShmem + threadIdx.x + threadIdx.y * 32]] = uint32_t(mainShmem[startOfLocalWorkQ + i] + (isGoldOffset * isGoldForLocQueue[i]) + 1);
                                    fbArgs.resultListPointerLocal[mainShmem[begSecRegShmem + threadIdx.x + threadIdx.y * 32]] = uint32_t((fbArgs.dbYLength * 32 * bitPos) + (threadIdx.y * 32) + (threadIdx.x));
                                    fbArgs.resultListPointerIterNumb[mainShmem[begSecRegShmem + threadIdx.x + threadIdx.y * 32]] = uint32_t(iterationNumb[0]);




                                }

                            };

                        }
                        /////////
                        pipeline.consumer_release();

                        /// /// cleaning 

                        sync(cta);

                        if (threadIdx.x == 9 && threadIdx.y == 2) {// this is how it is encoded wheather it is gold or segm block

         //executed in case of previous block
                            if (isBlockFull[i & 1] && i >= 0) {
                                //setting data in metadata that block is full
                                fbArgs.metaDataArrPointer[mainShmem[startOfLocalWorkQ + i] * fbArgs.metaData.metaDataSectionLength + 10 - (isGoldForLocQueue[i] * 2)] = true;
                            }
                            //resetting for some reason  block 0 gets as full even if it should not ...
                            isBlockFull[i & 1] = true;// mainShmem[startOfLocalWorkQ + i]>0;//!isPaddingPass;
                        };




                        //we do it only for non padding pass
                        if (threadIdx.x < 6 && threadIdx.y == 1 && !isPaddingPass) {
                            //executed in case of previous block
                            if (i >= 0) {

                                if (localBlockMetaData[(i & 1) * 20 + 13 + threadIdx.x] < isGoldOffset) {

                                    if (isAnythingInPadding[threadIdx.x]) {
                                        // holding data weather we have anything in padding 0)top  1)bottom, 2)left 3)right, 4)anterior, 5)posterior,
                                        fbArgs.metaDataArrPointer[localBlockMetaData[(i & 1) * 20 + 13 + threadIdx.x] * fbArgs.metaData.metaDataSectionLength + 12 - isGoldForLocQueue[i]] = 1;
                                    }

                                }
                            }
                            isAnythingInPadding[threadIdx.x] = false;
                        };






                        if (threadIdx.x == 7 && threadIdx.y == 0) {
                            //this will be executed only if fp or fn counters are bigger than 0 so not during first pass
                            if (localFpConter[0] > 0) {
                                fbArgs.metaDataArrPointer[mainShmem[startOfLocalWorkQ + i] * fbArgs.metaData.metaDataSectionLength + 3] += localFpConter[0];

                                blockFpConter[0] += localFpConter[0];
                                localFpConter[0] = 0;
                            }


                        };
                        if (threadIdx.x == 8 && threadIdx.y == 0) {

                            if (localFnConter[0] > 0) {
                                fbArgs.metaDataArrPointer[mainShmem[startOfLocalWorkQ + i] * fbArgs.metaData.metaDataSectionLength + 4] += localFnConter[0];

                                blockFnConter[0] += localFnConter[0];
                                localFnConter[0] = 0;
                            }
                        };

                        sync(cta);

                    }
                }

                //here we are after all of the blocks planned to be processed by this block are

                // just for pipeline to work
                pipeline.consumer_wait();



                pipeline.consumer_release();

            }

            sync(cta);

            //     updating global counters
            if (threadIdx.x == 0 && threadIdx.y == 0) {
                if (blockFpConter[0] > 0) {
                    atomicAdd(&(fbArgs.minMaxes[10]), (blockFpConter[0]));
                }
            };
            if (threadIdx.x == 1 && threadIdx.y == 0) {
                if (blockFnConter[0] > 0) {
                    //if (blockFnConter[0]>10) {
                    //    printf("Fn %d  ", blockFnConter[0]);
                    //}
                    atomicAdd(&(fbArgs.minMaxes[11]), (blockFnConter[0]));
                }
            };
            grid.sync();

            // in first thread block we zero work queue counter
            if (threadIdx.x == 2 && threadIdx.y == 0) {
                if (blockIdx.x == 0) {

                    fbArgs.minMaxes[9] = 0;
                }
            };

            grid.sync();
            /////////////////////////****************************************************************************************************************  
/////////////////////////****************************************************************************************************************  
/////////////////////////****************************************************************************************************************  
/////////////////////////****************************************************************************************************************  
/////////////////////////****************************************************************************************************************  
/// metadata pass










            // preparation loads
            if (threadIdx.x == 0 && threadIdx.y == 0) {
                fpFnLocCounter[0] = 0;
            }
            if (threadIdx.x == 1 && threadIdx.y == 0) {
                localWorkQueueCounter[0] = 0;
            }
            if (threadIdx.x == 2 && threadIdx.y == 0) {
                localWorkQueueCounter[0] = 0;
            }
            if (threadIdx.x == 3 && threadIdx.y == 0) {
                localWorkQueueCounter[0] = 0;

            }

            if (threadIdx.x == 0 && threadIdx.y == 1) {

                isGoldPassToContinue[0]
                    = ((fbArgs.minMaxes[7] * fbArgs.robustnessPercent) > fbArgs.minMaxes[10]);

            };

            if (threadIdx.x == 0 && threadIdx.y == 1) {

                isSegmPassToContinue[0]
                    = ((fbArgs.minMaxes[8] * fbArgs.robustnessPercent) > fbArgs.minMaxes[11]);
            };


            __syncthreads();

            /////////////////////////////////

            for (uint32_t linIdexMeta = blockIdx.x * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x
                ; linIdexMeta <= fbArgs.metaData.totalMetaLength
                ; linIdexMeta += (blockDim.x * blockDim.y * gridDim.x)
                ) {


                if (isPaddingPass == 0) {

                    //goldpass
                    if (isGoldPassToContinue[0] && fbArgs.metaDataArrPointer[linIdexMeta * fbArgs.metaData.metaDataSectionLength + 11]
                        && !fbArgs.metaDataArrPointer[linIdexMeta * fbArgs.metaData.metaDataSectionLength + 7]
                        && !fbArgs.metaDataArrPointer[linIdexMeta * fbArgs.metaData.metaDataSectionLength + 8]) {

                        mainShmem[atomicAdd_block(&localWorkQueueCounter[0], 1)] = linIdexMeta + (isGoldOffset);
                        //setting to be activated to 0 
                        fbArgs.metaDataArrPointer[linIdexMeta * fbArgs.metaData.metaDataSectionLength + 11] = 0;
                        //setting active to 1
                        fbArgs.metaDataArrPointer[linIdexMeta * fbArgs.metaData.metaDataSectionLength + 7] = 1;


                    };

                }
                //contrary to number it is when we are not in padding pass
                else {
                    //gold pass
                    if (isGoldPassToContinue[0] && fbArgs.metaDataArrPointer[linIdexMeta * fbArgs.metaData.metaDataSectionLength + 7]
                        && !fbArgs.metaDataArrPointer[linIdexMeta * fbArgs.metaData.metaDataSectionLength + 8]) {

                        mainShmem[atomicAdd_block(&localWorkQueueCounter[0], 1)] = linIdexMeta + (isGoldOffset);

                    };

                }
            }

            __syncthreads();

            if (localWorkQueueCounter[0] > 0) {
                if (threadIdx.x == 0 && threadIdx.y == 0) {
                    globalWorkQueueCounter[0] = atomicAdd(&(fbArgs.minMaxes[9]), (localWorkQueueCounter[0]));


                }
                __syncthreads();
                for (uint32_t linI = threadIdx.y * blockDim.x + threadIdx.x; linI < localWorkQueueCounter[0]; linI += blockDim.x * blockDim.y) {
                    fbArgs.workQueuePointer[globalWorkQueueCounter[0] + linI] = mainShmem[linI];
                }
                __syncthreads();

            }

            __syncthreads();

            if (threadIdx.x == 0 && threadIdx.y == 0) {

                localWorkQueueCounter[0] = 0;
            }
            __syncthreads();

            for (uint32_t linIdexMeta = blockIdx.x * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x
                ; linIdexMeta <= fbArgs.metaData.totalMetaLength
                ; linIdexMeta += (blockDim.x * blockDim.y * gridDim.x)
                ) {


                if (isPaddingPass == 0) {

                    //segm pass
                    if ((isSegmPassToContinue[0] && fbArgs.metaDataArrPointer[linIdexMeta * fbArgs.metaData.metaDataSectionLength + 12]
                        && !fbArgs.metaDataArrPointer[linIdexMeta * fbArgs.metaData.metaDataSectionLength + 9]
                        && !fbArgs.metaDataArrPointer[linIdexMeta * fbArgs.metaData.metaDataSectionLength + 10])) {



                        mainShmem[atomicAdd_block(&localWorkQueueCounter[0], 1)] = linIdexMeta;

                        //setting to be activated to 0 
                        fbArgs.metaDataArrPointer[linIdexMeta * fbArgs.metaData.metaDataSectionLength + 12] = 0;
                        //setting active to 1
                        fbArgs.metaDataArrPointer[linIdexMeta * fbArgs.metaData.metaDataSectionLength + 9] = 1;

                    }

                }
                //contrary to number it is when we are not in padding pass
                else {
                    //segm pass
                    if ((isSegmPassToContinue[0] && fbArgs.metaDataArrPointer[linIdexMeta * fbArgs.metaData.metaDataSectionLength + 9]
                        && !fbArgs.metaDataArrPointer[linIdexMeta * fbArgs.metaData.metaDataSectionLength + 10])) {



                        mainShmem[atomicAdd_block(&localWorkQueueCounter[0], 1)] = linIdexMeta;
                    }

                }
            }
            __syncthreads();

            if (localWorkQueueCounter[0] > 0) {
                if (threadIdx.x == 0 && threadIdx.y == 0) {
                    globalWorkQueueCounter[0] = atomicAdd(&(fbArgs.minMaxes[9]), (localWorkQueueCounter[0]));


                }
                __syncthreads();
                for (uint32_t linI = threadIdx.y * blockDim.x + threadIdx.x; linI < localWorkQueueCounter[0]; linI += blockDim.x * blockDim.y) {
                    fbArgs.workQueuePointer[globalWorkQueueCounter[0] + linI] = mainShmem[linI];

                }

            }



            grid.sync();
        }



    } while (isGoldPassToContinue[0] || isSegmPassToContinue[0]);


    //setting global iteration number to local one 
    if (blockIdx.x == 0) {
        if (threadIdx.x == 2 && threadIdx.y == 0) {
            fbArgs.metaData.minMaxes[13] = iterationNumb[0];
        }
    }
}



/*
get data from occupancy calculator API used to get optimal number of thread blocks and threads per thread block
*/
template <typename T>
inline occupancyCalcData getOccupancy() {

    occupancyCalcData res;

    int blockSize; // The launch configurator returned block size
    int minGridSize; // The minimum grid size needed to achieve the maximum occupancy for a full device launch
    int gridSize; // The actual grid size needed, based on input size

    // for min maxes kernel 
    cudaOccupancyMaxPotentialBlockSize(
        &minGridSize,
        &blockSize,
        (void*)getMinMaxes<T>,
        0);
    res.warpsNumbForMinMax = blockSize / 32;
    res.blockSizeForMinMax = minGridSize;

    // for min maxes kernel 
    cudaOccupancyMaxPotentialBlockSize(
        &minGridSize,
        &blockSize,
        (void*)boolPrepareKernel<T>,
        0);
    res.warpsNumbForboolPrepareKernel = blockSize / 32;
    res.blockSizeFoboolPrepareKernel = minGridSize;
    // for first meta pass kernel
    cudaOccupancyMaxPotentialBlockSize(
        &minGridSize,
        &blockSize,
        (void*)boolPrepareKernel<T>,
        0);
    res.theadsForFirstMetaPass = blockSize;
    res.blockForFirstMetaPass = minGridSize;
    //for main pass kernel
    cudaOccupancyMaxPotentialBlockSize(
        &minGridSize,
        &blockSize,
        (void*)mainPassKernel<T>,
        0);
    res.warpsNumbForMainPass = blockSize / 32;
    res.blockForMainPass = minGridSize;

    // res.blockForMainPass = 5;
     //res.blockForMainPass = 136;
     //res.warpsNumbForMainPass = 8;

    printf("warpsNumbForMainPass %d blockForMainPass %d  ", res.warpsNumbForMainPass, res.blockForMainPass);
    return res;
}













/*
TODO consider representing as a CUDA graph
executing Algorithm as CUDA graph  based on official documentation and
https://codingbyexample.com/2020/09/25/cuda-graph-usage/
*/
#pragma once
template <typename T>
ForBoolKernelArgs<T> executeHausdoff(ForFullBoolPrepArgs<T>& fFArgs, const int WIDTH, const int HEIGHT, const int DEPTH, occupancyCalcData& occData,
    cudaStream_t stream, bool resToSave = false) {

    // For Graph
    //cudaStream_t streamForGraph;
    //cudaGraph_t graph;
    //std::vector<cudaGraphNode_t> nodeDependencies;
    //cudaGraphNode_t memcpyNode, kernelNode;
    //cudaKernelNodeParams kernelNodeParams = { 0 };
    //  cudaMemcpyParams memcpyParams = { 0 };



    ForBoolKernelArgs<T> fbArgs = getArgsForKernel<T>(fFArgs, occData.warpsNumbForMainPass, occData.blockForMainPass, WIDTH, HEIGHT, DEPTH, stream);

    //checkCuda(cudaDeviceSynchronize(), "a1");

    //getMinMaxes << <blockSizeForMinMax, dim3(32, warpsNumbForMinMax) >> > ( minMaxes);
    getMinMaxes << <occData.blockSizeForMinMax, dim3(32, occData.warpsNumbForMinMax) >> > (fbArgs, fbArgs.minMaxes, fbArgs.goldArr.arrP, fbArgs.segmArr.arrP, fbArgs.metaData);

    //checkCuda(cudaDeviceSynchronize(), "a1b");

    fbArgs.metaData = allocateMemoryAfterMinMaxesKernel(fbArgs, fFArgs, stream);

    //checkCuda(cudaDeviceSynchronize(), "a2b");

    boolPrepareKernel << <occData.blockSizeFoboolPrepareKernel, dim3(32, occData.warpsNumbForboolPrepareKernel) >> > (
        fbArgs, fbArgs.metaData, fbArgs.origArrsPointer, fbArgs.metaDataArrPointer, fbArgs.goldArr.arrP, fbArgs.segmArr.arrP, fbArgs.minMaxes);

    //checkCuda(cudaDeviceSynchronize(), "a3");

    int fpPlusFn = allocateMemoryAfterBoolKernel(fbArgs, fFArgs, stream);

    //checkCuda(cudaDeviceSynchronize(), "a4");


    firstMetaPrepareKernel << <occData.blockForFirstMetaPass, occData.theadsForFirstMetaPass >> > (fbArgs, fbArgs.metaData, fbArgs.minMaxes, fbArgs.workQueuePointer, fbArgs.origArrsPointer, fbArgs.metaDataArrPointer);

    //checkCuda(cudaDeviceSynchronize(), "a5");

    void* kernel_args[] = { &fbArgs };
    cudaLaunchCooperativeKernel((void*)(mainPassKernel<int>), occData.blockForMainPass, dim3(32, occData.warpsNumbForMainPass), kernel_args);

    //checkCuda(cudaDeviceSynchronize(), "a6");

    if (resToSave) {
        copyResultstoCPU(fbArgs, fFArgs, stream);

    }
    cudaFreeAsync(fbArgs.resultListPointerMeta, stream);
    cudaFreeAsync(fbArgs.resultListPointerLocal, stream);
    cudaFreeAsync(fbArgs.resultListPointerIterNumb, stream);
    cudaFreeAsync(fbArgs.workQueuePointer, stream);
    cudaFreeAsync(fbArgs.origArrsPointer, stream);
    cudaFreeAsync(fbArgs.metaDataArrPointer, stream);
    cudaFreeAsync(fbArgs.mainArrAPointer, stream);
    cudaFreeAsync(fbArgs.mainArrBPointer, stream);

    return fbArgs;

}



#pragma once
template <typename T>
ForBoolKernelArgs<T> mainKernelsRun(ForFullBoolPrepArgs<T>& fFArgs, const int WIDTH, const int HEIGHT, const int DEPTH, cudaStream_t stream, bool resToSave = false
) {

    //cudaDeviceReset();
    cudaError_t syncErr;
    cudaError_t asyncErr;

    occupancyCalcData occData = getOccupancy<T>();

    //pointers ...
    ForBoolKernelArgs<T> fbArgs = executeHausdoff(fFArgs, WIDTH, HEIGHT, DEPTH, occData, resToSave, stream);

    checkCuda(cudaDeviceSynchronize(), "last ");

    /////////// error handling 
    syncErr = cudaGetLastError();
    asyncErr = cudaDeviceSynchronize();
    if (syncErr != cudaSuccess) printf("Error in syncErr: %s\n", cudaGetErrorString(syncErr));
    if (asyncErr != cudaSuccess) printf("Error in asyncErr: %s\n", cudaGetErrorString(asyncErr));


    cudaDeviceReset();

    return fbArgs;
}








void loadHDFIntoBoolArr(H5std_string FILE_NAME, H5std_string DATASET_NAME, bool*& data) {

    H5::H5File file(FILE_NAME, H5F_ACC_RDONLY);
    H5::DataSet dset = file.openDataSet(DATASET_NAME);
    /*
     * Get the class of the datatype that is used by the dataset.
     */
    H5T_class_t type_class = dset.getTypeClass();
    H5::DataSpace dspace = dset.getSpace();
    int rank = dspace.getSimpleExtentNdims();

    hsize_t dims[2];
    rank = dspace.getSimpleExtentDims(dims, NULL); // rank = 1
    printf("Datasize: %d \n ", dims[0]); // this is the correct number of values

    // Define the memory dataspace
    hsize_t dimsm[1];
    dimsm[0] = dims[0];
    H5::DataSpace memspace(1, dimsm);
    data = (bool*)calloc(dims[0], sizeof(bool));
    dset.read(data, H5::PredType::NATIVE_HBOOL, memspace, dspace);
    file.close();

}

template<typename T>
T FindMax(T* arr, size_t n)
{
    int max = arr[0];

    for (size_t j = 0; j < n; ++j) {
        if (arr[j] > max) {
            max = arr[j];
        }
    }
    return max;
}






void benchmarkMitura(bool* onlyBladderBoolFlat, bool* onlyLungsBoolFlat, const int WIDTH, const int HEIGHT, const int DEPTH, cudaStream_t stream1) {



    bool resultToCopy = true;
    //// some preparations and configuring
    MetaDataCPU metaData;
    size_t size = sizeof(unsigned int) * 20;
    unsigned int* minMaxesCPU = (unsigned int*)malloc(size);
    metaData.minMaxes = minMaxesCPU;

    ForFullBoolPrepArgs<bool> forFullBoolPrepArgs;
    forFullBoolPrepArgs.metaData = metaData;
    forFullBoolPrepArgs.numberToLookFor = true;
    forFullBoolPrepArgs.goldArr = get3dArrCPU(onlyBladderBoolFlat, WIDTH, HEIGHT, DEPTH);
    forFullBoolPrepArgs.segmArr = get3dArrCPU(onlyLungsBoolFlat, WIDTH, HEIGHT, DEPTH);

    occupancyCalcData occData = getOccupancy<bool>();

    //pointers ...

    //function invocation
    auto begin = std::chrono::high_resolution_clock::now();
    cudaDeviceSynchronize();

    ForBoolKernelArgs<bool> fbArgs = executeHausdoff(forFullBoolPrepArgs, WIDTH, HEIGHT, DEPTH, occData, stream1, resultToCopy);

    // ForBoolKernelArgs<bool> fbArgs = mainKernelsRun(forFullBoolPrepArgs, WIDTH, HEIGHT, DEPTH);
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();

    checkCuda(cudaDeviceSynchronize(), "a7a");


    std::cout << "Total elapsed time: ";
    std::cout << (double)(std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count() / (double)1000000000) << "s" << std::endl;
    checkCuda(cudaDeviceSynchronize(), "a7b");


    size_t sizeMinMax = sizeof(unsigned int) * 20;
    cudaMemcpy(minMaxesCPU, fbArgs.metaData.minMaxes, sizeMinMax, cudaMemcpyDeviceToHost);
    checkCuda(cudaDeviceSynchronize(), "a7c");

    printf("HD: %d \n", minMaxesCPU[13]);
    printf("debug sum : %d \n", minMaxesCPU[15]);


    printf("max iter numb %d  \n", FindMax(forFullBoolPrepArgs.resultListPointerIterNumb, (minMaxesCPU[7] + minMaxesCPU[8] + 50)));


    checkCuda(cudaDeviceSynchronize(), "a8");

    if (resultToCopy) {
        free(forFullBoolPrepArgs.resultListPointerMeta);
        free(forFullBoolPrepArgs.resultListPointerLocalCPU);
        free(forFullBoolPrepArgs.resultListPointerIterNumb);
    }

    checkCuda(cudaDeviceSynchronize(), "a9");

    // printf("debug sum : %d \n", minMaxesCPU[15]);


     // freeee
    free(onlyBladderBoolFlat);
    free(onlyLungsBoolFlat);


    checkCuda(cudaDeviceSynchronize(), "a10");


}







typedef unsigned char uchar;
typedef unsigned int uint;
#pragma once
class Volume {

private:
    bool* volume;
    int width, height, depth;
    int getLinearIndex(int x, int y, int z);
public:
    bool getVoxelValue(int x, int y, int z);
    bool getPixelValue(int x, int y);
    uint getWidth();
    uint getHeight();
    uint getDepth();
    bool* getVolume();
    void setVoxelValue(bool value, int x, int y, int z);
    void setPixelValue(bool value, int x, int y);
    Volume(int width, int height, int depth);
    Volume(int width, int height);
    void dispose();

};




#define CUDA_DEVICE_INDEX 0 //setting the index of your CUDA device

#define IS_3D 1 //setting this to 0 would grant a very slightly improvement on the performance if working with images only
#define CHEBYSHEV 0 //if not set to 1, then this algorithm would use an Euclidean-like metric, it is just an approximation. 
//It can be changed according to the structuring element
#pragma once
class HausdorffDistance {

private:
    void print(cudaError_t error, char* msg);

public:
    int computeDistance(Volume* img1, Volume* img2);

};


inline Volume::Volume(const int width, const int height, const int depth) {
    this->width = width; this->height = height; this->depth = depth;
    volume = (bool*)calloc(width * height * depth, sizeof(bool));
}

#pragma once
inline Volume::Volume(const int width, const int height) {
    this->width = width; this->height = height; this->depth = 1;
    volume = (bool*)calloc(width * height * depth, sizeof(bool));
}
#pragma once
inline int Volume::getLinearIndex(const int x, const int y, const int z) {
    const int a = 1, b = width, c = (width) * (height);
    return a * x + b * y + c * z;
}

inline uint Volume::getWidth() { return this->width; }
inline uint Volume::getHeight() { return this->height; }
inline uint Volume::getDepth() { return this->depth; }
inline bool* Volume::getVolume() { return this->volume; }
inline bool Volume::getPixelValue(int x, int y) { return this->volume[getLinearIndex(x, y, 0)]; }
#pragma once
inline bool Volume::getVoxelValue(int x, int y, int z) {
    return volume[getLinearIndex(x, y, z)];
}
#pragma once
inline void Volume::setPixelValue(bool value, const int x, const int y) {
    volume[getLinearIndex(x, y, 0)] = value;
}
#pragma once
inline void Volume::setVoxelValue(bool value, const int x, const int y, const int z) {
    volume[getLinearIndex(x, y, z)] = value;
}
#pragma once
inline void Volume::dispose() {
    free(volume);
}

typedef unsigned char uchar;
typedef unsigned int uint;

#pragma once
__device__ int finished; //global variable that contains a boolean which indicates when to stop the kernel processing
#pragma once
__constant__ __device__ int WIDTH, HEIGHT, DEPTH; //constant variables that contain the size of the volume


#pragma once
__global__ void dilate(const bool* IMG1, const bool* IMG2, const bool* img1Read, const bool* img2Read,
    bool* img1Write, bool* img2Write) {

    const int id = blockDim.x * blockIdx.x + threadIdx.x;
#if !IS_3D
    const int x = id % WIDTH, y = id / WIDTH;
#else
    const int x = id % WIDTH, y = (id / WIDTH) % HEIGHT, z = (id / WIDTH) / HEIGHT;
#endif

    if (id < WIDTH * HEIGHT * DEPTH) {


        if (img1Read[id]) {
            if (x + 1 < WIDTH) img1Write[id + 1] = true;
            if (x - 1 >= 0) img1Write[id - 1] = true;
            if (y + 1 < HEIGHT) img1Write[id + WIDTH] = true;
            if (y - 1 >= 0) img1Write[id - WIDTH] = true;
#if IS_3D //if working with 3d volumes, then the 3D part
            if (z + 1 < DEPTH) img1Write[id + WIDTH * HEIGHT] = true;
            if (z - 1 >= 0) img1Write[id - WIDTH * HEIGHT] = true;
#endif

#if CHEBYSHEV
            //diagonals
            if (x + 1 < WIDTH && y - 1 >= 0) img1Write[id - WIDTH + 1] = true;
            if (x - 1 >= 0 && y - 1 >= 0) img1Write[id - WIDTH - 1] = true;
            if (x + 1 < WIDTH && y + 1 < HEIGHT) img1Write[id + WIDTH + 1] = true;
            if (x - 1 >= 0 && y + 1 < HEIGHT) img1Write[id + WIDTH - 1] = true;
#if IS_3D //if working with 3d volumes, then the 3D part
            if (z + 1 < DEPTH && x + 1 < WIDTH && y - 1 >= 0) img1Write[id - WIDTH + 1 + WIDTH * HEIGHT] = true;
            if (z + 1 < DEPTH && x - 1 >= 0 && y - 1 >= 0) img1Write[id - WIDTH - 1 + WIDTH * HEIGHT] = true;
            if (z + 1 < DEPTH && x + 1 < WIDTH && y + 1 < HEIGHT) img1Write[id + WIDTH + 1 + WIDTH * HEIGHT] = true;
            if (z + 1 < DEPTH && x - 1 >= 0 && y + 1 < HEIGHT) img1Write[id + WIDTH - 1 + WIDTH * HEIGHT] = true;
            if (z - 1 >= 0 && x + 1 < WIDTH && y - 1 >= 0) img1Write[id - WIDTH + 1 - WIDTH * HEIGHT] = true;
            if (z - 1 >= 0 && x - 1 >= 0 && y - 1 >= 0) img1Write[id - WIDTH - 1 - WIDTH * HEIGHT] = true;
            if (z - 1 >= 0 && x + 1 < WIDTH && y + 1 < HEIGHT) img1Write[id + WIDTH + 1 - WIDTH * HEIGHT] = true;
            if (z - 1 >= 0 && x - 1 >= 0 && y + 1 < HEIGHT) img1Write[id + WIDTH - 1 - WIDTH * HEIGHT] = true;
#endif
#endif
        }


        if (img2Read[id]) {
            if (x + 1 < WIDTH) img2Write[id + 1] = true;
            if (x - 1 >= 0) img2Write[id - 1] = true;
            if (y + 1 < HEIGHT) img2Write[id + WIDTH] = true;
            if (y - 1 >= 0) img2Write[id - WIDTH] = true;
#if IS_3D //if working with 3d volumes, then the 3D part
            if (z + 1 < DEPTH) img2Write[id + WIDTH * HEIGHT] = true;
            if (z - 1 >= 0) img2Write[id - WIDTH * HEIGHT] = true;
#endif

#if CHEBYSHEV
            //diagonals
            if (x + 1 < WIDTH && y - 1 >= 0) img2Write[id - WIDTH + 1] = true;
            if (x - 1 >= 0 && y - 1 >= 0) img2Write[id - WIDTH - 1] = true;
            if (x + 1 < WIDTH && y + 1 < HEIGHT) img2Write[id + WIDTH + 1] = true;
            if (x - 1 >= 0 && y + 1 < HEIGHT) img2Write[id + WIDTH - 1] = true;
#if IS_3D //if working with 3d volumes, then the 3D part
            if (z + 1 < DEPTH && x + 1 < WIDTH && y - 1 >= 0) img2Write[id - WIDTH + 1 + WIDTH * HEIGHT] = true;
            if (z + 1 < DEPTH && x - 1 >= 0 && y - 1 >= 0) img2Write[id - WIDTH - 1 + WIDTH * HEIGHT] = true;
            if (z + 1 < DEPTH && x + 1 < WIDTH && y + 1 < HEIGHT) img2Write[id + WIDTH + 1 + WIDTH * HEIGHT] = true;
            if (z + 1 < DEPTH && x - 1 >= 0 && y + 1 < HEIGHT) img2Write[id + WIDTH - 1 + WIDTH * HEIGHT] = true;
            if (z - 1 >= 0 && x + 1 < WIDTH && y - 1 >= 0) img2Write[id - WIDTH + 1 - WIDTH * HEIGHT] = true;
            if (z - 1 >= 0 && x - 1 >= 0 && y - 1 >= 0) img2Write[id - WIDTH - 1 - WIDTH * HEIGHT] = true;
            if (z - 1 >= 0 && x + 1 < WIDTH && y + 1 < HEIGHT) img2Write[id + WIDTH + 1 - WIDTH * HEIGHT] = true;
            if (z - 1 >= 0 && x - 1 >= 0 && y + 1 < HEIGHT) img2Write[id + WIDTH - 1 - WIDTH * HEIGHT] = true;
#endif
#endif
        }


        //this is an atomic and computed to the finished global variable, if image 1 contains all of image 2 and image 2 contains all pixels of
        //image 1 then finished is true
        atomicAnd(&finished, (img2Read[id] || !IMG1[id]) && (img1Read[id] || !IMG2[id]));
    }
}

#pragma once
int HausdorffDistance::computeDistance(Volume* img1, Volume* img2) {

    const int height = (*img1).getHeight(), width = (*img1).getWidth(), depth = (*img1).getDepth();

    size_t size = width * height * depth * sizeof(bool);

    //getting details of your CUDA device
    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, CUDA_DEVICE_INDEX); //device index = 0, you can change it if you have more CUDA devices
    const int threadsPerBlock = props.maxThreadsPerBlock / 2;
    const int blocksPerGrid = (height * width * depth + threadsPerBlock - 1) / threadsPerBlock;


    //copying the dimensions to the GPU
    cudaMemcpyToSymbolAsync(WIDTH, &width, sizeof(width),0);
    cudaMemcpyToSymbolAsync(HEIGHT, &height, sizeof(height),0);
    cudaMemcpyToSymbolAsync(DEPTH, &depth, sizeof(depth),0);


    //allocating the input images on the GPU
    bool* d_img1, * d_img2;
    cudaMalloc(&d_img1, size);
    cudaMalloc(&d_img2, size);


    //copying the data to the allocated memory on the GPU
    cudaMemcpyAsync(d_img1, (*img1).getVolume(), size, cudaMemcpyHostToDevice);
    cudaMemcpyAsync(d_img2, (*img2).getVolume(), size, cudaMemcpyHostToDevice);


    //allocating the images that will be the processing ones
    bool* d_img1Write, * d_img1Read, * d_img2Write, * d_img2Read;
    cudaMalloc(&d_img1Write, size); cudaMalloc(&d_img1Read, size);
    cudaMalloc(&d_img2Write, size); cudaMalloc(&d_img2Read, size);


    //cloning the input images to these two image versions (write and read)
    cudaMemcpyAsync(d_img1Read, d_img1, size, cudaMemcpyDeviceToDevice);
    cudaMemcpyAsync(d_img2Read, d_img2, size, cudaMemcpyDeviceToDevice);
    cudaMemcpyAsync(d_img1Write, d_img1, size, cudaMemcpyDeviceToDevice);
    cudaMemcpyAsync(d_img2Write, d_img2, size, cudaMemcpyDeviceToDevice);



    //required variables to compute the distance
    int h_finished = false, t = true;
    int distance = -1;

    //where the magic happens
    while (!h_finished) {
        //reset the bool variable that verifies if the processing ended
        cudaMemcpyToSymbol(finished, &t, sizeof(h_finished));


        //lauching the verify kernel, which verifies if the processing finished
        dilate << < blocksPerGrid, threadsPerBlock >> > (d_img1, d_img2, d_img1Read, d_img2Read, d_img1Write, d_img2Write);

        //cudaDeviceSynchronize();

        //updating the imgRead (cloning imgWrite to imgRead)
        cudaMemcpy(d_img1Read, d_img1Write, size, cudaMemcpyDeviceToDevice);
        cudaMemcpy(d_img2Read, d_img2Write, size, cudaMemcpyDeviceToDevice);

        //copying the result back to host memory
        cudaMemcpyFromSymbol(&h_finished, finished, sizeof(h_finished));


        //incrementing the distance at each iteration
        distance++;
    }


    //freeing memory
    cudaFree(d_img1); cudaFree(d_img2);
    cudaFree(d_img1Write); cudaFree(d_img1Read);
    cudaFree(d_img2Write); cudaFree(d_img2Read);

    //resetting device
   // cudaDeviceReset();

    //print(cudaGetLastError(), "processing CUDA. Something may be wrong with your CUDA device.");

    return distance;

}
#pragma once
inline void HausdorffDistance::print(cudaError_t error, char* msg) {
    if (error != cudaSuccess)
    {
        printf("Error on %s ", msg);
        fprintf(stderr, "Error code: %s!\n", cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }
}



/*
benchmark for original code from  https://github.com/Oyatsumi/HausdorffDistanceComparison
*/
void benchmarkOliviera(bool* onlyBladderBoolFlat, bool* onlyLungsBoolFlat, const int WIDTH, const int HEIGHT
    , const int DEPTH) {
    Volume img1 = Volume(WIDTH, HEIGHT, DEPTH), img2 = Volume(WIDTH, HEIGHT, DEPTH);

    for (int x = 0; x < WIDTH; x++) {
        for (int y = 0; y < HEIGHT; y++) {
            for (int z = 0; z < DEPTH; z++) {
                img1.setVoxelValue(onlyLungsBoolFlat[x + y * WIDTH + z * WIDTH * HEIGHT], x, y, z);
                img2.setVoxelValue(onlyBladderBoolFlat[x + y * WIDTH + z * WIDTH * HEIGHT], x, y, z);
            }
        }
    }

    auto begin = std::chrono::high_resolution_clock::now();
    HausdorffDistance* hd = new HausdorffDistance();
    cudaDeviceSynchronize();

    int dist = (*hd).computeDistance(&img1, &img2);
    cudaDeviceSynchronize();

    auto end = std::chrono::high_resolution_clock::now();

    std::cout << "Total elapsed time: ";
    std::cout << (double)(std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count() / (double)1000000000) << "s" << std::endl;

    printf("HD: %d \n", dist);

    //freeing memory
    img1.dispose(); img2.dispose();

    //Datasize: 216530944
   //Datasize : 216530944
    //Total elapsed time : 2.62191s
    //HD : 234

}

void loadHDF(cudaStream_t stream) {



    const int WIDTH = 512;
    const int HEIGHT = 512;
    //    const int DEPTH = 536;

    int DEPTH = 826;

    const H5std_string FILE_NAMEonlyLungsBoolFlat("D:\\dataSets\\forMainHDF5\\forHausdorffTests.hdf5");
    const H5std_string FILE_NAMEonlyBladderBoolFlat("D:\\dataSets\\forMainHDF5\\forHausdorffTests.hdf5");

    const H5std_string DATASET_NAMEonlyLungsBoolFlat("onlyLungsBoolFlat");
    //const H5std_string DATASET_NAMEonlyLungsBoolFlat("onlyLungsBoolFlatB");
    // create a vector the same size as the dataset
    bool* onlyLungsBoolFlat;
    loadHDFIntoBoolArr(FILE_NAMEonlyLungsBoolFlat, DATASET_NAMEonlyLungsBoolFlat, onlyLungsBoolFlat);



    const H5std_string DATASET_NAMEonlyBladderBoolFlat("onlyBladderBoolFlat");
    //const H5std_string DATASET_NAMEonlyBladderBoolFlat("onlyBladderBoolFlatB");
    // create a vector the same size as the dataset
    bool* onlyBladderBoolFlat;
    loadHDFIntoBoolArr(FILE_NAMEonlyBladderBoolFlat, DATASET_NAMEonlyBladderBoolFlat, onlyBladderBoolFlat);
    //onlyBladderBoolFlat = (bool*)calloc(WIDTH* HEIGHT* DEPTH, sizeof(bool));

    //onlyBladderBoolFlat[0] = true;

    //benchmarkOliviera(onlyBladderBoolFlat, onlyLungsBoolFlat, WIDTH, HEIGHT, DEPTH);//125 
    //benchmarkMitura(onlyBladderBoolFlat, onlyLungsBoolFlat, WIDTH, HEIGHT, DEPTH);//124 or 259
    benchmarkMitura(onlyLungsBoolFlat, onlyBladderBoolFlat, WIDTH, HEIGHT, DEPTH, stream);//124 or 259


    //DEPTH = 536;
    //const H5std_string FILE_NAMEonlyLungsBoolFlatB("D:\\dataSets\\forMainHDF5\\forHausdorffTests.hdf5");

    //const H5std_string DATASET_NAMEonlyLungsBoolFlatB("onlyLungsBoolFlatB");
    //// create a vector the same size as the dataset
    //bool* onlyLungsBoolFlatB;
    //loadHDFIntoBoolArr(FILE_NAMEonlyLungsBoolFlatB, DATASET_NAMEonlyLungsBoolFlatB, onlyLungsBoolFlatB);


    //const H5std_string DATASET_NAMEonlyBladderBoolFlatB("onlyBladderBoolFlatB");
    //// create a vector the same size as the dataset
    //bool* onlyBladderBoolFlatB;
    //loadHDFIntoBoolArr(FILE_NAMEonlyBladderBoolFlat, DATASET_NAMEonlyBladderBoolFlatB, onlyBladderBoolFlatB);

    ////benchmarkOliviera(onlyBladderBoolFlatB, onlyLungsBoolFlatB, WIDTH, HEIGHT, DEPTH);//125 
    //benchmarkMitura(onlyBladderBoolFlatB, onlyLungsBoolFlatB, WIDTH, HEIGHT, DEPTH);//124 or 259

}



void loadHDFB(cudaStream_t stream) {



    const int WIDTH = 512;
    const int HEIGHT = 512;
    const int  DEPTH = 536;

    const H5std_string FILE_NAMEonlyLungsBoolFlat("D:\\dataSets\\forMainHDF5\\forHausdorffTests.hdf5");
    const H5std_string FILE_NAMEonlyBladderBoolFlat("D:\\dataSets\\forMainHDF5\\forHausdorffTests.hdf5");



    const H5std_string FILE_NAMEonlyLungsBoolFlatB("D:\\dataSets\\forMainHDF5\\forHausdorffTests.hdf5");

    const H5std_string DATASET_NAMEonlyLungsBoolFlatB("onlyLungsBoolFlatB");
    // create a vector the same size as the dataset
    bool* onlyLungsBoolFlatB;
    loadHDFIntoBoolArr(FILE_NAMEonlyLungsBoolFlatB, DATASET_NAMEonlyLungsBoolFlatB, onlyLungsBoolFlatB);


    const H5std_string DATASET_NAMEonlyBladderBoolFlatB("onlyBladderBoolFlatB");
    // create a vector the same size as the dataset
    bool* onlyBladderBoolFlatB;
    loadHDFIntoBoolArr(FILE_NAMEonlyBladderBoolFlat, DATASET_NAMEonlyBladderBoolFlatB, onlyBladderBoolFlatB);

    // benchmarkOliviera(onlyBladderBoolFlatB, onlyLungsBoolFlatB, WIDTH, HEIGHT, DEPTH);//125 
    benchmarkMitura(onlyBladderBoolFlatB, onlyLungsBoolFlatB, WIDTH, HEIGHT, DEPTH, stream);//124 or 259

}




void setCPU(bool* arr, int x, int y, int z, int xDim, int yDim) {
    arr[x + y * xDim + z * xDim * yDim] = true;
}

/*
void testAll() {



    //cudaDeviceReset();
    cudaError_t syncErr;
    cudaError_t asyncErr;

    const int WIDTH = 512;
    const int HEIGHT = 512;
    //    const int DEPTH = 536;

    int DEPTH = 826;

    bool resultToCopy = true;
    //// some preparations and configuring
    MetaDataCPU metaData;
    size_t size = sizeof(unsigned int) * 20;
    unsigned int* minMaxesCPU = (unsigned int*)malloc(size);
    metaData.minMaxes = minMaxesCPU;

    bool* arrA = (bool*)calloc(WIDTH * HEIGHT * DEPTH, sizeof(bool));
    bool* arrB = (bool*)calloc(WIDTH * HEIGHT * DEPTH, sizeof(bool));

    //for (int i = 0; i < 500; i++) {
    //    setCPU(arrB, i, i, 000, WIDTH, HEIGHT);
    //}


    for (int i = 0; i < 500;i++) {
        for (int j = 0; j < 500; j++) {
            setCPU(arrA, i, j, 0, WIDTH, HEIGHT);
        }
    }
    //setCPU(arrA, 2, 2, 0, WIDTH, HEIGHT);

   // setCPU(arrA, 0, 0, 300, WIDTH, HEIGHT);


    //for (int i = 0; i < 500; i++) {
    //    setCPU(arrB, i, i, 600, WIDTH, HEIGHT);
    //}

    //setCPU(arrB, 30, 30, 605, WIDTH, HEIGHT);
    //setCPU(arrB, 30, 36, 606, WIDTH, HEIGHT);
    //setCPU(arrB, 30, 39, 607, WIDTH, HEIGHT);
    //setCPU(arrB, 30, 12, 608, WIDTH, HEIGHT);
    //setCPU(arrB, 30, 33, 609, WIDTH, HEIGHT);
    //setCPU(arrB, 30, 66, 610, WIDTH, HEIGHT);
    setCPU(arrB, 0, 0, 500, WIDTH, HEIGHT);





    setCPU(arrB, 2, 88, 800, WIDTH, HEIGHT);
    setCPU(arrB, 99, 7, 801, WIDTH, HEIGHT);
    setCPU(arrB, 45, 77, 802, WIDTH, HEIGHT);
    //setCPU(arrB, 30, 332, 612, WIDTH, HEIGHT);



    ForFullBoolPrepArgs<bool> forFullBoolPrepArgs;
    forFullBoolPrepArgs.metaData = metaData;
    forFullBoolPrepArgs.numberToLookFor = true;
    forFullBoolPrepArgs.goldArr = get3dArrCPU(arrA, WIDTH, HEIGHT, DEPTH);
    forFullBoolPrepArgs.segmArr = get3dArrCPU(arrB, WIDTH, HEIGHT, DEPTH);

    ForBoolKernelArgs<bool> fbArgs = mainKernelsRun(forFullBoolPrepArgs, WIDTH, HEIGHT, DEPTH, resultToCopy);

    size_t sizeMinMax = sizeof(unsigned int) * 20;
    cudaMemcpy(minMaxesCPU, fbArgs.metaData.minMaxes, sizeMinMax, cudaMemcpyDeviceToHost);

    printf("HD: %d \n", minMaxesCPU[13]);

    free(arrA);
    free(arrB);

    printf("max iter numb %d  \n", FindMax(forFullBoolPrepArgs.resultListPointerIterNumb, (minMaxesCPU[7] + minMaxesCPU[8] + 50)));



    if (resultToCopy) {
        free(forFullBoolPrepArgs.resultListPointerMeta);
        free(forFullBoolPrepArgs.resultListPointerLocalCPU);
        free(forFullBoolPrepArgs.resultListPointerIterNumb);
    }

    checkCuda(cudaDeviceSynchronize(), "last ");

    /////////// error handling
    syncErr = cudaGetLastError();
    asyncErr = cudaDeviceSynchronize();
    if (syncErr != cudaSuccess) printf("Error in syncErr: %s\n", cudaGetErrorString(syncErr));
    if (asyncErr != cudaSuccess) printf("Error in asyncErr: %s\n", cudaGetErrorString(asyncErr));


}
*/


int main(void) {


    cudaStream_t stream1;
    cudaStreamCreate(&stream1);

    cudaStream_t stream2;
    cudaStreamCreate(&stream2);

    for (int i = 0; i < 10; i++) {
        loadHDF(stream1);
    }

    for (int i = 0; i < 10; i++) {
        loadHDFB(stream2);
    }

    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);
    //  testAll();


    return 0;  // successfully terminated
}

