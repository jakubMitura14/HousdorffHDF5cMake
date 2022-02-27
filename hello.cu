


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
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include "MetaDataOtherPasses.cu"
#include "MinMaxesKernel.cu"
#include "MainKernelMetaHelpers.cu"
#include <cooperative_groups/memcpy_async.h>
using namespace cooperative_groups;



//#include "hdf5Manag.cu"
#include <iostream>
#include <string>
#include <vector>
#define H5_BUILT_AS_DYNAMIC_LIB 1
#include <H5Cpp.h>






/*
gettinng  array for dilatations
basically arrays will alternate between iterations once one will be source other target then they will switch - we will decide upon knowing
wheather the iteration number is odd or even
*/
#pragma once
template <typename TXPI>
inline __device__ uint32_t* getSourceReduced(ForBoolKernelArgs<TXPI>& fbArgs, int(&iterationNumb)[1]) {


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
inline __device__ uint32_t* getTargetReduced(ForBoolKernelArgs<TXPPI>& fbArgs, int(&iterationNumb)[1]) {

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
inline __device__ uint32_t bitDilatate(uint32_t& x) {
    return ((x) >> 1) | (x) | ((x) << 1);
}

/*
return 1 if at given position of given number bit is set otherwise 0
*/
#pragma once
inline __device__ uint32_t isBitAt(uint32_t& numb, const int pos) {
    return (numb & (1 << (pos)));
}

#pragma once
inline uint32_t isBitAtCPU(uint32_t& numb, const int pos) {
    return (numb & (1 << (pos)));
}






//
///*
//given source and target uint32 it will check the bit of intrest  of source and set the target to bit of target intrest
//*/
//#pragma once
//inline __device__ void setBitTo(uint32_t source, uint8_t sourceBit, uint32_t resShared[32][32], uint8_t targetBit) {   
//    resShared[threadIdx.x][threadIdx.y] |= ((source >> sourceBit) & 1) << targetBit;
//   // return target;
//}

///////////////////////////////// new functions





/*
to iterate over the threads and given their position - checking edge cases do appropriate dilatations ...
works only for anterior - posterior lateral an medial dilatations
predicate - indicates what we consider border case here
paddingPos = integer marking which padding we are currently talking about(top ? bottom ? anterior ? ...)
padingVariedA, padingVariedB - eithr bitPos threadid X or Y depending what will be changing in this case

normalXChange, normalYchange - indicating which wntries we are intrested in if we are not at the boundary so how much to add to xand y thread position
metaDataCoordIndex - index where in the metadata of this block th linear index of neihjbouring block is present
targetShmemOffset - offset where loaded data needed for dilatation of outside of the block is present for example defining  register shmem one or 2 ...
*/
#pragma once
template <typename TXPI>
inline __device__ void dilatateHelperForTransverse(ForBoolKernelArgs<TXPI>& fbArgs, const bool predicate,
    const uint8_t  paddingPos, const   int8_t  normalXChange, const  int8_t normalYchange
    , uint32_t(&mainShmem)[lengthOfMainShmem], bool(&isAnythingInPadding)[6]
    , const uint8_t forBorderYcoord, const  uint8_t forBorderXcoord
    , const uint8_t metaDataCoordIndex, const uint32_t targetShmemOffset, uint32_t(&localBlockMetaData)[40], uint32_t& i
    , bool(&isGoldForLocQueue)[localWorkQueLength]) {



    //if (paddingPos == 3 && mainShmem[begSourceShmem + threadIdx.x + threadIdx.y * 32]>0 && isGoldForLocQueue[i] == 0 ) {
    //if ( mainShmem[begSourceShmem + threadIdx.x + threadIdx.y * 32]>0 && isGoldForLocQueue[i] == 1 ) {
    //
    //    printf("something in loaded from right idX %d idY %d  paddingPos %d \n", threadIdx.x, threadIdx.y , paddingPos );
    //}


    // so we first check for corner cases 
    if (predicate) {


        // now we need to load the data from the neigbouring blocks
        //first checking is there anything to look to 
        if (localBlockMetaData[(i & 1) * 20 + metaDataCoordIndex] < isGoldOffset) {


            //now we load - we already done earlier up and down so now we are considering only anterior, posterior , left , right possibilities
            if (mainShmem[begSourceShmem + threadIdx.x + threadIdx.y * 32] > 0) {
                isAnythingInPadding[paddingPos] = true;

            };



            mainShmem[begResShmem + threadIdx.x + threadIdx.y * 32] =
                mainShmem[begResShmem + threadIdx.x + threadIdx.y * 32]
                | mainShmem[targetShmemOffset + forBorderXcoord + forBorderYcoord * 32];

        };
    }
    else {//given we are not in corner case we need just to do the dilatation using biwise or with the data inside the block


        mainShmem[begResShmem + threadIdx.x + threadIdx.y * 32]
            = mainShmem[begSourceShmem + (threadIdx.x + normalXChange) + (threadIdx.y + normalYchange) * 32]
            | mainShmem[begResShmem + threadIdx.x + threadIdx.y * 32];

    }


}


#pragma once
inline __device__ void dilatateHelperTopDown(const uint8_t paddingPos,
    uint32_t(&mainShmem)[lengthOfMainShmem], bool(&isAnythingInPadding)[6], uint32_t(&localBlockMetaData)[40]
    , const uint8_t metaDataCoordIndex
    , const  uint8_t sourceBit
    , const uint8_t targetBit
    , const uint32_t targetShmemOffset, uint32_t& i
) {
    // now we need to load the data from the neigbouring blocks
    //first checking is there anything to look to 
    if (localBlockMetaData[(i & 1) * 20 + metaDataCoordIndex] < isGoldOffset) {
        if (isBitAt(mainShmem[begSourceShmem + threadIdx.x + threadIdx.y * 32], targetBit)) {
            // printf("setting padding top val %d \n ", isAnythingInPadding[0]);
            isAnythingInPadding[paddingPos] = true;
        };
        // if in bit of intrest of neighbour block is set
        mainShmem[begResShmem + threadIdx.x + threadIdx.y * 32] |= ((mainShmem[targetShmemOffset + threadIdx.x + threadIdx.y * 32] >> sourceBit) & 1) << targetBit;
    }

}



/*
we need to define here the function that will update the metadata result for the given block -
also if it is not padding pass we need to set the neighbouring blocks as to be activated according to the data in shmem
this will also include preparations for next round of iterations through blocks from work queue
isInPipeline - marks is it meant to be executed at the begining of the pipeline or after the pipeline
finilizing operations for last block
*/
#pragma once
inline __device__  void afterBlockClean(thread_block& cta
    , unsigned int(&worQueueStep)[1], uint32_t(&localBlockMetaData)[40]
    , uint32_t(&mainShmem)[lengthOfMainShmem], const uint32_t i, MetaDataGPU& metaData
    , thread_block_tile<32>& tile
    , unsigned int(&localFpConter)[1], unsigned int(&localFnConter)[1]
    , unsigned int(&blockFpConter)[1], unsigned int(&blockFnConter)[1]
    , uint32_t*& metaDataArr
    , bool(&isAnythingInPadding)[6], bool(&isBlockFull)[2], const bool isPaddingPass, bool(&isGoldForLocQueue)[localWorkQueLength], uint32_t(&lastI)[1]
) {

    if (i < UINT32_MAX) {

        if (threadIdx.x == 7 && threadIdx.y == 0) {// this is how it is encoded wheather it is gold or segm block
                       //this will be executed only if fp or fn counters are bigger than 0 so not during first pass
            if (localFpConter[0] >= 0) {
                metaDataArr[mainShmem[startOfLocalWorkQ + i] * metaData.metaDataSectionLength + 3] += localFpConter[0];
                blockFpConter[0] += localFpConter[0];
                localFpConter[0] = 0;
            }
        };
        if (threadIdx.x == 8 && threadIdx.y == 3) {

            if (localFnConter[0] >= 0) {
                metaDataArr[mainShmem[startOfLocalWorkQ + i] * metaData.metaDataSectionLength + 4] += localFnConter[0];

                blockFnConter[0] += localFnConter[0];
                localFnConter[0] = 0;
            }
        };
        if (threadIdx.x == 9 && threadIdx.y == 2) {// this is how it is encoded wheather it is gold or segm block

            //executed in case of previous block
            if (isBlockFull[i & 1] && i >= 0) {
                //setting data in metadata that block is full
                metaDataArr[mainShmem[startOfLocalWorkQ + i] * metaData.metaDataSectionLength + 10 - (isGoldForLocQueue[i] * 2)] = true;
            }
            //resetting for some reason  block 0 gets as full even if it should not ...
            isBlockFull[i & 1] = true;// mainShmem[startOfLocalWorkQ + i]>0;//!isPaddingPass;
        };




        //we do it only for non padding pass
        if (threadIdx.x < 6 && threadIdx.y == 1 && !isPaddingPass) {
            //executed in case of previous block
            if (i >= 0) {
                auto metadataTarget = localBlockMetaData[(i & 1) * 20 + 13 + threadIdx.x];

                if (metadataTarget < isGoldOffset) {

                    if (isAnythingInPadding[threadIdx.x]) {
                        // holding data weather we have anything in padding 0)top  1)bottom, 2)left 3)right, 4)anterior, 5)posterior,
                        metaDataArr[metadataTarget * metaData.metaDataSectionLength + 12 - isGoldForLocQueue[i]] = 1;
                    }

                }
            }
            isAnythingInPadding[tile.thread_rank()] = false;
        };

    };


}













//template <typename TKKI, typename forPipeline >
#pragma once
template <typename TKKI >
inline __device__ void mainDilatation(const bool isPaddingPass, ForBoolKernelArgs<TKKI>& fbArgs, uint32_t*& mainArrAPointer,
    uint32_t*& mainArrBPointer, MetaDataGPU& metaData
    , unsigned int*& minMaxes, uint32_t*& workQueue
    , uint32_t*& resultListPointerMeta, uint32_t*& resultListPointerLocal, uint32_t*& resultListPointerIterNumb,
    thread_block& cta, thread_block_tile<32>& tile, grid_group& grid, uint32_t(&mainShmem)[lengthOfMainShmem]
    , bool(&isAnythingInPadding)[6], bool(&isBlockFull)[2], int(&iterationNumb)[1], unsigned int(&globalWorkQueueOffset)[1]
    , unsigned int(&globalWorkQueueCounter)[1]
    , unsigned int(&localWorkQueueCounter)[1], unsigned int(&localTotalLenthOfWorkQueue)[1]
    , unsigned int(&localFpConter)[1]
    , unsigned int(&localFnConter)[1], unsigned int(&blockFpConter)[1]
    , unsigned int(&blockFnConter)[1], unsigned int(&resultfpOffset)[1]
    , unsigned int(&resultfnOffset)[1], unsigned int(&worQueueStep)[1]
    , unsigned int(&localMinMaxes)[5]
    , uint32_t(&localBlockMetaData)[40]
    , unsigned int(&fpFnLocCounter)[1]
    , bool(&isGoldPassToContinue)[1], bool(&isSegmPassToContinue)[1]
    , uint32_t*& origArrs, uint32_t*& metaDataArr, bool(&isGoldForLocQueue)[localWorkQueLength]
    , uint32_t(&lastI)[1]
    , cuda::pipeline<cuda::thread_scope_block>& pipeline
) {


    //initial cleaning  and initializations include loading min maxes
    if (tile.thread_rank() == 7 && tile.meta_group_rank() == 0 && !isPaddingPass) {
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
        isBlockFull[0] = true;
    };
    if (tile.thread_rank() == 9 && tile.meta_group_rank() == 1) {
        isBlockFull[1] = true;
    };

    if (tile.thread_rank() == 10 && tile.meta_group_rank() == 0) {
        fpFnLocCounter[0] = 0;
    };


    if (tile.thread_rank() == 10 && tile.meta_group_rank() == 2) {// this is how it is encoded wheather it is gold or segm block

        lastI[0] = UINT32_MAX;
    };


    if (tile.thread_rank() == 0 && tile.meta_group_rank() == 0) {
        localTotalLenthOfWorkQueue[0] = minMaxes[9];
        globalWorkQueueOffset[0] = floor((float)(localTotalLenthOfWorkQueue[0] / gridDim.x)) + 1;
        worQueueStep[0] = min(localWorkQueLength, globalWorkQueueOffset[0]);
    };

    if (tile.meta_group_rank() == 1) {
        cooperative_groups::memcpy_async(tile, (&localMinMaxes[0]), (&minMaxes[7]), cuda::aligned_size_t<4>(sizeof(unsigned int) * 5));
    }

    sync(cta);

    /// load work QueueData into shared memory 
    for (uint32_t bigloop = blockIdx.x * globalWorkQueueOffset[0]; bigloop < ((blockIdx.x + 1) * globalWorkQueueOffset[0]); bigloop += worQueueStep[0]) {
        // grid stride loop - sadly most of threads will be idle 
        /////////// loading to work queue
        if (((bigloop) < localTotalLenthOfWorkQueue[0]) && ((bigloop) < ((blockIdx.x + 1) * globalWorkQueueOffset[0]))) {

            for (uint16_t ii = cta.thread_rank(); ii < worQueueStep[0]; ii += cta.size()) {
                mainShmem[startOfLocalWorkQ + ii] = workQueue[bigloop + ii];
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
                (&metaDataArr[mainShmem[startOfLocalWorkQ] * metaData.metaDataSectionLength])
                , cuda::aligned_size_t<4>(sizeof(uint32_t) * 20), pipeline);

        }
        pipeline.producer_commit();



        for (uint32_t i = 0; i < worQueueStep[0]; i += 1) {
            if (((bigloop + i) < localTotalLenthOfWorkQueue[0]) && ((bigloop + i) < ((blockIdx.x + 1) * globalWorkQueueOffset[0]))) {
                //////////////// step 0  load main data and final processing of previous block
                              //loading main data for first dilatation
                               //IMPORTANT we need to keep a lot of variables constant here like is Anuthing in padding of fp count .. as the represent processing of previous block  - so do not modify them here ...

                pipeline.producer_acquire();
                cuda::memcpy_async(cta, &mainShmem[begSourceShmem], &getSourceReduced(fbArgs, iterationNumb)[
                    mainShmem[startOfLocalWorkQ + i] * metaData.mainArrSectionLength + metaData.mainArrXLength * (1 - isGoldForLocQueue[i])],
                    cuda::aligned_size_t<128>(sizeof(uint32_t) * metaData.mainArrXLength), pipeline);
                pipeline.producer_commit();

                pipeline.consumer_wait();


                afterBlockClean(cta, worQueueStep, localBlockMetaData, mainShmem, i - 1,
                    metaData, tile, localFpConter, localFnConter
                    , blockFpConter, blockFnConter
                    , metaDataArr, isAnythingInPadding, isBlockFull, isPaddingPass, isGoldForLocQueue, lastI);


                //needed for after block metadata update
                if (tile.thread_rank() == 0 && tile.meta_group_rank() == 0) {
                    lastI[0] = i;
                }

                pipeline.consumer_release();

                ///////// step 1 load top and process main data 
                                //load top 
                pipeline.producer_acquire();
                if (localBlockMetaData[(i & 1) * 20 + 13] < isGoldOffset) {
                    cuda::memcpy_async(cta, (&mainShmem[begfirstRegShmem]),
                        &getSourceReduced(fbArgs, iterationNumb)[localBlockMetaData[(i & 1) * 20 + 13]
                        * metaData.mainArrSectionLength + metaData.mainArrXLength * (1 - isGoldForLocQueue[i])],
                        cuda::aligned_size_t<128>(sizeof(uint32_t) * metaData.mainArrXLength)
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
                        * metaData.mainArrSectionLength + metaData.mainArrXLength * (1 - isGoldForLocQueue[i])],
                        cuda::aligned_size_t<128>(sizeof(uint32_t) * metaData.mainArrXLength)
                        , pipeline);
                }
                pipeline.producer_commit();
                //process top
                pipeline.consumer_wait();

                dilatateHelperTopDown(0, mainShmem, isAnythingInPadding, localBlockMetaData, 13
                    , 31, 0
                    , begfirstRegShmem, i);

                pipeline.consumer_release();

                /////////// step 3 load right  process bottom  
                pipeline.producer_acquire();
                if (localBlockMetaData[(i & 1) * 20 + 16] < isGoldOffset) {
                    cuda::memcpy_async(cta, (&mainShmem[begfirstRegShmem]),
                        &getSourceReduced(fbArgs, iterationNumb)[localBlockMetaData[(i & 1) * 20 + 16] * metaData.mainArrSectionLength + metaData.mainArrXLength * (1 - isGoldForLocQueue[i])],
                        cuda::aligned_size_t<128>(sizeof(uint32_t) * metaData.mainArrXLength)
                        , pipeline);
                }
                pipeline.producer_commit();
                //process bototm
                pipeline.consumer_wait();

                dilatateHelperTopDown(1, mainShmem, isAnythingInPadding, localBlockMetaData, 14
                    , 0, 31
                    , begSecRegShmem, i);

                pipeline.consumer_release();
                /////////// step 4 load left process right  
                                //load left 
                pipeline.producer_acquire();
                if (mainShmem[startOfLocalWorkQ + i] > 0) {
                    cuda::memcpy_async(cta, (&mainShmem[begSecRegShmem]),
                        &getSourceReduced(fbArgs, iterationNumb)[(mainShmem[startOfLocalWorkQ + i] - 1) * metaData.mainArrSectionLength + metaData.mainArrXLength * (1 - isGoldForLocQueue[i])],
                        cuda::aligned_size_t<128>(sizeof(uint32_t) * metaData.mainArrXLength)
                        , pipeline);
                }
                pipeline.producer_commit();
                //process right
                pipeline.consumer_wait();

                dilatateHelperForTransverse(fbArgs, (threadIdx.x == (fbArgs.dbXLength - 1)),
                    3, (1), (0), mainShmem, isAnythingInPadding
                    , threadIdx.y, 0
                    , 16, begfirstRegShmem, localBlockMetaData, i, isGoldForLocQueue);

                pipeline.consumer_release();

                /////// step 5 load anterior process left 
                                //load anterior
                pipeline.producer_acquire();
                if (localBlockMetaData[(i & 1) * 20 + 17] < isGoldOffset) {

                    cuda::memcpy_async(cta, (&mainShmem[begfirstRegShmem]),
                        &getSourceReduced(fbArgs, iterationNumb)[localBlockMetaData[(i & 1) * 20 + 17] * metaData.mainArrSectionLength + metaData.mainArrXLength * (1 - isGoldForLocQueue[i])],
                        cuda::aligned_size_t<128>(sizeof(uint32_t) * metaData.mainArrXLength)
                        , pipeline);
                }
                pipeline.producer_commit();
                //process left 
                pipeline.consumer_wait();

                dilatateHelperForTransverse(fbArgs, (threadIdx.x == 0),
                    2, (-1), (0), mainShmem, isAnythingInPadding
                    , threadIdx.y, 31
                    , 15, begSecRegShmem, localBlockMetaData, i, isGoldForLocQueue);

                pipeline.consumer_release();
                /////// step 6 load posterior process anterior 
                                //load posterior
                pipeline.producer_acquire();
                if (localBlockMetaData[(i & 1) * 20 + 18] < isGoldOffset) {


                    cuda::memcpy_async(cta, (&mainShmem[begSecRegShmem]),
                        &getSourceReduced(fbArgs, iterationNumb)[localBlockMetaData[(i & 1) * 20 + 18] * metaData.mainArrSectionLength + metaData.mainArrXLength * (1 - isGoldForLocQueue[i])],
                        cuda::aligned_size_t<128>(sizeof(uint32_t) * metaData.mainArrXLength)
                        , pipeline);
                }
                pipeline.producer_commit();

                //process anterior
                pipeline.consumer_wait();

                dilatateHelperForTransverse(fbArgs, (threadIdx.y == (fbArgs.dbYLength - 1)), 4
                    , (0), (1), mainShmem, isAnythingInPadding
                    , 0, threadIdx.x
                    , 17, begfirstRegShmem, localBlockMetaData, i, isGoldForLocQueue);
                pipeline.consumer_release();

                /////// step 7 
                               //load reference if needed or data for next iteration if there is such 
                                //process posterior, save data from res shmem to global memory also we mark weather block is full
                pipeline.producer_acquire();

                //if block should be validated we load data for validation
                if (localBlockMetaData[(i & 1) * 20 + ((1 - isGoldForLocQueue[i]) + 1)] //fp for gold and fn count for not gold
                > localBlockMetaData[(i & 1) * 20 + ((1 - isGoldForLocQueue[i]) + 3)]) {// so count is bigger than counter so we should validate
                    cuda::memcpy_async(cta, (&mainShmem[begfirstRegShmem]),
                        &origArrs[mainShmem[startOfLocalWorkQ + i] * metaData.mainArrSectionLength + metaData.mainArrXLength * (isGoldForLocQueue[i])], //we look for 
                        cuda::aligned_size_t<128>(sizeof(uint32_t) * metaData.mainArrXLength)
                        , pipeline);

                }
                else {//if we are not validating we immidiately start loading data for next loop
                    if (i + 1 < worQueueStep[0]) {
                        cuda::memcpy_async(cta, (&localBlockMetaData[((i + 1) & 1) * 20]),
                            (&metaDataArr[(mainShmem[startOfLocalWorkQ + 1 + i])
                                * metaData.metaDataSectionLength])
                            , cuda::aligned_size_t<4>(sizeof(uint32_t) * 20), pipeline);


                    }
                }


                pipeline.producer_commit();

                //processPosteriorAndSaveResShmem

                pipeline.consumer_wait();
                //dilatate posterior 
                dilatateHelperForTransverse(fbArgs, (threadIdx.y == 0), 5
                    , (0), (-1), mainShmem, isAnythingInPadding
                    , fbArgs.dbYLength - 1, threadIdx.x // we add offset depending on y dimension
                    , 18, begSecRegShmem, localBlockMetaData, i, isGoldForLocQueue);
                //now all data should be properly dilatated we save it to global memory
                //try save target reduced via mempcy async ...

                getTargetReduced(fbArgs, iterationNumb)[mainShmem[startOfLocalWorkQ + i] * metaData.mainArrSectionLength + metaData.mainArrXLength * (1 - isGoldForLocQueue[i])
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
                            (&metaDataArr[(mainShmem[startOfLocalWorkQ + 1 + i])
                                * metaData.metaDataSectionLength])
                            , cuda::aligned_size_t<4>(sizeof(uint32_t) * 20), pipeline);

                    }
                }
                pipeline.producer_commit();


                //validation - so looking for newly covered voxel for opposite array so new fps or new fns
                pipeline.consumer_wait();

                if (localBlockMetaData[(i & 1) * 20 + ((1 - isGoldForLocQueue[i]) + 1)] //fp for gold and fn count for not gold
                    > localBlockMetaData[(i & 1) * 20 + ((1 - isGoldForLocQueue[i]) + 3)]) {// so count is bigger than counter so we should validate
                        //mainShmem[begSourceShmem + threadIdx.x + threadIdx.y * 32] = 
                        //    ((~mainShmem[begSourceShmem + threadIdx.x + threadIdx.y * 32]) 
                        //        & mainShmem[begResShmem + threadIdx.x + threadIdx.y * 32]);



                        //we now look for bits prasent in both reference arrays and current one
                       // mainShmem[begSourceShmem + threadIdx.x + threadIdx.y * 32] = ((mainShmem[begSourceShmem + threadIdx.x + threadIdx.y * 32]) & mainShmem[begfirstRegShmem + threadIdx.x + threadIdx.y * 32]);

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
                            }
                            else {
                                mainShmem[begSecRegShmem + threadIdx.x + threadIdx.y * 32] = uint32_t(atomicAdd_block(&(localFnConter[0]), 1) + localBlockMetaData[(i & 1) * 20 + 5] + localBlockMetaData[(i & 1) * 20 + 4]);
                                //    printf("local fn counter add \n");

                            };
                            //   add results to global memory    
                            //we add one gere jjust to distinguish it from empty result
                            resultListPointerMeta[mainShmem[begSecRegShmem + threadIdx.x + threadIdx.y * 32]] = uint32_t(mainShmem[startOfLocalWorkQ + i] + (isGoldOffset * isGoldForLocQueue[i]) + 1);
                            resultListPointerLocal[mainShmem[begSecRegShmem + threadIdx.x + threadIdx.y * 32]] = uint32_t((fbArgs.dbYLength * 32 * bitPos) + (threadIdx.y * 32) + (threadIdx.x));
                            resultListPointerIterNumb[mainShmem[begSecRegShmem + threadIdx.x + threadIdx.y * 32]] = uint32_t(iterationNumb[0]);



                        }

                    };

                }
                /////////
                pipeline.consumer_release();
                sync(cta);


            }
        }

        //here we are after all of the blocks planned to be processed by this block are

//updating local counters of last local block (normally it is done at the bagining of the next block)
//but we need to check weather any block was processed at all
        pipeline.consumer_wait();

        afterBlockClean(cta, worQueueStep, localBlockMetaData, mainShmem, lastI[0],
            metaData, tile, localFpConter, localFnConter
            , blockFpConter, blockFnConter
            , metaDataArr, isAnythingInPadding, isBlockFull, isPaddingPass, isGoldForLocQueue, lastI);


        pipeline.consumer_release();

    }

    sync(cta);

    //     updating global counters
    if (tile.thread_rank() == 0 && tile.meta_group_rank() == 0) {
        if (blockFpConter[0] > 0) {
            atomicAdd(&(minMaxes[10]), (blockFpConter[0]));
        }
    };
    if (tile.thread_rank() == 1 && tile.meta_group_rank() == 0) {
        if (blockFnConter[0] > 0) {
            atomicAdd(&(minMaxes[11]), (blockFnConter[0]));
        }
    };
    // in first thread block we zero work queue counter
    if (threadIdx.x == 2 && threadIdx.y == 0) {
        if (blockIdx.x == 0) {

            minMaxes[9] = 0;
        }
    };


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
    if (tile.thread_rank() == 9 && tile.meta_group_rank() == 0) { iterationNumb[0] = -1; };
    if (tile.thread_rank() == 11 && tile.meta_group_rank() == 0) {
        isGoldPassToContinue[0] = true;
    };
    if (tile.thread_rank() == 12 && tile.meta_group_rank() == 0) {
        isSegmPassToContinue[0] = true;

    };


    //here we caclulate the offset for given block depending on length of the workqueue and number of the  available blocks in a grid
    // - this will give us number of work queue items per block - we will calculate offset on the basis of the block number



   // for (int t = 0; t < 3; t++) {
    do {

        //for (bool isPaddingPass = false; isPaddingPass; isPaddingPass = true) {
        for (uint8_t isPaddingPass = 0; isPaddingPass < 2; isPaddingPass++) {
            /////////////////////////****************************************************************************************************************  
            /////////////////////////****************************************************************************************************************  
            /////////////////////////****************************************************************************************************************  
            /////////////////////////****************************************************************************************************************  
            /////////////////////////****************************************************************************************************************  
            /// dilataions

    //initial cleaning  and initializations include loading min maxes
            if (tile.thread_rank() == 7 && tile.meta_group_rank() == 0 && !isPaddingPass) {
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
                isBlockFull[0] = true;
            };
            if (tile.thread_rank() == 9 && tile.meta_group_rank() == 1) {
                isBlockFull[1] = true;
            };

            if (tile.thread_rank() == 10 && tile.meta_group_rank() == 0) {
                fpFnLocCounter[0] = 0;
            };


            if (tile.thread_rank() == 10 && tile.meta_group_rank() == 2) {// this is how it is encoded wheather it is gold or segm block

                lastI[0] = UINT32_MAX;
            };


            if (tile.thread_rank() == 0 && tile.meta_group_rank() == 0) {
                localTotalLenthOfWorkQueue[0] = fbArgs.minMaxes[9];
                globalWorkQueueOffset[0] = floor((float)(localTotalLenthOfWorkQueue[0] / gridDim.x)) + 1;
                worQueueStep[0] = min(localWorkQueLength, globalWorkQueueOffset[0]);
            };

            if (tile.meta_group_rank() == 1) {
                cooperative_groups::memcpy_async(tile, (&localMinMaxes[0]), (&fbArgs.minMaxes[7]), cuda::aligned_size_t<4>(sizeof(unsigned int) * 5));
            }

            sync(cta);

            /// load work QueueData into shared memory 
            for (uint32_t bigloop = blockIdx.x * globalWorkQueueOffset[0]; bigloop < ((blockIdx.x + 1) * globalWorkQueueOffset[0]); bigloop += worQueueStep[0]) {
                // grid stride loop - sadly most of threads will be idle 
                /////////// loading to work queue
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



                for (uint32_t i = 0; i < worQueueStep[0]; i += 1) {
                    if (((bigloop + i) < localTotalLenthOfWorkQueue[0]) && ((bigloop + i) < ((blockIdx.x + 1) * globalWorkQueueOffset[0]))) {
                        //////////////// step 0  load main data and final processing of previous block
                                      //loading main data for first dilatation
                                       //IMPORTANT we need to keep a lot of variables constant here like is Anuthing in padding of fp count .. as the represent processing of previous block  - so do not modify them here ...

                        pipeline.producer_acquire();
                        cuda::memcpy_async(cta, &mainShmem[begSourceShmem], &getSourceReduced(fbArgs, iterationNumb)[
                            mainShmem[startOfLocalWorkQ + i] * fbArgs.metaData.mainArrSectionLength + fbArgs.metaData.mainArrXLength * (1 - isGoldForLocQueue[i])],
                            cuda::aligned_size_t<128>(sizeof(uint32_t) * fbArgs.metaData.mainArrXLength), pipeline);
                        pipeline.producer_commit();

                        pipeline.consumer_wait();


                        afterBlockClean(cta, worQueueStep, localBlockMetaData, mainShmem, i - 1,
                            fbArgs.metaData, tile, localFpConter, localFnConter
                            , blockFpConter, blockFnConter
                            , fbArgs.metaDataArrPointer, isAnythingInPadding, isBlockFull, isPaddingPass, isGoldForLocQueue, lastI);


                        //needed for after block metadata update
                        if (tile.thread_rank() == 0 && tile.meta_group_rank() == 0) {
                            lastI[0] = i;
                        }

                        pipeline.consumer_release();

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

                        dilatateHelperTopDown(0, mainShmem, isAnythingInPadding, localBlockMetaData, 13
                            , 31, 0
                            , begfirstRegShmem, i);

                        pipeline.consumer_release();

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

                        dilatateHelperTopDown(1, mainShmem, isAnythingInPadding, localBlockMetaData, 14
                            , 0, 31
                            , begSecRegShmem, i);

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

                        dilatateHelperForTransverse(fbArgs, (threadIdx.x == (fbArgs.dbXLength - 1)),
                            3, (1), (0), mainShmem, isAnythingInPadding
                            , threadIdx.y, 0
                            , 16, begfirstRegShmem, localBlockMetaData, i, isGoldForLocQueue);

                        pipeline.consumer_release();

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

                        dilatateHelperForTransverse(fbArgs, (threadIdx.x == 0),
                            2, (-1), (0), mainShmem, isAnythingInPadding
                            , threadIdx.y, 31
                            , 15, begSecRegShmem, localBlockMetaData, i, isGoldForLocQueue);

                        pipeline.consumer_release();
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

                        dilatateHelperForTransverse(fbArgs, (threadIdx.y == (fbArgs.dbYLength - 1)), 4
                            , (0), (1), mainShmem, isAnythingInPadding
                            , 0, threadIdx.x
                            , 17, begfirstRegShmem, localBlockMetaData, i, isGoldForLocQueue);
                        pipeline.consumer_release();

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
                        dilatateHelperForTransverse(fbArgs, (threadIdx.y == 0), 5
                            , (0), (-1), mainShmem, isAnythingInPadding
                            , fbArgs.dbYLength - 1, threadIdx.x // we add offset depending on y dimension
                            , 18, begSecRegShmem, localBlockMetaData, i, isGoldForLocQueue);
                        //now all data should be properly dilatated we save it to global memory
                        //try save target reduced via mempcy async ...

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


                        //validation - so looking for newly covered voxel for opposite array so new fps or new fns
                        pipeline.consumer_wait();

                        if (localBlockMetaData[(i & 1) * 20 + ((1 - isGoldForLocQueue[i]) + 1)] //fp for gold and fn count for not gold
                            > localBlockMetaData[(i & 1) * 20 + ((1 - isGoldForLocQueue[i]) + 3)]) {// so count is bigger than counter so we should validate
                                //mainShmem[begSourceShmem + threadIdx.x + threadIdx.y * 32] = 
                                //    ((~mainShmem[begSourceShmem + threadIdx.x + threadIdx.y * 32]) 
                                //        & mainShmem[begResShmem + threadIdx.x + threadIdx.y * 32]);



                                //we now look for bits prasent in both reference arrays and current one
                               // mainShmem[begSourceShmem + threadIdx.x + threadIdx.y * 32] = ((mainShmem[begSourceShmem + threadIdx.x + threadIdx.y * 32]) & mainShmem[begfirstRegShmem + threadIdx.x + threadIdx.y * 32]);

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
                                    }
                                    else {
                                        mainShmem[begSecRegShmem + threadIdx.x + threadIdx.y * 32] = uint32_t(atomicAdd_block(&(localFnConter[0]), 1) + localBlockMetaData[(i & 1) * 20 + 5] + localBlockMetaData[(i & 1) * 20 + 4]);
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
                        sync(cta);


                    }
                }

                //here we are after all of the blocks planned to be processed by this block are

        //updating local counters of last local block (normally it is done at the bagining of the next block)
        //but we need to check weather any block was processed at all
                sync(cta);
                pipeline.consumer_wait();

                afterBlockClean(cta, worQueueStep, localBlockMetaData, mainShmem, lastI[0],
                    fbArgs.metaData, tile, localFpConter, localFnConter
                    , blockFpConter, blockFnConter
                    , fbArgs.metaDataArrPointer, isAnythingInPadding, isBlockFull, isPaddingPass, isGoldForLocQueue, lastI);


                pipeline.consumer_release();

            }

            sync(cta);

            //     updating global counters
            if (tile.thread_rank() == 0 && tile.meta_group_rank() == 0) {
                if (blockFpConter[0] > 0) {
                    atomicAdd(&(fbArgs.minMaxes[10]), (blockFpConter[0]));
                }
            };
            if (tile.thread_rank() == 1 && tile.meta_group_rank() == 0) {
                if (blockFnConter[0] > 0) {
                    atomicAdd(&(fbArgs.minMaxes[11]), (blockFnConter[0]));
                }
            };
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
            metadataPass(fbArgs, !isPaddingPass
                , mainShmem, globalWorkQueueOffset, globalWorkQueueCounter
                , localWorkQueueCounter, localTotalLenthOfWorkQueue, localMinMaxes
                , fpFnLocCounter, isGoldPassToContinue, isSegmPassToContinue, cta, tile
                , fbArgs.metaData, fbArgs.minMaxes, fbArgs.workQueuePointer, fbArgs.metaDataArrPointer);




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
ForBoolKernelArgs<T> executeHausdoffGraph(ForFullBoolPrepArgs<T> fFArgs, const int WIDTH, const int HEIGHT, const int DEPTH, occupancyCalcData occData) {

    // For Graph
    cudaStream_t streamForGraph;
    cudaGraph_t graph;
    std::vector<cudaGraphNode_t> nodeDependencies;
    cudaGraphNode_t memcpyNode, kernelNode;
    cudaKernelNodeParams kernelNodeParams = { 0 };
    //  cudaMemcpyParams memcpyParams = { 0 };



    ForBoolKernelArgs<T> fbArgs = getArgsForKernel<T>(fFArgs, occData.warpsNumbForMainPass, occData.blockForMainPass, WIDTH, HEIGHT, DEPTH);

    checkCuda(cudaDeviceSynchronize(), "a1");

    //getMinMaxes << <blockSizeForMinMax, dim3(32, warpsNumbForMinMax) >> > ( minMaxes);
    getMinMaxes << <occData.blockSizeForMinMax, dim3(32, occData.warpsNumbForMinMax) >> > (fbArgs, fbArgs.minMaxes, fbArgs.goldArr.arrP, fbArgs.segmArr.arrP, fbArgs.metaData);

    checkCuda(cudaDeviceSynchronize(), "a1b");

    fbArgs.metaData = allocateMemoryAfterMinMaxesKernel(fbArgs, fFArgs);

    checkCuda(cudaDeviceSynchronize(), "a2b");

    boolPrepareKernel << <occData.blockSizeFoboolPrepareKernel, dim3(32, occData.warpsNumbForboolPrepareKernel) >> > (
        fbArgs, fbArgs.metaData, fbArgs.origArrsPointer, fbArgs.metaDataArrPointer, fbArgs.goldArr.arrP, fbArgs.segmArr.arrP, fbArgs.minMaxes);

    checkCuda(cudaDeviceSynchronize(), "a3");

    int fpPlusFn = allocateMemoryAfterBoolKernel(fbArgs, fFArgs);

    checkCuda(cudaDeviceSynchronize(), "a4");


    firstMetaPrepareKernel << <occData.blockForFirstMetaPass, occData.theadsForFirstMetaPass >> > (fbArgs, fbArgs.metaData, fbArgs.minMaxes, fbArgs.workQueuePointer, fbArgs.origArrsPointer, fbArgs.metaDataArrPointer);

    checkCuda(cudaDeviceSynchronize(), "a5");

    void* kernel_args[] = { &fbArgs };
    cudaLaunchCooperativeKernel((void*)(mainPassKernel<int>), occData.blockForMainPass, dim3(32, occData.warpsNumbForMainPass), kernel_args);


    cudaFreeAsync(fbArgs.resultListPointerMeta, 0);
    cudaFreeAsync(fbArgs.resultListPointerLocal, 0);
    cudaFreeAsync(fbArgs.resultListPointerIterNumb, 0);
    cudaFreeAsync(fbArgs.workQueuePointer, 0);
    cudaFreeAsync(fbArgs.origArrsPointer, 0);
    cudaFreeAsync(fbArgs.metaDataArrPointer, 0);
    cudaFreeAsync(fbArgs.mainArrAPointer, 0);
    cudaFreeAsync(fbArgs.mainArrBPointer, 0);

    return fbArgs;

}



#pragma once
template <typename T>
ForBoolKernelArgs<T> mainKernelsRun(ForFullBoolPrepArgs<T> fFArgs, const int WIDTH, const int HEIGHT, const int DEPTH
) {

    //cudaDeviceReset();
    cudaError_t syncErr;
    cudaError_t asyncErr;

    occupancyCalcData occData = getOccupancy<T>();

    //pointers ...


    ForBoolKernelArgs<T> fbArgs = executeHausdoffGraph(fFArgs, WIDTH, HEIGHT, DEPTH, occData);




    checkCuda(cudaDeviceSynchronize(), "last ");

    /////////// error handling 
    syncErr = cudaGetLastError();
    asyncErr = cudaDeviceSynchronize();
    if (syncErr != cudaSuccess) printf("Error in syncErr: %s\n", cudaGetErrorString(syncErr));
    if (asyncErr != cudaSuccess) printf("Error in asyncErr: %s\n", cudaGetErrorString(asyncErr));


    cudaDeviceReset();

    return fbArgs;
}





inline void setArrCPUB(bool* arrCPU, int x, int y, int z, int  Nx, int Ny) {

    arrCPU[x + y * Nx + z * Nx * Ny] = true;
};






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





void benchmarkMitura(bool* onlyBladderBoolFlat, bool* onlyLungsBoolFlat, const int WIDTH, const int HEIGHT, const int DEPTH) {

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

    // ForBoolKernelArgs<bool> fbArgs = executeHausdoff(forFullBoolPrepArgs, WIDTH, HEIGHT, DEPTH, occData);

    ForBoolKernelArgs<bool> fbArgs = mainKernelsRun(forFullBoolPrepArgs, WIDTH, HEIGHT, DEPTH);
    auto end = std::chrono::high_resolution_clock::now();

    std::cout << "Total elapsed time: ";
    std::cout << (double)(std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count() / (double)1000000000) << "s" << std::endl;


    size_t sizeMinMax = sizeof(unsigned int) * 20;
    cudaMemcpy(minMaxesCPU, fbArgs.metaData.minMaxes, sizeMinMax, cudaMemcpyDeviceToHost);

    printf("HD: %d \n", minMaxesCPU[13]);


    // freeee
    free(onlyBladderBoolFlat);
    free(onlyLungsBoolFlat);




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
    cudaMemcpyToSymbolAsync(WIDTH, &width, sizeof(width));
    cudaMemcpyToSymbolAsync(HEIGHT, &height, sizeof(height));
    cudaMemcpyToSymbolAsync(DEPTH, &depth, sizeof(depth));


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
    cudaDeviceReset();

    print(cudaGetLastError(), "processing CUDA. Something may be wrong with your CUDA device.");

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
    int dist = (*hd).computeDistance(&img1, &img2);
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

void loadHDF() {
    const int WIDTH = 512;
    const int HEIGHT = 512;
    //    const int DEPTH = 536;

    int DEPTH = 826;

    const H5std_string FILE_NAMEonlyLungsBoolFlat("D:\\dataSets\\forMainHDF5\\forHausdorffTests.hdf5");

    const H5std_string DATASET_NAMEonlyLungsBoolFlat("onlyLungsBoolFlat");
    //const H5std_string DATASET_NAMEonlyLungsBoolFlat("onlyLungsBoolFlatB");
    // create a vector the same size as the dataset
    bool* onlyLungsBoolFlat;
    loadHDFIntoBoolArr(FILE_NAMEonlyLungsBoolFlat, DATASET_NAMEonlyLungsBoolFlat, onlyLungsBoolFlat);

    const H5std_string FILE_NAMEonlyBladderBoolFlat("D:\\dataSets\\forMainHDF5\\forHausdorffTests.hdf5");


    const H5std_string DATASET_NAMEonlyBladderBoolFlat("onlyBladderBoolFlat");
    //const H5std_string DATASET_NAMEonlyBladderBoolFlat("onlyBladderBoolFlatB");
    // create a vector the same size as the dataset
    bool* onlyBladderBoolFlat;
    loadHDFIntoBoolArr(FILE_NAMEonlyBladderBoolFlat, DATASET_NAMEonlyBladderBoolFlat, onlyBladderBoolFlat);
    //onlyBladderBoolFlat = (bool*)calloc(WIDTH* HEIGHT* DEPTH, sizeof(bool));

    //onlyBladderBoolFlat[0] = true;

    // benchmarkOliviera(onlyBladderBoolFlat, onlyLungsBoolFlat, WIDTH, HEIGHT, DEPTH);//125 

    benchmarkMitura(onlyBladderBoolFlat, onlyLungsBoolFlat, WIDTH, HEIGHT, DEPTH);//124 or 259


    DEPTH = 536;
    const H5std_string FILE_NAMEonlyLungsBoolFlatB("D:\\dataSets\\forMainHDF5\\forHausdorffTests.hdf5");

    const H5std_string DATASET_NAMEonlyLungsBoolFlatB("onlyLungsBoolFlatB");
    // create a vector the same size as the dataset
    bool* onlyLungsBoolFlatB;
    loadHDFIntoBoolArr(FILE_NAMEonlyLungsBoolFlatB, DATASET_NAMEonlyLungsBoolFlatB, onlyLungsBoolFlatB);


    const H5std_string DATASET_NAMEonlyBladderBoolFlatB("onlyBladderBoolFlatB");
    // create a vector the same size as the dataset
    bool* onlyBladderBoolFlatB;
    loadHDFIntoBoolArr(FILE_NAMEonlyBladderBoolFlat, DATASET_NAMEonlyBladderBoolFlatB, onlyBladderBoolFlatB);

    //benchmarkOliviera(onlyBladderBoolFlatB, onlyLungsBoolFlatB, WIDTH, HEIGHT, DEPTH);//125 
    benchmarkMitura(onlyBladderBoolFlatB, onlyLungsBoolFlatB, WIDTH, HEIGHT, DEPTH);//124 or 259

}











int main(void) {

    //  const int WIDTH = atoi(argv[1]), HEIGHT = WIDTH, DEPTH = 1;
   //   Volume img1 = Volume(WIDTH, HEIGHT, DEPTH), img2 = Volume(WIDTH, HEIGHT, DEPTH);
   // testMainPasswes();
    loadHDF();



    return 0;  // successfully terminated
}



