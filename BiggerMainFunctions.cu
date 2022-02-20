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

/////////// loading functions





////////////////////MAin
/*
loading data about this block to shmem
*/
template <typename TXPI>
inline __device__  void loadMain(ForBoolKernelArgs<TXPI>& fbArgs, thread_block& cta, uint32_t(&localBlockMetaData)[40]
    , uint32_t(&mainShmem)[lengthOfMainShmem], cuda::pipeline<cuda::thread_scope_block>& pipeline
    , uint32_t*& metaDataArr, MetaDataGPU& metaData, uint32_t& i, thread_block_tile<32>& tile
    , bool(&isGoldForLocQueue)[localWorkQueLength], int(&iterationNumb)[1]) {

    pipeline.producer_acquire();
    cuda::memcpy_async(cta, &mainShmem[begSourceShmem], &getSourceReduced(fbArgs, iterationNumb)[
        mainShmem[startOfLocalWorkQ + i] * metaData.mainArrSectionLength + metaData.mainArrXLength * (1 - isGoldForLocQueue[i])],
        cuda::aligned_size_t<128>(sizeof(uint32_t) * metaData.mainArrXLength), pipeline);
    pipeline.producer_commit();


}

/*
process data about this block 
*/
template <typename TXPI>
inline __device__  void processMain(ForBoolKernelArgs<TXPI>& fbArgs, thread_block& cta, uint32_t(&localBlockMetaData)[40]
    , uint32_t(&mainShmem)[lengthOfMainShmem], cuda::pipeline<cuda::thread_scope_block>& pipeline
    , uint32_t*& metaDataArr, MetaDataGPU& metaData, uint32_t& i, thread_block_tile<32>& tile
    , bool(&isGoldForLocQueue)[localWorkQueLength], int(&iterationNumb)[1], bool(&isBlockFull)[1]) {

    pipeline.consumer_wait();
    //if ((((~mainShmem[begSourceShmem + threadIdx.x + threadIdx.y * 32]))  > 0)
//    || mainShmem[begSourceShmem + threadIdx.x + threadIdx.y * 32]==0
//    ) {
   // isBlockFull[0] = false;
    //    }
    //if (__popc(mainShmem[begSourceShmem + threadIdx.x + threadIdx.y * 32])<32) {
    //
    //    isBlockFull[0] = false;
    //}


    mainShmem[begResShmem + threadIdx.x + threadIdx.y * 32] = bitDilatate(mainShmem[begSourceShmem + threadIdx.x + threadIdx.y * 32]);
    //marking weather block is already full and no more dilatations are possible 


    pipeline.consumer_release();


}

////////////////TOP
/*
loading data about block above to shmem
*/
template <typename TXPI>
inline __device__  void loadTop(ForBoolKernelArgs<TXPI>& fbArgs, thread_block& cta, uint32_t(&localBlockMetaData)[40]
    , uint32_t(&mainShmem)[lengthOfMainShmem], cuda::pipeline<cuda::thread_scope_block>& pipeline
    , uint32_t*& metaDataArr, MetaDataGPU& metaData, uint32_t& i, thread_block_tile<32>& tile
    , bool(&isGoldForLocQueue)[localWorkQueLength], int(&iterationNumb)[1]) {

    pipeline.producer_acquire();
    if (localBlockMetaData[(i & 1) * 20+13] < isGoldOffset) {
        cuda::memcpy_async(cta, (&mainShmem[begfirstRegShmem]),
            &getSourceReduced(fbArgs, iterationNumb)[localBlockMetaData[(i & 1) * 20 + 13]
            * metaData.mainArrSectionLength + metaData.mainArrXLength * (1 - isGoldForLocQueue[i])],
            cuda::aligned_size_t<128>(sizeof(uint32_t) * metaData.mainArrXLength)
            , pipeline);
    }
    pipeline.producer_commit();

}


/*
loading data about block above to shmem
*/
template <typename TXPI>
inline __device__  void processTop(ForBoolKernelArgs<TXPI>& fbArgs, thread_block& cta, uint32_t(&localBlockMetaData)[40]
    , uint32_t(&mainShmem)[lengthOfMainShmem], cuda::pipeline<cuda::thread_scope_block>& pipeline
    , uint32_t*& metaDataArr, MetaDataGPU& metaData, uint32_t& i, thread_block_tile<32>& tile
    , bool(&isGoldForLocQueue)[localWorkQueLength], int(&iterationNumb)[1], bool(&isAnythingInPadding)[6] ) {

    pipeline.consumer_wait();

    dilatateHelperTopDown(0, mainShmem, isAnythingInPadding, localBlockMetaData, 13
        , 31, 0
        , begfirstRegShmem,i);

    pipeline.consumer_release();

}

/////BOTTOM
template <typename TXPI>
inline __device__  void loadBottom(ForBoolKernelArgs<TXPI>& fbArgs, thread_block& cta, uint32_t(&localBlockMetaData)[40]
    , uint32_t(&mainShmem)[lengthOfMainShmem], cuda::pipeline<cuda::thread_scope_block>& pipeline
    , uint32_t*& metaDataArr, MetaDataGPU& metaData, uint32_t& i, thread_block_tile<32>& tile
    , bool(&isGoldForLocQueue)[localWorkQueLength], int(&iterationNumb)[1], bool(&isAnythingInPadding)[6]) {

    pipeline.producer_acquire();
    if (localBlockMetaData[(i & 1) * 20+14] < isGoldOffset) {
        cuda::memcpy_async(cta, (&mainShmem[begSecRegShmem]),
            &getSourceReduced(fbArgs, iterationNumb)[localBlockMetaData[(i & 1) * 20 + 14] 
            * metaData.mainArrSectionLength + metaData.mainArrXLength * (1 - isGoldForLocQueue[i])],
            cuda::aligned_size_t<128>(sizeof(uint32_t) * metaData.mainArrXLength)
            , pipeline);
    }
    pipeline.producer_commit();

}

template <typename TXPI>
inline __device__  void processBottom(ForBoolKernelArgs<TXPI>& fbArgs, thread_block& cta, uint32_t(&localBlockMetaData)[40]
    , uint32_t(&mainShmem)[lengthOfMainShmem], cuda::pipeline<cuda::thread_scope_block>& pipeline
    , uint32_t*& metaDataArr, MetaDataGPU& metaData, uint32_t& i, thread_block_tile<32>& tile
    , bool(&isGoldForLocQueue)[localWorkQueLength], int(&iterationNumb)[1], bool(&isAnythingInPadding)[6]) {

    pipeline.consumer_wait();

    dilatateHelperTopDown(1, mainShmem, isAnythingInPadding, localBlockMetaData, 14
        , 0, 31
        , begSecRegShmem,i);

    pipeline.consumer_release();

}






///////////// right
template <typename TXPI>
inline __device__  void loadRight(ForBoolKernelArgs<TXPI>& fbArgs, thread_block& cta, uint32_t(&localBlockMetaData)[40]
    , uint32_t(&mainShmem)[lengthOfMainShmem], cuda::pipeline<cuda::thread_scope_block>& pipeline
    , uint32_t*& metaDataArr, MetaDataGPU& metaData, uint32_t& i, thread_block_tile<32>& tile
    , bool(&isGoldForLocQueue)[localWorkQueLength], int(&iterationNumb)[1], bool(&isAnythingInPadding)[6]) {



    pipeline.producer_acquire();
    if (localBlockMetaData[(i & 1) * 20+16] < isGoldOffset) {
        cuda::memcpy_async(cta, (&mainShmem[begfirstRegShmem]),
            &getSourceReduced(fbArgs, iterationNumb)[localBlockMetaData[(i & 1) * 20+16] * metaData.mainArrSectionLength + metaData.mainArrXLength * (1 - isGoldForLocQueue[i])], 
            cuda::aligned_size_t<128>(sizeof(uint32_t) * metaData.mainArrXLength)
            , pipeline);
    }
    pipeline.producer_commit();
}


template <typename TXPI>
inline __device__  void processRight(ForBoolKernelArgs<TXPI>& fbArgs, thread_block& cta, uint32_t(&localBlockMetaData)[40]
    , uint32_t(&mainShmem)[lengthOfMainShmem], cuda::pipeline<cuda::thread_scope_block>& pipeline
    , uint32_t*& metaDataArr, MetaDataGPU& metaData, uint32_t& i, thread_block_tile<32>& tile
    , bool(&isGoldForLocQueue)[localWorkQueLength], int(&iterationNumb)[1], bool(&isAnythingInPadding)[6]) {


    pipeline.consumer_wait();

    dilatateHelperForTransverse(fbArgs,(threadIdx.x == (fbArgs.dbXLength - 1)),
        3, (1), (0), mainShmem, isAnythingInPadding
        , threadIdx.y, 0
        , 16, begfirstRegShmem, localBlockMetaData,i);

    pipeline.consumer_release();
}



///////////// left
template <typename TXPI>
inline __device__  void loadLeft(ForBoolKernelArgs<TXPI>& fbArgs, thread_block& cta, uint32_t(&localBlockMetaData)[40]
    , uint32_t(&mainShmem)[lengthOfMainShmem], cuda::pipeline<cuda::thread_scope_block>& pipeline
    , uint32_t*& metaDataArr, MetaDataGPU& metaData, uint32_t& i, thread_block_tile<32>& tile
    , bool(&isGoldForLocQueue)[localWorkQueLength], int(&iterationNumb)[1], bool(&isAnythingInPadding)[6]) {



    pipeline.producer_acquire();
    if (mainShmem[startOfLocalWorkQ + i]>0) {
        cuda::memcpy_async(cta, (&mainShmem[begSecRegShmem]),
            &getSourceReduced(fbArgs, iterationNumb)[(mainShmem[startOfLocalWorkQ + i]-1) * metaData.mainArrSectionLength + metaData.mainArrXLength * (1 - isGoldForLocQueue[i])],
            cuda::aligned_size_t<128>(sizeof(uint32_t) * metaData.mainArrXLength)
            , pipeline);
    }
    pipeline.producer_commit();
}


template <typename TXPI>
inline __device__  void processLeft(ForBoolKernelArgs<TXPI>& fbArgs, thread_block& cta, uint32_t(&localBlockMetaData)[40]
    , uint32_t(&mainShmem)[lengthOfMainShmem], cuda::pipeline<cuda::thread_scope_block>& pipeline
    , uint32_t*& metaDataArr, MetaDataGPU& metaData, uint32_t& i, thread_block_tile<32>& tile
    , bool(&isGoldForLocQueue)[localWorkQueLength], int(&iterationNumb)[1], bool(&isAnythingInPadding)[6]) {


    pipeline.consumer_wait();

    dilatateHelperForTransverse(fbArgs,(threadIdx.x == 0),
        2, (-1), (0), mainShmem, isAnythingInPadding
        , threadIdx.y, 31
        , 15, begSecRegShmem, localBlockMetaData,i);

    pipeline.consumer_release();
}

///////////// anterior
template <typename TXPI>
inline __device__  void loadAnterior(ForBoolKernelArgs<TXPI>& fbArgs, thread_block& cta, uint32_t(&localBlockMetaData)[40]
    , uint32_t(&mainShmem)[lengthOfMainShmem], cuda::pipeline<cuda::thread_scope_block>& pipeline
    , uint32_t*& metaDataArr, MetaDataGPU& metaData, uint32_t& i, thread_block_tile<32>& tile
    , bool(&isGoldForLocQueue)[localWorkQueLength], int(&iterationNumb)[1], bool(&isAnythingInPadding)[6]) {

    pipeline.producer_acquire();
    if (localBlockMetaData[(i & 1) * 20+17] < isGoldOffset ) {

        cuda::memcpy_async(cta, (&mainShmem[begfirstRegShmem]),
            &getSourceReduced(fbArgs, iterationNumb)[localBlockMetaData[(i & 1) * 20 + 17] * metaData.mainArrSectionLength + metaData.mainArrXLength * (1 - isGoldForLocQueue[i])],
            cuda::aligned_size_t<128>(sizeof(uint32_t) * metaData.mainArrXLength)
            , pipeline);
    }
    pipeline.producer_commit();
}


template <typename TXPI>
inline __device__  void processAnterior(ForBoolKernelArgs<TXPI>& fbArgs, thread_block& cta, uint32_t(&localBlockMetaData)[40]
    , uint32_t(&mainShmem)[lengthOfMainShmem], cuda::pipeline<cuda::thread_scope_block>& pipeline
    , uint32_t*& metaDataArr, MetaDataGPU& metaData, uint32_t& i, thread_block_tile<32>& tile
    , bool(&isGoldForLocQueue)[localWorkQueLength], int(&iterationNumb)[1], bool(&isAnythingInPadding)[6]) {

    pipeline.consumer_wait();

    dilatateHelperForTransverse(fbArgs,(threadIdx.y == (fbArgs.dbYLength - 1)), 4
        , (0), (1), mainShmem, isAnythingInPadding
        , 0, threadIdx.x
        , 17, begfirstRegShmem, localBlockMetaData, i);
    pipeline.consumer_release();
}

///////////// posterior
template <typename TXPI>
inline __device__  void loadPosterior(ForBoolKernelArgs<TXPI>& fbArgs, thread_block& cta, uint32_t(&localBlockMetaData)[40]
    , uint32_t(&mainShmem)[lengthOfMainShmem], cuda::pipeline<cuda::thread_scope_block>& pipeline
    , uint32_t*& metaDataArr, MetaDataGPU& metaData, uint32_t& i, thread_block_tile<32>& tile
    , bool(&isGoldForLocQueue)[localWorkQueLength], int(&iterationNumb)[1], bool(&isAnythingInPadding)[6]) {

    pipeline.producer_acquire();
    if (localBlockMetaData[(i & 1) * 20+18] < isGoldOffset) {


        cuda::memcpy_async(cta, (&mainShmem[begSecRegShmem]),
            &getSourceReduced(fbArgs, iterationNumb)[localBlockMetaData[(i & 1) * 20 + 18] * metaData.mainArrSectionLength + metaData.mainArrXLength * (1 - isGoldForLocQueue[i])],
            cuda::aligned_size_t<128>(sizeof(uint32_t) * metaData.mainArrXLength)
            , pipeline);
    }
    pipeline.producer_commit();
}





//////////// last load 

/*
load reference if needed or data for next iteration if there is such
*/
template <typename TXPI>
inline __device__  void lastLoad(ForBoolKernelArgs<TXPI>& fbArgs, thread_block& cta, uint32_t(&localBlockMetaData)[40]
    , uint32_t(&mainShmem)[lengthOfMainShmem], cuda::pipeline<cuda::thread_scope_block>& pipeline
    , uint32_t*& metaDataArr, MetaDataGPU& metaData, uint32_t& i, thread_block_tile<32>& tile
    , bool(&isGoldForLocQueue)[localWorkQueLength], int(&iterationNumb)[1], bool(&isAnythingInPadding)[6]
    , uint32_t*& origArrs, unsigned int (&worQueueStep)[1]) {

    pipeline.producer_acquire();
      
    //if block should be validated we load data for validation
    if (localBlockMetaData[(i & 1) * 20+((1 - isGoldForLocQueue[i]) + 1)] //fp for gold and fn count for not gold
                   > localBlockMetaData[(i & 1) * 20+((1 - isGoldForLocQueue[i]) + 3)]) {// so count is bigger than counter so we should validate
        cuda::memcpy_async(cta, (&mainShmem[begfirstRegShmem]),
            &origArrs[mainShmem[startOfLocalWorkQ + i] * metaData.mainArrSectionLength + metaData.mainArrXLength * (isGoldForLocQueue[i])], //we look for 
            cuda::aligned_size_t<128>(sizeof(uint32_t) * metaData.mainArrXLength)
            , pipeline);
      
    }
    else {//if we are not validating we immidiately start loading data for next loop
        if (i + 1 <= worQueueStep[0]) {
            loadMetaDataToShmem(cta, localBlockMetaData, mainShmem, pipeline, metaDataArr, metaData, 1, i);
        }
    }


    pipeline.producer_commit();
}

template <typename TXPI>
inline __device__  void processPosteriorAndSaveResShmem(ForBoolKernelArgs<TXPI>& fbArgs, thread_block& cta
    , uint32_t(&localBlockMetaData)[40]
    , uint32_t(&mainShmem)[lengthOfMainShmem], cuda::pipeline<cuda::thread_scope_block>& pipeline
    , uint32_t*& metaDataArr, MetaDataGPU& metaData, uint32_t& i, thread_block_tile<32>& tile
    , bool(&isGoldForLocQueue)[localWorkQueLength], int(&iterationNumb)[1], bool(&isAnythingInPadding)[6],
    bool(&isBlockFull)[1]) {

    pipeline.consumer_wait();
    //dilatate posterior 
    dilatateHelperForTransverse(fbArgs, (threadIdx.y == 0), 5
        , (0), (-1), mainShmem, isAnythingInPadding
        , fbArgs.dbYLength - 1, threadIdx.x // we add offset depending on y dimension
        , 18, begSecRegShmem, localBlockMetaData, i);
    //now all data should be properly dilatated we save it to global memory
    //try save target reduced via mempcy async ...

    getTargetReduced(fbArgs, iterationNumb)[mainShmem[startOfLocalWorkQ + i] * metaData.mainArrSectionLength + metaData.mainArrXLength * (1 - isGoldForLocQueue[i])
        + threadIdx.x + threadIdx.y * 32]
        = mainShmem[begResShmem + threadIdx.x + threadIdx.y * 32];

    //TODO remove 

    //for (uint8_t bitPos = 0; bitPos < 32; bitPos++) {
    //    if (threadIdx.x == 0 && threadIdx.y == 0) {

    //    //if any bit here is set it means it should be added to result list 
    //    if (isBitAt(mainShmem[begResShmem + threadIdx.x + threadIdx.y * 32], bitPos)) {
    //        //if (mainShmem[startOfLocalWorkQ + i] * 32 + bitPos>130) {
    //            printf("bit set loc %d isGold %d \n", mainShmem[startOfLocalWorkQ + i] * 32 + bitPos, isGoldForLocQueue[i]);
    //        //}
    //    }
    //    
    //    }
    //}
    
    
    
    //if (!(localBlockMetaData[(i & 1) * 20 + ((1 - isGoldForLocQueue[i]) + 1)] //fp for gold and fn count for not gold
    //> localBlockMetaData[(i & 1) * 20 + ((1 - isGoldForLocQueue[i]) + 3)])) {// so count is bigger than counter so we should validate
    //    mainShmem[begResShmem + threadIdx.x + threadIdx.y * 32] = 0;
    //    mainShmem[begSourceShmem + threadIdx.x + threadIdx.y * 32] = 0;
    //}


    pipeline.consumer_release();
}


//////////// validation

template <typename TXPI>
inline __device__  void validate(ForBoolKernelArgs<TXPI>& fbArgs, thread_block& cta, uint32_t(&localBlockMetaData)[40]
    , uint32_t(&mainShmem)[lengthOfMainShmem], cuda::pipeline<cuda::thread_scope_block>& pipeline
    , uint32_t*& metaDataArr, MetaDataGPU& metaData, uint32_t& i, thread_block_tile<32>& tile
    , bool(&isGoldForLocQueue)[localWorkQueLength], int(&iterationNumb)[1], bool(&isAnythingInPadding)[6]
    , bool(&isBlockFull)[1]
, unsigned int (&localFpConter)[1], unsigned int (&localFnConter)[1]
, uint32_t*& resultListPointerMeta, uint32_t*& resultListPointerLocal, uint32_t*& resultListPointerIterNumb

) {

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
           // if (isBitAt(mainShmem[begSourceShmem + threadIdx.x + threadIdx.y * 32], bitPos)) {
                //first we add to the resList
                //TODO consider first passing it into shared memory and then async mempcy ...
                //we use offset plus number of results already added (we got earlier count from global memory now we just atomically add locally)
                unsigned int old = 0;
                ////// IMPORTANT for some reason in order to make it work resultfnOffset and resultfnOffset swith places
                if (isGoldForLocQueue[i]) {
                    old = atomicAdd_block(&(localFpConter[0]), 1) + localBlockMetaData[(i & 1) * 20 + 6] + localBlockMetaData[(i & 1) * 20 + 3];
                }
                else {
                    old = atomicAdd_block(&(localFnConter[0]), 1) + localBlockMetaData[(i & 1) * 20 + 5] + localBlockMetaData[(i & 1) * 20 + 4];
                //    printf("local fn counter add \n");

                };
                //   add results to global memory    
                //we add one gere jjust to distinguish it from empty result
                resultListPointerMeta[old] = uint32_t(mainShmem[startOfLocalWorkQ + i] + (isGoldOffset * isGoldForLocQueue[i]) + 1);
                resultListPointerLocal[old] = uint32_t((fbArgs.dbYLength * 32 * bitPos) + (threadIdx.y * 32) + (threadIdx.x));
                resultListPointerIterNumb[old] = uint32_t(iterationNumb[0]);

                   //printf("rrrrresult i %d  meta %d isGold %d old %d localFpConter %d localFnConter %d fpOffset %d fnOffset %d linIndUpdated %d  localInd %d  xLoc %d yLoc %d zLoc %d \n"
                   //    ,i
                   //    ,mainShmem[startOfLocalWorkQ + i]
                   //    , isGoldForLocQueue[i]
                   //    , old
                   //    , localFpConter[0]
                   //    , localFnConter[0]
                   //    , localBlockMetaData[(i & 1) * 20+ 5]
                   //    , localBlockMetaData[(i & 1) * 20+6]
                   //    , uint32_t(mainShmem[startOfLocalWorkQ + i] + isGoldOffset * isGoldForLocQueue[i])
                   //    , uint32_t((fbArgs.dbYLength * 32 * bitPos) + (threadIdx.y * 32) + (threadIdx.x))
                   //    , threadIdx.x
                   //    , threadIdx.y
                   //    , bitPos
                   //);


                   printf("\n rrrrresult meta %d isGold %d old %d  xLoc %d yLoc %d zLoc %d iterNumbb %d \n"
                       , mainShmem[startOfLocalWorkQ + i]
                       , isGoldForLocQueue[i]
                       , old
                       , threadIdx.x
                       , threadIdx.y
                       , bitPos
                       , iterationNumb[0]
                   );


            }

        };
        //mainShmem[begResShmem + threadIdx.x + threadIdx.y * 32] = 0;
        //mainShmem[begSourceShmem + threadIdx.x + threadIdx.y * 32] = 0;

    }
}
