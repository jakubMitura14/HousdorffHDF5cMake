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

template <typename TXPI>
inline __device__  void loadRightLeft(ForBoolKernelArgs<TXPI> fbArgs, thread_block& cta, uint32_t localBlockMetaData[]
    , uint32_t mainShmem[lengthOfMainShmem], cuda::pipeline<cuda::thread_scope_block>& pipeline
    , uint32_t* metaDataArr, MetaDataGPU& metaData, uint32_t& i, thread_block_tile<32>& tile
    , bool isGoldForLocQueue[localWorkQueLength], int iterationNumb[1]) {

    if (mainShmem[startOfLocalWorkQ + i] < (metaData.totalMetaLength - 1)) {
        cooperative_groups::memcpy_async(tile, (&mainShmem[begSMallRegShmemB]),
            &getSourceReduced(fbArgs, iterationNumb)[
                (mainShmem[startOfLocalWorkQ + i] + 1) * metaData.mainArrSectionLength + metaData.mainArrXLength * (1 - isGoldForLocQueue[i])
                    + tile.meta_group_rank() * 32], //we look for indicies 0,32,64... up to metaData.mainArrXLength
            cuda::aligned_size_t<4>(sizeof(uint32_t))
                    );
    }
}




////////////////////MAin
/*
loading data about this block to shmem
*/
template <typename TXPI>
inline __device__  void loadMain(ForBoolKernelArgs<TXPI>& fbArgs, thread_block& cta, uint32_t*& localBlockMetaData
    , uint32_t*& mainShmem, cuda::pipeline<cuda::thread_scope_block>& pipeline
    , uint32_t*& metaDataArr, MetaDataGPU& metaData, uint32_t& i, thread_block_tile<32>& tile
    , bool*& isGoldForLocQueue, int*& iterationNumb) {

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
inline __device__  void processMain(ForBoolKernelArgs<TXPI>& fbArgs, thread_block& cta, uint32_t*& localBlockMetaData
    , uint32_t*& mainShmem, cuda::pipeline<cuda::thread_scope_block>& pipeline
    , uint32_t*& metaDataArr, MetaDataGPU& metaData, uint32_t& i, thread_block_tile<32>& tile
    , bool*& isGoldForLocQueue, int*& iterationNumb) {

    pipeline.consumer_wait();

    mainShmem[begResShmem + threadIdx.x + threadIdx.y * 32] = bitDilatate(mainShmem[begSourceShmem + threadIdx.x + threadIdx.y * 32]);

    pipeline.consumer_release();


}



////////////////TOP
/*
loading data about block above to shmem
*/
template <typename TXPI>
inline __device__  void loadTop(ForBoolKernelArgs<TXPI>& fbArgs, thread_block& cta, uint32_t*& localBlockMetaData
    , uint32_t*& mainShmem, cuda::pipeline<cuda::thread_scope_block>& pipeline
    , uint32_t*& metaDataArr, MetaDataGPU& metaData, uint32_t& i, thread_block_tile<32>& tile
    , bool*& isGoldForLocQueue, int*& iterationNumb) {

    pipeline.producer_acquire();
    if (localBlockMetaData[13] < isGoldOffset) {
        cuda::memcpy_async(cta, (&mainShmem[begfirstRegShmem]),
            &getSourceReduced(fbArgs, iterationNumb)[localBlockMetaData[13] * metaData.mainArrSectionLength + metaData.mainArrXLength * (1 - isGoldForLocQueue[i])], //we look for indicies 0,32,64... up to metaData.mainArrXLength
            cuda::aligned_size_t<128>(sizeof(uint32_t) * metaData.mainArrXLength)
            , pipeline);
    }
    pipeline.producer_commit();

}


/*
loading data about block above to shmem
*/
template <typename TXPI>
inline __device__  void processTop(ForBoolKernelArgs<TXPI>& fbArgs, thread_block& cta, uint32_t*& localBlockMetaData
    , uint32_t*& mainShmem, cuda::pipeline<cuda::thread_scope_block>& pipeline
    , uint32_t*& metaDataArr, MetaDataGPU& metaData, uint32_t& i, thread_block_tile<32>& tile
    , bool*& isGoldForLocQueue, int*& iterationNumb, bool*& isAnythingInPadding) {

    pipeline.consumer_wait();

    dilatateHelperTopDown(0, mainShmem, isAnythingInPadding, localBlockMetaData, 13
        , 31, 0
        , begfirstRegShmem);

    pipeline.consumer_release();

}

/////BOTTOM
template <typename TXPI>
inline __device__  void loadBottom(ForBoolKernelArgs<TXPI>& fbArgs, thread_block& cta, uint32_t*& localBlockMetaData
    , uint32_t*& mainShmem, cuda::pipeline<cuda::thread_scope_block>& pipeline
    , uint32_t*& metaDataArr, MetaDataGPU& metaData, uint32_t& i, thread_block_tile<32>& tile
    , bool*& isGoldForLocQueue, int*& iterationNumb, bool*& isAnythingInPadding) {

    pipeline.producer_acquire();
    if (localBlockMetaData[14] < isGoldOffset) {
        cuda::memcpy_async(cta, (&mainShmem[begSecRegShmem]),
            &getSourceReduced(fbArgs, iterationNumb)[
                localBlockMetaData[14] * metaData.mainArrSectionLength + metaData.mainArrXLength * (1 - isGoldForLocQueue[i])], //we look for indicies 0,32,64... up to metaData.mainArrXLength
            cuda::aligned_size_t<128>(sizeof(uint32_t) * metaData.mainArrXLength)
                    , pipeline);
    }
    pipeline.producer_commit();

}


