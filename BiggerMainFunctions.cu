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


//if (mainShmem[startOfLocalWorkQ + i] > 0) {
//    cuda::memcpy_async(tile, (&mainShmem[begSMallRegShmemA + tile.meta_group_rank()]),
//        &getSourceReduced(fbArgs, iterationNumb)[
//            (mainShmem[startOfLocalWorkQ + i] - 1) * metaData.mainArrSectionLength + metaData.mainArrXLength * (1 - isGoldForLocQueue[i])
//                //we look for indicies 31,63... up to metaData.mainArrXLength
//                + (tile.meta_group_rank() * 32) + 31]
//        , cuda::aligned_size_t<4>(sizeof(uint32_t)), pipeline);
//
//}


//
//
//pipeline.producer_acquire();
////load data of interst form block to the right
//loadRightLeft(fbArgs, cta, localBlockMetaData, mainShmem, pipeline, metaDataArr, metaData, i, tile, isGoldForLocQueue, iterationNumb);
//
//
//////load data of interst form block to the left
////if (mainShmem[startOfLocalWorkQ + i] > 0) {
////    cuda::memcpy_async(tile, (&mainShmem[begSMallRegShmemA + tile.meta_group_rank()]),
////        &getSourceReduced(fbArgs, iterationNumb)[
////            (mainShmem[startOfLocalWorkQ + i] - 1) * metaData.mainArrSectionLength + metaData.mainArrXLength * (1 - isGoldForLocQueue[i])
////                //we look for indicies 31,63... up to metaData.mainArrXLength
////                + (tile.meta_group_rank() * 32) + 31]
////        , cuda::aligned_size_t<4>(sizeof(uint32_t)), pipeline);
//
////}
//pipeline.producer_commit();