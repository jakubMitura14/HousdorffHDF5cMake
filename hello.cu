
#include "cuda_runtime.h"
#include "CPUAllocations.cu"
#include "MetaData.cu"

#include "ExceptionManagUtils.cu"
#include "CooperativeGroupsUtils.cu"
#include "MainPassFunctions.cu"
#include <cstdint>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cooperative_groups/memcpy_async.h>
#include <cuda/pipeline>


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
#include "testAll.cu"
using namespace cooperative_groups;

using namespace cooperative_groups;

#include <iostream>
#include <string>
#include <vector>
#include <H5Cpp.h>
using namespace H5;
using std::cout;
using std::endl;
#include <string>
#include "forBench/Volume.h"
#include "forBench/HausdorffDistance.cuh"
#include "forBench/HausdorffDistance.cu"





//using std::cout;
//using std::endl;
//#include <string>
//#include "Volume.cuh"
//#include "HausdorffDistance.cuh"
//#include "HausdorffDistance.cu"
//
//#include <iostream>
//#include <string>
//#include <vector>




/*
becouse we need a lot of the additional memory spaces to minimize memory consumption allocations will be postponed after first kernel run enabling
*/
#pragma once
template <typename ZZR>
inline int allocateMemoryAfterBoolKernel(ForBoolKernelArgs<ZZR>& gpuArgs, ForFullBoolPrepArgs<ZZR>& cpuArgs,
    uint32_t*& resultListPointerMeta
    , uint32_t*& resultListPointerLocal
    , uint32_t*& resultListPointerIterNumb,
    uint32_t*& origArrsPointer,
    uint32_t*& mainArrAPointer,
    uint32_t*& mainArrBPointer, MetaDataGPU& metaData, array3dWithDimsGPU<ZZR>& goldArr, array3dWithDimsGPU<ZZR>& segmArr) {

    //free no longer needed arrays
    cudaFreeAsync(goldArr.arrP, 0);
    cudaFreeAsync(segmArr.arrP, 0);

    //copy on cpu
    size_t size = sizeof(unsigned int) * 20;
    cudaMemcpy(cpuArgs.metaData.minMaxes, gpuArgs.metaData.minMaxes, size, cudaMemcpyDeviceToHost);

    unsigned int fpPlusFn = cpuArgs.metaData.minMaxes[7] + cpuArgs.metaData.minMaxes[8];


    size = sizeof(uint32_t) * (fpPlusFn + 50);


    cudaMallocAsync(&resultListPointerLocal, size, 0);
    cudaMallocAsync(&resultListPointerIterNumb, size, 0);
    cudaMallocAsync(&resultListPointerMeta, size, 0);


    auto xRange = metaData.metaXLength;
    auto yRange = metaData.MetaYLength;
    auto zRange = metaData.MetaZLength;




    size_t sizeB = metaData.totalMetaLength * metaData.mainArrSectionLength * sizeof(uint32_t);

    //printf("size of reduced main arr %d total meta len %d mainArrSectionLen %d  \n", sizeB, metaData.totalMetaLength, metaData.mainArrSectionLength);

    cudaMallocAsync(&mainArrAPointer, sizeB, 0);
    cudaMemcpyAsync(mainArrAPointer, origArrsPointer, sizeB, cudaMemcpyDeviceToDevice, 0);


    cudaMallocAsync(&mainArrBPointer, sizeB, 0);
    cudaMemcpyAsync(mainArrBPointer, origArrsPointer, sizeB, cudaMemcpyDeviceToDevice, 0);

    //just in order set it to 0
    uint32_t* resultListPointerMetaCPU = (uint32_t*)calloc(fpPlusFn + 50, sizeof(uint32_t));
    cudaMemcpyAsync(resultListPointerMeta, resultListPointerMetaCPU, size, cudaMemcpyHostToDevice, 0);
    free(resultListPointerMetaCPU);




    return fpPlusFn;
};




#pragma once
template <typename ZZR>
inline MetaDataGPU allocateMemoryAfterMinMaxesKernel(ForBoolKernelArgs<ZZR>& gpuArgs, ForFullBoolPrepArgs<ZZR>& cpuArgs,
    uint32_t*& workQueue, unsigned int* minMaxes, MetaDataGPU& metaData, uint32_t*& origArr
    , uint32_t*& metaDataArr) {
    ////reduced arrays


    //copy on cpu
    size_t size = sizeof(unsigned int) * 20;
    cudaMemcpy(cpuArgs.metaData.minMaxes, minMaxes, size, cudaMemcpyDeviceToHost);

    //read an modify
    //1)maxX 2)minX 3)maxY 4) minY 5) maxZ 6) minZ
    //7)global FP count; 8)global FN count
    unsigned int xRange = cpuArgs.metaData.minMaxes[1] - cpuArgs.metaData.minMaxes[2] + 1;
    unsigned int yRange = cpuArgs.metaData.minMaxes[3] - cpuArgs.metaData.minMaxes[4] + 1;
    unsigned int zRange = cpuArgs.metaData.minMaxes[5] - cpuArgs.metaData.minMaxes[6] + 1;
    unsigned int totalMetaLength = (xRange) * (yRange) * (zRange);
    /* printf("in allocateMemoryAfterMinMaxesKernel totalMetaLength  %d  xRange %d yRange %d zRange %d \n"
         , totalMetaLength
         , (xRange)
         , (yRange)
         , (zRange));*/

         //updating size informations
    metaData.metaXLength = xRange;
    metaData.MetaYLength = yRange;
    metaData.MetaZLength = zRange;
    metaData.totalMetaLength = totalMetaLength;

    //cpuArgs.metaData.metaXLength = xRange;
    //cpuArgs.metaData.MetaYLength = yRange;
    //cpuArgs.metaData.MetaZLength = zRange;
    //cpuArgs.metaData.totalMetaLength = totalMetaLength;
    //saving min maxes
    metaData.maxX = cpuArgs.metaData.minMaxes[1];
    metaData.minX = cpuArgs.metaData.minMaxes[2];
    metaData.maxY = cpuArgs.metaData.minMaxes[3];
    metaData.minY = cpuArgs.metaData.minMaxes[4];
    metaData.maxZ = cpuArgs.metaData.minMaxes[5];
    metaData.minZ = cpuArgs.metaData.minMaxes[6];





    //int i = 1;
    //printf("maxX %d  [%d]\n", cpuArgs.metaData.minMaxes[i], i);
    //i = 2;
    //printf("minX %d  [%d]\n", cpuArgs.metaData.minMaxes[i], i);
    //i = 3;
    //printf("maxY %d  [%d]\n", cpuArgs.metaData.minMaxes[i], i);
    //i = 4;
    //printf("minY %d  [%d]\n", cpuArgs.metaData.minMaxes[i], i);
    //i = 5;
    //printf("maxZ %d  [%d]\n", cpuArgs.metaData.minMaxes[i], i);
    //i = 6;
    //printf("minZ %d  [%d]\n", cpuArgs.metaData.minMaxes[i], i);

  /*  int ii = 7;
    printf("global FP count %d  [%d]\n", cpuArgs.metaData.minMaxes[ii], ii);
    ii = 8;
    printf("global FN count %d  [%d]\n", cpuArgs.metaData.minMaxes[ii], ii);
    ii = 9;
    printf("workQueueCounter %d  [%d]\n", cpuArgs.metaData.minMaxes[ii], ii);
    ii = 10;
    printf("resultFP globalCounter %d  [%d]\n", cpuArgs.metaData.minMaxes[ii], ii);
    ii = 11;
    printf("resultFn globalCounter %d  [%d]\n", cpuArgs.metaData.minMaxes[ii], ii);
    ii = 12;
    printf("global offset counter %d  [%d]\n", cpuArgs.metaData.minMaxes[ii], ii);*/










    //allocating needed memory
    // main array
    unsigned int mainArrXLength = gpuArgs.dbXLength * gpuArgs.dbYLength;
    unsigned int mainArrSectionLength = (mainArrXLength * 2);
    metaData.mainArrXLength = mainArrXLength;
    metaData.mainArrSectionLength = mainArrSectionLength;

    size_t sizeB = totalMetaLength * mainArrSectionLength * sizeof(uint32_t);


    //cudaMallocAsync(&mainArr, sizeB, 0);
    size_t sizeorigArr = totalMetaLength * (mainArrXLength * 2) * sizeof(uint32_t);
    cudaMallocAsync(&origArr, sizeorigArr, 0);
    size_t sizemetaDataArr = totalMetaLength * (20) * sizeof(uint32_t) + 100;
    cudaMallocAsync(&metaDataArr, sizemetaDataArr, 0);


    size_t sizeC = (totalMetaLength * 2 * sizeof(uint32_t) + 50);
    //cudaMallocAsync(&workQueue, size, 0);
    cudaMallocAsync(&workQueue, sizeC, 0);
    // printf("in allocateMemoryAfterMinMaxesKernel workQueu size  %d isGold constant value %d  \n", totalMetaLength * 2  + 50, isGoldOffset );

    return metaData;
};





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

            //if (paddingPos == 2 && isGoldForLocQueue[i] == 0) {
            //    printf("b padding begining  in processs left  \n"
            //    );

            //}

            //now we load - we already done earlier up and down so now we are considering only anterior, posterior , left , right possibilities
            if (mainShmem[begSourceShmem + threadIdx.x + threadIdx.y * 32] > 0) {
                isAnythingInPadding[paddingPos] = true;

                //if (paddingPos == 3 && isGoldForLocQueue[i] == 0) {
                //    printf("c padding begining  in processs right  \n"
                //    );

                //}

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
        //if (paddingPos==0) {               

        //    //mainShmem[begResShmem + threadIdx.x + threadIdx.y * 32] |= ((mainShmem[targetShmemOffset + threadIdx.x + threadIdx.y * 32] >> sourceBit) & 1) << targetBit;
        //    mainShmem[begResShmem + threadIdx.x + threadIdx.y * 32] |= ((mainShmem[targetShmemOffset + threadIdx.x + threadIdx.y * 32] & uint32_t(1)));
        //}
        //else {
        //  
        //   // mainShmem[begResShmem + threadIdx.x + threadIdx.y * 32] |= ((mainShmem[targetShmemOffset + threadIdx.x + threadIdx.y * 32] >> sourceBit) & 1) << targetBit;
        //      mainShmem[begResShmem + threadIdx.x + threadIdx.y * 32] |= ((mainShmem[targetShmemOffset + threadIdx.x + threadIdx.y * 32] & uint32_t(2147483648)));
        //
        //}


        //mainShmem[begResShmem + threadIdx.x + threadIdx.y * 32] =
        //    mainShmem[begResShmem + threadIdx.x + threadIdx.y * 32]
        //    | (mainShmem[targetShmemOffset + threadIdx.x + threadIdx.y * 32] & numberWithCorrBitSetInNeigh);

    }

}


//inline __device__  void lastLoad(ForBoolKernelArgs<TXPPI> fbArgs, thread_block cta//some needed CUDA objects
//    , unsigned int worQueueStep[1], uint32_t localBlockMetaData[(i & 1) * 20+]
//    , uint32_t mainShmem[], uint32_t i, MetaDataGPU metaData
//) {


//
///*
//constitutes end of pipeline  where we load data for next iteration if such is present
//*/
//template <typename TXPPI>
//inline __device__  void lastLoad(ForBoolKernelArgs<TXPPI> fbArgs, thread_block& cta//some needed CUDA objects
//    , unsigned int worQueueStep[1], uint32_t localBlockMetaData[(i & 1) * 20+]
//    , uint32_t mainShmem[], uint32_t i, MetaDataGPU metaData, uint32_t* metaDataArr
//) {
//
//    if (i + 1 <= worQueueStep[0]) {
//        cuda::memcpy_async(cta, (&localBlockMetaData[(i & 1) * 20+0]),
//            (&metaDataArr[(mainShmem[startOfLocalWorkQ + i - isGoldOffset * (mainShmem[startOfLocalWorkQ + i] >= isGoldOffset))
//                * metaData.metaDataSectionLength]])
//            , cuda::aligned_size_t<4>(sizeof(uint32_t) * 20), pipeline);
//    }
//
//
//};

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
    , bool(&isAnythingInPadding)[6], bool(&isBlockFull)[1], const bool isPaddingPass, bool(&isGoldForLocQueue)[localWorkQueLength], uint32_t(&lastI)[1]
) {



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
        if (isBlockFull[0] && i > 0) {
            //setting data in metadata that block is full
           // metaDataArr[mainShmem[startOfLocalWorkQ + i] * metaData.metaDataSectionLength + 10 - (isGoldForLocQueue[i] * 2)] = true;
        }
        //resetting for some reason  block 0 gets as full even if it should not ...
        isBlockFull[0] = true;// mainShmem[startOfLocalWorkQ + i]>0;//!isPaddingPass;
    };




    //we do it only for non padding pass
    if (threadIdx.x < 6 && threadIdx.y == 1 && !isPaddingPass) {
        //executed in case of previous block
        if (i >= 0) {
            auto metadataTarget = localBlockMetaData[(i & 1) * 20 + 13 + threadIdx.x];

            if (metadataTarget < isGoldOffset) {

                if (isAnythingInPadding[threadIdx.x]) {
                    // holding data weather we have anything in padding 0)top  1)bottom, 2)left 3)right, 4)anterior, 5)posterior,

                    //if (threadIdx.x == 4   ) {
                    //    printf(" padding in end  processs anterior  at the end of linMeta %d  isGold %d \n"
                    //    , metadataTarget
                    //        , isGoldForLocQueue[i]
                    //    );

                    //}


                    //if (threadIdx.x == 5) {
                    //    printf(" padding in end  processs posterior  at the end of linMeta %d  isGold %d \n"
                    //        , metadataTarget
                    //        , isGoldForLocQueue[i]
                    //    );

                    //}



                   // printf( "in setting paddings metadata target %d  full index %d  \n", metadataTarget, metadataTarget * metaData.metaDataSectionLength + 12 - isGoldForLocQueue[i]);
 /*                   if (metadataTarget>0 && metadataTarget < metaData.totalMetaLength) {
                        metaDataArr[localBlockMetaData[(i & 1) * 20 + 13 + threadIdx.x] * metaData.metaDataSectionLength + 12 - isGoldForLocQueue[i]] = 1;
                    }*/
                    //if (metadataTarget > 0 && metadataTarget < metaData.totalMetaLength) {
                    metaDataArr[metadataTarget * metaData.metaDataSectionLength + 12 - isGoldForLocQueue[i]] = 1;
                    //}


                }

            }
        }
        isAnythingInPadding[tile.thread_rank()] = false;
    };
    //if (tile.thread_rank() == 0 && tile.meta_group_rank() == 3) {// this is how it is encoded wheather it is gold or segm block

    //    if (i >= 0) {
    //        lastI[0] = UINT32_MAX;
    //    };
    //}

}





////////////////// with pipeline ofr barrier

/*
initial cleaning  and initializations of dilatation kernel

*/
#pragma once
inline __device__  void dilBlockInitialClean(thread_block_tile<32>& tile,
    const  bool isPaddingPass, int(&iterationNumb)[1],
    unsigned int(&localWorkQueueCounter)[1], unsigned int(&blockFpConter)[1],
    unsigned int(&blockFnConter)[1], unsigned int(&localFpConter)[1],
    unsigned int(&localFnConter)[1], bool(&isBlockFull)[1],
    unsigned int(&fpFnLocCounter)[1],
    unsigned int(&localTotalLenthOfWorkQueue)[1], unsigned int(&globalWorkQueueOffset)[1]
    , unsigned int(&worQueueStep)[1], unsigned int*& minMaxes, unsigned int(&localMinMaxes)[5], uint32_t(&lastI)[1])
{

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
}



/*
load work que from global memory
*/
#pragma once
inline __device__  void loadWorkQueue(thread_block& cta, uint32_t(&mainShmem)[lengthOfMainShmem], uint32_t*& workQueue
    , bool(&isGoldForLocQueue)[localWorkQueLength], uint32_t& bigloop, unsigned int(&worQueueStep)[1]) {

    //to do change into barrier

    //cuda::memcpy_async(cta, (&mainShmem[startOfLocalWorkQ]), (&workQueue[bigloop])
    //    , cuda::aligned_size_t<4>(sizeof(uint32_t) * worQueueStep[0]), pipeline);

    for (uint16_t ii = cta.thread_rank(); ii < worQueueStep[0]; ii += cta.size()) {
        mainShmem[startOfLocalWorkQ + ii] = workQueue[bigloop + ii];
        isGoldForLocQueue[ii] = (mainShmem[startOfLocalWorkQ + ii] >= isGoldOffset);
        mainShmem[startOfLocalWorkQ + ii] = mainShmem[startOfLocalWorkQ + ii] - isGoldOffset * isGoldForLocQueue[ii];

    }
}


/*
loads metadata of given block to meta data
*/
#pragma once
inline __device__  void loadMetaDataToShmem(thread_block& cta, uint32_t(&localBlockMetaData)[40]
    , uint32_t(&mainShmem)[lengthOfMainShmem], cuda::pipeline<cuda::thread_scope_block>& pipeline
    , uint32_t*& metaDataArr, MetaDataGPU& metaData, const uint8_t toAdd, uint32_t& ii) {

    //cuda::memcpy_async(cta, (&localBlockMetaData[(ii&1)*20]),
    //    (&metaDataArr[(mainShmem[startOfLocalWorkQ + toAdd+ii])
    //        * metaData.metaDataSectionLength])
    //    , cuda::aligned_size_t<4>(sizeof(uint32_t) * 20), pipeline);

    cuda::memcpy_async(cta, (&localBlockMetaData[((ii + 1) & 1) * 20]),
        (&metaDataArr[(mainShmem[startOfLocalWorkQ + toAdd + ii])
            * metaData.metaDataSectionLength])
        , cuda::aligned_size_t<4>(sizeof(uint32_t) * 20), pipeline);


}





////////////////////MAin
/*
loading data about this block to shmem
*/
#pragma once
template <typename TXPI>
inline __device__  void loadMain(ForBoolKernelArgs<TXPI>& fbArgs, thread_block& cta, uint32_t(&localBlockMetaData)[40]
    , uint32_t(&mainShmem)[lengthOfMainShmem], cuda::pipeline<cuda::thread_scope_block>& pipeline
    , uint32_t*& metaDataArr, MetaDataGPU& metaData, uint32_t& i, thread_block_tile<32>& tile
    , bool(&isGoldForLocQueue)[localWorkQueLength], int(&iterationNumb)[1]) {

    pipeline.producer_acquire();
    //auto inMainLineMeta = mainShmem[startOfLocalWorkQ + i] ;
    //auto inMainFullIndex = inMainLineMeta * metaData.mainArrSectionLength + metaData.mainArrXLength * (1 - isGoldForLocQueue[i]);
    //printf("inMain load full index %d \n ", inMainFullIndex);

    //cuda::memcpy_async(cta, &mainShmem[begSourceShmem], &getSourceReduced(fbArgs, iterationNumb)[inMainFullIndex],
    //    cuda::aligned_size_t<128>(sizeof(uint32_t) * metaData.mainArrXLength), pipeline);
    //pipeline.producer_commit();


    cuda::memcpy_async(cta, &mainShmem[begSourceShmem], &getSourceReduced(fbArgs, iterationNumb)[
        mainShmem[startOfLocalWorkQ + i] * metaData.mainArrSectionLength + metaData.mainArrXLength * (1 - isGoldForLocQueue[i])],
        cuda::aligned_size_t<128>(sizeof(uint32_t) * metaData.mainArrXLength), pipeline);
    pipeline.producer_commit();


}

/*
process data about this block
*/
#pragma once
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


    //if (mainShmem[begSourceShmem + threadIdx.x + threadIdx.y * 32] > 0 && isGoldForLocQueue[i] == 1) {

    //    printf("something in loaded  in main load idX %d idY %d  \n", threadIdx.x, threadIdx.y);
    //}


    //if (getSourceReduced(fbArgs, iterationNumb)[
     //   mainShmem[startOfLocalWorkQ + i] * metaData.mainArrSectionLength + metaData.mainArrXLength * (1 - isGoldForLocQueue[i])+ threadIdx.x + threadIdx.y * 32] > 0 && isGoldForLocQueue[i] == 0) {
    //if (isGoldForLocQueue[i] == 1) {
    //    if ( threadIdx.x + threadIdx.y * 32 ==0 ) {
    //        printf("in lin meta  %d looking for non zero  looking for index starting  %d  mainArrSection %d \n"
    //            , mainShmem[startOfLocalWorkQ + i]
    //            ,4* metaData.mainArrSectionLength + threadIdx.x + threadIdx.y * 32
    //        , metaData.mainArrSectionLength
    //        , );
    //    }
    //    //printf("aaain main load idX %d idY vall  %d \n", threadIdx.x, threadIdx.y, fbArgs.mainArrBPointer[4 * metaData.mainArrSectionLength + threadIdx.x + threadIdx.y * 32]);

    //    for (int ii = 0; ii < 6; ii++) {
    //        //if (fbArgs.mainArrBPointer[ii * metaData.mainArrSectionLength + metaData.mainArrXLength + threadIdx.x + threadIdx.y * 32] > 0) {
    //        if (fbArgs.mainArrBPointer[ii * metaData.mainArrSectionLength + threadIdx.x + threadIdx.y * 32] > 0) {

    //            printf("something in traditionally loaded  in main load idX %d idY %d ii %d \n", threadIdx.x, threadIdx.y, ii);
    //        }
    //    }

    //}


    mainShmem[begResShmem + threadIdx.x + threadIdx.y * 32] = bitDilatate(mainShmem[begSourceShmem + threadIdx.x + threadIdx.y * 32]);
    //marking weather block is already full and no more dilatations are possible 


    pipeline.consumer_release();


}

////////////////TOP
/*
loading data about block above to shmem
*/
#pragma once
template <typename TXPI>
inline __device__  void loadTop(ForBoolKernelArgs<TXPI>& fbArgs, thread_block& cta, uint32_t(&localBlockMetaData)[40]
    , uint32_t(&mainShmem)[lengthOfMainShmem], cuda::pipeline<cuda::thread_scope_block>& pipeline
    , uint32_t*& metaDataArr, MetaDataGPU& metaData, uint32_t& i, thread_block_tile<32>& tile
    , bool(&isGoldForLocQueue)[localWorkQueLength], int(&iterationNumb)[1]) {

    pipeline.producer_acquire();
    if (localBlockMetaData[(i & 1) * 20 + 13] < isGoldOffset) {
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
#pragma once
template <typename TXPI>
inline __device__  void processTop(ForBoolKernelArgs<TXPI>& fbArgs, thread_block& cta, uint32_t(&localBlockMetaData)[40]
    , uint32_t(&mainShmem)[lengthOfMainShmem], cuda::pipeline<cuda::thread_scope_block>& pipeline
    , uint32_t*& metaDataArr, MetaDataGPU& metaData, uint32_t& i, thread_block_tile<32>& tile
    , bool(&isGoldForLocQueue)[localWorkQueLength], int(&iterationNumb)[1], bool(&isAnythingInPadding)[6]) {

    pipeline.consumer_wait();

    dilatateHelperTopDown(0, mainShmem, isAnythingInPadding, localBlockMetaData, 13
        , 31, 0
        , begfirstRegShmem, i);

    pipeline.consumer_release();

}

/////BOTTOM
#pragma once
template <typename TXPI>
inline __device__  void loadBottom(ForBoolKernelArgs<TXPI>& fbArgs, thread_block& cta, uint32_t(&localBlockMetaData)[40]
    , uint32_t(&mainShmem)[lengthOfMainShmem], cuda::pipeline<cuda::thread_scope_block>& pipeline
    , uint32_t*& metaDataArr, MetaDataGPU& metaData, uint32_t& i, thread_block_tile<32>& tile
    , bool(&isGoldForLocQueue)[localWorkQueLength], int(&iterationNumb)[1], bool(&isAnythingInPadding)[6]) {

    pipeline.producer_acquire();
    if (localBlockMetaData[(i & 1) * 20 + 14] < isGoldOffset) {
        cuda::memcpy_async(cta, (&mainShmem[begSecRegShmem]),
            &getSourceReduced(fbArgs, iterationNumb)[localBlockMetaData[(i & 1) * 20 + 14]
            * metaData.mainArrSectionLength + metaData.mainArrXLength * (1 - isGoldForLocQueue[i])],
            cuda::aligned_size_t<128>(sizeof(uint32_t) * metaData.mainArrXLength)
            , pipeline);
    }
    pipeline.producer_commit();

}
#pragma once
template <typename TXPI>
inline __device__  void processBottom(ForBoolKernelArgs<TXPI>& fbArgs, thread_block& cta, uint32_t(&localBlockMetaData)[40]
    , uint32_t(&mainShmem)[lengthOfMainShmem], cuda::pipeline<cuda::thread_scope_block>& pipeline
    , uint32_t*& metaDataArr, MetaDataGPU& metaData, uint32_t& i, thread_block_tile<32>& tile
    , bool(&isGoldForLocQueue)[localWorkQueLength], int(&iterationNumb)[1], bool(&isAnythingInPadding)[6]) {

    pipeline.consumer_wait();

    dilatateHelperTopDown(1, mainShmem, isAnythingInPadding, localBlockMetaData, 14
        , 0, 31
        , begSecRegShmem, i);

    pipeline.consumer_release();

}






///////////// right
#pragma once
template <typename TXPI>
inline __device__  void loadRight(ForBoolKernelArgs<TXPI>& fbArgs, thread_block& cta, uint32_t(&localBlockMetaData)[40]
    , uint32_t(&mainShmem)[lengthOfMainShmem], cuda::pipeline<cuda::thread_scope_block>& pipeline
    , uint32_t*& metaDataArr, MetaDataGPU& metaData, uint32_t& i, thread_block_tile<32>& tile
    , bool(&isGoldForLocQueue)[localWorkQueLength], int(&iterationNumb)[1], bool(&isAnythingInPadding)[6]) {



    pipeline.producer_acquire();
    if (localBlockMetaData[(i & 1) * 20 + 16] < isGoldOffset) {
        cuda::memcpy_async(cta, (&mainShmem[begfirstRegShmem]),
            &getSourceReduced(fbArgs, iterationNumb)[localBlockMetaData[(i & 1) * 20 + 16] * metaData.mainArrSectionLength + metaData.mainArrXLength * (1 - isGoldForLocQueue[i])],
            cuda::aligned_size_t<128>(sizeof(uint32_t) * metaData.mainArrXLength)
            , pipeline);
    }
    pipeline.producer_commit();
}

#pragma once
template <typename TXPI>
inline __device__  void processRight(ForBoolKernelArgs<TXPI>& fbArgs, thread_block& cta, uint32_t(&localBlockMetaData)[40]
    , uint32_t(&mainShmem)[lengthOfMainShmem], cuda::pipeline<cuda::thread_scope_block>& pipeline
    , uint32_t*& metaDataArr, MetaDataGPU& metaData, uint32_t& i, thread_block_tile<32>& tile
    , bool(&isGoldForLocQueue)[localWorkQueLength], int(&iterationNumb)[1], bool(&isAnythingInPadding)[6]) {


    pipeline.consumer_wait();

    dilatateHelperForTransverse(fbArgs, (threadIdx.x == (fbArgs.dbXLength - 1)),
        3, (1), (0), mainShmem, isAnythingInPadding
        , threadIdx.y, 0
        , 16, begfirstRegShmem, localBlockMetaData, i, isGoldForLocQueue);

    pipeline.consumer_release();
}



///////////// left
#pragma once
template <typename TXPI>
inline __device__  void loadLeft(ForBoolKernelArgs<TXPI>& fbArgs, thread_block& cta, uint32_t(&localBlockMetaData)[40]
    , uint32_t(&mainShmem)[lengthOfMainShmem], cuda::pipeline<cuda::thread_scope_block>& pipeline
    , uint32_t*& metaDataArr, MetaDataGPU& metaData, uint32_t& i, thread_block_tile<32>& tile
    , bool(&isGoldForLocQueue)[localWorkQueLength], int(&iterationNumb)[1], bool(&isAnythingInPadding)[6]) {



    pipeline.producer_acquire();
    if (mainShmem[startOfLocalWorkQ + i] > 0) {
        cuda::memcpy_async(cta, (&mainShmem[begSecRegShmem]),
            &getSourceReduced(fbArgs, iterationNumb)[(mainShmem[startOfLocalWorkQ + i] - 1) * metaData.mainArrSectionLength + metaData.mainArrXLength * (1 - isGoldForLocQueue[i])],
            cuda::aligned_size_t<128>(sizeof(uint32_t) * metaData.mainArrXLength)
            , pipeline);
    }
    pipeline.producer_commit();
}

#pragma once
template <typename TXPI>
inline __device__  void processLeft(ForBoolKernelArgs<TXPI>& fbArgs, thread_block& cta, uint32_t(&localBlockMetaData)[40]
    , uint32_t(&mainShmem)[lengthOfMainShmem], cuda::pipeline<cuda::thread_scope_block>& pipeline
    , uint32_t*& metaDataArr, MetaDataGPU& metaData, uint32_t& i, thread_block_tile<32>& tile
    , bool(&isGoldForLocQueue)[localWorkQueLength], int(&iterationNumb)[1], bool(&isAnythingInPadding)[6]) {


    pipeline.consumer_wait();

    dilatateHelperForTransverse(fbArgs, (threadIdx.x == 0),
        2, (-1), (0), mainShmem, isAnythingInPadding
        , threadIdx.y, 31
        , 15, begSecRegShmem, localBlockMetaData, i, isGoldForLocQueue);

    pipeline.consumer_release();
}

///////////// anterior
#pragma once
template <typename TXPI>
inline __device__  void loadAnterior(ForBoolKernelArgs<TXPI>& fbArgs, thread_block& cta, uint32_t(&localBlockMetaData)[40]
    , uint32_t(&mainShmem)[lengthOfMainShmem], cuda::pipeline<cuda::thread_scope_block>& pipeline
    , uint32_t*& metaDataArr, MetaDataGPU& metaData, uint32_t& i, thread_block_tile<32>& tile
    , bool(&isGoldForLocQueue)[localWorkQueLength], int(&iterationNumb)[1], bool(&isAnythingInPadding)[6]) {

    pipeline.producer_acquire();
    if (localBlockMetaData[(i & 1) * 20 + 17] < isGoldOffset) {

        cuda::memcpy_async(cta, (&mainShmem[begfirstRegShmem]),
            &getSourceReduced(fbArgs, iterationNumb)[localBlockMetaData[(i & 1) * 20 + 17] * metaData.mainArrSectionLength + metaData.mainArrXLength * (1 - isGoldForLocQueue[i])],
            cuda::aligned_size_t<128>(sizeof(uint32_t) * metaData.mainArrXLength)
            , pipeline);
    }
    pipeline.producer_commit();
}

#pragma once
template <typename TXPI>
inline __device__  void processAnterior(ForBoolKernelArgs<TXPI>& fbArgs, thread_block& cta, uint32_t(&localBlockMetaData)[40]
    , uint32_t(&mainShmem)[lengthOfMainShmem], cuda::pipeline<cuda::thread_scope_block>& pipeline
    , uint32_t*& metaDataArr, MetaDataGPU& metaData, uint32_t& i, thread_block_tile<32>& tile
    , bool(&isGoldForLocQueue)[localWorkQueLength], int(&iterationNumb)[1], bool(&isAnythingInPadding)[6]) {

    pipeline.consumer_wait();

    dilatateHelperForTransverse(fbArgs, (threadIdx.y == (fbArgs.dbYLength - 1)), 4
        , (0), (1), mainShmem, isAnythingInPadding
        , 0, threadIdx.x
        , 17, begfirstRegShmem, localBlockMetaData, i, isGoldForLocQueue);
    pipeline.consumer_release();
}

///////////// posterior
#pragma once
template <typename TXPI>
inline __device__  void loadPosterior(ForBoolKernelArgs<TXPI>& fbArgs, thread_block& cta, uint32_t(&localBlockMetaData)[40]
    , uint32_t(&mainShmem)[lengthOfMainShmem], cuda::pipeline<cuda::thread_scope_block>& pipeline
    , uint32_t*& metaDataArr, MetaDataGPU& metaData, uint32_t& i, thread_block_tile<32>& tile
    , bool(&isGoldForLocQueue)[localWorkQueLength], int(&iterationNumb)[1], bool(&isAnythingInPadding)[6]) {

    pipeline.producer_acquire();
    if (localBlockMetaData[(i & 1) * 20 + 18] < isGoldOffset) {


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
#pragma once
template <typename TXPI>
inline __device__  void lastLoad(ForBoolKernelArgs<TXPI>& fbArgs, thread_block& cta, uint32_t(&localBlockMetaData)[40]
    , uint32_t(&mainShmem)[lengthOfMainShmem], cuda::pipeline<cuda::thread_scope_block>& pipeline
    , uint32_t*& metaDataArr, MetaDataGPU& metaData, uint32_t& i, thread_block_tile<32>& tile
    , bool(&isGoldForLocQueue)[localWorkQueLength], int(&iterationNumb)[1], bool(&isAnythingInPadding)[6]
    , uint32_t*& origArrs, unsigned int(&worQueueStep)[1]) {

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
            //auto metaDataIndex = mainShmem[startOfLocalWorkQ + 1 + i];
            //printf(" metaDataIndex  to copy metadata for next %d \n", metaDataIndex);
            //cuda::memcpy_async(cta, (&localBlockMetaData[((i + 1) & 1) * 20]),
            //    (&metaDataArr[metaDataIndex * metaData.metaDataSectionLength])
            //    , cuda::aligned_size_t<4>(sizeof(uint32_t) * 20), pipeline);

            cuda::memcpy_async(cta, (&localBlockMetaData[((i + 1) & 1) * 20]),
                (&metaDataArr[(mainShmem[startOfLocalWorkQ + 1 + i])
                    * metaData.metaDataSectionLength])
                , cuda::aligned_size_t<4>(sizeof(uint32_t) * 20), pipeline);


        }
    }


    pipeline.producer_commit();
}
#pragma once
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
        , 18, begSecRegShmem, localBlockMetaData, i, isGoldForLocQueue);
    //now all data should be properly dilatated we save it to global memory
    //try save target reduced via mempcy async ...

    getTargetReduced(fbArgs, iterationNumb)[mainShmem[startOfLocalWorkQ + i] * metaData.mainArrSectionLength + metaData.mainArrXLength * (1 - isGoldForLocQueue[i])
        + threadIdx.x + threadIdx.y * 32]
        = mainShmem[begResShmem + threadIdx.x + threadIdx.y * 32];

    //TODO remove 
    //if (blockIdx.x == 0) {
    //    for (uint8_t bitPos = 0; bitPos < 32; bitPos++) {
    //        if (threadIdx.x == 0 && threadIdx.y == 0) {

    //            //if any bit here is set it means it should be added to result list 
    //            if (isBitAt(mainShmem[begResShmem + threadIdx.x + threadIdx.y * 32], bitPos)) {
    //                if (mainShmem[startOfLocalWorkQ + i] * 32 + bitPos>200) {
    //                printf("bit set loc %d isGold %d \n", mainShmem[startOfLocalWorkQ + i] * 32 + bitPos, isGoldForLocQueue[i]);
    //            }
    //             }

    //        }
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
#pragma once
template <typename TXPI>
inline __device__  void validate(ForBoolKernelArgs<TXPI>& fbArgs, thread_block& cta, uint32_t(&localBlockMetaData)[40]
    , uint32_t(&mainShmem)[lengthOfMainShmem], cuda::pipeline<cuda::thread_scope_block>& pipeline
    , uint32_t*& metaDataArr, MetaDataGPU& metaData, uint32_t& i, thread_block_tile<32>& tile
    , bool(&isGoldForLocQueue)[localWorkQueLength], int(&iterationNumb)[1], bool(&isAnythingInPadding)[6]
    , bool(&isBlockFull)[1]
    , unsigned int(&localFpConter)[1], unsigned int(&localFnConter)[1]
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


                //printf("\n rrrrresult meta %d isGold %d old %d  xLoc %d yLoc %d zLoc %d iterNumbb %d \n"
                //    , mainShmem[startOfLocalWorkQ + i]
                //    , isGoldForLocQueue[i]
                //    , old
                //    , threadIdx.x
                //    , threadIdx.y
                //    , bitPos
                //    , iterationNumb[0]
                //);


            }

        };
        //mainShmem[begResShmem + threadIdx.x + threadIdx.y * 32] = 0;
        //mainShmem[begSourceShmem + threadIdx.x + threadIdx.y * 32] = 0;

    }
}










//template <typename TKKI, typename forPipeline >
#pragma once
template <typename TKKI >
inline __device__ void mainDilatation(const bool isPaddingPass, ForBoolKernelArgs<TKKI>& fbArgs, uint32_t*& mainArrAPointer,
    uint32_t*& mainArrBPointer, MetaDataGPU& metaData
    , unsigned int*& minMaxes, uint32_t*& workQueue
    , uint32_t*& resultListPointerMeta, uint32_t*& resultListPointerLocal, uint32_t*& resultListPointerIterNumb,
    thread_block& cta, thread_block_tile<32>& tile, grid_group& grid, uint32_t(&mainShmem)[lengthOfMainShmem]
    , bool(&isAnythingInPadding)[6], bool(&isBlockFull)[1], int(&iterationNumb)[1], unsigned int(&globalWorkQueueOffset)[1]
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
    dilBlockInitialClean(tile, isPaddingPass, iterationNumb, localWorkQueueCounter, blockFpConter,
        blockFnConter, localFpConter, localFnConter, isBlockFull
        , fpFnLocCounter,
        localTotalLenthOfWorkQueue, globalWorkQueueOffset
        , worQueueStep, minMaxes, localMinMaxes, lastI);
    sync(cta);

    /// load work QueueData into shared memory 
    for (uint32_t bigloop = blockIdx.x * globalWorkQueueOffset[0]; bigloop < ((blockIdx.x + 1) * globalWorkQueueOffset[0]); bigloop += worQueueStep[0]) {
        // grid stride loop - sadly most of threads will be idle 
        /////////// loading to work queue
        if (((bigloop) < localTotalLenthOfWorkQueue[0]) && ((bigloop) < ((blockIdx.x + 1) * globalWorkQueueOffset[0]))) {

            loadWorkQueue(cta, mainShmem, workQueue, isGoldForLocQueue, bigloop, worQueueStep);
        }
        //now all of the threads in the block needs to have the same i value so we will increment by 1 we are preloading to the pipeline block metaData
        ////##### pipeline Step 0

        sync(cta);
        ////TODO(remove) krowa
        //if (((bigloop) < localTotalLenthOfWorkQueue[0]) && ((bigloop) < ((blockIdx.x + 1) * globalWorkQueueOffset[0]))) {
        //    if (cta.thread_rank()< worQueueStep[0]) {
        //        printf("work que just after load %d index %d global index %d zerooLoc %d localTotalLenthOfWorkQueue[0] %d really in wor q 0 %d \n"
        //            , mainShmem[startOfLocalWorkQ+ cta.thread_rank()]
        //            , cta.thread_rank()
        //            , bigloop + cta.thread_rank()
        //            , mainShmem[startOfLocalWorkQ]
        //            , localTotalLenthOfWorkQueue[0]
        //            , workQueue[bigloop] - isGoldOffset * isGoldForLocQueue[0] );
        //    }
        //}
        //sync(cta);




        //loading metadata
        pipeline.producer_acquire();
        if (((bigloop) < localTotalLenthOfWorkQueue[0]) && ((bigloop) < ((blockIdx.x + 1) * globalWorkQueueOffset[0]))) {

            auto FirstindexToLoadFromWQ = mainShmem[startOfLocalWorkQ];

            cuda::memcpy_async(cta, (&localBlockMetaData[0]),
                (&metaDataArr[FirstindexToLoadFromWQ * metaData.metaDataSectionLength])
                , cuda::aligned_size_t<4>(sizeof(uint32_t) * 20), pipeline);


            //cuda::memcpy_async(cta, (&localBlockMetaData[0]),
            //    (&metaDataArr[(mainShmem[startOfLocalWorkQ])
            //        * metaData.metaDataSectionLength])
            //    , cuda::aligned_size_t<4>(sizeof(uint32_t) * 20), pipeline);

            //loadMetaDataToShmem(cta, localBlockMetaData, mainShmem, pipeline, metaDataArr, metaData, 0, 0);

        }
        pipeline.producer_commit();



        for (uint32_t i = 0; i < worQueueStep[0]; i += 1) {
            if (((bigloop + i) < localTotalLenthOfWorkQueue[0]) && ((bigloop + i) < ((blockIdx.x + 1) * globalWorkQueueOffset[0]))) {

                //if (tile.thread_rank() == 0 && tile.meta_group_rank() == 0) {                      
                //    printf("\n linMeta beg %d is gold %d is padding pass %d\n ", mainShmem[startOfLocalWorkQ + i], isGoldForLocQueue[i], isPaddingPass);
                //};

                // if (tile.thread_rank() == 0 && tile.meta_group_rank() == 0 && isGoldForLocQueue[i]==0 ) {
                //    printf("\n linMeta beg %d is gold %d is padding pass %d\n ", mainShmem[startOfLocalWorkQ + i], isGoldForLocQueue[i], isPaddingPass);
                //};

//////////////// step 0  load main data and final processing of previous block
               //loading main data for first dilatation
                //IMPORTANT we need to keep a lot of variables constant here like is Anuthing in padding of fp count .. as the represent processing of previous block  - so do not modify them here ...
                loadMain(fbArgs, cta, localBlockMetaData, mainShmem, pipeline, metaDataArr, metaData, i, tile, isGoldForLocQueue, iterationNumb
                );

                pipeline.consumer_wait();
                afterBlockClean(cta, worQueueStep, localBlockMetaData, mainShmem, i - 1,
                    metaData, tile, localFpConter, localFnConter
                    , blockFpConter, blockFnConter
                    , metaDataArr, isAnythingInPadding, isBlockFull, isPaddingPass, isGoldForLocQueue, lastI);
                //needed for after block metadata update
                if (tile.thread_rank() == 0 && tile.meta_group_rank() == 3) {
                    lastI[0] = i;
                }

                pipeline.consumer_release();

                ///////// step 1 load top and process main data 
                               //load top 
                loadTop(fbArgs, cta, localBlockMetaData, mainShmem, pipeline, metaDataArr, metaData, i, tile, isGoldForLocQueue, iterationNumb);
                //process main
                processMain(fbArgs, cta, localBlockMetaData, mainShmem, pipeline, metaDataArr, metaData, i, tile, isGoldForLocQueue, iterationNumb, isBlockFull);
                ///////// step 2 load bottom and process top 
                loadBottom(fbArgs, cta, localBlockMetaData, mainShmem, pipeline, metaDataArr, metaData, i, tile, isGoldForLocQueue, iterationNumb, isAnythingInPadding);
                //process top
                processTop(fbArgs, cta, localBlockMetaData, mainShmem, pipeline, metaDataArr, metaData, i, tile, isGoldForLocQueue, iterationNumb, isAnythingInPadding);
                /////////// step 3 load right  process bottom  
                loadRight(fbArgs, cta, localBlockMetaData, mainShmem, pipeline, metaDataArr, metaData, i, tile, isGoldForLocQueue, iterationNumb, isAnythingInPadding);
                //process bototm
                processBottom(fbArgs, cta, localBlockMetaData, mainShmem, pipeline, metaDataArr, metaData, i, tile, isGoldForLocQueue, iterationNumb, isAnythingInPadding);
                /////////// step 4 load left process right  

                loadLeft(fbArgs, cta, localBlockMetaData, mainShmem, pipeline, metaDataArr, metaData, i, tile, isGoldForLocQueue, iterationNumb, isAnythingInPadding);
                processRight(fbArgs, cta, localBlockMetaData, mainShmem, pipeline, metaDataArr, metaData, i, tile, isGoldForLocQueue, iterationNumb, isAnythingInPadding);
                /////// step 5 load anterior process left 
                loadAnterior(fbArgs, cta, localBlockMetaData, mainShmem, pipeline, metaDataArr, metaData, i, tile, isGoldForLocQueue, iterationNumb, isAnythingInPadding);
                processLeft(fbArgs, cta, localBlockMetaData, mainShmem, pipeline, metaDataArr, metaData, i, tile, isGoldForLocQueue, iterationNumb, isAnythingInPadding);
                /////// step 6 load posterior process anterior 
                loadPosterior(fbArgs, cta, localBlockMetaData, mainShmem, pipeline, metaDataArr, metaData, i, tile, isGoldForLocQueue, iterationNumb, isAnythingInPadding);
                processAnterior(fbArgs, cta, localBlockMetaData, mainShmem, pipeline, metaDataArr, metaData, i, tile, isGoldForLocQueue, iterationNumb, isAnythingInPadding);
                /////// step 7 
                // 

                            //    sync(cta);

                                //load reference if needed or data for next iteration if there is such 
                                //process posterior, save data from res shmem to global memory also we mark weather block is full
                lastLoad(fbArgs, cta, localBlockMetaData, mainShmem, pipeline, metaDataArr, metaData, i, tile, isGoldForLocQueue, iterationNumb, isAnythingInPadding, origArrs, worQueueStep);
                processPosteriorAndSaveResShmem(fbArgs, cta, localBlockMetaData, mainShmem, pipeline, metaDataArr, metaData, i, tile, isGoldForLocQueue, iterationNumb, isAnythingInPadding, isBlockFull);
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

                validate(fbArgs, cta, localBlockMetaData, mainShmem, pipeline, metaDataArr, metaData, i, tile, isGoldForLocQueue, iterationNumb, isAnythingInPadding, isBlockFull, localFpConter, localFnConter, resultListPointerMeta, resultListPointerLocal, resultListPointerIterNumb);
                /////////
                pipeline.consumer_release();

                //  sync(cta);

                  //pipeline.producer_acquire();

                  //pipeline.producer_commit();

                  //pipeline.consumer_wait();

                  //getTargetReduced(fbArgs, iterationNumb)[mainShmem[startOfLocalWorkQ + i] * metaData.mainArrSectionLength + metaData.mainArrXLength * (1 - isGoldForLocQueue[i])
                  //    + threadIdx.x + threadIdx.y * 32]
                  //    = mainShmem[begResShmem + threadIdx.x + threadIdx.y * 32];

                  //pipeline.consumer_release();

            }
        }

        //here we are after all of the blocks planned to be processed by this block are

//updating local counters of last local block (normally it is done at the bagining of the next block)
//but we need to check weather any block was processed at all
        pipeline.consumer_wait();

        if (lastI[0] < UINT32_MAX) {
            afterBlockClean(cta, worQueueStep, localBlockMetaData, mainShmem, lastI[0],
                metaData, tile, localFpConter, localFnConter
                , blockFpConter, blockFnConter
                , metaDataArr, isAnythingInPadding, isBlockFull, isPaddingPass, isGoldForLocQueue, lastI);

        }
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

        if (blockIdx.x == 0) {
            printf("maxX % d minX % d maxY % d  minY % d maxZ % d minZ % d global FP count % d global FN count % d  total meta len %d \n"
                , fbArgs.minMaxes[1]
                , fbArgs.minMaxes[2]
                , fbArgs.minMaxes[3]
                , fbArgs.minMaxes[4]
                , fbArgs.minMaxes[5]
                , fbArgs.minMaxes[6]
                , fbArgs.minMaxes[7]
                , fbArgs.minMaxes[8]
                , fbArgs.metaData.totalMetaLength

            );
        }

    };


    //here we caclulate the offset for given block depending on length of the workqueue and number of the  available blocks in a grid
    // - this will give us number of work queue items per block - we will calculate offset on the basis of the block number



   // for (int t = 0; t < 3; t++) {
    do {
        //if (threadIdx.x == 2 && threadIdx.y == 0) {
        //    if (blockIdx.x == 0) {
        //        printf("************  iter nuumb %d \n", iterationNumb[0]);
        //        //  fbArgs.metaData.minMaxes[13] = iterationNumb[0];
        //    }
        //};

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
        /*  if (blockIdx.x == 0) {
              if (threadIdx.x == 2 && threadIdx.y == 0) {
                  printf("b iter nuumb %d \n", iterationNumb[0]);
              }
          }*/
          ///////////// loading work queue for padding dilatations
        metadataPass(fbArgs, true, 11, 7, 8,
            12, 9, 10
            , mainShmem, globalWorkQueueOffset, globalWorkQueueCounter
            , localWorkQueueCounter, localTotalLenthOfWorkQueue, localMinMaxes
            , fpFnLocCounter, isGoldPassToContinue, isSegmPassToContinue, cta, tile
            , fbArgs.metaData, fbArgs.minMaxes, fbArgs.workQueuePointer, fbArgs.metaDataArrPointer);




        //////////// padding dilatations
        grid.sync();
        //if (blockIdx.x == 0) {
        //    if (threadIdx.x == 2 && threadIdx.y == 0) {
        //        printf("c iter nuumb %d \n", iterationNumb[0]);
        //    }
        //}
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
        /*  if (blockIdx.x == 0) {
              if (threadIdx.x == 2 && threadIdx.y == 0) {
                  printf("d iter nuumb %d \n", iterationNumb[0]);
              }
          }*/
          ////////////////////////main metadata pass
        metadataPass(fbArgs, false, 7, 8, 8,
            9, 10, 8
            , mainShmem, globalWorkQueueOffset, globalWorkQueueCounter
            , localWorkQueueCounter, localTotalLenthOfWorkQueue, localMinMaxes
            , fpFnLocCounter, isGoldPassToContinue, isSegmPassToContinue, cta, tile
            , fbArgs.metaData, fbArgs.minMaxes, fbArgs.workQueuePointer, fbArgs.metaDataArrPointer);
        grid.sync();

    } while (isGoldPassToContinue[0] || isSegmPassToContinue[0]);
    //}
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
    if (blockIdx.x == 0) {
        if (threadIdx.x == 2 && threadIdx.y == 0) {
            fbArgs.metaData.minMaxes[13] = iterationNumb[0];
        }
    }
}





#pragma once
template <typename T>
ForBoolKernelArgs<T> mainKernelsRun(ForFullBoolPrepArgs<T> fFArgs, uint32_t*& reducedResCPU
    , uint32_t*& resultListPointerMetaCPU
    , uint32_t*& resultListPointerLocalCPU
    , uint32_t*& resultListPointerIterNumbCPU
    , uint32_t*& metaDataArrPointerCPU
    , uint32_t*& workQueuePointerCPU
    , uint32_t*& origArrsCPU
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
    //blockForMainPass = 1;
    blockSizeForMinMax = 1;





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

    cudaMallocAsync(&goldArrPointer, sizeMainArr, 0);
    cudaMallocAsync(&segmArrPointer, sizeMainArr, 0);

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
    cudaMallocAsync(&minMaxes, sizeminMaxes, 0);




    checkCuda(cudaDeviceSynchronize(), "a0b");
    ForBoolKernelArgs<T> fbArgs = getArgsForKernel<T>(fFArgs, goldArrPointer, segmArrPointer, minMaxes, warpsNumbForMainPass, blockForMainPass, WIDTH, HEIGHT, DEPTH);
    fbArgs.metaData.minMaxes = minMaxes;
    fbArgs.minMaxes = minMaxes;


    fbArgs.goldArr = goldArr;
    fbArgs.segmArr = segmArr;


    ////preparation kernel

    // initialize, then launch

    checkCuda(cudaDeviceSynchronize(), "a1");


    //getMinMaxes << <blockSizeForMinMax, dim3(32, warpsNumbForMinMax) >> > ( minMaxes);
    getMinMaxes << <blockSizeForMinMax, dim3(32, warpsNumbForMinMax) >> > (fbArgs, minMaxes, goldArrPointer, segmArrPointer, fbArgs.metaData);

    checkCuda(cudaDeviceSynchronize(), "a1b");


    checkCuda(cudaDeviceSynchronize(), "a2a");

    fbArgs.metaData = allocateMemoryAfterMinMaxesKernel(fbArgs, fFArgs, workQueuePointer, minMaxes, fbArgs.metaData, origArrsPointer, metaDataArrPointer);

    checkCuda(cudaDeviceSynchronize(), "a2b");

    boolPrepareKernel << <blockSizeFoboolPrepareKernel, dim3(32, warpsNumbForboolPrepareKernel) >> > (fbArgs, fbArgs.metaData, origArrsPointer, metaDataArrPointer, goldArrPointer, segmArrPointer, minMaxes);
    //  //uint32_t* origArrs, uint32_t* metaDataArr     metaDataArr[linIdexMeta * metaData.metaDataSectionLength     metaDataOffset

    checkCuda(cudaDeviceSynchronize(), "a3");



    int fpPlusFn = allocateMemoryAfterBoolKernel(fbArgs, fFArgs, resultListPointerMeta, resultListPointerLocal, resultListPointerIterNumb, origArrsPointer, mainArrAPointer, mainArrBPointer, fbArgs.metaData, goldArr, segmArr);




    checkCuda(cudaDeviceSynchronize(), "a4");

    //cudaFreeAsync(goldArrPointer, 0);
    //cudaFreeAsync(segmArrPointer, 0);

    firstMetaPrepareKernel << <blockForFirstMetaPass, theadsForFirstMetaPass >> > (fbArgs, fbArgs.metaData, minMaxes, workQueuePointer, origArrsPointer, metaDataArrPointer);

    checkCuda(cudaDeviceSynchronize(), "a5");
    //void* kernel_args[] = { &fbArgs, mainArrPointer,&metaData,minMaxes, workQueuePointer,resultListPointerMeta,resultListPointerLocal, resultListPointerIterNumb };



    //fbArgs.goldArr = goldArr;
    //fbArgs.segmArr = segmArr;
    //fbArgs.metaData = metaData;

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


    auto metaData = fbArgs.metaData;
    size_t sizeMinnMax = sizeof(unsigned int) * 20;

    cudaMemcpy(fFArgs.metaData.minMaxes, minMaxes, sizeMinnMax, cudaMemcpyDeviceToHost);

    //copy to CPU
    size_t sizeCPU = metaData.totalMetaLength * fbArgs.metaData.mainArrSectionLength * sizeof(uint32_t);
    reducedResCPU = (uint32_t*)calloc(metaData.totalMetaLength * metaData.mainArrSectionLength, sizeof(uint32_t));
    cudaMemcpy(reducedResCPU, mainArrAPointer, sizeCPU, cudaMemcpyDeviceToHost);

    origArrsCPU = (uint32_t*)calloc(metaData.totalMetaLength * metaData.mainArrSectionLength, sizeof(uint32_t));
    cudaMemcpy(origArrsCPU, origArrsPointer, sizeCPU, cudaMemcpyDeviceToHost);


    size_t sizeRes = sizeof(uint32_t) * (fpPlusFn + 50);


    resultListPointerMetaCPU = (uint32_t*)calloc(fpPlusFn + 50, sizeof(uint32_t));
    resultListPointerLocalCPU = (uint32_t*)calloc(fpPlusFn + 50, sizeof(uint32_t));
    resultListPointerIterNumbCPU = (uint32_t*)calloc(fpPlusFn + 50, sizeof(uint32_t));
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








inline void setArrCPUB(bool* arrCPU, int x, int y, int z, int  Nx, int Ny) {

    arrCPU[x + y * Nx + z * Nx * Ny] = true;
};



//testing loopMeta function in order to execute test unhash proper function in loopMeta
#pragma once
extern "C" inline void testMainPasswes() {
    // threads and blocks for bool kernel
    const int blocks = 17;
    const int xThreadDim = 32;
    const int yThreadDim = 12;
    const dim3 threads = dim3(xThreadDim, yThreadDim);
    // threads and blocks for first metadata pass
    int threadsFirstMetaDataPass = 32;
    int blocksFirstMetaDataPass = 10;



    //datablock dimensions
    int dbXLength = xThreadDim;
    int dbYLength = 5;
    int dbZLength = 32;



    //threads and blocks for main pass 
    dim3 threadsMainPass = dim3(dbXLength, dbYLength);
    int blocksMainPass = 7;
    //threads and blocks for padding pass 
    dim3 threadsPaddingPass = dim3(32, 11);
    int blocksPaddingPass = 13;
    //threads and blocks for non first metadata passes 
    int threadsOtherMetaDataPasses = 32;
    int blocksOtherMetaDataPasses = 7;


    int minMaxesLength = 20;



    //metadata
    const int metaXLength = 5;//8
    const int MetaYLength = 10;//30
    const int MetaZLength = 30;//8


    const int totalLength = metaXLength * MetaYLength * MetaZLength;

    /*   int*** h_tensor;
       h_tensor = alloc_tensorToZeros<int>(metaXLength, MetaYLength, MetaZLength);*/

    int i, j, k, value = 0;

    const int mainXLength = dbXLength * metaXLength;
    const int mainYLength = 1200;//dbYLength * MetaYLength;
    const int mainZLength = dbZLength * MetaZLength;


    //main data arrays
    bool* goldArr = alloc_tensorToZeros<bool>(mainXLength, mainYLength, mainZLength);

    bool* segmArr = alloc_tensorToZeros<bool>(mainXLength, mainYLength, mainZLength);
    MetaDataCPU metaData;
    metaData.metaXLength = metaXLength;
    metaData.MetaYLength = MetaYLength;
    metaData.MetaZLength = MetaZLength;
    metaData.totalMetaLength = totalLength;


    size_t size = sizeof(unsigned int) * 20;
    unsigned int* minMaxesCPU = (unsigned int*)malloc(size);
    metaData.minMaxes = minMaxesCPU;

    int workQueueAndRLLength = 200;
    int workQueueWidth = 4;
    int resultListWidth = 5;
    //allocating to semiarbitrrary size 
    auto workQueuePointer = alloc_tensorToZeros<uint32_t>(workQueueAndRLLength, workQueueWidth, 1);




    // arguments to pass
    ForFullBoolPrepArgs<bool> forFullBoolPrepArgs;
    forFullBoolPrepArgs.metaData = metaData;
    forFullBoolPrepArgs.numberToLookFor = 2;
    forFullBoolPrepArgs.dbXLength = dbXLength;
    forFullBoolPrepArgs.dbYLength = dbYLength;
    forFullBoolPrepArgs.dbZLength = dbZLength;
    forFullBoolPrepArgs.goldArr = get3dArrCPU(goldArr, mainXLength, mainYLength, mainZLength);
    forFullBoolPrepArgs.segmArr = get3dArrCPU(segmArr, mainXLength, mainYLength, mainZLength);
    forFullBoolPrepArgs.threads = threads;
    forFullBoolPrepArgs.blocks = blocks;

    forFullBoolPrepArgs.threadsFirstMetaDataPass = threadsFirstMetaDataPass;
    forFullBoolPrepArgs.blocksFirstMetaDataPass = blocksFirstMetaDataPass;

    forFullBoolPrepArgs.threadsMainPass = threadsMainPass;
    forFullBoolPrepArgs.blocksMainPass = blocksMainPass;

    forFullBoolPrepArgs.threadsPaddingPass = threadsPaddingPass;
    forFullBoolPrepArgs.blocksPaddingPass = blocksPaddingPass;

    forFullBoolPrepArgs.threadsOtherMetaDataPasses = threadsOtherMetaDataPasses;
    forFullBoolPrepArgs.blocksOtherMetaDataPasses = blocksOtherMetaDataPasses;

    //populate segm  and gold Arr


    auto arrGoldObj = forFullBoolPrepArgs.goldArr;
    auto arrSegmObj = forFullBoolPrepArgs.segmArr;




    for (int i = 0; i < 1; i++) {
        setArrCPUB(segmArr, i, i, 0, mainXLength, mainYLength);//
    }

    for (int i = 0; i < 1; i++) {
        setArrCPUB(goldArr, i, i, 900, mainXLength, mainYLength);//
    }
    /* int x = 0;
     int y = 900;
     int z = 0;
     int Nx = 5 * 32;
     goldArr[x + y * Nx] = true;*/


     //900 - 720
     //800 - 640

    // goldArr[ 300 * 32 ] = true;


    // goldArr[300 * 32] = true;

     //int lenn = 900;
     //goldArr[0] = true;
     //segmArr[lenn] = true;
     //goldArr[lenn] = true;
     //segmArr[lenn] = true;
     //segmArr[49*32] = true;



     //int plane = mainXLength * mainYLength;

     //for (int y = 0; y < mainXLength * (mainYLength / 2); y++) {
     //    goldArr[y] = true;

     //}

     ////segmArr[plane+1] = true;

     //int offset = plane * 3 * dbZLength;
     ////for (int y = offset; y < offset + mainXLength * (mainYLength / 2); y++) {
     ////	segmArr[y] = true;

     ////}



     ////
     //offset = mainXLength * mainYLength * mainZLength - (plane * 4);
     //for (int y = offset; y < offset + mainXLength * (mainYLength / 2); y++) {
     //    segmArr[y] = true;

     //}




     //int pointsNumber = 0;
     //int& pointsNumberRef = pointsNumber;
     //forTestPointStruct allPointsA[] = {
     //	// meta 2,2,2 only gold points not in result after 2 dilataions
     //getTestPoint(
     //2,2,2//x,y,z
     //,true//isGold
     //,0,0,0//xMeta,yMeta,Zmeta
     //,dbXLength,dbYLength,dbZLength,pointsNumberRef)
     //};

     /*
     maxX 2  [1]
 minX 1  [2]
 maxY 1  [3]
 minY 0  [4]
 maxZ 5  [5]
 minZ 2  [6]
     */


    printf("\n aaa \n");

    uint32_t* resultListPointerMetaCPU;
    uint32_t* resultListPointerLocalCPU;
    uint32_t* resultListPointerIterNumbCPU;
    uint32_t* metaDataArrPointerCPU;
    uint32_t* workQueuePointerCPU;

    uint32_t* reducedResCPU;
    uint32_t* origArrsCPU;



    ForBoolKernelArgs<bool> fbArgs = mainKernelsRun(forFullBoolPrepArgs, reducedResCPU, resultListPointerMetaCPU
        , resultListPointerLocalCPU, resultListPointerIterNumbCPU
        , metaDataArrPointerCPU, workQueuePointerCPU, origArrsCPU, mainXLength, mainYLength, mainZLength
    );

    //for (int outer = 0; outer< ceil(lenn/ int(32)); outer++ ) {
    //	for (int u = 0; u < 32; u++) {
    //		int coord = outer * 32 + u;


    //		//3printf("set %d in %d \n  ", (reducedResCPU[u] >0), u);
    //	}
    //}







    //for (uint32_t linIdexMeta = 0; linIdexMeta < fbArgs.metaData.totalMetaLength; linIdexMeta += 1) {
    //	//we get from linear index  the coordinates of the metadata block of intrest
    //	uint8_t xMeta = linIdexMeta % fbArgs.metaData.metaXLength;
    //	uint8_t zMeta = floor((float)(linIdexMeta / (fbArgs.metaData.metaXLength * fbArgs.metaData.MetaYLength)));
    //	uint8_t yMeta = floor((float)((linIdexMeta - ((zMeta * fbArgs.metaData.metaXLength * fbArgs.metaData.MetaYLength) + xMeta)) / fbArgs.metaData.metaXLength));

    //	for (int locPos = 0; locPos < 32 * fbArgs.dbYLength; locPos++) {
    //		auto col = reducedResCPU[linIdexMeta * fbArgs.metaData.mainArrSectionLength + locPos];
    //		if (col > 0) {
    //			for (uint8_t bitPos = 0; bitPos < 32; bitPos++) {
    //				int x = locPos % 32 + xMeta * fbArgs.dbXLength;
    //				int y = int(floor((float)(locPos / 32)) + yMeta * fbArgs.dbYLength);
    //				int z = bitPos + zMeta * fbArgs.dbZLength;

    //				if (y==0 && z==0) {
    //					if (isBitAtCPU(col, bitPos)) {
    //						printf("point gold set at x %d y %d z %d  \n"
    //							, locPos % 32 + xMeta * fbArgs.dbXLength
    //							, int(floor((float)(locPos / 32)) + yMeta * fbArgs.dbYLength)
    //							, bitPos + zMeta * fbArgs.dbZLength
    //						);
    //					}
    //				}
    //			}
    //		}
    //	}


    //	//for (int locPos = 32 * fbArgs.dbYLength; locPos < 32 * 2 * fbArgs.dbYLength; locPos++) {
    //	//	auto col = reducedResCPU[linIdexMeta * fbArgs.metaData.mainArrSectionLength + locPos];
    //	//	if (col > 0) {
    //	//		for (uint8_t bitPos = 0; bitPos < 32; bitPos++) {
    //	//			if (isBitAtCPU(col, bitPos)) {
    //	//				int locPosB = locPos - 32 * fbArgs.dbYLength;
    //	//				int x = locPosB % 32 + xMeta * fbArgs.dbXLength;
    //	//				int y = int(floor((float)(locPosB / 32)) + yMeta * fbArgs.dbYLength);
    //	//				int z = bitPos + zMeta * fbArgs.dbZLength;
    //	//				if (y == 0 && z == 0) {

    //	//					printf("point segm  set at x %d y %d z %d  \n"
    //	//						, locPosB % 32 + xMeta * fbArgs.dbXLength
    //	//						, int(floor((float)(locPosB / 32)) + yMeta * fbArgs.dbYLength)
    //	//						, bitPos + zMeta * fbArgs.dbZLength
    //	//					);
    //	//				}
    //	//			}
    //	//		}
    //	//	}
    //	//}


    //}















    //testDilatations(fbArgs, allPointsA, );






    //printFromReduced(fbArgs, reducedResCPU);
    //printIsBlockActiveEtc(fbArgs, metaDataArrPointerCPU, fbArgs.metaData);


    //for (int wQi = 0; wQi < minMaxesCPU[9]; wQi ++ ) {
    //	printf("in work q %d  \n ", workQueuePointerCPU[wQi] - isGoldOffset * (workQueuePointerCPU[wQi] >= isGoldOffset) );
    //}

    //for (int wQi = 0; wQi < 700; wQi++) {
    //	if (metaDataArrPointerCPU[wQi]==1) {
    //		printf("\n in metadaArr i %d  \n ", wQi);
    //	}
    //}

    //info in padding AND range 14 linMeta 2 new block adress 30   inMetadataArrIndex 612
    //	info in padding AND range 15 linMeta 2 new block adress 1   inMetadataArrIndex 32
    //	info in padding AND range 14 linMeta 0 new block adress 28   inMetadataArrIndex 571

//printf(" for cpu results ranges xMeta %d yMeta %d zMeta %d ", fbArgs.metaData.metaXLength, fbArgs.metaData.MetaYLength, fbArgs.metaData.MetaZLength);


    unsigned int xRange = minMaxesCPU[1] - minMaxesCPU[2] + 1;
    unsigned int yRange = minMaxesCPU[3] - minMaxesCPU[4] + 1;
    unsigned int zRange = minMaxesCPU[5] - minMaxesCPU[6] + 1;

    printf("before results xRange %d yRange %d zRange %d \n", xRange, yRange, zRange);
    dbYLength = 12;
    for (int i = 0; i < 5; i++) {
        if (resultListPointerLocalCPU[i] > 0 || resultListPointerMetaCPU[i] > 0) {
            uint32_t linIdexMeta = resultListPointerMetaCPU[i] - (isGoldOffset * (resultListPointerMetaCPU[i] >= isGoldOffset)) - 1;
            uint32_t xMeta = linIdexMeta % xRange;
            uint32_t zMeta = uint32_t(floor((float)(linIdexMeta / (xRange * yRange))));
            uint32_t yMeta = uint32_t(floor((float)((linIdexMeta - ((zMeta * xRange * yRange) + xMeta)) / xRange)));

            uint32_t linLocal = resultListPointerLocalCPU[i];
            uint32_t xLoc = linLocal % 32;
            uint32_t zLoc = uint32_t(floor((float)(linLocal / (32 * dbYLength))));
            uint32_t yLoc = uint32_t(floor((float)((linLocal - ((zLoc * 32 * dbYLength) + xLoc)) / 32)));


            uint32_t x = xMeta * 32 + xLoc;
            uint32_t y = yMeta * dbYLength + yLoc;
            uint32_t z = zMeta * 32 + zLoc;
            uint32_t iterNumb = resultListPointerIterNumbCPU[i];

            printf("resullt linIdexMeta %d x %d y %d z %d  xMeta %d yMeta %d zMeta %d xLoc %d yLoc %d zLoc %d linLocal %d  iterNumb %d \n"
                , linIdexMeta
                , x, y, z
                , xMeta, yMeta, zMeta
                , xLoc, yLoc, zLoc
                , linLocal
                , iterNumb


            );



        }
    }





    printf("\n **************************************** \n");

    i = 1;
    printf("maxX %d  [%d]\n", minMaxesCPU[i], i);
    i = 2;
    printf("minX %d  [%d]\n", minMaxesCPU[i], i);
    i = 3;
    printf("maxY %d  [%d]\n", minMaxesCPU[i], i);
    i = 4;
    printf("minY %d  [%d]\n", minMaxesCPU[i], i);
    i = 5;
    printf("maxZ %d  [%d]\n", minMaxesCPU[i], i);
    i = 6;
    printf("minZ %d  [%d]\n", minMaxesCPU[i], i);

    int ii = 7;
    printf("global FP count %d  [%d]\n", minMaxesCPU[ii], ii);
    ii = 8;
    printf("global FN count %d  [%d]\n", minMaxesCPU[ii], ii);
    ii = 9;
    printf("workQueueCounter %d  [%d]\n", minMaxesCPU[ii], ii);
    ii = 10;
    printf("resultFP globalCounter %d  [%d]\n", minMaxesCPU[ii], ii);
    ii = 11;
    printf("resultFn globalCounter %d  [%d]\n", minMaxesCPU[ii], ii);
    ii = 12;
    printf("global offset counter %d  [%d]\n", minMaxesCPU[ii], ii);

    ii = 13;
    printf("globalIterationNumb %d  [%d]\n", minMaxesCPU[ii], ii);
    ii = 17;
    printf("suum debug %d  [%d]\n", minMaxesCPU[ii], ii);





    //i, j, k, value = 0;
    //i = 31;
    //j = 12;
    //for (k = 0; k < MetaZLength; k++) {
    //	goldArr[k][j][i] = 1;
    //	if (reducedGold[k][j][i] > 0) {
    //		for (int tt = 0; tt < 32; tt++) {
    //			if ((reducedGold[k][j][i] & (1 << (tt)))) {
    //				printf("found in reduced fp  [%d]\n", k * 32 + tt);

    //			}
    //		}

    //	}
    //}


    //		i, j, k, value = 0;
    //for (i = 0; i < mainXLength; i++) {
    //	for (j = 0; j < mainYLength; j++) {
    //		for (k = 0; k < MetaZLength; k++) {
    //			//goldArr[k][j][i] = 1;
    //			if (reducedGold[k][j][i] > 0) {
    //				for (int tt = 0; tt < 32; tt++) {
    //					if ((reducedGold[k][j][i] & (1 << (tt)))) {
    //						printf("found in reduced fp  [%d][%d][%d]\n", i, j, k * 32 + tt);

    //					}
    //				}

    //			}
    //		}
    //	}
    //}






    //minMaxes.arrP[0][0][10] + minMaxes.arrP[0][0][11]

    //int sumDebug = 0;
    //for (int ji = 0; ji < 8000; ji++) {
    //	if (forDebugArr[0][0][ji]==1) {
    //		sumDebug += forDebugArr[0][0][ji];
    //		//printf("for debug %d i %d \n", forDebugArr[0][0][ji],ji);
    //	}
    //}
    //printf("\n sumDebug %d \n", sumDebug);


//
//
//	//	for (int ji = 0; ji < minMaxes.arrP[0][0][10] + minMaxes.arrP[0][0][11]; ji++) {
//		for (int ji = 0; ji < 10; ji++) {
//    if (forFullBoolPrepArgs.metaData.resultList.arrP[0][2][ji] + forFullBoolPrepArgs.metaData.resultList.arrP[0][1][ji]  > 0) {
//   	 int x = forFullBoolPrepArgs.metaData.resultList.arrP[0][0][ji];
//	 int y = forFullBoolPrepArgs.metaData.resultList.arrP[0][1][ji];
//	 int z = forFullBoolPrepArgs.metaData.resultList.arrP[0][2][ji];
//	 int isGold = forFullBoolPrepArgs.metaData.resultList.arrP[0][3][ji];
//	 int iternumb = forFullBoolPrepArgs.metaData.resultList.arrP[0][4][ji];
//
//	 //uint32_t x = forFullBoolPrepArgs.metaData.resultList.arrP[ji][0][0];
//	 //uint32_t y = forFullBoolPrepArgs.metaData.resultList.arrP[ji][1][0];
//	 //uint32_t z = forFullBoolPrepArgs.metaData.resultList.arrP[ji][2][0];
//	 //uint32_t isGold = forFullBoolPrepArgs.metaData.resultList.arrP[ji][3][0];
//	 //uint32_t iternumb = forFullBoolPrepArgs.metaData.resultList.arrP[ji][4][0];
//
//
//   	 if (iternumb!=9) {
//   		 printf("result  in point  %d %d %d isGold %d iteration %d \n "
//   			 , x
//   			 , y
//   			 , z
//   			 , isGold
//   			 , iternumb);
//   	 }
//   	 else {
//   		 printf("**");
//   	 }
//
//    }
//}






     //for (int i = 0; i < workQueueAndRLLength; i++) {

        // if (workQueuePointer[0][2][i] > 0) {
        //	 printf("work queue [%d][%d][%d] = [%d][%d][%d][%d]\n"
        //		 , 0, 0, i
        //		 , workQueuePointer[0][0][i]
        //		 , workQueuePointer[0][1][i]
        //		 , workQueuePointer[0][2][i]
        //		 , workQueuePointer[0][3][i]
        //	 );
        // }

     //}






    printf("cleaaning");



    free(goldArr);
    free(segmArr);


    free(resultListPointerMetaCPU);
    free(resultListPointerLocalCPU);
    free(resultListPointerIterNumbCPU);
    free(metaDataArrPointerCPU);
    free(workQueuePointerCPU);

    free(reducedResCPU);
    free(origArrsCPU);



}










void loadHDFIntoBoolArr(H5std_string FILE_NAME, H5std_string DATASET_NAME, bool*& data) {
    /*
     * Open the specified file and the specified dataset in the file.
     */
    H5File file(FILE_NAME, H5F_ACC_RDONLY);
    DataSet dset = file.openDataSet(DATASET_NAME);
    /*
     * Get the class of the datatype that is used by the dataset.
     */
    H5T_class_t type_class = dset.getTypeClass();
    DataSpace dspace = dset.getSpace();
    int rank = dspace.getSimpleExtentNdims();


    hsize_t dims[2];
    rank = dspace.getSimpleExtentDims(dims, NULL); // rank = 1
    cout << "Datasize: " << dims[0] << endl; // this is the correct number of values

    // Define the memory dataspace
    hsize_t dimsm[1];
    dimsm[0] = dims[0];
    DataSpace memspace(1, dimsm);



    data = (bool*)calloc(dims[0], sizeof(bool));




    dset.read(data, PredType::NATIVE_HBOOL, memspace, dspace);


    //int sum = 0;
    //for (int i = 0; i < dims[0]; i++) {
    //    sum += data[i];
    //}
    //printf("suuum %d \n  ", sum);


    file.close();

}



/*
benchmark for original code from  https://github.com/Oyatsumi/HausdorffDistanceComparison
*/
void benchmarkOliviera(bool* onlyBladderBoolFlat, bool* onlyLungsBoolFlat, const int WIDTH, const int HEIGHT, const int DEPTH) {
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
    //HD : 234 or 274 


    //reversed args
         //Total elapsed time : 1.44947s
     //    HD : 146

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
   // forFullBoolPrepArgs.goldArr = get3dArrCPU(onlyBladderBoolFlat, WIDTH, DEPTH, HEIGHT);
    forFullBoolPrepArgs.segmArr = get3dArrCPU(onlyLungsBoolFlat, WIDTH, HEIGHT, DEPTH);
   // forFullBoolPrepArgs.segmArr = get3dArrCPU(onlyLungsBoolFlat, WIDTH, DEPTH, HEIGHT);
    /// for debugging
    uint32_t* resultListPointerMetaCPU;
    uint32_t* resultListPointerLocalCPU;
    uint32_t* resultListPointerIterNumbCPU;
    uint32_t* metaDataArrPointerCPU;
    uint32_t* workQueuePointerCPU;
    uint32_t* reducedResCPU;
    uint32_t* origArrsCPU;


    //function invocation
    auto begin = std::chrono::high_resolution_clock::now();

    ForBoolKernelArgs<bool> fbArgs = mainKernelsRun(forFullBoolPrepArgs, reducedResCPU, resultListPointerMetaCPU
        , resultListPointerLocalCPU, resultListPointerIterNumbCPU
        , metaDataArrPointerCPU, workQueuePointerCPU, origArrsCPU, WIDTH, HEIGHT, DEPTH
    );


    //ForBoolKernelArgs<bool> fbArgs = mainKernelsRun(forFullBoolPrepArgs, reducedResCPU, resultListPointerMetaCPU
    //    , resultListPointerLocalCPU, resultListPointerIterNumbCPU
    //    , metaDataArrPointerCPU, workQueuePointerCPU, origArrsCPU, WIDTH,  DEPTH, HEIGHT
    //);


    auto end = std::chrono::high_resolution_clock::now();

    std::cout << "Total elapsed time: ";
    std::cout << (double)(std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count() / (double)1000000000) << "s" << std::endl;


    size_t sizeMinMax = sizeof(unsigned int) * 20;
    cudaMemcpy(minMaxesCPU, fbArgs.metaData.minMaxes, sizeMinMax, cudaMemcpyDeviceToHost);

    printf("HD: %d \n", minMaxesCPU[13]);


    // freeee
    free(onlyBladderBoolFlat);
    free(onlyLungsBoolFlat);


    free(resultListPointerMetaCPU);
    free(resultListPointerLocalCPU);
    free(resultListPointerIterNumbCPU);
    free(metaDataArrPointerCPU);
    free(workQueuePointerCPU);

    free(reducedResCPU);
    free(origArrsCPU);

}



void loadHDF() {
    const int WIDTH = 512;
    const int HEIGHT = 512;
    const int DEPTH = 826;





	//main data arrays
	//bool* onlyBladderBoolFlat = alloc_tensorToZeros<bool>(WIDTH, HEIGHT, DEPTH);

	//bool* onlyLungsBoolFlat = alloc_tensorToZeros<bool>(WIDTH, HEIGHT, DEPTH);

 //   onlyBladderBoolFlat[0] = true;
 //   onlyLungsBoolFlat[500] = true;
    const H5std_string FILE_NAMEonlyLungsBoolFlat("C:\\Users\\1\\PycharmProjects\\pythonProject3\\mytestfile.hdf5");
    const H5std_string DATASET_NAMEonlyLungsBoolFlat("onlyLungsBoolFlat");
    // create a vector the same size as the dataset
    bool* onlyLungsBoolFlat;
    loadHDFIntoBoolArr(FILE_NAMEonlyLungsBoolFlat, DATASET_NAMEonlyLungsBoolFlat, onlyLungsBoolFlat);

    const H5std_string FILE_NAMEonlyBladderBoolFlat("C:\\Users\\1\\PycharmProjects\\pythonProject3\\mytestfile.hdf5");
    const H5std_string DATASET_NAMEonlyBladderBoolFlat("onlyBladderBoolFlat");
    // create a vector the same size as the dataset
    bool* onlyBladderBoolFlat;
    loadHDFIntoBoolArr(FILE_NAMEonlyBladderBoolFlat, DATASET_NAMEonlyBladderBoolFlat, onlyBladderBoolFlat);

  //   benchmarkOliviera(onlyBladderBoolFlat, onlyLungsBoolFlat, WIDTH, HEIGHT, DEPTH);
   //  benchmarkOliviera(onlyBladderBoolFlat, onlyLungsBoolFlat, WIDTH, DEPTH, HEIGHT);
    benchmarkMitura(onlyBladderBoolFlat, onlyLungsBoolFlat, WIDTH,  DEPTH, HEIGHT);




}



int main(void) {

    //  const int WIDTH = atoi(argv[1]), HEIGHT = WIDTH, DEPTH = 1;
   //   Volume img1 = Volume(WIDTH, HEIGHT, DEPTH), img2 = Volume(WIDTH, HEIGHT, DEPTH);
   // testMainPasswes();
    loadHDF();



    return 0;  // successfully terminated
}