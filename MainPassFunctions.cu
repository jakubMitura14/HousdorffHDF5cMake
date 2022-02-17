#pragma once


#include "cuda_runtime.h"
#include "CPUAllocations.cu"
#include "MetaData.cu"
#include "IterationUtils.cu"
#include "ExceptionManagUtils.cu"
#include "CooperativeGroupsUtils.cu"
#include <cstdint>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
using namespace cooperative_groups;



/*
gettinng  array for dilatations
basically arrays will alternate between iterations once one will be source other target then they will switch - we will decide upon knowing 
wheather the iteration number is odd or even
*/
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
template <typename TXPPI>
inline __device__ uint32_t* getTargetReduced(ForBoolKernelArgs<TXPPI>& fbArgs, int(&iterationNumb)[1]) {

    if ((iterationNumb[0] & 1) == 0) {
        //printf(" BB ");

      return fbArgs.mainArrBPointer;

    }
    else {       
       // printf(" AA ");

       return fbArgs.mainArrAPointer  ;

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
inline __device__ void dilatateHelperForTransverse(const bool predicate,
    const uint8_t  paddingPos, const   int8_t  normalXChange, const  int8_t normalYchange
, uint32_t (&mainShmem)[lengthOfMainShmem], bool(&isAnythingInPadding)[6]
,const uint8_t forBorderYcoord,const  uint8_t forBorderXcoord
,const uint8_t metaDataCoordIndex,const uint32_t targetShmemOffset , uint32_t (&localBlockMetaData)[40], uint32_t& i ) {
    // so we first check for corner cases 
    if (predicate) {
        // now we need to load the data from the neigbouring blocks
        //first checking is there anything to look to 
        if (localBlockMetaData[(i & 1) * 20+metaDataCoordIndex] < isGoldOffset) {
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


         mainShmem[begResShmem+threadIdx.x+threadIdx.y*32] 
        = mainShmem[begSourceShmem+(threadIdx.x+ normalXChange)+(threadIdx.y+ normalYchange)*32]
             | mainShmem[begResShmem+threadIdx.x+threadIdx.y*32];
    
    }
   

}


#pragma once
inline __device__ void dilatateHelperTopDown( const uint8_t paddingPos, 
uint32_t(&mainShmem)[lengthOfMainShmem], bool(&isAnythingInPadding)[6], uint32_t(&localBlockMetaData)[40]
,const uint8_t metaDataCoordIndex
,const  uint8_t sourceBit 
, const uint8_t targetBit
, const uint32_t targetShmemOffset, uint32_t& i
) {
       // now we need to load the data from the neigbouring blocks
       //first checking is there anything to look to 
       if (localBlockMetaData[(i & 1) * 20+metaDataCoordIndex]< isGoldOffset) {
           if (isBitAt(mainShmem[begSourceShmem+ threadIdx.x + threadIdx.y * 32], targetBit)) {
                              // printf("setting padding top val %d \n ", isAnythingInPadding[0]);
                              isAnythingInPadding[paddingPos] = true;
           };
           // if in bit of intrest of neighbour block is set

     mainShmem[begResShmem + threadIdx.x + threadIdx.y * 32] |= ((mainShmem[targetShmemOffset + threadIdx.x + threadIdx.y * 32] >> sourceBit) & 1) << targetBit;



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





inline __device__  void afterBlockClean(thread_block& cta
    , unsigned int(&worQueueStep)[1], uint32_t (&localBlockMetaData)[40]
    , uint32_t(&mainShmem)[lengthOfMainShmem],const uint32_t i, MetaDataGPU& metaData
    , thread_block_tile<32>& tile
    , unsigned int(&localFpConter)[1], unsigned int(&localFnConter)[1]
    , unsigned int(&blockFpConter)[1], unsigned int (&blockFnConter)[1]
    , uint32_t*& metaDataArr
    , bool (&isAnythingInPadding)[6],bool (&isBlockFull)[1],const bool isPaddingPass, bool (&isGoldForLocQueue)[localWorkQueLength], uint32_t(&lastI)[1]
   ) {



    if (tile.thread_rank() == 7 && tile.meta_group_rank() == 0) {// this is how it is encoded wheather it is gold or segm block
                    //this will be executed only if fp or fn counters are bigger than 0 so not during first pass
        if (localFpConter[0] >= 0) {
            metaDataArr[mainShmem[startOfLocalWorkQ + i] * metaData.metaDataSectionLength + 3] += localFpConter[0];
            blockFpConter[0] += localFpConter[0];
            localFpConter[0] = 0;
        }
    };
    if (tile.thread_rank() == 8 && tile.meta_group_rank() == 0) {// this is how it is encoded wheather it is gold or segm block

        if (localFnConter[0] >= 0) {
            metaDataArr[mainShmem[startOfLocalWorkQ + i] * metaData.metaDataSectionLength + 4] += localFnConter[0];

            blockFnConter[0] += localFnConter[0];
            localFnConter[0] = 0;
        }
    };
    if (tile.thread_rank() == 9 && tile.meta_group_rank() == 2) {// this is how it is encoded wheather it is gold or segm block

        //executed in case of previous block
        if (isBlockFull[0] && i >= 0) {
            //setting data in metadata that block is full
            metaDataArr[mainShmem[startOfLocalWorkQ + i] * metaData.metaDataSectionLength + 10 - (isGoldForLocQueue[i] * 2)] = true;
        }
        //resetting
        isBlockFull[0] = true;
    };



    
    //we do it only for non padding pass
    if (tile.thread_rank() < 6 && tile.meta_group_rank() == 1 && !isPaddingPass) {   
        //executed in case of previous block
        if (i>=0) {

          /*  if (isAnythingInPadding[tile.thread_rank()]) {
                printf("info in padding %d linMeta %d \n ", 13 + tile.thread_rank(), mainShmem[startOfLocalWorkQ + i]);

            }*/

            if (localBlockMetaData[(i & 1) * 20+   13+tile.thread_rank()] < isGoldOffset) {
                //printf("info in range %d linMeta %d \n ", 13 + tile.thread_rank(), mainShmem[startOfLocalWorkQ + i]);

                if (isAnythingInPadding[tile.thread_rank()]) {
                    metaDataArr[localBlockMetaData[(i & 1) * 20 + 13 + tile.thread_rank()] * metaData.metaDataSectionLength + 12 - isGoldForLocQueue[i]] = 1;
                    //printf("info in padding AND range %d linMeta %d \n ", 13 + tile.thread_rank(), mainShmem[startOfLocalWorkQ + i]);

                }
                
            }
        }
        isAnythingInPadding[0] = false;
    };



}





////////////////// with pipeline ofr barrier

/*
initial cleaning  and initializations of dilatation kernel

*/
inline __device__  void dilBlockInitialClean(thread_block_tile<32>& tile,
    const  bool isPaddingPass, int(&iterationNumb)[1],
    unsigned int(&localWorkQueueCounter)[1], unsigned int(&blockFpConter)[1],
    unsigned int(&blockFnConter)[1], unsigned int(&localFpConter)[1],
    unsigned int(&localFnConter)[1],bool (&isBlockFull)[1], 
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
inline __device__  void loadWorkQueue(uint32_t(&mainShmem)[lengthOfMainShmem], uint32_t*& workQueue
, bool(&isGoldForLocQueue)[localWorkQueLength], uint32_t& bigloop, unsigned int(&worQueueStep)[1]) {

    //to do change into barrier

    //cuda::memcpy_async(cta, (&mainShmem[startOfLocalWorkQ]), (&workQueue[bigloop])
    //    , cuda::aligned_size_t<4>(sizeof(uint32_t) * worQueueStep[0]), pipeline);


    for (uint16_t ii = 0; ii < worQueueStep[0]; ii++) {
        mainShmem[startOfLocalWorkQ + ii] = workQueue[bigloop + ii];
        isGoldForLocQueue[ii] = (mainShmem[startOfLocalWorkQ + ii] >= isGoldOffset);
        mainShmem[startOfLocalWorkQ + ii] = mainShmem[startOfLocalWorkQ + ii] - isGoldOffset * isGoldForLocQueue[ii];

    }
}


/*
loads metadata of given block to meta data 
*/
inline __device__  void loadMetaDataToShmem(thread_block& cta, uint32_t(&localBlockMetaData)[40]
    , uint32_t(&mainShmem)[lengthOfMainShmem], cuda::pipeline<cuda::thread_scope_block>& pipeline
, uint32_t*& metaDataArr, MetaDataGPU& metaData, const uint8_t toAdd, uint32_t& ii) {
   
    //cuda::memcpy_async(cta, (&localBlockMetaData[(ii&1)*20]),
    //    (&metaDataArr[(mainShmem[startOfLocalWorkQ + toAdd+ii])
    //        * metaData.metaDataSectionLength])
    //    , cuda::aligned_size_t<4>(sizeof(uint32_t) * 20), pipeline);

    cuda::memcpy_async(cta, (&localBlockMetaData[((ii+1) & 1) * 20]),
        (&metaDataArr[(mainShmem[startOfLocalWorkQ + toAdd + ii])
            * metaData.metaDataSectionLength])
        , cuda::aligned_size_t<4>(sizeof(uint32_t) * 20), pipeline);


}
