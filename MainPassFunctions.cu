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
inline __device__ uint32_t* getSourceReduced(ForBoolKernelArgs<TXPI> fbArgs, uint32_t iterationNumb[1]) {


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
inline __device__ uint32_t* getTargetReduced(ForBoolKernelArgs<TXPPI> fbArgs, uint32_t iterationNumb[1]) {

    if ((iterationNumb[0] & 1) == 0) {
      return fbArgs.mainArrBPointer;
    }
    else {       
       return fbArgs.mainArrAPointer  ;
    }

}


/*
dilatation up and down - using bitwise operators
*/
#pragma once
inline __device__ uint32_t bitDilatate(uint32_t x) {
    return ((x) >> 1) | (x) | ((x) << 1);
}

/*
return 1 if at given position of given number bit is set otherwise 0 
*/
#pragma once
inline __device__ uint32_t isBitAt(uint32_t numb, int pos) {
    return (numb & (1 << (pos)));
}


inline uint32_t isBitAtCPU(uint32_t numb, int pos) {
    return (numb & (1 << (pos)));
}






#pragma once
inline __device__ void setNextBlockAsIsToBeActivated(coalesced_group active, char* tensorslice,
    int paddingNumb, uint16_t localWorkQueue[localWorkQueLength][4], uint16_t i, 
    int xMetaChange, int yMetaChange, int zMetaChange
    ,array3dWithDimsGPU targetArr,bool isAnythingInPadding[6], bool isInRagePred
) {
    //if (isToBeExecutedOnActive(active, paddingNumb)) {
    //    printf("\n setting neighbour of %d %d %d to active- %d %d %d padding numb %d  isAnyInPadding %d\n"
    //        , localWorkQueue[i][0], localWorkQueue[i][1], localWorkQueue[i][2]
    //        , localWorkQueue[i][0] + xMetaChange, localWorkQueue[i][1] + yMetaChange, localWorkQueue[i][2] + zMetaChange
    //        , paddingNumb , isAnythingInPadding[paddingNumb]
    //    );
    //}

    if (isAnythingInPadding[paddingNumb] && isToBeExecutedOnActive(active, paddingNumb) && isInRagePred) {


      //  printf(" \n saving to be actvated  xMeta %d yMeta %d zMeta %d isGold %d \n ", localWorkQueue[i][0] + xMetaChange, localWorkQueue[i][1] + yMetaChange, localWorkQueue[i][2] + zMetaChange, localWorkQueue[i][3]);


        getTensorRow<bool>(tensorslice, targetArr, targetArr.Ny, localWorkQueue[i][1] + yMetaChange, localWorkQueue[i][2] + zMetaChange)[localWorkQueue[i][0] + xMetaChange] = true;
    };

}


#pragma once
inline __device__ void setNextBlocksActivity( char* tensorslice,
    uint16_t localWorkQueue[localWorkQueLength][4], uint16_t i, array3dWithDimsGPU targetArr
    , bool isAnythingInPadding[6], coalesced_group active) {
    //0)top  1)bottom, 2)left 3)right, 4)anterior, 5)posterior, 
    //top
    setNextBlockAsIsToBeActivated(active, tensorslice, 0, localWorkQueue, i, 0, 0, -1, targetArr, isAnythingInPadding
    , localWorkQueue[i][2]>0);
    //bottom
    setNextBlockAsIsToBeActivated(active, tensorslice, 1, localWorkQueue, i, 0, 0, 1, targetArr, isAnythingInPadding
    , localWorkQueue[i][2]<(targetArr.Nz-1));
    //left
    setNextBlockAsIsToBeActivated(active, tensorslice, 2, localWorkQueue, i, -1, 0, 0, targetArr, isAnythingInPadding
    , localWorkQueue[i][0]>0);
    //right
    setNextBlockAsIsToBeActivated(active, tensorslice, 3, localWorkQueue, i, 1, 0, 0, targetArr, isAnythingInPadding
        , localWorkQueue[i][0] < (targetArr.Nx - 1));
    //anterior
    setNextBlockAsIsToBeActivated(active, tensorslice, 4, localWorkQueue, i, 0, 1, 0, targetArr, isAnythingInPadding
        , localWorkQueue[i][1] < (targetArr.Ny - 1));
    //posterior
    setNextBlockAsIsToBeActivated(active, tensorslice, 5, localWorkQueue, i, 0, -1, 0, targetArr, isAnythingInPadding
    , localWorkQueue[i][1] > 0);



}

/*
given source and target uint32 it will check the bit of intrest  of source and set the target to bit of target intrest
*/
#pragma once
inline __device__ void setBitTo(uint32_t source, uint8_t sourceBit, uint32_t resShared[32][32], uint8_t targetBit) {   
    resShared[threadIdx.x][threadIdx.y] |= ((source >> sourceBit) & 1) << targetBit;
   // return target;
}

///////////////////////////////// new functions


/*
calculate index in main shmem where array that is source for this dilatation round is present
*/
#pragma once
inline __device__ uint16_t getIndexForSourceShmem(MetaDataGPU metaData, uint32_t mainShmem[lengthOfMainShmem]
    ,  uint16_t i, bool isGold){
    return  metaData.mainArrXLength * 
    ((1 -isGold)// here calculating offset depending on what iteration and is gold;
        + (mainShmem[startOfLocalWorkQ + i] - (UINT16_MAX * (isGold))) * metaData.mainArrSectionLength )  ;// offset depending on linear index of metadata block of intrest

}
#pragma once
inline __device__ uint16_t getFullIndexForSourceShmemTotal(MetaDataGPU metaData, uint32_t mainShmem[lengthOfMainShmem]
    , uint16_t i, bool isGold) {
    return  (( (mainShmem[startOfLocalWorkQ + i] - UINT16_MAX * isGold) >0)* (-32)) // we check weather there is anything to the left - not on left border if so we load left 32 entries
        + getIndexForSourceShmem(metaData, mainShmem,  i, isGold);
}




/*
getting index where we should put first load - so data about this block and if apply block to the left and right
*/
#pragma once
inline __device__ uint16_t getIndexOfShmemToFirstLoad(uint32_t mainShmem[lengthOfMainShmem], uint16_t i, bool isGold) {
    return  (((mainShmem[startOfLocalWorkQ + i] - UINT16_MAX 
        * (mainShmem[startOfLocalWorkQ + i] >= UINT16_MAX)) > 0)* (-32)) + begSourceShmem;
}

/*
calculating where to put the data from res shmem - so data after dilatation back to global memory
*/
#pragma once
inline __device__ uint16_t getLengthOfShmemToFirstLoad(MetaDataGPU metaData, uint32_t mainShmem[lengthOfMainShmem]
    , uint16_t i, bool isGold) {
    return    (metaData.mainArrXLength + 32 * (((mainShmem[startOfLocalWorkQ + i] - UINT16_MAX * (isGold)) > 0)
        + ((mainShmem[startOfLocalWorkQ + i] - UINT16_MAX * (isGold)) < (metaData.totalMetaLength - 1))));// offset depending on linear index of this block
}




/*
calculate index in main shmem where array that is source for this dilatation round is present in the neighboutring block ...
*/
#pragma once
inline __device__ uint16_t getIndexForNeighbourForShmem(MetaDataGPU metaData, uint32_t mainShmem[lengthOfMainShmem]
    , uint32_t iterationNumb[1], uint32_t isGold[1], uint16_t currLinIndM[1], uint16_t localBlockMetaData[19],  size_t inMetaIndex) {
       return  metaData.mainArrXLength * 
    ((1 - (isGold[1]) )// here calculating offset depending on what iteration and is gold;
        + (localBlockMetaData[inMetaIndex]) * metaData.mainArrSectionLength )  ;// offset depending on linear index of metadata block of intrest
}

/*
calculating where to put the data from res shmem - so data after dilatation back to global memory
*/
inline __device__ uint16_t getIndexForSaveResShmem(MetaDataGPU metaData, uint32_t mainShmem[lengthOfMainShmem]
    , uint32_t iterationNumb[1], uint32_t isGold[1], uint16_t currLinIndM[1], uint16_t localBlockMetaData[19]) {
    return  metaData.mainArrXLength * (isGold[1])// here calculating offset depending on what iteration and is gold;
        + (currLinIndM[0] * metaData.mainArrSectionLength);// offset depending on linear index of this block
}



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
inline __device__ void dilatateHelperForTransverse(bool predicate,
    uint8_t paddingPos,    int8_t  normalXChange, int8_t normalYchange
, uint32_t mainShmem[], bool isAnythingInPadding[6]
,uint8_t forBorderYcoord, uint8_t forBorderXcoord
,uint8_t metaDataCoordIndex, uint16_t targetShmemOffset , uint16_t localBlockMetaData[20]) {
    // so we first check for corner cases 
    if (predicate) {
        // now we need to load the data from the neigbouring blocks
        //first checking is there anything to look to 
        if (localBlockMetaData[metaDataCoordIndex] < UINT16_MAX) {
            //now we load - we already done earlier up and down so now we are considering only anterior, posterior , left , right possibilities
            if (mainShmem[threadIdx.x + threadIdx.y * 32] > 0) {
                isAnythingInPadding[paddingPos] = true;
            };
            mainShmem[begResShmem + threadIdx.x + threadIdx.y * 32] =
                mainShmem[begResShmem + threadIdx.x + threadIdx.y * 32]
                | mainShmem[targetShmemOffset + forBorderXcoord + forBorderYcoord * 32];

        };
    }
    else {//given we are not in corner case we need just to do the dilatation using biwise or with the data inside the block


         mainShmem[begResShmem+threadIdx.x+threadIdx.y*32] 
        = mainShmem[(threadIdx.x+ normalXChange)+(threadIdx.y+ normalYchange)*32] | mainShmem[begResShmem+threadIdx.x+threadIdx.y*32];
    
    }
   

}


#pragma once
inline __device__ void dilatateHelperTopDown( uint8_t paddingPos, 
uint32_t* mainShmem, bool isAnythingInPadding[6], uint16_t localBlockMetaData[20]
,uint8_t metaDataCoordIndex
, uint32_t numberbitOfIntrestInBlock // represent a uint32 number that has a bit of intrest in this block set and all others 0 
, uint32_t numberWithCorrBitSetInNeigh// represent a uint32 number that has a bit of intrest in neighbouring block set and all others 0 
, uint16_t targetShmemOffset
) {
       // now we need to load the data from the neigbouring blocks
       //first checking is there anything to look to 
       if (localBlockMetaData[metaDataCoordIndex]< UINT16_MAX) {
           //now we load - we already done earlier up and down so now we are considering only anterior, posterior , left , right possibilities
           if (mainShmem[threadIdx.x + threadIdx.y * 32] & numberbitOfIntrestInBlock) {
                              // printf("setting padding top val %d \n ", isAnythingInPadding[0]);
                              isAnythingInPadding[0] = true;
           };
           mainShmem[begResShmem + threadIdx.x + threadIdx.y * 32] =
               mainShmem[begResShmem + threadIdx.x + threadIdx.y * 32]
               | (mainShmem[targetShmemOffset + threadIdx.x + threadIdx.y * 32] & numberWithCorrBitSetInNeigh);

       }   

}


//inline __device__  void lastLoad(ForBoolKernelArgs<TXPPI> fbArgs, thread_block cta//some needed CUDA objects
//    , unsigned int worQueueStep[1], uint16_t localBlockMetaData[]
//    , uint32_t mainShmem[], uint16_t i, MetaDataGPU metaData
//) {


//
///*
//constitutes end of pipeline  where we load data for next iteration if such is present
//*/
//template <typename TXPPI>
//inline __device__  void lastLoad(ForBoolKernelArgs<TXPPI> fbArgs, thread_block& cta//some needed CUDA objects
//    , unsigned int worQueueStep[1], uint16_t localBlockMetaData[]
//    , uint32_t mainShmem[], uint16_t i, MetaDataGPU metaData, uint16_t* metaDataArr
//) {
//
//    if (i + 1 <= worQueueStep[0]) {
//        cuda::memcpy_async(cta, (&localBlockMetaData[0]),
//            (&metaDataArr[(mainShmem[startOfLocalWorkQ + i - UINT16_MAX * (mainShmem[startOfLocalWorkQ + i] >= UINT16_MAX))
//                * metaData.metaDataSectionLength]])
//            , cuda::aligned_size_t<4>(sizeof(uint16_t) * 20), pipeline);
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





inline __device__  void afterBlockClean(thread_block cta
    , unsigned int worQueueStep[1], uint16_t localBlockMetaDataOld[6]
    , uint32_t mainShmem[], uint16_t i, MetaDataGPU metaData
    , thread_block_tile<32> tile
    , unsigned int localFpConter[1], unsigned int localFnConter[1]
    , unsigned int blockFpConter[1], unsigned int blockFnConter[1]
    , uint16_t* metaDataArr, uint16_t oldLinIndM[1], uint32_t oldIsGold[1]
    , bool isAnythingInPadding[6],bool isBlockFull[1], bool isPaddingPass) {



    if (tile.thread_rank() == 7 && tile.meta_group_rank() == 0) {// this is how it is encoded wheather it is gold or segm block
                    //this will be executed only if fp or fn counters are bigger than 0 so not during first pass
        if (localFpConter[0] > 0) {
            metaDataArr[oldLinIndM[0] * metaData.metaDataSectionLength + 3] += localFpConter[0];
            blockFpConter[0] += localFpConter[0];
            localFpConter[0] = 0;
        }
    };
    if (tile.thread_rank() == 8 && tile.meta_group_rank() == 0) {// this is how it is encoded wheather it is gold or segm block

        if (localFnConter[0] > 0) {
            metaDataArr[oldLinIndM[0] * metaData.metaDataSectionLength + 4] += localFnConter[0];

            blockFnConter[0] += localFnConter[0];
            localFnConter[0] = 0;
        }
    };
    if (tile.thread_rank() == 9 && tile.meta_group_rank() == 2) {// this is how it is encoded wheather it is gold or segm block

        //executed in case of previous block
        if (isBlockFull[0] && i > 0) {
            //setting data in metadata that block is full
            metaDataArr[oldLinIndM[0] * metaData.metaDataSectionLength + 10 - (oldIsGold[0] * 2)] = true;
        }
        //resetting
        isBlockFull[0] = true;
    };


    //we do it only for non padding pass
    if (tile.thread_rank() < 6 && tile.meta_group_rank() == 1 && !isPaddingPass) {   
        //executed in case of previous block
        if (i>0) {
            if (localBlockMetaDataOld[tile.thread_rank()] < UINT16_MAX) {
                metaDataArr[localBlockMetaDataOld[tile.thread_rank()] * metaData.metaDataSectionLength + 12 - oldIsGold[0]] = isAnythingInPadding[tile.thread_rank()];
            }
        }

        isAnythingInPadding[0] = false;
    };



}






