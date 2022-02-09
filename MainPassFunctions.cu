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
inline __device__ uint32_t* getTargetReduced(ForBoolKernelArgs<TXPI> fbArgs, uint32_t iterationNumb[1]) {

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
inline __device__ uint16_t getIndexForSourceShmem(MetaDataGPU metaData, uint32_t mainShmem[lengthOfMainShmem]
    , uint32_t iterationNumb[1], uint16_t i){
    return  metaData.mainArrXLength * 
    (1 - (mainShmem[startOfLocalWorkQ + i] >= UINT16_MAX))// here calculating offset depending on what iteration and is gold;
        + (mainShmem[startOfLocalWorkQ + i] - (UINT16_MAX * (mainShmem[startOfLocalWorkQ + i] >= UINT16_MAX))) * metaData.mainArrSectionLength   ;// offset depending on linear index of metadata block of intrest

}


/*
calculate index in main shmem where array that is source for this dilatation round is present in the neighboutring block ...
*/
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
    return  metaData.mainArrXLength *
        (1 - (isGold[1]) * 2))// here calculating offset depending on what iteration and is gold;
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
    uint8_t paddingPos,    uint8_t  normalXChange, uint8_t normalYchange
, uint32_t mainShmem[], bool isAnythingInPadding[6]
,uint8_t forBorderYcoord, uint8_t forBorderXcoord
,uint8_t metaDataCoordIndex, uint16_t targetShmemOffset , uint16_t localBlockMetaData[20]) {
    // so we first check for corner cases 
    if (predicate) {
        // now we need to load the data from the neigbouring blocks
        //first checking is there anything to look to 
        if (localBlockMetaData[metaDataCoordIndex]< UINT16_MAX) {
            //now we load - we already done earlier up and down so now we are considering only anterior, posterior , left , right possibilities
            if (mainShmem[threadIdx.x+threadIdx.y*32] > 0) {
                isAnythingInPadding[paddingPos] = true;
            };
            mainShmem[begResShmem + threadIdx.x + threadIdx.y * 32] =
                mainShmem[begResShmem + threadIdx.x + threadIdx.y * 32]
                | mainShmem[targetShmemOffset + forBorderXcoord + forBorderYcoord * 32];

        }
    }
    else {//given we are not in corner case we need just to do the dilatation using biwise or with the data inside the block
        mainShmem[begResShmem+threadIdx.x+threadIdx.y*32] 
        = mainShmem[(threadIdx.x+ normalXChange)+(threadIdx.y+ normalYchange)*32] | mainShmem[begResShmem+threadIdx.x+threadIdx.y*32];
    
    }
   

}


#pragma once
template <typename TXTOI>
inline __device__ void dilatateHelperTopDown( uint8_t paddingPos, 
, uint32_t mainShmem[], bool isAnythingInPadding[6], localBlockMetaData
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
           mainShmem[begResShmem+threadIdx.x+threadIdx.y*32] = 
               mainShmem[begResShmem+threadIdx.x+threadIdx.y*32]
                   | (mainShmem[targetShmemOffset+forBorderXcoord+forBorderYcoord*32] & numberWithCorrBitSetInNeigh )

       }   

}
//
//
//
///*
//in pipeline defined to load data for next step and simultaneously process the previous step data  
//used for left,right,anterior,posterior dilatations
//*/
//inline __device__  void loadNextAndProcessPreviousSides(pipeline,cta//some needed CUDA objects
//localBlockMetaData,mainShmem,iterationNumb,isGold, currLinIndM// shared memory arrays used block wide
//, metaData,mainArr, //pointers to arrays with data
////now some variables needed to load data  
//    uint8_t metaDataCoordIndexToLoad // where is the index describing linear index of the neighbour in direction of intrest
//    ,uint16_t targetShmemOffset //offset defined in shared memory used to load data into 
//    , shape // shape and alignment of data in load - inludes length of data
////now variables needed for dilatations
//    uint8_t metaDataCoordIndexToProcess // where is the index describing linear index of the neighbour in direction of intrest
//    ,uint16_t sourceShmemOffset //offset defined in shared memory used to process  data from 
//,bool predicate // defining when our thread is a corner case and need to load data from outside of the block
//,uint8_t paddingPos,// needed to know wheather block in given direction should be marked as to be activated
//uint8_t  normalXChange, uint8_t normalYchange
//, uint8_t forBorderYcoord, uint8_t forBorderXcoord
//
//){
//
//krowa rethink weather pipeline.producer_acquire() and commit should not be inside the if statements for border cases
//
//               pipeline.producer_acquire();
//                       if (localBlockMetaData[metaDataCoordIndexToLoad]<UINT16_MAX) {
//                           cooperative_groups::memcpy_async(cta, (&mainShmem[begSecRegShmem]),
//                              (&mainArr[getIndexForNeighbourForShmem(metaData, mainShmem, iterationNumb, isGold, currLinIndM, localBlockMetaData,metaDataCoordIndexToLoad )]) 
//                              , shape, pipeline);
//
//                       }
//                     
//               pipeline.producer_commit();
//               //compute 
//                    //if we want to do left riaght, anterior , posterior dilatations
//                  dilatateHelperForTransverse(predicate), paddingPos, normalXChange, normalYchange, mainShmem
//                     , isAnythingInPadding,  iterationNumb,forBorderYcoord, forBorderXcoord,metaDataCoordIndexToProcess,sourceShmemOffset );
//  
//                     
//                     
//              
//}
//
//
/*
constitutes end of pipeline  where we load data for next iteration if such is present
*/
inline __device__  void lastLoad(cta//some needed CUDA objects
worQueueStep, localBlockMetaData, mainArr, mainShmem, i, metaData
){
              if (i + 1<= worQueueStep[0]) {
                  cuda::memcpy_async(cta, (&localBlockMetaData[0]), (&mainArr[(mainShmem[startOfLocalWorkQ+1+i] - UINT16_MAX * (mainShmem[startOfLocalWorkQ+i+1] >= UINT16_MAX)) 
                  * metaData.mainArrSectionLength + metaData.metaDataOffset])
                      , cuda::aligned_size_t<4>(sizeof(uint32_t) * 18), pipeline);
              }
}

/// we need to define here the function that will update the metadata result for the given block - also if it is not padding pass we need to set the neighbouring blocks as to be activated according to the data in shmem


  if (tile.thread_rank() == 1 && tile.meta_group_rank() == 0) {
      blockFpConter[0]+=localFpConter[0]
        localFpConter[0] = 0;
    };
    if (tile.thread_rank() == 2 && tile.meta_group_rank() == 0) {
        blockFnConter[0]+=localFnConter[0] ;
        localFnConter[0]=0;
    };
    if (tile.thread_rank() == 3 && tile.meta_group_rank() == 0) {
        localFpConter[0] = 0;
    };
    if (tile.thread_rank() == 4 && tile.meta_group_rank() == 0) {
        localFnConter[0] = 0;
    };
          add info about increase fp or fn count to metadata block and to block variable in thread block






