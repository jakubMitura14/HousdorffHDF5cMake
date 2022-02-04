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
gettinng source array for dilatations
basically arrays will alternate between iterations once one will be source other target then they will switch - we will decide upon knowing 
wheather the iteration number is odd or even
*/
template <typename TXPI>
inline __device__ array3dWithDimsGPU getSourceReduced(ForBoolKernelArgs<TXPI> fbArgs
    , uint16_t localWorkQueue[localWorkQueLength][4], uint16_t i, unsigned int iterationNumb[1]) {


    if ((iterationNumb[0] & 1) == 0) {
        if (localWorkQueue[i][3] == 1) {
            return fbArgs.reducedGoldPrev;
        }
        else {
            return fbArgs.reducedSegmPrev;
        }
    }
    else {       
        if (localWorkQueue[i][3] == 1) {
            return fbArgs.reducedGold;
        }
        else {
            return fbArgs.reducedSegm;
        }    
    }


}
/*
gettinng target array for dilatations
*/
template <typename TXPPI>
inline __device__ array3dWithDimsGPU getTargetReduced(ForBoolKernelArgs<TXPPI> fbArgs
    , uint16_t localWorkQueue[localWorkQueLength][4], uint16_t i, unsigned int iterationNumb[1]) {


    if ((iterationNumb[0] & 1) != 0) {
        if (localWorkQueue[i][3] == 1) {
            return fbArgs.reducedGoldPrev;
        }
        else {
            return fbArgs.reducedSegmPrev;
        }
    }
    else {     
        if (localWorkQueue[i][3] == 1) {
            return fbArgs.reducedGold;
        }
        else {
            return fbArgs.reducedSegm;
        }
    }


}
/*
loading data from appropriate reduce Arr to shared memory 
*/
#pragma once
template <typename TXI>
inline __device__ void loadDataToShmem(ForBoolKernelArgs<TXI> fbArgs, char* tensorslice, uint32_t sourceShared[32][32], array3dWithDimsGPU sourceReduced
, uint16_t localWorkQueue[localWorkQueLength][4], uint16_t i ) {
      sourceShared[threadIdx.x][ threadIdx.y]  
          = getTensorRow<uint32_t>(tensorslice, sourceReduced, sourceReduced.Ny
              , localWorkQueue[i][1] * fbArgs.dbYLength+ threadIdx.y
              , localWorkQueue[i][2])[localWorkQueue[i][0] *fbArgs.dbXLength+ threadIdx.x];
    //if (sourceShared[threadIdx.x][threadIdx.y] > 0) {
    //    printf("non zero in idX %d idY %d \n ", threadIdx.x, threadIdx.y);
    //}
}

/*
in order to be later able to analyze paddings we will save copy of the currently dilatated array 
(before dilatation) to global memory
*/
//template <typename TPYXI>
//inline __device__ void fromShmemToGlobal(ForBoolKernelArgs<TPYXI> fbArgs, char* tensorslice, uint32_t sourceShared[32][32], array3dWithDimsGPU target
//    , uint16_t localWorkQueue[localWorkQueLength][4], uint16_t i
//) {
//    
//    getTensorRow<uint32_t>(tensorslice, target, target.Ny, yMeta * fbArgs.dbYLength + threadIdx.y, zMeta)[xMeta * fbArgs.dbXLength + threadIdx.x]= sourceShared[threadIdx.x][ threadIdx.y];
//}
//


/*
saving dilatated data to global memory
*/
#pragma once
template <typename TXTI>
inline __device__ void saveToDilatationArr(ForBoolKernelArgs<TXTI> fbArgs, char* tensorslice, uint32_t resShared[32][32], array3dWithDimsGPU resDilatated
    , uint16_t localWorkQueue[localWorkQueLength][4], uint16_t i
) {
    //if (resShared[threadIdx.x][threadIdx.y]>0) {
    //    printf("non zero in saving  in idX %d idY %d zMeta %d \n ", threadIdx.x, threadIdx.y, localWorkQueue[i][2]);

    //}
    //    getTensorRow<uint32_t>(tensorslice, resDilatated, resDilatated.Ny,localWorkQueue[i][1] * fbArgs.dbYLength + threadIdx.y, localWorkQueue[i][2])[localWorkQueue[i][0] * fbArgs.dbXLength + threadIdx.x]; 

    getTensorRow<uint32_t>(tensorslice, resDilatated, resDilatated.Ny, localWorkQueue[i][1] * fbArgs.dbYLength + threadIdx.y, localWorkQueue[i][2])[localWorkQueue[i][0] * fbArgs.dbXLength + threadIdx.x]
    = resShared[threadIdx.x][ threadIdx.y];
}




///*
//checking in metadata weather block need to be validated
//*/
//#pragma once
//inline __device__ void isBlockToBeValidatedd(char* tensorslice, bool isBlockToBeValidated[1], array3dWithDimsGPU sourceReduced
//    , uint16_t xMeta, uint16_t yMeta, uint16_t zMeta)
//{
//    isBlockToBeValidated[0] = getTensorRow<bool>(tensorslice, sourceReduced, sourceReduced.Ny, yMeta , zMeta)[xMeta ];
//}
//

/*
marking that block is already full*/
#pragma once
inline __device__ void markIsBlockFull(char* tensorslice
    , uint16_t localWorkQueue[localWorkQueLength][4], uint16_t i, bool isBlockFull, array3dWithDimsGPU targetMeta, coalesced_group active)
{
    if (isBlockFull && isToBeExecutedOnActive(active, 8)) {
        
      //  printf("set block as full  %d %d %d " , localWorkQueue[i][0], localWorkQueue[i][1], localWorkQueue[i][2]);

        getTensorRow<bool>(tensorslice, targetMeta, targetMeta.Ny, localWorkQueue[i][1], localWorkQueue[i][2])[localWorkQueue[i][0]] = true;
    }
}

/*
set the fp or fn counters of metadata
*/
#pragma once
inline __device__ void updateMetaCounters(char* tensorslice
    , uint16_t xMeta, uint16_t yMeta, uint16_t zMeta, uint16_t isGold,   array3dWithDimsGPU targetMeta,unsigned int fpOrFnCount,  coalesced_group active)
{
    if ( isToBeExecutedOnActive(active, 9)) {
        getTensorRow<unsigned int>(tensorslice, targetMeta, targetMeta.Ny, yMeta, zMeta)[xMeta] += fpOrFnCount;
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


/*
to iterate over the threads and given their position - checking edge cases do appropriate dilatations ...
predicate - indicates what we consider border case here
paddingPos = integer marking which padding we are currently talking about(top ? bottom ? anterior ? ...)
padingVariedA, padingVariedB - eithr bitPos threadid X or Y depending what will be changing in this case

normalXChange, normalYchange - indicating which wntries we are intrested in if we are not at the boundary so how much to add to xand y thread position
*/
#pragma once
template <typename TXTOI>
inline __device__ void dilatateHelper(bool predicate,
    int paddingPos,   int  padingVariedB, int  normalXChange, int normalYchange
, uint32_t sourceShared[32][32], uint32_t resShared[32][32], bool isAnythingInPadding[6]
,bool predicateToLoadOutside, char* tensorslice, ForBoolKernelArgs<TXTOI> fbArgs, uint16_t localWorkQueue[localWorkQueLength][4], uint16_t i
, unsigned int iterationNumb[1], uint8_t forBorderYcoord, uint8_t forBorderXcoord) {
   


    // so we first check for corner cases 
    if (predicate) {
        // now we need to load the data from the neigbouring blocks
        //first checking is there anything to look to 
        if (predicateToLoadOutside) {
            //now we load - we already done earlier up and down so now we are considering only anterior, posterior , left , right possibilities
            if (sourceShared[threadIdx.x][threadIdx.y] > 0) {
                isAnythingInPadding[paddingPos] = true;
            };
            //printf("looking padding currMetaX %d currMetaY %d currMetaZ %d X %d Y %d padding pos  paddingPos %d value %d  ;  %d  xChange %d   y Change %d  \n"
            //   ,  localWorkQueue[i][0], localWorkQueue[i][1], localWorkQueue[i][2]
            //, (localWorkQueue[i][0] + normalXChange) * fbArgs.dbXLength + forBorderXcoord
            //    , (localWorkQueue[i][1] + normalYchange)* fbArgs.dbYLength + forBorderYcoord 
            //        , paddingPos,  isAnythingInPadding[paddingPos], sourceShared[threadIdx.x][threadIdx.y]
            //, normalXChange, normalYchange
            //);

            //printf("looking padding xChange %d yChange %d currMetaX %d currMetaY %d currMetaZ %d new X %d new Y %d value %d  \n"
            //    , normalXChange, normalYchange, localWorkQueue[i][0], localWorkQueue[i][1], localWorkQueue[i][2]
            //, (localWorkQueue[i][0] + normalXChange) * fbArgs.dbXLength + forBorderXcoord
            //    , (localWorkQueue[i][1] + normalYchange)* fbArgs.dbYLength + forBorderYcoord 
            //    , getTensorRow<uint32_t>(tensorslice, getSourceReduced(fbArgs, localWorkQueue, i, iterationNumb)
            //        , fbArgs.reducedGold.Ny, (localWorkQueue[i][1] + normalYchange) * fbArgs.dbYLength + forBorderYcoord
            //        , localWorkQueue[i][2])[(localWorkQueue[i][0] + normalXChange) * fbArgs.dbXLength + forBorderXcoord]);


            resShared[threadIdx.x][threadIdx.y] = 
                resShared[threadIdx.x][threadIdx.y]
                    | getTensorRow<uint32_t>(tensorslice, getSourceReduced(fbArgs, localWorkQueue, i, iterationNumb)
                        , fbArgs.reducedGold.Ny, (localWorkQueue[i][1] + normalYchange) * fbArgs.dbYLength + forBorderYcoord
                            , localWorkQueue[i][2])[(localWorkQueue[i][0]+ normalXChange) * fbArgs.dbXLength + forBorderXcoord];
            ;

        }
    }
    else {//given we are not in corner case we need just to do the dilatation using biwise or 
        resShared[threadIdx.x][threadIdx.y] = sourceShared[threadIdx.x+ normalXChange][threadIdx.y+ normalYchange] | resShared[threadIdx.x][threadIdx.y];
    
    }
   

}




inline __device__ void clearisAnythingInPadding (bool isAnythingInPadding[6]) {

    auto active = coalesced_threads();
    #pragma unroll
    for (int ii; ii < 6; ii++) {
        if (isToBeExecutedOnActive(active, ii)) { isAnythingInPadding[ii] = 0; };
    };
}

/**
loading some data on single threads to shared memory that can be needed by all blocks 
*/

#pragma once
template <typename TYXI>
inline __device__ void loadSmallVars(ForBoolKernelArgs<TYXI> fbArgs, char* tensorslice
    , unsigned int resultfpOffset[1], unsigned int resultfnOffset[1], bool isBlockToBeValidated[1]
    ,uint16_t xMeta, uint16_t yMeta, uint16_t zMeta,uint16_t isGold, coalesced_group active
    , unsigned int localFpConter[1], unsigned int localFnConter[1]
) {

    //is to be validates
    if (isToBeExecutedOnActive(active, 0) && isGold == 1) {
       
        //printf("\n isToBeValidatedFp %d count %d counter %d     %d xMeta %d yMeta %d zMeta %d \n  ",
        //    getTensorRow<unsigned int>(tensorslice, fbArgs.metaData.fpCounter, fbArgs.metaData.fpCounter.Ny, yMeta, zMeta)[xMeta]
        //    < getTensorRow<unsigned int>(tensorslice, fbArgs.metaData.fpCount, fbArgs.metaData.fpCount.Ny, yMeta, zMeta)[xMeta]
        //    , getTensorRow<unsigned int>(tensorslice, fbArgs.metaData.fpCount, fbArgs.metaData.fpCount.Ny, yMeta, zMeta)[xMeta]
        //    , getTensorRow<unsigned int>(tensorslice, fbArgs.metaData.fpCounter, fbArgs.metaData.fpCounter.Ny, yMeta, zMeta)[xMeta]
        //    , xMeta, yMeta, zMeta);


   /*     getTensorRow<bool>(tensorslice, fbArgs.metaData.isToBeValidatedFp, fbArgs.metaData.isToBeValidatedFp.Ny, yMeta, zMeta)[xMeta]
            = (getTensorRow<unsigned int>(tensorslice, fbArgs.metaData.fpCounter, fbArgs.metaData.fpCounter.Ny, yMeta, zMeta)[xMeta]
                < getTensorRow<unsigned int>(tensorslice, fbArgs.metaData.fpCount, fbArgs.metaData.fpCount.Ny, yMeta, zMeta)[xMeta]);*/

        isBlockToBeValidated[0] = (getTensorRow<unsigned int>(tensorslice, fbArgs.metaData.fpCounter, fbArgs.metaData.fpCounter.Ny, yMeta, zMeta)[xMeta]
            < getTensorRow<unsigned int>(tensorslice, fbArgs.metaData.fpCount, fbArgs.metaData.fpCount.Ny, yMeta, zMeta)[xMeta]);
       // isBlockToBeValidated[0] = getTensorRow<unsigned int>(tensorslice, fbArgs.metaData.isToBeValidatedFp, fbArgs.metaData.fpOffset.Ny, yMeta, zMeta)[xMeta];
    };
    if (isToBeExecutedOnActive(active, 1) && isGold == 0) {
       
    //    printf("\n isToBeValidated Fn  %d count %d counter %d     xMeta %d yMeta %d zMeta %d   \n  ",
    //getTensorRow<unsigned int>(tensorslice, fbArgs.metaData.fnCounter, fbArgs.metaData.fnCounter.Ny, yMeta, zMeta)[xMeta]
    //< getTensorRow<unsigned int>(tensorslice, fbArgs.metaData.fnCount, fbArgs.metaData.fnCount.Ny, yMeta, zMeta)[xMeta]
    //, getTensorRow<unsigned int>(tensorslice, fbArgs.metaData.fnCount, fbArgs.metaData.fnCount.Ny, yMeta, zMeta)[xMeta]
    //, getTensorRow<unsigned int>(tensorslice, fbArgs.metaData.fnCounter, fbArgs.metaData.fnCounter.Ny, yMeta, zMeta)[xMeta]
    //, xMeta, yMeta, zMeta);



   /*     getTensorRow<bool>(tensorslice, fbArgs.metaData.isToBeValidatedFn, fbArgs.metaData.isToBeValidatedFn.Ny, yMeta, zMeta)[xMeta]
            = (getTensorRow<unsigned int>(tensorslice, fbArgs.metaData.fnCounter, fbArgs.metaData.fnCounter.Ny, yMeta, zMeta)[xMeta]
                < getTensorRow<unsigned int>(tensorslice, fbArgs.metaData.fnCount, fbArgs.metaData.fnCount.Ny, yMeta, zMeta)[xMeta]);
  */      
        isBlockToBeValidated[0] = (getTensorRow<unsigned int>(tensorslice, fbArgs.metaData.fnCounter, fbArgs.metaData.fnCounter.Ny, yMeta, zMeta)[xMeta]
            < getTensorRow<unsigned int>(tensorslice, fbArgs.metaData.fnCount, fbArgs.metaData.fnCount.Ny, yMeta, zMeta)[xMeta]);
        //isBlockToBeValidated[0] = getTensorRow<unsigned int>(tensorslice, fbArgs.metaData.isToBeValidatedFn, fbArgs.metaData.fpOffset.Ny, yMeta, zMeta)[xMeta];
    };
    //offsets
    if (isToBeExecutedOnActive(active, 2)) {// && isGold == 1
        resultfpOffset[0] = getTensorRow<unsigned int>(tensorslice, fbArgs.metaData.fpOffset, fbArgs.metaData.fpOffset.Ny, yMeta, zMeta)[xMeta];
      //  printf("\n resultfpOffset[0] %d xMeta %d yMeta %d  zMeta %d \n ", resultfpOffset[0], xMeta, yMeta, zMeta);

    };
    if (isToBeExecutedOnActive(active, 3) ) {//&& isGold == 0
       //printf("\n resultfnOffset[0] %d xMeta %d yMeta %d  zMeta %d \n ", resultfnOffset[0], xMeta, yMeta, zMeta);


        resultfnOffset[0] = getTensorRow<unsigned int>(tensorslice, fbArgs.metaData.fnOffset, fbArgs.metaData.fnOffset.Ny, yMeta, zMeta)[xMeta];
    };
    // block counters
    if (isToBeExecutedOnActive(active, 4) && isGold == 1) {
        //auto xx = getTensorRow<unsigned int>(tensorslice, fbArgs.metaData.fpCounter, fbArgs.metaData.fpCounter.Ny, yMeta, zMeta)[xMeta];
        //printf("setting ");

        localFpConter[0] = getTensorRow<unsigned int>(tensorslice, fbArgs.metaData.fpCounter, fbArgs.metaData.fpCounter.Ny, yMeta, zMeta)[xMeta];
    };
    if (isToBeExecutedOnActive(active, 5) && isGold == 0) {
        localFnConter[0] = getTensorRow<unsigned int>(tensorslice, fbArgs.metaData.fnCounter, fbArgs.metaData.fnCounter.Ny, yMeta, zMeta)[xMeta];
    };



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


/*
now we will  additionally get bottom bit of block above and top of block below given they exist
*/
#pragma once
template <typename TXTIO>
inline __device__ void checkBlockToUpAndBottom (ForBoolKernelArgs<TXTIO> fbArgs, char* tensorslice,
    uint16_t localWorkQueue[localWorkQueLength][4], uint16_t i, array3dWithDimsGPU sourceArr, uint32_t resShared[32][32]) {
 
    //looking up
    if (localWorkQueue[i][2] > 0) {//boundary check
     //auto  xx =   getTensorRow<unsigned int>(tensorslice, sourceArr, sourceArr.Ny, localWorkQueue[i][1] * fbArgs.dbYLength + threadIdx.y, localWorkQueue[i][2] - 1)[localWorkQueue[i][0] * fbArgs.dbXLength + threadIdx.x]

      // printf(" looking up  ");
        //source
        setBitTo(getTensorRow<uint32_t>(tensorslice, sourceArr, sourceArr.Ny, localWorkQueue[i][1] * fbArgs.dbYLength + threadIdx.y, localWorkQueue[i][2] - 1)[localWorkQueue[i][0] * fbArgs.dbXLength + threadIdx.x]
            , (fbArgs.dbZLength - 1) //sourceBit
            , resShared//target
            , 0//target bit
        );

    };
    //look down 
    if (localWorkQueue[i][2] < (fbArgs.metaData.MetaZLength - 1)) {//boundary check
        //source
        setBitTo(getTensorRow<uint32_t>(tensorslice, sourceArr, sourceArr.Ny, localWorkQueue[i][1] * fbArgs.dbYLength + threadIdx.y, localWorkQueue[i][2] + 1)[localWorkQueue[i][0] * fbArgs.dbXLength + threadIdx.x]
            , 0 //sourceBit
            , resShared//target
            , (fbArgs.dbZLength - 1)//target bit
        );

    };


}


template <typename TXYYOI>
inline __device__ void clearShmemBeforeDilatation(ForBoolKernelArgs<TXYYOI> fbArgs, char* tensorslice, unsigned int blockFpConter[1], unsigned int blockFnConter[1]
    , unsigned int localWorkQueueCounter[1], unsigned int localFpConter[1], unsigned int localFnConter[1]
) {

    auto activeD = coalesced_threads();
    //resetting
    if (isToBeExecutedOnActive(activeD, 3)) {
        localWorkQueueCounter[0] = 0;
    };
    if (isToBeExecutedOnActive(activeD, 4)) {
        localFpConter[0] = 0;
    };
    if (isToBeExecutedOnActive(activeD, 5)) {
        localFnConter[0] = 0;
    };

}




/*
establish wheather we still need dilatations in both passes
*/
template <typename TXTJIOP>
inline __device__ void checkIsToBeDilatated(ForBoolKernelArgs<TXTJIOP> fbArgs, char* tensorslice, bool isGoldPassToContinue[1], bool isSegmPassToContinue[1]) {
    auto activeE = coalesced_threads();
    if (isToBeExecutedOnActive(activeE, 0)) {
        isGoldPassToContinue[0] = (ceilf(getTensorRow<unsigned int>(tensorslice, fbArgs.metaData.minMaxes, 1, 0, 0)[7] * fbArgs.robustnessPercent)
    > getTensorRow<unsigned int>(tensorslice, fbArgs.metaData.minMaxes, 1, 0, 0)[10]);
   
   //     isGoldPassToContinue[0] = (getTensorRow<unsigned int>(tensorslice, fbArgs.metaData.minMaxes, 1, 0, 0)[7] 
   // > getTensorRow<unsigned int>(tensorslice, fbArgs.metaData.minMaxes, 1, 0, 0)[10]);

    }
    if (isToBeExecutedOnActive(activeE, 1)) {
       
        //TODO() remove 
     /*   auto xx = getTensorRow<unsigned int>(tensorslice, fbArgs.metaData.minMaxes, 1, 0, 0)[8];
        unsigned int counter = getTensorRow<unsigned int>(tensorslice, fbArgs.metaData.minMaxes, 1, 0, 0)[11];
        printf("\n  setting is to be dilatated   global fn count %d times robustness %f counter %f is to be accepted %d \n",xx
            , ceilf((float)xx * fbArgs.robustnessPercent), counter,  ( ceilf(xx* 0.95)> counter));
        */


       
        isSegmPassToContinue[0] = (ceilf(getTensorRow<unsigned int>(tensorslice, fbArgs.metaData.minMaxes, 1, 0, 0)[8] * fbArgs.robustnessPercent)
            > getTensorRow<unsigned int>(tensorslice, fbArgs.metaData.minMaxes, 1, 0, 0)[11]);


    //    isSegmPassToContinue[0] = (getTensorRow<unsigned int>(tensorslice, fbArgs.metaData.minMaxes, 1, 0, 0)[8]
      //      > getTensorRow<unsigned int>(tensorslice, fbArgs.metaData.minMaxes, 1, 0, 0)[11]);

    }

}

/*
update global fp and fn counters and resets shared memory values after dilatations*/
template <typename TXTJIOI>
inline __device__ void updateGlobalCountersAndClear(ForBoolKernelArgs<TXTJIOI> fbArgs, char* tensorslice, unsigned int blockFpConter[1], unsigned int blockFnConter[1]
    , unsigned int localWorkQueueCounter[1], unsigned int localFpConter[1], unsigned int localFnConter[1]
) {
  
    auto activeD = coalesced_threads();
    if (isToBeExecutedOnActive(activeD, 6)) {
        //if (blockFpConter[0]>0) {
        //    printf("\n adding to global fp counter  %d \n", blockFpConter[0]);

        //}
        atomicAdd(&(getTensorRow<unsigned int>(tensorslice, fbArgs.metaData.minMaxes, 1, 0, 0)[10]), (blockFpConter[0]));
       // blockFpConter[0] = 0;
    };
    if (isToBeExecutedOnActive(activeD, 7)) {
        //if (blockFnConter[0]) {
        //    printf("\n adding to global fn counter  %d \n", blockFnConter[0]);
        //}
       // printf("\n  block fn counter %d curr value %d \n", blockFnConter[0], getTensorRow<unsigned int>(tensorslice, fbArgs.metaData.minMaxes, 1, 0, 0)[11]);
        atomicAdd(&(getTensorRow<unsigned int>(tensorslice, fbArgs.metaData.minMaxes, 1, 0, 0)[11]), (blockFnConter[0]));
     //   blockFnConter[0] = 0;
    };

    if (isToBeExecutedOnActive(activeD, 8)) {
        // printf("\n  block fn counter %d curr value %d \n", blockFnConter[0], getTensorRow<unsigned int>(tensorslice, fbArgs.metaData.minMaxes, 1, 0, 0)[11]);
        getTensorRow<unsigned int>(tensorslice, fbArgs.metaData.minMaxes, 1, 0, 0)[9] = 0;
    };
    //resetting
    //if (isToBeExecutedOnActive(activeD, 3)) {
    //    localWorkQueueCounter[0] = 0;
    //};
    //if (isToBeExecutedOnActive(activeD, 4)) {
    //    localFpConter[0] = 0;
    //};
    //if (isToBeExecutedOnActive(activeD, 5)) {
    //    localFnConter[0] = 0;
    //};
    //if (isToBeExecutedOnActive(activeD, 5)) {
    //    localFnConter[0] = 0;
    //};

}




///////////////////////////////// bigger functions









