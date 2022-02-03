

#include "CPUAllocations.cu"
#include "MetaData.cu"
#include "IterationUtils.cu"
#include "ExceptionManagUtils.cu"
#include "CooperativeGroupsUtils.cu"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cooperative_groups/memcpy_async.h>

using namespace cooperative_groups;

/*
given appropriate cudaPitchedPtr and ForFullBoolPrepArgs will return ForBoolKernelArgs
*/
#pragma once
template <typename TCC>
inline ForBoolKernelArgs<TCC> getArgsForKernel(ForFullBoolPrepArgs<TCC> mainFunArgs, array3dWithDimsGPU forDebugArr
    , array3dWithDimsGPU goldArr
    , array3dWithDimsGPU segmArr
) {

    ForBoolKernelArgs<TCC> res;
    res.metaData = allocateMetaDataOnGPU(mainFunArgs.metaData);
    res.forDebugArr = forDebugArr;
    res.goldArr = goldArr;
    res.segmArr = segmArr;

    res.numberToLookFor = mainFunArgs.numberToLookFor;
    res.dbXLength = mainFunArgs.dbXLength;
    res.dbYLength = mainFunArgs.dbYLength;
    res.dbZLength = mainFunArgs.dbZLength;



    return res;
}


/*
setting the linear index of metadata blocks that are in given direction if there is no such (out of range) we will save it as UINT32_MAX
*/
template <typename TCC>
__device__ inline void setNeighbourBlocks(ForBoolKernelArgs<TCC> fbArgs,uint8_t idX, uint8_t inArrIndex, bool predicate, uint32_t toAdd , uint16_t linIdexMeta) {

    if ((threadIdx.x == idX) && (threadIdx.y == 0)) {
        if (predicate) {
            fbArgs.mainArr[linIdexMeta * fbArgs.mainArrSectionLength + fbArgs.metaDataOffset + inArrIndex] = linIdexMeta - toAdd;
        }
        else {
            fbArgs.mainArr[linIdexMeta * fbArgs.mainArrSectionLength + fbArgs.metaDataOffset + inArrIndex] = UINT32_MAX;
        }
    };
}





/*
iteration over metadata - becouse metadata may be small and to maximize occupancy we use linear index and then clalculate xMeta,ymeta,zMeta from this linear index ...
*/
#pragma once
template <typename TYU>
__device__ void metaDataIter(ForBoolKernelArgs<TYU> fbArgs) {

    ////////////some initializations
    bool goldBool = false;
    bool segmBool = false;
    bool isNotEmpty = false;
  
    thread_block cta = this_thread_block();
    thread_block_tile<32> tile = tiled_partition<32>(cta);
    uint16_t sumFp = 0;
    uint16_t sumFn = 0;
   


    //shared memory

    //TODO() make it dynamically sized 
    __shared__ uint32_t sharedForGold[1024];
    __shared__ uint32_t sharedForSegm[1024];
    //for storing fp and fn sums to later accumulate it to global values
    __shared__ uint32_t fpSFnS[2];

    __shared__ bool anyInGold[1];
    __shared__ bool anyInSegm[1];
    //__shared__ uint32_t reduction_s[32];
    //1)maxX 2)minX 3)maxY 4) minY 5) maxZ 6) minZ
    __shared__ int minMaxesInShmem[7];

    if ((threadIdx.x == 1) && (threadIdx.y == 1)) { fpSFnS[0] = 0; };
    if ((threadIdx.x == 2) && (threadIdx.y == 1)) { fpSFnS[1] = 0; };
    if ((threadIdx.x == 3) && (threadIdx.y == 1)) { anyInGold[1] = false; };
    if ((threadIdx.x == 4) && (threadIdx.y == 1)) { anyInSegm[1] = false; };

    //we need to load also min and maxes of metdata 


    __syncthreads();

    /////////////////////////


    //main metadata iteration
    for (uint16_t linIdexMeta = blockIdx.x; linIdexMeta < fbArgs.metaData.totalMetaLength; linIdexMeta += gridDim.x) {
        //we get from linear index  the coordinates of the metadata block of intrest
        uint8_t xMeta = linIdexMeta % fbArgs.metaData.metaXLength;
        uint8_t zMeta = floor((float)(linIdexMeta / (fbArgs.metaData.metaXLength * fbArgs.metaData.MetaYLength)));
        uint8_t yMeta = floor((float)((linIdexMeta - ((zMeta * fbArgs.metaData.metaXLength * fbArgs.metaData.MetaYLength) + xMeta)) / fbArgs.metaData.metaXLength));
        //reset
        isNotEmpty = false;
        sumFp = 0;
        sumFn = 0;
        anyInGold[0] = false;
        anyInSegm[0] = false;
        //iterating over data block

        for (uint8_t xLoc = threadIdx.x; xLoc < fbArgs.dbXLength; xLoc += blockDim.x) {
            uint16_t x = (xMeta+ fbArgs.minX)* fbArgs.dbXLength + xLoc;//absolute position
            for (uint8_t yLoc = threadIdx.y; yLoc < fbArgs.dbYLength; yLoc += blockDim.y) {
                uint16_t  y = (yMeta+fbArgs.minY) * fbArgs.dbYLength + yLoc;//absolute position
                if (y < fbArgs.goldArr.Ny && x < fbArgs.goldArr.Nz) {

                    // resetting 
                    sharedForGold[xLoc + yLoc * fbArgs.dbXLength] = 0;
                    sharedForSegm[xLoc + yLoc * fbArgs.dbXLength] = 0;
        

                    for (uint8_t zLoc = 0; zLoc < fbArgs.dbZLength; zLoc++) {
                        uint16_t z = (zMeta+ fbArgs.minZ)* fbArgs.dbZLength + zLoc;//absolute position
                        if (z < fbArgs.goldArr.Nx) {
                            char* tensorslice;

                            //first array gold
                            goldBool = (getTensorRow<TYU>(tensorslice, fbArgs.goldArr, fbArgs.goldArr.Ny, y, z)[x] == fbArgs.numberToLookFor);

                            // now segmentation  array
                            segmBool = (getTensorRow<TYU>(tensorslice, fbArgs.segmArr, fbArgs.goldArr.Ny, y, z)[x] == fbArgs.numberToLookFor);
                            // setting bits
                            sharedForGold[xLoc+yLoc* fbArgs.dbXLength] |= goldBool << zLoc;
                            sharedForSegm[xLoc+yLoc* fbArgs.dbXLength] |= segmBool << zLoc;
                            // setting value of local boolean marking that any of the entries was evaluated to true in either of arrays
                            isNotEmpty = (isNotEmpty || (goldBool || segmBool));
                            sumFp += (!goldBool && segmBool);
                            sumFn += (goldBool && !segmBool);
                            if (goldBool)  anyInGold[0] = true;
                            if (segmBool)  anyInSegm[0] = true;

                        }
                    }
                }
            }
        }

        

    
        isNotEmpty = __syncthreads_or(isNotEmpty);
        //copy data to global memory from shmem
        cooperative_groups::memcpy_async(cta, (&fbArgs.mainArr[linIdexMeta * fbArgs.mainArrSectionLength]), (sharedForGold), (sizeof(uint32_t) * blockDim.x * blockDim.y) );
        cooperative_groups::memcpy_async(cta, (&fbArgs.mainArr[linIdexMeta * fbArgs.mainArrSectionLength + fbArgs.mainArrXLength]), (sharedForSegm), (sizeof(uint32_t) * blockDim.x * blockDim.y) );

        cooperative_groups::memcpy_async(cta, (&fbArgs.mainArr[linIdexMeta * fbArgs.mainArrSectionLength + 2*fbArgs.mainArrXLength]), (sharedForGold), (sizeof(uint32_t) * blockDim.x * blockDim.y) );
        cooperative_groups::memcpy_async(cta, (&fbArgs.mainArr[linIdexMeta * fbArgs.mainArrSectionLength + 3*fbArgs.mainArrXLength]), (sharedForSegm), (sizeof(uint32_t) * blockDim.x * blockDim.y) );

        cooperative_groups::memcpy_async(cta, (&fbArgs.mainArr[linIdexMeta * fbArgs.mainArrSectionLength + 4 * fbArgs.mainArrXLength]), (sharedForGold), (sizeof(uint32_t) * blockDim.x * blockDim.y));
        cooperative_groups::memcpy_async(cta, (&fbArgs.mainArr[linIdexMeta * fbArgs.mainArrSectionLength + 5 * fbArgs.mainArrXLength]), (sharedForSegm), (sizeof(uint32_t) * blockDim.x * blockDim.y));

        //// no need of synchronizations we are exportin data here only 


        /////adding the block and total number of the Fp's and Fn's 
        sumFp = reduce(tile, sumFp, plus<uint16_t>());
        sumFn = reduce(tile, sumFn, plus<uint16_t>());
        //reusing shared memory and adding accumulated values from tiles
        if (tile.thread_rank() == 0) {
            sharedForGold[tile.meta_group_rank()] = sumFp;
            sharedForSegm[tile.meta_group_rank()] = sumFn;
        }
        sync(cta);//waiting so shared memory will be loaded evrywhere
        //on single thread we do last sum reduction
        auto active = coalesced_threads();
        //gold
        if ((threadIdx.x == 0) && (threadIdx.y == 0) && isNotEmpty) {
            sharedForGold[33] = 0;//reset
            for (int i = 0; i < tile.meta_group_size(); i += 1) {
                sharedForGold[33] += sharedForGold[i];
            };
            fpSFnS[0] += sharedForGold[33];// will be needed later for global set
            fbArgs.mainArr[linIdexMeta * fbArgs.mainArrSectionLength + fbArgs.metaDataOffset + 1] = sharedForGold[33];

           // getTensorRow<unsigned int>(tensorslice, fbArgs.metaData.fpCount, fbArgs.metaData.fpCount.Ny, yMeta, zMeta)[xMeta] = sharedForGold[1][0];
        }
        //segm
       // if (isToBeExecutedOnActive(active, 1) && isNotEmpty) {
        if ((threadIdx.x == 0) && (threadIdx.y == 1) && isNotEmpty) {
            sharedForSegm[33] = 0;//reset
            for (int i = 0; i < tile.meta_group_size(); i += 1) {
                sharedForSegm[33] += sharedForSegm[i];
            };
            fpSFnS[1] += sharedForSegm[33];// will be needed later for global set
            //setting metadata
            fbArgs.mainArr[linIdexMeta * fbArgs.mainArrSectionLength+ fbArgs.metaDataOffset + 2] = sharedForSegm[33];

           // getTensorRow<unsigned int>(tensorslice, fbArgs.metaData.fnCount, fbArgs.metaData.fnCount.Ny, yMeta, zMeta)[xMeta] = sharedForSegm[1][0];

        }

        //marking as active 
//FP pass
        if ((threadIdx.x == 0) && (threadIdx.y == 0) && isNotEmpty && anyInGold[0]) { 
         //   printf("\n set activeee in gold xMeta %d yMeta %d  zMeta %d \n",xMeta,yMeta,zMeta);
           // getTensorRow<bool>(tensorslice, fbArgs.metaData.isActiveGold, fbArgs.metaData.isActiveGold.Ny, yMeta, zMeta)[xMeta] = true;
            fbArgs.mainArr[linIdexMeta * fbArgs.mainArrSectionLength + fbArgs.metaDataOffset + 8] = 1;

        };
        //FN pass
        if ((threadIdx.x == 1) && (threadIdx.y == 0) && isNotEmpty && anyInSegm[0]) {
         //   printf("\n set activeee in segm xMeta %d yMeta %d  zMeta %d \n", xMeta, yMeta, zMeta);
           // getTensorRow<bool>(tensorslice, fbArgs.metaData.isActiveSegm, fbArgs.metaData.isActiveSegm.Ny, yMeta, zMeta)[xMeta] = true;
            fbArgs.mainArr[linIdexMeta * fbArgs.mainArrSectionLength + fbArgs.metaDataOffset + 9] = 1;

        };


        //after we streamed over all block we save also information about indicies of the surrounding blocks - given they are in range if not UINT32_MAX will be saved 
        //top
        //setNeighbourBlocks(fbArgs, idX, inArrIndex, predicate, toAdd)

        setNeighbourBlocks(fbArgs, 3, 13, (zMeta > 0), (-(fbArgs.metaData.metaXLength * fbArgs.metaData.MetaYLength)), linIdexMeta);//top
        setNeighbourBlocks(fbArgs, 4, 14, (zMeta < (fbArgs.metaData.MetaZLength - 1)), (fbArgs.metaData.metaXLength* fbArgs.metaData.MetaYLength), linIdexMeta);//bottom

        setNeighbourBlocks(fbArgs, 6 ,15, (xMeta > 0), (-1), linIdexMeta);//left
        setNeighbourBlocks(fbArgs, 7, 16, (xMeta < (fbArgs.metaData.metaXLength - 1)), 1, linIdexMeta);//right

        setNeighbourBlocks(fbArgs, 8, 17, (yMeta < (fbArgs.metaData.MetaYLength - 1)), fbArgs.metaData.metaXLength, linIdexMeta);//anterior
        setNeighbourBlocks(fbArgs, 9, 18, (yMeta > 0), (-fbArgs.metaData.metaXLength), linIdexMeta);//posterior


        sync(cta); // just to reduce the warp divergence


    }
    sync(cta);


    //setting global fp and fn
    if ((threadIdx.x == 0) && (threadIdx.y == 0)) {
        atomicAdd(&(fbArgs.metaData.minMaxes[7]), fpSFnS[0]);
    };

    if ((threadIdx.x == 1) && (threadIdx.y == 0)) {
          atomicAdd(&(fbArgs.metaData.minMaxes[8]), fpSFnS[1]);

    };





}



/*
collecting all needed functions for GPU execution to prepare data from calculating Housedorff distance
*/
#pragma once
template <typename TYO>
__global__ void boolPrepareKernel(ForBoolKernelArgs<TYO> fbArgs) {
    metaDataIter(fbArgs);
}

