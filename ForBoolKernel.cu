

#include "CPUAllocations.cu"
#include "MetaData.cu"
#include "IterationUtils.cu"
#include "ExceptionManagUtils.cu"
#include "CooperativeGroupsUtils.cu"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cooperative_groups/memcpy_async.h>
#include <cuda/barrier>

using namespace cooperative_groups;

/*
given appropriate cudaPitchedPtr and ForFullBoolPrepArgs will return ForBoolKernelArgs
*/
#pragma once
template <typename TCC>
inline ForBoolKernelArgs<TCC> getArgsForKernel(ForFullBoolPrepArgs<TCC> mainFunArgs, array3dWithDimsGPU forDebugArr
    , array3dWithDimsGPU goldArr
    , array3dWithDimsGPU segmArr
    ,unsigned int*& minMaxes
    ,int warpsNumbForMainPass,int blockForMainPass
) {

    ForBoolKernelArgs<TCC> res;
    res.metaData = allocateMetaDataOnGPU(mainFunArgs.metaData, minMaxes);
    res.forDebugArr = forDebugArr;
    res.goldArr = goldArr;
    res.segmArr = segmArr;

    res.numberToLookFor = mainFunArgs.numberToLookFor;
    res.dbXLength = 32;
    res.dbYLength = warpsNumbForMainPass;
    res.dbZLength = 32;



    return res;
}


/*
setting the linear index of metadata blocks that are in given direction if there is no such (out of range) we will save it as UINT32_MAX
*/
template <typename TCC>
__device__ inline void setNeighbourBlocks(ForBoolKernelArgs<TCC> fbArgs,uint8_t idX, uint8_t inArrIndex, bool predicate, uint32_t toAdd
    , uint32_t linIdexMeta , MetaDataGPU metaData, uint32_t localBlockMetaData[20]) {

    if ((threadIdx.x == idX) && (threadIdx.y == 0)) {
        if (predicate) {
            localBlockMetaData[inArrIndex] = (linIdexMeta + toAdd);
        }
        else {
            localBlockMetaData[inArrIndex] = isGoldOffset;
        }
    };
}





/*
iteration over metadata - becouse metadata may be small and to maximize occupancy we use linear index and then clalculate xMeta,ymeta,zMeta from this linear index ...
*/
#pragma once
template <typename TYU>
__device__ void metaDataIter(ForBoolKernelArgs<TYU> fbArgs
    , MetaDataGPU metaData, uint32_t* origArrs, uint32_t* metaDataArr) {

    ////////////some initializations
    bool goldBool = false;
    bool segmBool = false;
    bool isNotEmpty = false;
  
    thread_block cta = this_thread_block();
    thread_block_tile<32> tile = tiled_partition<32>(cta);
    uint32_t sumFp = 0;
    uint32_t sumFn = 0;
   
    auto pipeline = cuda::make_pipeline();


    //shared memory

    //TODO() make it dynamically sized 
    __shared__ uint32_t sharedForGold[1024];
    __shared__ uint32_t sharedForSegm[1024];


    //for storing fp and fn sums to later accumulate it to global values
    __shared__ uint32_t fpSFnS[2];
    __shared__ uint32_t localBlockMetaData[20];

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
    __shared__ cuda::barrier<cuda::thread_scope::thread_scope_block> barrier;
    if (cta.thread_rank() == 0) {
        init(&barrier, cta.size()); // Friend function initializes barrier
    }


    sync(cta);

    /////////////////////////


    //main metadata iteration
    for (uint32_t linIdexMeta = blockIdx.x; linIdexMeta < metaData.totalMetaLength; linIdexMeta += gridDim.x) {
        //we get from linear index  the coordinates of the metadata block of intrest
        uint8_t xMeta = linIdexMeta % metaData.metaXLength;
        uint8_t zMeta = floor((float)(linIdexMeta / (metaData.metaXLength * metaData.MetaYLength)));
        uint8_t yMeta = floor((float)((linIdexMeta - ((zMeta * metaData.metaXLength * metaData.MetaYLength) + xMeta)) / metaData.metaXLength));
        //reset
        isNotEmpty = false;
        sumFp = 0;
        sumFn = 0;
        anyInGold[0] = false;
        anyInSegm[0] = false;
        //iterating over data block
        sync(cta);
        for (uint8_t xLoc = threadIdx.x; xLoc < fbArgs.dbXLength; xLoc += blockDim.x) {
            uint32_t x = (xMeta+ metaData.minX)* fbArgs.dbXLength + xLoc;//absolute position
            for (uint8_t yLoc = threadIdx.y; yLoc < fbArgs.dbYLength; yLoc += blockDim.y) {
                uint32_t  y = (yMeta+ metaData.minY) * fbArgs.dbYLength + yLoc;//absolute position
                if (y < fbArgs.goldArr.Ny && x < fbArgs.goldArr.Nz) {

                    // resetting 
                    sharedForGold[xLoc + yLoc * fbArgs.dbXLength] = 0;
                    sharedForSegm[xLoc + yLoc * fbArgs.dbXLength] = 0;
        

                    for (uint8_t zLoc = 0; zLoc < fbArgs.dbZLength; zLoc++) {
                        uint32_t z = (zMeta+ metaData.minZ)* fbArgs.dbZLength + zLoc;//absolute position
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
                           
                            //if (goldBool) {
                            //    printf("in kernel  gold x %d y %d z %d linearLocal %d linIdexMeta %d\n", x, y, z, xLoc + yLoc * fbArgs.dbXLength, linIdexMeta);
                            //}

                            //if (segmBool) {
                            //    printf("in kernel  segm  x %d y %d z %d linearLocal %d linIdexMeta %d\n", x, y, z, xLoc + yLoc * fbArgs.dbXLength, linIdexMeta);
                            //}


                        }
                    }
                }

                //if (sharedForGold[xLoc + yLoc * fbArgs.dbXLength] > 0) {
                //    printf("in kernel Metax %d yMeta %d zMeta %d linearLocal %d linIdexMeta %d column %d \n"
                //        , xMeta, yMeta, zMeta,  xLoc + yLoc * fbArgs.dbXLength, linIdexMeta
                //    , sharedForGold[xLoc + yLoc * fbArgs.dbXLength]);
                //}


            }
        }
        //reset local metadata
        if ((threadIdx.x <20) && (threadIdx.y == 0)) {
            localBlockMetaData[threadIdx.x]=0;
        }
        

    
        isNotEmpty = __syncthreads_or(isNotEmpty);
        //exporting to global memory
        for (uint8_t xLoc = threadIdx.x; xLoc < fbArgs.dbXLength; xLoc += blockDim.x) {
            uint32_t x = (xMeta + metaData.minX) * fbArgs.dbXLength + xLoc;//absolute position
            for (uint8_t yLoc = threadIdx.y; yLoc < fbArgs.dbYLength; yLoc += blockDim.y) {
                uint32_t  y = (yMeta + metaData.minY) * fbArgs.dbYLength + yLoc;//absolute position
                if (y < fbArgs.goldArr.Ny && x < fbArgs.goldArr.Nz) {
                    origArrs[linIdexMeta * metaData.mainArrSectionLength + yLoc * 32 + xLoc] = sharedForGold[yLoc * 32 + xLoc];
                    origArrs[linIdexMeta * metaData.mainArrSectionLength + yLoc * 32 + xLoc + metaData.mainArrXLength] = sharedForSegm[yLoc * 32 + xLoc];


                }
            }
        }

     //   sync(cta);

        //copy data to global memory from shmem

        //mainArr[linIdexMeta * metaData.mainArrSectionLength + threadIdx.x + threadIdx.y * metaData.metaXLength] = sharedForGold[threadIdx.x + threadIdx.y * metaData.metaXLength];
        //cooperative_groups::memcpy_async(cta, (mainArr), (sharedForGold), (sizeof(uint32_t) *2) );
       

        //cuda::memcpy_async(cta, (&origArrs[linIdexMeta * metaData.mainArrSectionLength]) , (sharedForGold), sizeof(uint32_t) * cta.size(), barrier);
        //barrier.arrive_and_wait(); // Waits for all copies to complete

    
       // cuda::memcpy_async(cta, (&origArrs[linIdexMeta * metaData.mainArrSectionLength]), (sharedForGoldB), (sizeof(uint32_t) * blockDim.x * blockDim.y), barrier);
       //barrier.arrive_and_wait(); // Waits for all copies to complete

       //cuda::memcpy_async(cta, (&origArrs[linIdexMeta * metaData.mainArrSectionLength + metaData.mainArrXLength]), (sharedForSegmB), (sizeof(uint32_t) * blockDim.x * blockDim.y), barrier);
       //barrier.arrive_and_wait(); // Waits for all copies to complete

       //cuda::memcpy_async(cta, (&mainArr[linIdexMeta * metaData.mainArrSectionLength ]), (sharedForGoldB), (sizeof(uint32_t) * blockDim.x * blockDim.y), barrier);
       // barrier.arrive_and_wait(); // Waits for all copies to complete

       // cuda::memcpy_async(cta, (&mainArr[linIdexMeta * metaData.mainArrSectionLength + metaData.mainArrXLength*1]), (sharedForSegmB), (sizeof(uint32_t) * blockDim.x * blockDim.y) , barrier);
       // barrier.arrive_and_wait(); // Waits for all copies to complete

       // cuda::memcpy_async(cta, (&mainArr[linIdexMeta * metaData.mainArrSectionLength + metaData.mainArrXLength*2]), (sharedForGoldB), (sizeof(uint32_t) * blockDim.x * blockDim.y), barrier);
       // barrier.arrive_and_wait(); // Waits for all copies to complete

       // cuda::memcpy_async(cta, (&mainArr[linIdexMeta * metaData.mainArrSectionLength + metaData.mainArrXLength*3]), (sharedForSegmB), (sizeof(uint32_t) * blockDim.x * blockDim.y), barrier);
       // barrier.arrive_and_wait(); // Waits for all copies to complete

       sync(cta);



        /////adding the block and total number of the Fp's and Fn's 
        sumFp = reduce(tile, sumFp, plus<uint32_t>());
        sumFn = reduce(tile, sumFn, plus<uint32_t>());
        //reusing shared memory and adding accumulated values from tiles
        if (tile.thread_rank() == 0) {
            sharedForGold[tile.meta_group_rank()] = sumFp;
            sharedForSegm[tile.meta_group_rank()] = sumFn;
        }
        sync(cta);//waiting so shared memory will be loaded evrywhere
        //on single thread we do last sum reduction
        auto active = coalesced_threads();

        //if ((threadIdx.x == 0) && (threadIdx.y == 0)) {
        //    printf("xMeta %d yMeta %d zMeta %d \n", xMeta, yMeta, zMeta);
        //}

        if ((threadIdx.x == 0) && (threadIdx.y == 0) && isNotEmpty) {
            sharedForGold[33] = 0;//reset
            for (int i = 0; i < tile.meta_group_size(); i += 1) {
                sharedForGold[33] += sharedForGold[i];
 /*               if (sharedForGold[i]>0) {
                    printf("adding sharedForGold[i] %d in gold \n ", sharedForGold[i]);
                }*/

            };
            fpSFnS[0] += sharedForGold[33];// will be needed later for global set
            //metaDataArr[linIdexMeta * metaData.metaDataSectionLength + 1] = sharedForGold[33];
            localBlockMetaData[1] = sharedForGold[33];

           // getTensorRow<unsigned int>(tensorslice, metaData.fpCount, metaData.fpCount.Ny, yMeta, zMeta)[xMeta] = sharedForGold[1][0];
        }
       // if (isToBeExecutedOnActive(active, 1) && isNotEmpty) {
        if ((threadIdx.x == 0) && (threadIdx.y == 1) && isNotEmpty) {


            sharedForSegm[33] = 0;//reset
            for (int i = 0; i < tile.meta_group_size(); i += 1) {
                sharedForSegm[33] += sharedForSegm[i];
            };
            fpSFnS[1] += sharedForSegm[33];// will be needed later for global set
            //setting metadata
            localBlockMetaData[2] = sharedForSegm[33];


           // getTensorRow<unsigned int>(tensorslice, metaData.fnCount, metaData.fnCount.Ny, yMeta, zMeta)[xMeta] = sharedForSegm[1][0];

        }

        //marking as active 
//FP pass
        if ((threadIdx.x == 0) && (threadIdx.y == 0) && isNotEmpty && anyInGold[0]) { 
            localBlockMetaData[7] = 1;
           // printf("in bool kernel mark fp as sctive linIdexMeta %d in index  %d \n  ", linIdexMeta);

        };
        //FN pass
        if ((threadIdx.x == 1) && (threadIdx.y == 0) && isNotEmpty && anyInSegm[0]) {
            //printf("in bool kernel mark fn as sctive linIdexMeta %d in index  %d \n  ", linIdexMeta);
            localBlockMetaData[9] = 1;

        };


        //after we streamed over all block we save also information about indicies of the surrounding blocks - given they are in range if not UINT32_MAX will be saved 
        //top



        setNeighbourBlocks(fbArgs, 3, 13, (zMeta > 0), (-(metaData.metaXLength * metaData.MetaYLength)), linIdexMeta, metaData, localBlockMetaData);//top
        setNeighbourBlocks(fbArgs, 4, 14, (zMeta < (metaData.MetaZLength - 1)), (metaData.metaXLength* metaData.MetaYLength), linIdexMeta, metaData, localBlockMetaData);//bottom

        setNeighbourBlocks(fbArgs, 6 ,15, (xMeta > 0), (-1), linIdexMeta, metaData, localBlockMetaData);//left
        setNeighbourBlocks(fbArgs, 7, 16, (xMeta < (metaData.metaXLength - 1)), 1, linIdexMeta, metaData, localBlockMetaData);//right

        setNeighbourBlocks(fbArgs, 8, 17, (yMeta < (metaData.MetaYLength - 1)), metaData.metaXLength, linIdexMeta, metaData, localBlockMetaData);//anterior
        setNeighbourBlocks(fbArgs, 9, 18, (yMeta > 0), (-metaData.metaXLength), linIdexMeta, metaData, localBlockMetaData);//posterior

  if ((threadIdx.x <20) && (threadIdx.y == 0)) {
metaDataArr[linIdexMeta * metaData.metaDataSectionLength+ threadIdx.x]= localBlockMetaData[threadIdx.x];
    };

        sync(cta); // just to reduce the warp divergence
        
        // copy metadata to global memory

        //cuda::memcpy_async(cta, &metaDataArr[linIdexMeta * metaData.metaDataSectionLength], (&localBlockMetaData[0]), (sizeof(uint32_t) * 20), barrier);
       // barrier.arrive_and_wait(); // Waits for all copies to complete

    }
    sync(cta);


    //setting global fp and fn
    if ((threadIdx.x == 0) && (threadIdx.y == 0)) {
      /*  printf("metaData.totalMetaLength %d metaData.mainArrSectionLength %d metaData.metaXLength %d \n"
            , metaData.totalMetaLength, metaData.mainArrSectionLength, metaData.metaXLength);*/

        atomicAdd(&(metaData.minMaxes[7]), fpSFnS[0]);
    };

    if ((threadIdx.x == 1) && (threadIdx.y == 0)) {
          atomicAdd(&(metaData.minMaxes[8]), fpSFnS[1]);

    };
   



}



/*
collecting all needed functions for GPU execution to prepare data from calculating Housedorff distance
*/
#pragma once
template <typename TYO>
__global__ void boolPrepareKernel(ForBoolKernelArgs<TYO> fbArgs
    , MetaDataGPU metaData, uint32_t* origArrs, uint32_t* metaDataArr) {
    metaDataIter(fbArgs,  metaData, origArrs, metaDataArr);
}

