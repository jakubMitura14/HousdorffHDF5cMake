

#include "CPUAllocations.cu"
#include "MetaData.cu"
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
inline ForBoolKernelArgs<TCC> getArgsForKernel(ForFullBoolPrepArgs<TCC>& mainFunArgs
    , TCC*& goldArrPointer
    , TCC*& segmArrPointer
    , unsigned int* minMaxes
    , int& warpsNumbForMainPass, int& blockForMainPass
    , const int xLen, const int yLen, const int zLen
) {
    // size_t sizeMainArr = sizeof(TCC) * WIDTH * HEIGHT * DEPTH;
    //cudaMallocAsync(&goldArrPointer, sizeMainArr, 0);
    // cudaMallocAsync(&segmArrPointer, sizeMainArr, 0);



    // cudaMalloc(&goldArrPointer, sizeMainArr);
    // cudaMalloc(&segmArrPointer, sizeMainArr);

    // array3dWithDimsGPU<TCC> goldArr;
    // array3dWithDimsGPU<TCC> segmArr;

    // goldArr.arrP = goldArrPointer;
    // goldArr.Nx = WIDTH;
    // goldArr.Ny = HEIGHT;
    // goldArr.Nz = DEPTH;



    // segmArr.arrP = segmArrPointer;
    // segmArr.Nx = WIDTH;
    // segmArr.Ny = HEIGHT;
    // segmArr.Nz = DEPTH;


    ForBoolKernelArgs<TCC> res;
    res.metaData = allocateMetaDataOnGPU(mainFunArgs.metaData, minMaxes);
    //res.goldArr = goldArr;
    //res.segmArr = segmArr;

    res.numberToLookFor = mainFunArgs.numberToLookFor;
    res.dbXLength = 32;
    res.dbYLength = warpsNumbForMainPass;
    res.dbZLength = 32;

    printf("in setting bool args ylen %d dbYlen %d calculated meta %d  \n ", yLen, res.dbYLength, int(ceil(yLen / res.dbYLength)));
    res.metaData.metaXLength = int(ceil(xLen / res.dbXLength));
    res.metaData.MetaYLength = int(ceil(yLen / res.dbYLength));;
    res.metaData.MetaZLength = int(ceil(zLen / res.dbZLength));;
    res.metaData.minX = 0;
    res.metaData.minY = 0;
    res.metaData.minZ = 0;
    res.metaData.maxX = res.metaData.metaXLength;
    res.metaData.maxY = res.metaData.MetaYLength;
    res.metaData.maxZ = res.metaData.MetaZLength;


    res.metaData.totalMetaLength = res.metaData.metaXLength * res.metaData.MetaYLength * res.metaData.MetaZLength;


    return res;
}


/*
setting the linear index of metadata blocks that are in given direction if there is no such (out of range) we will save it as UINT32_MAX
*/
template <typename TCC>
__device__ inline void setNeighbourBlocks(ForBoolKernelArgs<TCC> fbArgs
    , uint8_t idX, uint8_t inArrIndex, bool predicate, uint32_t toAdd
    , uint32_t linIdexMeta, MetaDataGPU metaData, uint32_t localBlockMetaData[20]) {

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
template <typename TYO>
__global__ void boolPrepareKernel(ForBoolKernelArgs<TYO> fbArgs
    , MetaDataGPU metaData, uint32_t* origArrs, uint32_t* metaDataArr, TYO* goldArr, TYO* segmArr, unsigned int* minMaxes) {

    ////////////some initializations
    bool goldBool = false;
    bool segmBool = false;
    bool isNotEmpty = false;

    thread_block cta = this_thread_block();
    thread_block_tile<32> tile = tiled_partition<32>(cta);
    uint32_t sumFp = 0;
    uint32_t sumFn = 0;

    //auto pipeline = cuda::make_pipeline();


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
    if ((threadIdx.x == 4) && (threadIdx.y == 1)) {
        anyInSegm[1] = false;

        if (blockIdx.x == 0) {
            printf("in bool kernel  dims meta in bool kernel Meta X %d MetaY %d metaZ %d dbXSize %d dbYsize %d dbZsize %d minX %d minY %d minZ \n "
                , metaData.metaXLength, metaData.MetaYLength, metaData.MetaZLength
                , fbArgs.dbXLength, fbArgs.dbYLength, fbArgs.dbZLength
                , metaData.minX, metaData.minY, metaData.minZ
            );
        }

    };



    sync(cta);

    /////////////////////////


    //main metadata iteration
    for (uint32_t linIdexMeta = blockIdx.x; linIdexMeta < metaData.totalMetaLength; linIdexMeta += gridDim.x) {
        //we get from linear index  the coordinates of the metadata block of intrest
        int xMeta = int(linIdexMeta % (metaData.metaXLength));
        int zMeta = int(floor((float)(linIdexMeta / (metaData.metaXLength * metaData.MetaYLength))));
        int yMeta = int(floor((float)((linIdexMeta - ((zMeta * metaData.metaXLength * metaData.MetaYLength) + xMeta)) / metaData.metaXLength)));
        //reset
        isNotEmpty = false;
        sumFp = 0;
        sumFn = 0;
        anyInGold[0] = false;
        anyInSegm[0] = false;
        //iterating over data block
        sync(cta);
        for (uint8_t xLoc = threadIdx.x; xLoc < fbArgs.dbXLength; xLoc += blockDim.x) {
            uint32_t x = (xMeta + metaData.minX) * fbArgs.dbXLength + xLoc;//absolute position
            for (uint8_t yLoc = threadIdx.y; yLoc < fbArgs.dbYLength; yLoc += blockDim.y) {
                uint32_t  y = (yMeta + metaData.minY) * fbArgs.dbYLength + yLoc;//absolute position
                if (y < fbArgs.goldArr.Ny && x < fbArgs.goldArr.Nx) {

                    // resetting 
                    sharedForGold[xLoc + yLoc * fbArgs.dbXLength] = 0;
                    sharedForSegm[xLoc + yLoc * fbArgs.dbXLength] = 0;


                    for (uint8_t zLoc = 0; zLoc < fbArgs.dbZLength; zLoc++) {
                        uint32_t z = (zMeta + metaData.minZ) * fbArgs.dbZLength + zLoc;//absolute position
                        if (z < fbArgs.goldArr.Nz) {
                            //char* tensorslice;

                            //first array gold
                            bool goldBool = goldArr[x + y * fbArgs.goldArr.Nx + z * fbArgs.goldArr.Nx * fbArgs.goldArr.Ny] == fbArgs.numberToLookFor;
                            bool segmBool = segmArr[x + y * fbArgs.segmArr.Nx + z * fbArgs.segmArr.Nx * fbArgs.segmArr.Ny] == fbArgs.numberToLookFor;
                            //goldBool = true;

                            // setting bits
                            sharedForGold[xLoc + yLoc * fbArgs.dbXLength] |= goldBool << zLoc;
                            sharedForSegm[xLoc + yLoc * fbArgs.dbXLength] |= segmBool << zLoc;
                            // setting value of local boolean marking that any of the entries was evaluated to true in either of arrays
                            isNotEmpty = (isNotEmpty || (goldBool || segmBool));
                            sumFp += (!goldBool && segmBool);
                            sumFn += (goldBool && !segmBool);
                            if (goldBool)  anyInGold[0] = true;
                            if (segmBool)  anyInSegm[0] = true;

                            //if (goldBool) {
                            //    printf("in kernel  gold x %d y %d z %d    xMeta %d yMeta %d zMeta %d counted ymeta %d linmeta %d \n", x, y, z, xMeta, yMeta, zMeta
                            //        , int(floor((float)((linIdexMeta - ((zMeta * metaData.metaXLength * metaData.MetaYLength) + xMeta)) / metaData.metaXLength)))
                            //        , linIdexMeta);
                            //}

                            //if (segmBool) {
                            //    printf("in kernel  segm  x %d y %d z %d    xMeta %d yMeta %d zMeta %d counted ymeta %d linmeta %d \n", x, y, z, xMeta, yMeta, zMeta
                            //        , int(floor((float)((linIdexMeta - ((zMeta * metaData.metaXLength * metaData.MetaYLength) + xMeta)) / metaData.metaXLength)))
                            //        , linIdexMeta);
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
        if ((threadIdx.x < 20) && (threadIdx.y == 0)) {
            localBlockMetaData[threadIdx.x] = 0;
        }



        isNotEmpty = __syncthreads_or(isNotEmpty);
        //exporting to global memory
        for (uint8_t xLoc = threadIdx.x; xLoc < fbArgs.dbXLength; xLoc += blockDim.x) {
            uint32_t x = (xMeta + metaData.minX) * fbArgs.dbXLength + xLoc;//absolute position
            for (uint8_t yLoc = threadIdx.y; yLoc < fbArgs.dbYLength; yLoc += blockDim.y) {
                uint32_t  y = (yMeta + metaData.minY) * fbArgs.dbYLength + yLoc;//absolute position
                if (y < fbArgs.goldArr.Ny && x < fbArgs.goldArr.Nx) {

                    origArrs[linIdexMeta * metaData.mainArrSectionLength + yLoc * 32 + xLoc] = sharedForGold[yLoc * 32 + xLoc];
                    origArrs[linIdexMeta * metaData.mainArrSectionLength + yLoc * 32 + xLoc + metaData.mainArrXLength] = sharedForSegm[yLoc * 32 + xLoc];


                }
            }
        }

        //   sync(cta);



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

        }

        //marking as active 
//FP pass
        if ((threadIdx.x == 0) && (threadIdx.y == 0) && isNotEmpty && anyInGold[0]) {
            localBlockMetaData[7] = 1;

        };
        //FN pass
        if ((threadIdx.x == 1) && (threadIdx.y == 0) && isNotEmpty && anyInSegm[0]) {
            localBlockMetaData[9] = 1;

        };


        //after we streamed over all block we save also information about indicies of the surrounding blocks - given they are in range if not UINT32_MAX will be saved 
        //top



        setNeighbourBlocks(fbArgs, 3, 13, (zMeta > 0), (-(metaData.metaXLength * metaData.MetaYLength)), linIdexMeta, metaData, localBlockMetaData);//top
        setNeighbourBlocks(fbArgs, 4, 14, (zMeta < (metaData.MetaZLength - 1)), (metaData.metaXLength * metaData.MetaYLength), linIdexMeta, metaData, localBlockMetaData);//bottom

        setNeighbourBlocks(fbArgs, 6, 15, (xMeta > 0), (-1), linIdexMeta, metaData, localBlockMetaData);//left
        setNeighbourBlocks(fbArgs, 7, 16, (xMeta < (metaData.metaXLength - 1)), 1, linIdexMeta, metaData, localBlockMetaData);//right

        setNeighbourBlocks(fbArgs, 8, 17, (yMeta < (metaData.MetaYLength - 1)), metaData.metaXLength, linIdexMeta, metaData, localBlockMetaData);//anterior
        setNeighbourBlocks(fbArgs, 9, 18, (yMeta > 0), (-metaData.metaXLength), linIdexMeta, metaData, localBlockMetaData);//posterior

        if ((threadIdx.x < 20) && (threadIdx.y == 0)) {
            metaDataArr[linIdexMeta * metaData.metaDataSectionLength + threadIdx.x] = localBlockMetaData[threadIdx.x];
        };

        sync(cta); // just to reduce the warp divergence

        // copy metadata to global memory

        //cuda::memcpy_async(cta, &metaDataArr[linIdexMeta * metaData.metaDataSectionLength], (&localBlockMetaData[0]), (sizeof(uint32_t) * 20), barrier);
       // barrier.arrive_and_wait(); // Waits for all copies to complete

    }



    sync(cta);


    //setting global fp and fn
    if ((threadIdx.x == 0) && (threadIdx.y == 0)) {
        atomicAdd(&(minMaxes[7]), fpSFnS[0]);
    };

    if ((threadIdx.x == 1) && (threadIdx.y == 0)) {
        atomicAdd(&(minMaxes[8]), fpSFnS[1]);

    };




}



///*
//collecting all needed functions for GPU execution to prepare data from calculating Housedorff distance
//*/
//#pragma once
//template <typename TYO>
//__global__ void boolPrepareKernel(ForBoolKernelArgs<TYO> fbArgs
//    , MetaDataGPU metaData, uint32_t* origArrs, uint32_t* metaDataArr) {
//    metaDataIter(fbArgs,  metaData, origArrs, metaDataArr);
//}

