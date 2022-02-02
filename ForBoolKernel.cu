

#include "CPUAllocations.cu"
#include "MetaData.cu"
#include "IterationUtils.cu"
#include "ExceptionManagUtils.cu"
#include "CooperativeGroupsUtils.cu"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
using namespace cooperative_groups;

/*
given appropriate cudaPitchedPtr and ForFullBoolPrepArgs will return ForBoolKernelArgs
*/
#pragma once
template <typename TCC>
inline ForBoolKernelArgs<TCC> getArgsForKernel(ForFullBoolPrepArgs<int> mainFunArgs, array3dWithDimsGPU forDebugArr
    , array3dWithDimsGPU goldArr
    , array3dWithDimsGPU segmArr
    //, array3dWithDimsGPU reducedGold
    //, array3dWithDimsGPU reducedSegm
    //, array3dWithDimsGPU reducedGoldRef
    //, array3dWithDimsGPU reducedSegmRef
    //, array3dWithDimsGPU reducedGoldPrev
    //, array3dWithDimsGPU reducedSegmPrev

) {

    ForBoolKernelArgs<TCC> res;
    res.metaData = allocateMetaDataOnGPU(mainFunArgs.metaData);
    res.forDebugArr = forDebugArr;
    res.goldArr = goldArr;
    res.segmArr = segmArr;
    //allocate the reduced arrays
    //res.reducedGold = reducedGold;
    //res.reducedSegm = reducedSegm;
    //res.reducedGoldPrev = reducedGoldPrev;
    //res.reducedSegmPrev = reducedSegmPrev;

    //res.reducedGoldRef = reducedGoldRef;
    //res.reducedSegmRef = reducedSegmRef;

    res.numberToLookFor = mainFunArgs.numberToLookFor;
    res.dbXLength = mainFunArgs.dbXLength;
    res.dbYLength = mainFunArgs.dbYLength;
    res.dbZLength = mainFunArgs.dbZLength;



    return res;
}





// helper functions and utilities to work with CUDA from https://github.com/NVIDIA/cuda-samples


/*
iteration over data block given metadata coordinates
*/
#pragma once
template <typename TPI>
__device__ void fillReduCedArr(ForBoolKernelArgs<TPI> fbArgs,
    int minMaxesInShmem[6], uint16_t& sumFp, uint16_t& sumFn,
    bool& isNotEmpty, uint8_t xMeta, uint8_t yMeta, uint8_t zMeta, char* tensorslice
    , uint16_t& z, uint16_t& y, uint16_t& x, bool& goldBool, bool& segmBool
    , uint32_t sharedForGold[32][32], uint32_t sharedForSegm[32][32]
    , uint8_t& xLoc, uint8_t& yLoc, uint8_t& zLoc
    , bool anyInGold[1], bool anyInSegm[1]) {

    //setting  bits for reduced representation 
    sharedForGold[xLoc][yLoc] |= goldBool << zLoc;
    sharedForSegm[xLoc][yLoc] |= segmBool << zLoc;
    // setting value of local boolean marking that any of the entries was evaluated to true in either of arrays
    isNotEmpty = (isNotEmpty || (goldBool || segmBool));
    sumFp += (!goldBool && segmBool);
    sumFn += (goldBool && !segmBool);
    if (goldBool)  anyInGold[0] = true;
    if (segmBool)  anyInSegm[0] = true;


    //if (goldBool && !segmBool) {
    //    printf("nnnnnnnnnnnn  fn x %d y %d z %d    xMeta [%d] yMeta [%d] zMeta [%d]  \n", x, y, z, xMeta, yMeta, zMeta);
    //}
    //if (!goldBool && segmBool) {
    //    printf("pppppppp  fp x %d y %d z %d    xMeta [%d] yMeta [%d] zMeta [%d]  \n", x, y, z, xMeta, yMeta, zMeta);
    //}

}


/*
iteration over data block given metadata coordinates
*/

#pragma once
template <typename TYI>
__device__ void dataBlockIter(ForBoolKernelArgs<TYI> fbArgs, thread_block cta, thread_block_tile<32> tile,
    int minMaxesInShmem[6], uint16_t& sumFp, uint16_t& sumFn,
    bool& isNotEmpty, uint8_t xMeta, uint8_t yMeta, uint8_t zMeta, char* tensorslice
    , uint16_t& z, uint16_t& y, uint16_t& x, bool& goldBool, bool& segmBool
    , uint32_t sharedForGold[32][32], uint32_t sharedForSegm[32][32], uint32_t fpSFnS[2]
    , bool anyInGold[1], bool anyInSegm[1]
) {
    //now we need to iterate over the data in the data block voxel by voxel
    for (uint8_t xLoc = threadIdx.x; xLoc < fbArgs.dbXLength; xLoc += blockDim.x) {
        x = xMeta * fbArgs.dbXLength + xLoc;//absolute position
        for (uint8_t yLoc = threadIdx.y; yLoc < fbArgs.dbYLength; yLoc += blockDim.y) {
            y = yMeta * fbArgs.dbYLength + yLoc;//absolute position
            if (y < fbArgs.goldArr.Ny && x < fbArgs.goldArr.Nz) {

                // resetting 
                sharedForGold[xLoc][yLoc] = 0;
                sharedForSegm[xLoc][yLoc] = 0;
                isNotEmpty = false;
                sumFp = 0;
                sumFn = 0;
                anyInGold[0] = false;
                anyInSegm[0] = false;

                for (uint8_t zLoc = 0; zLoc < fbArgs.dbZLength; zLoc++) {
                    z = zMeta * fbArgs.dbZLength + zLoc;//absolute position
                    if (z < fbArgs.goldArr.Nx) {
                        //first array gold
                        uint8_t& zLocRef = zLoc; uint8_t& yLocRef = yLoc; uint8_t& xLocRef = xLoc;

                        goldBool = (getTensorRow<TYI>(tensorslice, fbArgs.goldArr, fbArgs.goldArr.Ny, y, z)[x] == fbArgs.numberToLookFor);

                        // now segmentation  array
                        segmBool = (getTensorRow<TYI>(tensorslice, fbArgs.segmArr, fbArgs.goldArr.Ny, y, z)[x] == fbArgs.numberToLookFor);
                        // setting bits
                        fillReduCedArr(fbArgs, minMaxesInShmem, sumFp, sumFn, isNotEmpty, xMeta, yMeta, zMeta
                            , tensorslice, z, y, x, goldBool, segmBool, sharedForGold, sharedForSegm
                            , xLocRef, yLocRef, zLocRef, anyInGold, anyInSegm);
                    }
                }
                //after we streamed over all z layers we need to save it into reduced representation arrays
                getTensorRow<uint32_t>(tensorslice, fbArgs.reducedGold, fbArgs.reducedGold.Ny, y, zMeta)[x] = sharedForGold[xLoc][yLoc];
                getTensorRow<uint32_t>(tensorslice, fbArgs.reducedSegm, fbArgs.reducedSegm.Ny, y, zMeta)[x] = sharedForSegm[xLoc][yLoc];
                // TODO() establish is it faster that way or better at the end do mempcy async
                //getTensorRow<uint32_t>(tensorslice, fbArgs.reducedGoldRef, fbArgs.reducedGoldRef.Ny, y, zMeta)[x] = sharedForGold[xLoc][yLoc];
                //getTensorRow<uint32_t>(tensorslice, fbArgs.reducedSegmRef, fbArgs.reducedSegmRef.Ny, y, zMeta)[x] = sharedForSegm[xLoc][yLoc];

                //getTensorRow<uint32_t>(tensorslice, fbArgs.reducedGoldPrev, fbArgs.reducedGoldPrev.Ny, y, zMeta)[x] = sharedForGold[xLoc][yLoc];
                //getTensorRow<uint32_t>(tensorslice, fbArgs.reducedSegmPrev, fbArgs.reducedSegmPrev.Ny, y, zMeta)[x] = sharedForSegm[xLoc][yLoc];
                //we establish wheather this block is not empty if it is not - we will mark it as active
                isNotEmpty = __syncthreads_or(isNotEmpty);


                /////adding the block and total number of the Fp's and Fn's 
                sumFp = reduce(tile, sumFp, plus<uint16_t>());
                sumFn = reduce(tile, sumFn, plus<uint16_t>());
                //reusing shared memory and adding accumulated values from tiles
                if (tile.thread_rank() == 0) {
                    sharedForGold[0][tile.meta_group_rank()] = sumFp;
                    sharedForSegm[0][tile.meta_group_rank()] = sumFn;
                }
                sync(cta);//waiting so shared memory will be loaded evrywhere
                //on single thread we do last sum reduction
                auto active = coalesced_threads();
                //gold
                if ((threadIdx.x == 0) && (threadIdx.y == 0) && isNotEmpty) {
                    //if (isToBeExecutedOnActive(active, 0) && isNotEmpty) {
                    sharedForGold[1][0] = 0;//reset
                    for (int i = 0; i < tile.meta_group_size(); i += 1) {
                        sharedForGold[1][0] += sharedForGold[0][i];
                    };
                    fpSFnS[0] += sharedForGold[1][0];// will be needed later for global set
                   // printf("adding fps %d  xMeta [%d] yMeta [%d] zMeta [%d]  \n", sharedForGold[1][0], xMeta, yMeta, zMeta);

               //     printf("locMeta x %d y %d z %d fp %d \n ", xMeta, yMeta, zMeta, sharedForGold[1][0]);

                    //setting metadata
                    //printf("\n in bool kernel fp count  x %d y %d z %d fp %d \n ", xMeta, yMeta, zMeta, sharedForGold[1][0]);

                    getTensorRow<unsigned int>(tensorslice, fbArgs.metaData.fpCount, fbArgs.metaData.fpCount.Ny, yMeta, zMeta)[xMeta] = sharedForGold[1][0];
                }
                //segm
               // if (isToBeExecutedOnActive(active, 1) && isNotEmpty) {
                if ((threadIdx.x == 0) && (threadIdx.y == 1) && isNotEmpty) {
                    sharedForSegm[1][0] = 0;//reset
                    for (int i = 0; i < tile.meta_group_size(); i += 1) {
                        sharedForSegm[1][0] += sharedForSegm[0][i];
                    };
                    fpSFnS[1] += sharedForSegm[1][0];// will be needed later for global set
                    //setting metadata

                   // printf("\n in bool kernel fn count  x %d y %d z %d fn %d \n " , xMeta, yMeta, zMeta, sharedForSegm[1][0]);
                    getTensorRow<unsigned int>(tensorslice, fbArgs.metaData.fnCount, fbArgs.metaData.fnCount.Ny, yMeta, zMeta)[xMeta] = sharedForSegm[1][0];

                }
                /////////////////// setting min and maxes
//    //1)maxX 2)minX 3)maxY 4) minY 5) maxZ 6) minZ
                if (isToBeExecutedOnActive(active, 2) && isNotEmpty) { minMaxesInShmem[1] = max(xMeta, minMaxesInShmem[1]); };
                if (isToBeExecutedOnActive(active, 3) && isNotEmpty) { minMaxesInShmem[2] = min(xMeta, minMaxesInShmem[2]); };

                if (isToBeExecutedOnActive(active, 4) && isNotEmpty) { minMaxesInShmem[3] = max(yMeta, minMaxesInShmem[3]); };
                if (isToBeExecutedOnActive(active, 5) && isNotEmpty) { minMaxesInShmem[4] = min(yMeta, minMaxesInShmem[4]); };

                if (isToBeExecutedOnActive(active, 6) && isNotEmpty) { minMaxesInShmem[5] = max(zMeta, minMaxesInShmem[5]); };
                if (isToBeExecutedOnActive(active, 7) && isNotEmpty) { minMaxesInShmem[6] = min(zMeta, minMaxesInShmem[6]); };

                //marking as active 
//FP pass
                if (isToBeExecutedOnActive(active, 8) && isNotEmpty && anyInGold[0]) {  //&& anyInGold[0]
                 //   printf("\n set activeee in gold xMeta %d yMeta %d  zMeta %d \n",xMeta,yMeta,zMeta);
                    getTensorRow<bool>(tensorslice, fbArgs.metaData.isActiveGold, fbArgs.metaData.isActiveGold.Ny, yMeta, zMeta)[xMeta] = true;

                };
                //FN pass
                if (isToBeExecutedOnActive(active, 9) && isNotEmpty && anyInSegm[0]) { // && anyInSegm[0]
                 //   printf("\n set activeee in segm xMeta %d yMeta %d  zMeta %d \n", xMeta, yMeta, zMeta);
                    getTensorRow<bool>(tensorslice, fbArgs.metaData.isActiveSegm, fbArgs.metaData.isActiveSegm.Ny, yMeta, zMeta)[xMeta] = true;

                };




                sync(cta); // just to reduce the warp divergence




            }
        }

    }
}


///*					if ((threadIdx.x == 0) && (threadIdx.y == 0)) {
//                        printf("dataBlockIter xMeta [%d] yMeta [%d] zMeta [%d]  \n",  xMeta, yMeta, zMeta);
//                        }*/
//
//
//                        /*					if (goldBool) {
//                                                printf("goldBool x %d y %d z %d    xMeta [%d] yMeta [%d] zMeta [%d]  \n", x, y, z, xMeta, yMeta, zMeta);
//                                            }*/
//
//                                            //now segmentation array
//
//
//                                            //tensorrow = getTensorRow(tensorslice, fbArgs.segmArr, fbArgs.mainArrYLength, y, z);
//                                            //segmBool = (tensorrow[x] == fbArgs.numberToLookFor);
//                                            //
//                                            //
//                                            //fillReduCedArr(fbArgs, xMeta, yMeta, zMeta, tensorslice, tensorrow, z, y, x
//                                            //	, zLocRef, yLocRef, xLocRef, goldBool, segmBool);
//                //if (x==1 && y==2 && z==3) {
//                //    atomicAdd(&tensorrow[x], 1);
//                //}
//                                            //debug
//
//                //printf("dataBlockIter x %d y %d z %d  curr %d ||   xLoc %d yLoc %d zLoc %d xMeta [%d] yMeta [%d] zMeta [%d]  dbX %d dbY %d  dbZ %d  \n", x, y, z, tensorrow[x], xLoc, yLoc, zLoc, xMeta, yMeta, zMeta, fbArgs.dbXLength, fbArgs.dbYLength, fbArgs.dbZLength);
//
//
//      /*              if (tensorrow[x] > 0) {
//                        printf("dataBlockIter x %d y %d z %d  xLoc %d yLoc %d zLoc %d xMeta [%d] yMeta [%d] zMeta [%d]  dbX %d dbY %d  dbZ %d  \n", x, y, z, xLoc, yLoc, zLoc, xMeta, yMeta, zMeta, fbArgs.dbXLength, fbArgs.dbYLength, fbArgs.dbZLength);
//                    }*/
//
//                    //tensorrow = getTensorRow(tensorslice, fbArgs.goldArr, fbArgs.mainArrYLength,  y,  z);
//                    //tensorrow[x] += 1;
//
//                    //debug
//                    //tensorslice = ((char*)fbArgs.forDebugArr.ptr) + z * fbArgs.forDebugArr.pitch * fbArgs.dYLength;
//                    //tensorrow = (int*)(tensorslice + y * fbArgs.forDebugArr.pitch);
//                    //tensorrow[x] += 1;
//
//                    //printf("dataBlockIter %d tensorrow[x]    xMeta [%d] yMeta [%d] zMeta [%d]  \n", tensorrow[x], xMeta, yMeta, zMeta);
//
//                    //array segmentation output




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
    bool& goldBoolRef = goldBool;
    bool& segmBoolRef = segmBool;
    bool& isNotEmptyRef = isNotEmpty;
    thread_block cta = this_thread_block();
    thread_block_tile<32> tile = tiled_partition<32>(cta);

    uint16_t z;	 uint16_t x;    uint16_t y;	 uint16_t sumFpOrig;    uint16_t sumFnOrig;
    uint16_t& refZ = z; uint16_t& refX = x; uint16_t& refY = y; uint16_t& sumFp = sumFpOrig; uint16_t& sumFn = sumFnOrig;


    uint8_t xMeta; uint8_t zMeta; uint8_t yMeta;
    char* tensorslice;


    //shared memory

    //TODO() make it dynamically sized 
    __shared__ uint32_t sharedForGold[32][32];
    __shared__ uint32_t sharedForSegm[32][32];
    //for storing fp and fn sums to later accumulate it to global values
    __shared__ uint32_t fpSFnS[2];

    __shared__ bool anyInGold[1];
    __shared__ bool anyInSegm[1];
    //__shared__ uint32_t reduction_s[32];
    //1)maxX 2)minX 3)maxY 4) minY 5) maxZ 6) minZ
    __shared__ int minMaxesInShmem[7];

    if ((threadIdx.x == 1) && (threadIdx.y == 0)) { minMaxesInShmem[1] = 0; };
    if ((threadIdx.x == 2) && (threadIdx.y == 0)) { minMaxesInShmem[2] = 1000; };

    if ((threadIdx.x == 3) && (threadIdx.y == 0)) { minMaxesInShmem[3] = 0; };
    if ((threadIdx.x == 4) && (threadIdx.y == 0)) { minMaxesInShmem[4] = 1000; };

    if ((threadIdx.x == 5) && (threadIdx.y == 0)) { minMaxesInShmem[5] = 0; };
    if ((threadIdx.x == 0) && (threadIdx.y == 1)) { minMaxesInShmem[6] = 1000; };
    if ((threadIdx.x == 1) && (threadIdx.y == 1)) { fpSFnS[0] = 0; };
    if ((threadIdx.x == 2) && (threadIdx.y == 1)) { fpSFnS[1] = 0; };
    if ((threadIdx.x == 3) && (threadIdx.y == 1)) { anyInGold[1] = false; };
    if ((threadIdx.x == 4) && (threadIdx.y == 1)) { anyInSegm[1] = false; };

    __syncthreads();

    /////////////////////////


    //main metadata iteration
    for (auto linIdexMeta = blockIdx.x; linIdexMeta < fbArgs.metaData.totalMetaLength; linIdexMeta += gridDim.x) {
        //we get from linear index  the coordinates of the metadata block of intrest
        xMeta = linIdexMeta % fbArgs.metaData.metaXLength;
        zMeta = floor((float)(linIdexMeta / (fbArgs.metaData.metaXLength * fbArgs.metaData.MetaYLength)));
        yMeta = floor((float)((linIdexMeta - ((zMeta * fbArgs.metaData.metaXLength * fbArgs.metaData.MetaYLength) + xMeta)) / fbArgs.metaData.metaXLength));
        //iterating over data block
        dataBlockIter(fbArgs, cta, tile, minMaxesInShmem, sumFp, sumFn, isNotEmptyRef
            , xMeta, yMeta, zMeta, tensorslice, refZ, refY, refX
            , goldBoolRef, segmBoolRef, sharedForGold, sharedForSegm, fpSFnS, anyInGold, anyInSegm);


        /*
           //if ((threadIdx.x == 0) && (threadIdx.y == 0)) {
           //	printf("dataBlockIter xMeta [%d] yMeta [%d] zMeta [%d]  \n",  xMeta, yMeta, zMeta);

           //}
           //unhash for debugging
           /// checking are all covered
           ////debugging loop
           //char* tensorslice;    int* tensorrow;
           ////tensorslice = ((char*)fbArgs.forDebugArr.ptr) + zMeta * fbArgs.forDebugArr.pitch * fbArgs.MetaYLength;
           ////tensorrow = (int*)(tensorslice + yMeta * fbArgs.forDebugArr.pitch);
           //////tensorrow[xMeta] += 1;
           ////if ((threadIdx.x == 0 )&& (threadIdx.y == 0)) {
           ////    atomicAdd(&tensorrow[xMeta], 1);
           ////}
           //// checking
           //tensorslice = ((char*)fbArgs.forDebugArr.ptr) + 0 * fbArgs.forDebugArr.pitch * fbArgs.MetaYLength;
           //tensorrow = (int*)(tensorslice + 0 * fbArgs.forDebugArr.pitch);
           ////tensorrow[xMeta] += 1;
           //if ((threadIdx.x == 0) && (threadIdx.y == 0)) {
           //    atomicAdd(&tensorrow[0], 1);
           //}


           //printf("metaCudaTensor[%d][%d][%d] = %d\n", i, j, k, tensorrow[i]);
   */

    }
    sync(cta);
    ////// completing reductions of fp and fns


    /// setting min maxes 
    //1)maxX 2)minX 3)maxY 4) minY 5) maxZ 6) minZ

    auto active = coalesced_threads();
    //int prev;
    //if (g.thread_rank() == 0) {
    //    prev = atomicAdd(p, g.num_threads());
    //}
    //active.thread_rank()
    //    active.num_threads

    if (isToBeExecutedOnActive(active, 0)) {
        //printf("in minMaxes internal  %d \n", minMaxesInShmem[0]);
        //getTensorRow<unsigned int>(tensorslice, fbArgs.metaData.minMaxes, fbArgs.metaData.minMaxes.Ny, 0, 0)[0] = 61;
        atomicMax(&(getTensorRow<unsigned int>(tensorslice, fbArgs.metaData.minMaxes, 1, 0, 0)[1]), minMaxesInShmem[1]);
    };

    if (isToBeExecutedOnActive(active, 1)) {

        atomicMin(&(getTensorRow<unsigned int>(tensorslice, fbArgs.metaData.minMaxes, 1, 0, 0)[2]), minMaxesInShmem[2]);
    };

    if (isToBeExecutedOnActive(active, 2)) {
        atomicMax(&(getTensorRow<unsigned int>(tensorslice, fbArgs.metaData.minMaxes, 1, 0, 0)[3]), minMaxesInShmem[3]);
    };

    if (isToBeExecutedOnActive(active, 3)) {
        atomicMin(&(getTensorRow<unsigned int>(tensorslice, fbArgs.metaData.minMaxes, 1, 0, 0)[4]), minMaxesInShmem[4]);
    };



    if (isToBeExecutedOnActive(active, 4)) {
        atomicMax(&(getTensorRow<unsigned int>(tensorslice, fbArgs.metaData.minMaxes, 1, 0, 0)[5]), minMaxesInShmem[5]);
    };

    if (isToBeExecutedOnActive(active, 5)) {
        atomicMin(&(getTensorRow<unsigned int>(tensorslice, fbArgs.metaData.minMaxes, 1, 0, 0)[6]), minMaxesInShmem[6]);
    };

    //setting global fp and fn
    if (isToBeExecutedOnActive(active, 6)) {
        //printf("internal last fp  %d \n", fpSFnS[0]);
        atomicAdd(&(getTensorRow<unsigned int>(tensorslice, fbArgs.metaData.minMaxes, 1, 0, 0)[7]), fpSFnS[0]);
    };

    if (isToBeExecutedOnActive(active, 7)) {
        //if ((threadIdx.x == 1) && (threadIdx.y == 0)) {
       //if (active.thread_rank() == 7 && active.meta_group_rank() == 0) {
       //     printf("internal last fn  %d idX %d  idY %d tile meta size %d \n", fpSFnS[1], threadIdx.x, threadIdx.y, tile.meta_group_size());

        atomicAdd(&(getTensorRow<unsigned int>(tensorslice, fbArgs.metaData.minMaxes, 1, 0, 0)[8]), fpSFnS[1]);

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


//
//#pragma once
//extern "C" inline bool boolPrepare(ForFullBoolPrepArgs<int> fFArgs) {
//
//
//    cudaError_t syncErr;
//    cudaError_t asyncErr;
//
//
//
//
//
//    //for debugging
//    array3dWithDimsGPU forDebug = allocate3dInGPU(fFArgs.forDebugArr);
//    //main arrays allocations
//    array3dWithDimsGPU goldArr = allocate3dInGPU(fFArgs.goldArr);
//
//    array3dWithDimsGPU segmArr = allocate3dInGPU(fFArgs.segmArr);
//    ////reduced arrays
//    array3dWithDimsGPU reducedGold = allocate3dInGPU(fFArgs.reducedGold);
//    array3dWithDimsGPU reducedSegm = allocate3dInGPU(fFArgs.reducedSegm);
//
//
//
//
//    array3dWithDimsGPU paddingsStore = allocate3dInGPU(fFArgs.paddingsStore);
//
//
//
//
//
//
//    ForBoolKernelArgs<int> fbArgs = getArgsForKernel<int>(fFArgs, forDebug, goldArr, segmArr, reducedGold, reducedSegm, paddingsStore);
//
//
//    boolPrepareKernel <<< fFArgs.blocks, fFArgs.threads >>> (fbArgs);
//
//    checkCuda(cudaDeviceSynchronize(), "just after boolPrepareKernel");
//
//
//
//
//    //deviceTohost
//
//    copyDeviceToHost3d(forDebug, fFArgs.forDebugArr);
//
//
//    copyDeviceToHost3d(goldArr, fFArgs.goldArr);
//    copyDeviceToHost3d(segmArr, fFArgs.segmArr);
//
//    copyDeviceToHost3d(reducedGold, fFArgs.reducedGold);
//    copyDeviceToHost3d(reducedSegm, fFArgs.reducedSegm);
//
//
//    copyMetaDataToCPU(fFArgs.metaData, fbArgs.metaData);
//
//
//
//    checkCuda(cudaDeviceSynchronize(), "just after copy device to host");
//    //cudaGetLastError();
//
//    cudaFree(forDebug.arrPStr.ptr);
//    cudaFree(goldArr.arrPStr.ptr);
//    cudaFree(segmArr.arrPStr.ptr);
//    cudaFree(reducedGold.arrPStr.ptr);
//    cudaFree(reducedSegm.arrPStr.ptr);
//
//
//    freeMetaDataGPU(fbArgs.metaData);
//
//
//    /*
// * Catch errors for both the kernel launch above and any
// * errors that occur during the asynchronous `doubleElements`
// * kernel execution.
// */
//
//    syncErr = cudaGetLastError();
//    asyncErr = cudaDeviceSynchronize();
//
//    /*
//     * Print errors should they exist.
//     */
//
//    if (syncErr != cudaSuccess) printf("Error in syncErr: %s\n", cudaGetErrorString(syncErr));
//    if (asyncErr != cudaSuccess) printf("Error in asyncErr: %s\n", cudaGetErrorString(asyncErr));
//
//
//
//    return true;
//}
