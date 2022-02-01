

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
) {

    ForBoolKernelArgs<TCC> res;
    MetaDataGPU resMeta;



    setArrCPU<unsigned int>(metaDataCPU.minMaxes, 1, 0, 0, 0, false);
    setArrCPU<unsigned int>(metaDataCPU.minMaxes, 2, 0, 0, 1000, false);
    setArrCPU<unsigned int>(metaDataCPU.minMaxes, 3, 0, 0, 0, false);
    setArrCPU<unsigned int>(metaDataCPU.minMaxes, 4, 0, 0, 1000, false);
    setArrCPU<unsigned int>(metaDataCPU.minMaxes, 5, 0, 0, 0, false);
    setArrCPU<unsigned int>(metaDataCPU.minMaxes, 6, 0, 0, 1000, false);
    setArrCPU<unsigned int>(metaDataCPU.minMaxes, 7, 0, 0, 0, false);
    setArrCPU<unsigned int>(metaDataCPU.minMaxes, 8, 0, 0, 0, false);
    setArrCPU<unsigned int>(metaDataCPU.minMaxes, 9, 0, 0, 0, false);
    setArrCPU<unsigned int>(metaDataCPU.minMaxes, 10, 0, 0, 0, false);
    setArrCPU<unsigned int>(metaDataCPU.minMaxes, 11, 0, 0, 0, false);
    setArrCPU<unsigned int>(metaDataCPU.minMaxes, 12, 0, 0, 1, false);
    setArrCPU<unsigned int>(metaDataCPU.minMaxes, 13, 0, 0, 0, false);
    setArrCPU<unsigned int>(metaDataCPU.minMaxes, 14, 0, 0, 0, false);
    setArrCPU<unsigned int>(metaDataCPU.minMaxes, 15, 0, 0, 0, false);
    setArrCPU<unsigned int>(metaDataCPU.minMaxes, 16, 0, 0, 0, false);
    setArrCPU<unsigned int>(metaDataCPU.minMaxes, 17, 0, 0, 0, false);


    resMeta.minMaxes = allocate3dInGPU(metaDataCPU.minMaxes);

    res.metaData = resMeta;
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

    char* tensorslice;
    uint16_t sumFp = 0;
    uint16_t sumFn = 0;

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
    if ((threadIdx.x == 1) && (threadIdx.y == 1)) { fpSFnS[0] = 0; };
    if ((threadIdx.x == 2) && (threadIdx.y == 1)) { fpSFnS[1] = 0; };
    if ((threadIdx.x == 3) && (threadIdx.y == 1)) { anyInGold[1] = false; };
    if ((threadIdx.x == 4) && (threadIdx.y == 1)) { anyInSegm[1] = false; };

    __syncthreads();

    /////////////////////////


    //main metadata iteration
    for (auto linIdexMeta = blockIdx.x; linIdexMeta < fbArgs.metaData.totalMetaLength; linIdexMeta += gridDim.x) {
        //we get from linear index  the coordinates of the metadata block of intrest
        uint8_t xMeta = linIdexMeta % fbArgs.metaData.metaXLength;
        uint8_t zMeta = floor((float)(linIdexMeta / (fbArgs.metaData.metaXLength * fbArgs.metaData.MetaYLength)));
        uint8_t yMeta = floor((float)((linIdexMeta - ((zMeta * fbArgs.metaData.metaXLength * fbArgs.metaData.MetaYLength) + xMeta)) / fbArgs.metaData.metaXLength));
        //iterating over data block
        //now we need to iterate over the data in the data block voxel by voxel
        for (uint8_t xLoc = threadIdx.x; xLoc < fbArgs.dbXLength; xLoc += blockDim.x) {
            uint16_t x = xMeta * fbArgs.dbXLength + xLoc;//absolute position
            for (uint8_t yLoc = threadIdx.y; yLoc < fbArgs.dbYLength; yLoc += blockDim.y) {
                uint16_t y = yMeta * fbArgs.dbYLength + yLoc;//absolute position
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
                        uint16_t z = zMeta * fbArgs.dbZLength + zLoc;//absolute position
                        if (z < fbArgs.goldArr.Nx) {
                            //first array gold
                            uint8_t& zLocRef = zLoc; uint8_t& yLocRef = yLoc; uint8_t& xLocRef = xLoc;

                            goldBool = (getTensorRow<TYI>(tensorslice, fbArgs.goldArr, fbArgs.goldArr.Ny, y, z)[x] == fbArgs.numberToLookFor);

                            // now segmentation  array
                            segmBool = (getTensorRow<TYI>(tensorslice, fbArgs.segmArr, fbArgs.goldArr.Ny, y, z)[x] == fbArgs.numberToLookFor);
                            // setting bits
                             //setting  bits for reduced representation 
                            sharedForGold[xLoc][yLoc] |= goldBool << zLoc;
                            sharedForSegm[xLoc][yLoc] |= segmBool << zLoc;
                            // setting value of local boolean marking that any of the entries was evaluated to true in either of arrays
                            isNotEmpty = (isNotEmpty || (goldBool || segmBool));
                            sumFp += (!goldBool && segmBool);
                            sumFn += (goldBool && !segmBool);
                            if (goldBool)  anyInGold[0] = true;
                            if (segmBool)  anyInSegm[0] = true;
                        }
                    }
                    ////after we streamed over all z layers we need to save it into reduced representation arrays
                    //getTensorRow<uint32_t>(tensorslice, fbArgs.reducedGold, fbArgs.reducedGold.Ny, y, zMeta)[x] = sharedForGold[xLoc][yLoc];
                    //getTensorRow<uint32_t>(tensorslice, fbArgs.reducedSegm, fbArgs.reducedSegm.Ny, y, zMeta)[x] = sharedForSegm[xLoc][yLoc];
                    //// TODO() establish is it faster that way or better at the end do mempcy async
                    //getTensorRow<uint32_t>(tensorslice, fbArgs.reducedGoldRef, fbArgs.reducedGoldRef.Ny, y, zMeta)[x] = sharedForGold[xLoc][yLoc];
                    //getTensorRow<uint32_t>(tensorslice, fbArgs.reducedSegmRef, fbArgs.reducedSegmRef.Ny, y, zMeta)[x] = sharedForSegm[xLoc][yLoc];

                    getTensorRow<uint32_t>(tensorslice, fbArgs.reducedGoldPrev, fbArgs.reducedGoldPrev.Ny, y, zMeta)[x] = sharedForGold[xLoc][yLoc];
                    getTensorRow<uint32_t>(tensorslice, fbArgs.reducedSegmPrev, fbArgs.reducedSegmPrev.Ny, y, zMeta)[x] = sharedForSegm[xLoc][yLoc];
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
                        getTensorRow<unsigned int>(tensorslice, fbArgs.metaData.fpCount, fbArgs.metaData.fpCount.Ny, yMeta, zMeta)[xMeta] = sharedForGold[1][0];
                    }
                    //segm
                   // if (isToBeExecutedOnActive(active, 1) && isNotEmpty) {
                    if ((threadIdx.x == 0) && (threadIdx.y == 1) && isNotEmpty) {
                        sharedForSegm[1][0] = 0;//reset
                        for (uint8_t i = 0; i < tile.meta_group_size(); i += 1) {
                            sharedForSegm[1][0] += sharedForSegm[0][i];
                        };
                        fpSFnS[1] += sharedForSegm[1][0];// will be needed later for global set
                        //setting metadata
                        getTensorRow<unsigned int>(tensorslice, fbArgs.metaData.fnCount, fbArgs.metaData.fnCount.Ny, yMeta, zMeta)[xMeta] = sharedForSegm[1][0];

                    }
                    /////////////////// setting min and maxes

                    //marking as active 
    //FP pass
                    if (isToBeExecutedOnActive(active, 8) && isNotEmpty && anyInGold[0]) {  //&& anyInGold[0]
                        getTensorRow<bool>(tensorslice, fbArgs.metaData.isActiveGold, fbArgs.metaData.isActiveGold.Ny, yMeta, zMeta)[xMeta] = true;

                    };
                    //FN pass
                    if (isToBeExecutedOnActive(active, 9) && isNotEmpty && anyInSegm[0]) { // && anyInSegm[0]
                        getTensorRow<bool>(tensorslice, fbArgs.metaData.isActiveSegm, fbArgs.metaData.isActiveSegm.Ny, yMeta, zMeta)[xMeta] = true;

                    };




                    sync(cta); // just to reduce the warp divergence




                }
            }

        }

    }
    sync(cta);
    ////// completing reductions of fp and fns



    auto active = coalesced_threads();

   
//setting global fp and fn
    if (isToBeExecutedOnActive(active, 0)) {
        atomicAdd(&(getTensorRow<unsigned int>(tensorslice, fbArgs.metaData.minMaxes, 1, 0, 0)[7]), fpSFnS[0]);
    };

    if (isToBeExecutedOnActive(active, 1)) {
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
