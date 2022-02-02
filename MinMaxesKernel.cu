

#include "CPUAllocations.cu"
#include "MetaData.cu"
#include "IterationUtils.cu"
#include "ExceptionManagUtils.cu"
#include "CooperativeGroupsUtils.cu"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
using namespace cooperative_groups;





// helper functions and utilities to work with CUDA from https://github.com/NVIDIA/cuda-samples



/*
iteration over metadata - becouse metadata may be small and to maximize occupancy we use linear index and then clalculate xMeta,ymeta,zMeta from this linear index ...
*/
#pragma once
template <typename TYU>
__device__ void metaDataIterB(ForBoolKernelArgs<TYU> fbArgs) {

    ////////////some initializations
    thread_block cta = this_thread_block();
    thread_block_tile<32> tile = tiled_partition<32>(cta);
     

    char* tensorslice;


    //shared memory
    bool isNotEmpty = false;
    __shared__ bool anyInGold[1];
    //__shared__ uint32_t reduction_s[32];
    //1)maxX 2)minX 3)maxY 4) minY 5) maxZ 6) minZ
    __shared__ uint8_t minMaxesInShmem[8];
    __shared__ uint8_t minXInShmem[1];
    //__shared__ bool [2000];

    if ((threadIdx.x == 1)) { minMaxesInShmem[1] = 0; };
    if ((threadIdx.x == 2)) { minMaxesInShmem[2] = 1000; };

    if ((threadIdx.x == 3)) { minMaxesInShmem[3] = 0; };
    if ((threadIdx.x == 4) ) { minMaxesInShmem[4] = 1000; };

    if ((threadIdx.x == 5) ) { minMaxesInShmem[5] = 0; };
    if ((threadIdx.x == 0)) { minMaxesInShmem[6] = 1000; };

    //if ((threadIdx.x == 3) && (threadIdx.y == 1)) { anyInGold[1] = false; };

    __syncthreads();

    /////////////////////////


    //main metadata iteration
    for (auto linIdexMeta = blockIdx.x; linIdexMeta < fbArgs.metaData.totalMetaLength; linIdexMeta += gridDim.x) {
        //if (threadIdx.x == 0) { anyInGold[0] = false; };
        //we get from linear index  the coordinates of the metadata block of intrest
        uint8_t xMeta = linIdexMeta % fbArgs.metaData.metaXLength;
        uint8_t zMeta = floor((float)(linIdexMeta / (fbArgs.metaData.metaXLength * fbArgs.metaData.MetaYLength)));
        uint8_t yMeta = floor((float)((linIdexMeta - ((zMeta * fbArgs.metaData.metaXLength * fbArgs.metaData.MetaYLength) + xMeta)) / fbArgs.metaData.metaXLength));
         isNotEmpty = __syncthreads_or(isNotEmpty);;
        if (threadIdx.x == 0) {
            printf("linIdexMeta %d xMeta %d yMeta %d zMeta %d metaXLength %d MetaYLength %d totalMetaLength %d anyInGold[0] %d gold Nx %d nY %d nZ %d segm Nx %d Ny %d Nz %d    \n "
                , linIdexMeta, xMeta, yMeta, zMeta, fbArgs.metaData.metaXLength, fbArgs.metaData.MetaYLength, fbArgs.metaData.totalMetaLength, isNotEmpty
            , fbArgs.goldArr.Nx, fbArgs.goldArr.Ny, fbArgs.goldArr.Nz , fbArgs.segmArr.Nx, fbArgs.segmArr.Ny, fbArgs.segmArr.Nz
            );
        }
        sync(cta);
        isNotEmpty = false;
        //iterating over data block
        for (uint8_t xLoc = threadIdx.x; xLoc < fbArgs.dbXLength; xLoc += blockDim.x) {
            uint16_t x = xMeta * fbArgs.dbXLength + xLoc;//absolute position
            for (uint8_t yLoc = threadIdx.y; yLoc < fbArgs.dbYLength; yLoc += blockDim.y) {
                uint16_t  y = yMeta * fbArgs.dbYLength + yLoc;//absolute position
                if (y < fbArgs.goldArr.Ny && x < fbArgs.goldArr.Nz) {

                //   //  resetting 


                    for (uint8_t zLoc = 0; zLoc < fbArgs.dbZLength; zLoc++) {
                        uint16_t z = zMeta * fbArgs.dbZLength + zLoc;//absolute position
                        if (z < fbArgs.goldArr.Nx) {
                            //first array gold
                            // setting bits

                            if ((getTensorRow<TYU>(tensorslice, fbArgs.goldArr, fbArgs.goldArr.Ny, y, z)[x] == fbArgs.numberToLookFor)
                                || (getTensorRow<TYU>(tensorslice, fbArgs.segmArr, fbArgs.goldArr.Ny, y, z)[x] == fbArgs.numberToLookFor)) {
                                isNotEmpty = true;
                            }

                            if (threadIdx.x == 0 && isNotEmpty) {
                                printf("x %d y%d z %d linIdexMeta %d xMeta %d yMeta %d zMeta %d metaXLength %d MetaYLength %d totalMetaLength %d anyInGold[0] \n "
                                    , x, y, z, linIdexMeta, xMeta, yMeta, zMeta, fbArgs.metaData.metaXLength, fbArgs.metaData.MetaYLength, fbArgs.metaData.totalMetaLength, isNotEmpty
                                    , fbArgs.goldArr.Nx, fbArgs.goldArr.Ny, fbArgs.goldArr.Nz, fbArgs.segmArr.Nx, fbArgs.segmArr.Ny, fbArgs.segmArr.Nz
                                );
                            }


                        }

                    }

                }
            }
        }



      //  isNotEmpty = __syncthreads_or(isNotEmpty);



        ///////////////// setting min and maxes
    //1)maxX 2)minX 3)maxY 4) minY 5) maxZ 6) minZ
       // if (((threadIdx.x == 0)) && anyInGold[0]) { minXInShmem[0] = xMeta; }
                //if (((threadIdx.x == 0)) && anyInGold[0]) { minMaxesInShmem[1] = max(xMeta, minMaxesInShmem[1]); };
                //if (((threadIdx.x == 1)) && anyInGold[0]) { minMaxesInShmem[2] = min(xMeta, minMaxesInShmem[2]); };

                //if (((threadIdx.x == 2)) && anyInGold[0]) { minMaxesInShmem[3] = max(yMeta, minMaxesInShmem[3]); };
                //if (((threadIdx.x == 3) ) && anyInGold[0]) { minMaxesInShmem[4] = min(yMeta, minMaxesInShmem[4]); };

                //if (((threadIdx.x == 4) ) && anyInGold[0]) { minMaxesInShmem[5] = max(zMeta, minMaxesInShmem[5]); };
                //if (((threadIdx.x == 5) ) && anyInGold[0]) { minMaxesInShmem[6] = min(zMeta, minMaxesInShmem[6]); };

                //if (((threadIdx.x == 6)) && anyInGold[0]) { anyInGold[0] = false; };

                //if (anyInGold[0]) { minMaxesInShmem[1] = max(xMeta, minMaxesInShmem[1]); };
                //if ( anyInGold[0]) { minMaxesInShmem[2] = min(xMeta, minMaxesInShmem[2]); };

                //if (anyInGold[0]) { minMaxesInShmem[3] = max(yMeta, minMaxesInShmem[3]); };
                //if (anyInGold[0]) { minMaxesInShmem[4] = min(yMeta, minMaxesInShmem[4]); };

                //if (anyInGold[0]) { minMaxesInShmem[5] = max(zMeta, minMaxesInShmem[5]); };
                //if (anyInGold[0]) { minMaxesInShmem[6] = min(zMeta, minMaxesInShmem[6]); };
                //if (isNotEmpty) { isNotEmpty = false; };
                //sync(cta);
 }
    sync(cta);


    auto active = coalesced_threads();
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






}



/*
collecting all needed functions for GPU execution to prepare data from calculating Housedorff distance
*/
#pragma once
template <typename TYO>
__global__ void getMinMaxes(ForBoolKernelArgs<TYO> fbArgs) {
    metaDataIterB(fbArgs);
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
