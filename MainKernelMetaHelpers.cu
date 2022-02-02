/*
becouse we need a lot of the additional memory spaces to minimize memory consumption allocations will be postponed after first kernel run enabling
*/
#pragma once
template <typename ZZR>
inline void allocateMemoryAfterBoolKernel(ForBoolKernelArgs<ZZR> gpuArgs, ForFullBoolPrepArgs<ZZR> cpuArgs, uint16_t*& resultListPointer) {
    //copy on cpu
    copyDeviceToHost3d(gpuArgs.metaData.minMaxes, cpuArgs.metaData.minMaxes);
    //read an modify
    //1)maxX 2)minX 3)maxY 4) minY 5) maxZ 6) minZ
    //7)global FP count; 8)global FN count
    unsigned int fpPlusFn = cpuArgs.metaData.minMaxes.arrP[0][0][7] + cpuArgs.metaData.minMaxes.arrP[0][0][8];

    size_t size = sizeof(uint16_t) * 5 * fpPlusFn + 1;
    cudaMallocAsync(&resultListPointer, size, 0);
    gpuArgs.metaData.resultList = resultListPointer;


    // cudaFreeAsync(gpuArgs.metaData.resultList, 0);

     //cudaFree(resultListPointer);


};




#pragma once
template <typename ZZR>
inline void allocateMemoryAfterMinMaxesKernel(ForBoolKernelArgs<ZZR> gpuArgs, ForFullBoolPrepArgs<ZZR> cpuArgs) {
    ////reduced arrays
    array3dWithDimsGPU reducedGold;
    array3dWithDimsGPU reducedSegm;

    array3dWithDimsGPU reducedGoldRef;
    array3dWithDimsGPU reducedSegmRef;


    array3dWithDimsGPU reducedGoldPrev;
    array3dWithDimsGPU reducedSegmPrev;


    //copy on cpu
    copyDeviceToHost3d(gpuArgs.metaData.minMaxes, cpuArgs.metaData.minMaxes);
    //read an modify
    //1)maxX 2)minX 3)maxY 4) minY 5) maxZ 6) minZ
    //7)global FP count; 8)global FN count
    unsigned int xRange = cpuArgs.metaData.minMaxes.arrP[0][0][1] - cpuArgs.metaData.minMaxes.arrP[0][0][2];
    unsigned int yRange = cpuArgs.metaData.minMaxes.arrP[0][0][3] - cpuArgs.metaData.minMaxes.arrP[0][0][4];
    unsigned int zRange = cpuArgs.metaData.minMaxes.arrP[0][0][5] - cpuArgs.metaData.minMaxes.arrP[0][0][6];

    //allocating needed memory
    reducedGold = getArrGpu<uint32_t>(xRange * cpuArgs.dbXLength, yRange * cpuArgs.dbYLength, zRange * cpuArgs.dbZLength);
    reducedSegm = getArrGpu<uint32_t>(xRange * cpuArgs.dbXLength, yRange * cpuArgs.dbYLength, zRange * cpuArgs.dbZLength);
    reducedGoldRef = getArrGpu<uint32_t>(xRange * cpuArgs.dbXLength, yRange * cpuArgs.dbYLength, zRange * cpuArgs.dbZLength);
    reducedSegmRef = getArrGpu<uint32_t>(xRange * cpuArgs.dbXLength, yRange * cpuArgs.dbYLength, zRange * cpuArgs.dbZLength);
    reducedGoldPrev = getArrGpu<uint32_t>(xRange * cpuArgs.dbXLength, yRange * cpuArgs.dbYLength, zRange * cpuArgs.dbZLength);
    reducedSegmPrev = getArrGpu<uint32_t>(xRange * cpuArgs.dbXLength, yRange * cpuArgs.dbYLength, zRange * cpuArgs.dbZLength);
    allocateMetaDataOnGPU(xRange, yRange, zRange);
    //unsigned int fpPlusFn = fFArgs.metaData.minMaxes.arrP[0][0][7] + fFArgs.metaData.minMaxes.arrP[0][0][8];
    //uint16_t* resultListPointer;
    //size_t size = sizeof(uint16_t) * 5 * fpPlusFn + 1;
    //cudaMallocAsync(&resultListPointer, size, 0);
    //fbArgs.metaData.resultList = resultListPointer;


};



//
//#pragma once
//template <typename ZZR>
//inline void calculateOccupancy(ForBoolKernelArgs<ZZR> gpuArgs, ForFullBoolPrepArgs<ZZR> cpuArgs) {
//   
//
//    int numBlocks; // Occupancy in terms of active blocks
//    int blockSize = 32;
//    // These variables are used to convert occupancy to warps
//    int device;
//    cudaDeviceProp prop;
//    int activeWarps;
//    int maxWarps;
//    cudaGetDevice(&device);
//    cudaGetDeviceProperties(&prop, device);
//
//    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
//        &numBlocks,
//        MyKernel,
//        blockSize,
//        0);
//    activeWarps = numBlocks * blockSize / prop.warpSize;
//    maxWarps = prop.maxThreadsPerMultiProcessor / prop.warpSize;
//    std::cout << "Occupancy: " << (double)activeWarps / maxWarps * 100 << "%" <<
//        std::endl;
//
//
//
//
//
//
//};


