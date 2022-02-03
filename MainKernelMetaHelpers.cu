/*
becouse we need a lot of the additional memory spaces to minimize memory consumption allocations will be postponed after first kernel run enabling
*/
#pragma once
template <typename ZZR>
inline void allocateMemoryAfterBoolKernel(ForBoolKernelArgs<ZZR> gpuArgs, ForFullBoolPrepArgs<ZZR> cpuArgs, uint32_t*& resultListPointer) {
    //copy on cpu
    size_t size = sizeof(unsigned int) * 20;
    cudaMemcpy(cpuArgs.metaData.minMaxes, gpuArgs.metaData.minMaxes, size, cudaMemcpyDeviceToHost);

    unsigned int fpPlusFn = cpuArgs.metaData.minMaxes[7] + cpuArgs.metaData.minMaxes[8];

    size = sizeof(uint32_t) * 5 * fpPlusFn + 1;
    cudaMallocAsync(&resultListPointer, size, 0);
    gpuArgs.metaData.resultList = resultListPointer;


    // cudaFreeAsync(gpuArgs.metaData.resultList, 0);

     //cudaFree(resultListPointer);


};




#pragma once
template <typename ZZR>
inline void allocateMemoryAfterMinMaxesKernel(ForBoolKernelArgs<ZZR> gpuArgs, ForFullBoolPrepArgs<ZZR> cpuArgs
            , uint32_t*& mainArr, uint32_t*& workQueue, unsigned int* minMaxes
) {
    ////reduced arrays


    //copy on cpu
    size_t size = sizeof(unsigned int) * 20;
    cudaMemcpy( cpuArgs.metaData.minMaxes, minMaxes, size, cudaMemcpyDeviceToHost);

    //read an modify
    //1)maxX 2)minX 3)maxY 4) minY 5) maxZ 6) minZ
    //7)global FP count; 8)global FN count
    unsigned int xRange = cpuArgs.metaData.minMaxes[1] - cpuArgs.metaData.minMaxes[2];
    unsigned int yRange = cpuArgs.metaData.minMaxes[3] - cpuArgs.metaData.minMaxes[4];
    unsigned int zRange = cpuArgs.metaData.minMaxes[5] - cpuArgs.metaData.minMaxes[6];
    unsigned int totalMetaLength = xRange* yRange* zRange;

 
    //updating size informations
    gpuArgs.metaData.metaXLength = xRange;
    gpuArgs.metaData.MetaYLength = yRange;
    gpuArgs.metaData.MetaZLength = zRange;
    gpuArgs.metaData.totalMetaLength = totalMetaLength;

    cpuArgs.metaData.metaXLength = xRange;
    cpuArgs.metaData.MetaYLength = yRange;
    cpuArgs.metaData.MetaZLength = zRange;
    cpuArgs.metaData.totalMetaLength = totalMetaLength;
    //saving min maxes
    gpuArgs.maxX = cpuArgs.metaData.minMaxes[1];
    gpuArgs.minX = cpuArgs.metaData.minMaxes[2];
    gpuArgs.maxY = cpuArgs.metaData.minMaxes[3];
    gpuArgs.minY = cpuArgs.metaData.minMaxes[4];
    gpuArgs.maxZ = cpuArgs.metaData.minMaxes[5];
    gpuArgs.minZ = cpuArgs.metaData.minMaxes[6];

    //allocating needed memory
    // main array
    unsigned int mainArrXLength = cpuArgs.dbXLength * cpuArgs.dbYLength;
    unsigned int mainArrSectionLength = (mainArrXLength * 6) + 18;
    gpuArgs.mainArrXLength = mainArrXLength;
    gpuArgs.mainArrSectionLength = mainArrSectionLength;
    gpuArgs.metaDataOffset = (mainArrXLength * 6);
    
    size_t sizeB = totalMetaLength * mainArrSectionLength * sizeof(uint32_t);
    std::cout <<"size  ";
    std::cout << (totalMetaLength * mainArrSectionLength * sizeof(uint32_t))/1000000000;
    std::cout << "\n";
    cudaMallocAsync(&mainArr, sizeB, 0);
    //workqueue

    size_t sizeC = (totalMetaLength * sizeof(uint32_t));
   //cudaMallocAsync(&workQueue, size, 0);
   cudaMalloc(&workQueue, size);


};




#pragma once
template <typename ZZR>
inline void printForDebug(ForBoolKernelArgs<ZZR> gpuArgs, ForFullBoolPrepArgs<ZZR> cpuArgs) {
    // getting arrays allocated on  cpu to be able to print and test them easier
    size_t size = sizeof(unsigned int) * 20;
    unsigned int* minMaxesCPU = (unsigned int*)malloc(size);


};


