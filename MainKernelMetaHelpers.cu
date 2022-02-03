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
inline MetaDataGPU allocateMemoryAfterMinMaxesKernel(ForBoolKernelArgs<ZZR> gpuArgs, ForFullBoolPrepArgs<ZZR> cpuArgs
            , uint32_t*& mainArr, uint32_t*& workQueue, unsigned int* minMaxes, MetaDataGPU metaData
) {
    ////reduced arrays


    //copy on cpu
    size_t size = sizeof(unsigned int) * 20;
    cudaMemcpy( cpuArgs.metaData.minMaxes, minMaxes, size, cudaMemcpyDeviceToHost);

    //read an modify
    //1)maxX 2)minX 3)maxY 4) minY 5) maxZ 6) minZ
    //7)global FP count; 8)global FN count
    unsigned int xRange = cpuArgs.metaData.minMaxes[1] - cpuArgs.metaData.minMaxes[2]+1;
    unsigned int yRange = cpuArgs.metaData.minMaxes[3] - cpuArgs.metaData.minMaxes[4]+1;
    unsigned int zRange = cpuArgs.metaData.minMaxes[5] - cpuArgs.metaData.minMaxes[6]+1;
    unsigned int totalMetaLength = xRange* yRange* zRange;

 
    //updating size informations
    metaData.metaXLength = xRange;
    metaData.MetaYLength = yRange;
    metaData.MetaZLength = zRange;
    metaData.totalMetaLength = totalMetaLength;

    cpuArgs.metaData.metaXLength = xRange;
    cpuArgs.metaData.MetaYLength = yRange;
    cpuArgs.metaData.MetaZLength = zRange;
    cpuArgs.metaData.totalMetaLength = totalMetaLength;
    //saving min maxes
    metaData.maxX = cpuArgs.metaData.minMaxes[1];
    metaData.minX = cpuArgs.metaData.minMaxes[2];
    metaData.maxY = cpuArgs.metaData.minMaxes[3];
    metaData.minY = cpuArgs.metaData.minMaxes[4];
    metaData.maxZ = cpuArgs.metaData.minMaxes[5];
    metaData.minZ = cpuArgs.metaData.minMaxes[6];

    //allocating needed memory
    // main array
    unsigned int mainArrXLength = cpuArgs.dbXLength * cpuArgs.dbYLength;
    unsigned int mainArrSectionLength = (mainArrXLength * 6) + 18;
    metaData.mainArrXLength = mainArrXLength;
    metaData.mainArrSectionLength = mainArrSectionLength;
    metaData.metaDataOffset = (mainArrXLength * 6);
    
    size_t sizeB = totalMetaLength * mainArrSectionLength * sizeof(uint32_t);
    std::cout <<"totalMetaLength  ";
    std::cout << totalMetaLength;
    std::cout << "\n";


    //std::cout << "xRange  ";
    //std::cout << xRange;
    //std::cout << "\n";

    //std::cout << "yRange  ";
    //std::cout << yRange;
    //std::cout << "\n";

    //std::cout << "zRange  ";
    //std::cout << zRange;
    //std::cout << "\n";


    cudaMallocAsync(&mainArr, sizeB, 0);
    //workqueue

    size_t sizeC = (totalMetaLength * sizeof(uint32_t));
   //cudaMallocAsync(&workQueue, size, 0);
   cudaMalloc(&workQueue, size);

   return metaData;
};




#pragma once
template <typename ZZR>
inline void printForDebug(ForBoolKernelArgs<ZZR> gpuArgs, ForFullBoolPrepArgs<ZZR> cpuArgs, uint32_t* resultListPointer
    , uint32_t* mainArrPointer, uint32_t* workQueuePointer, MetaDataGPU metaData) {
    // getting arrays allocated on  cpu to be able to print and test them easier
    size_t size = sizeof(uint32_t) * metaData.totalMetaLength* metaData.mainArrSectionLength;
    uint32_t* mainArrCPU = (uint32_t*)malloc(size);


    for (int linIdexMeta = 0; linIdexMeta < metaData.totalMetaLength; linIdexMeta++) {
        

    }

};


