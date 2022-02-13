/*
becouse we need a lot of the additional memory spaces to minimize memory consumption allocations will be postponed after first kernel run enabling
*/
#pragma once
template <typename ZZR>
inline int allocateMemoryAfterBoolKernel(ForBoolKernelArgs<ZZR> gpuArgs, ForFullBoolPrepArgs<ZZR> cpuArgs, 
    uint32_t*& resultListPointerMeta
    ,uint32_t*& resultListPointerLocal
    ,uint32_t*& resultListPointerIterNumb,
    uint32_t*& origArrsPointer,
    uint32_t*& mainArrAPointer,
    uint32_t*& mainArrBPointer, MetaDataGPU metaData,array3dWithDimsGPU goldArr, array3dWithDimsGPU segmArr) {
    
    //free no longer needed arrays
    cudaFreeAsync(goldArr.arrPStr.ptr, 0);
    cudaFreeAsync(segmArr.arrPStr.ptr, 0);
    
    //copy on cpu
    size_t size = sizeof(unsigned int) * 20;
    cudaMemcpy(cpuArgs.metaData.minMaxes, gpuArgs.metaData.minMaxes, size, cudaMemcpyDeviceToHost);

    unsigned int fpPlusFn = cpuArgs.metaData.minMaxes[7] + cpuArgs.metaData.minMaxes[8];


    size = sizeof(uint32_t)* (fpPlusFn + 50);



    //cudaMalloc(&resultListPointerLocal, size);
    //cudaMalloc(&resultListPointerIterNumb, size);
    //cudaMalloc(&resultListPointerMeta, size);

    cudaMallocAsync(&resultListPointerLocal, size, 0);
    cudaMallocAsync(&resultListPointerIterNumb, size, 0);
    cudaMallocAsync(&resultListPointerMeta, size, 0);


   auto xRange  = metaData.metaXLength ;
   auto yRange =  metaData.MetaYLength ;
   auto zRange = metaData.MetaZLength ;
    
    

    
    size_t sizeB = metaData.totalMetaLength * metaData.mainArrSectionLength * sizeof(uint32_t);

    cudaMallocAsync(&mainArrAPointer, sizeB, 0);
    cudaMemcpyAsync(mainArrAPointer, origArrsPointer, sizeB, cudaMemcpyDeviceToDevice,0);

    
    cudaMallocAsync(&mainArrBPointer, sizeB, 0);
    cudaMemcpyAsync(mainArrBPointer, origArrsPointer, sizeB, cudaMemcpyDeviceToDevice,0);

    
   // size_t sizeorigArr = totalMetaLength * (mainArrXLength * 2) * sizeof(uint32_t);
    
   // metaData.resultList = resultListPointer;


    // cudaFreeAsync(gpuArgs.metaData.resultList, 0);

     //cudaFree(resultListPointer);

    return fpPlusFn;
};




#pragma once
template <typename ZZR>
inline MetaDataGPU allocateMemoryAfterMinMaxesKernel(ForBoolKernelArgs<ZZR> gpuArgs, ForFullBoolPrepArgs<ZZR> cpuArgs,
             uint32_t*& workQueue, unsigned int* minMaxes, MetaDataGPU metaData, uint32_t*& origArr
    , uint32_t*& metaDataArr) {
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
    unsigned int mainArrXLength = gpuArgs.dbXLength * gpuArgs.dbYLength;
    unsigned int mainArrSectionLength = (mainArrXLength * 2);
    metaData.mainArrXLength = mainArrXLength;
    metaData.mainArrSectionLength = mainArrSectionLength;
    
    size_t sizeB = totalMetaLength * mainArrSectionLength * sizeof(uint32_t);


    //cudaMallocAsync(&mainArr, sizeB, 0);
    size_t sizeorigArr = totalMetaLength * (mainArrXLength * 2) * sizeof(uint32_t);
    cudaMallocAsync(&origArr, sizeorigArr, 0);
    size_t sizemetaDataArr = totalMetaLength * (20) * sizeof(uint32_t);
    cudaMallocAsync(&metaDataArr, sizemetaDataArr, 0);

    
    size_t sizeC = (totalMetaLength * sizeof(uint32_t));
   //cudaMallocAsync(&workQueue, size, 0);
    cudaMallocAsync(&workQueue, sizeC,0);

   return metaData;
};



