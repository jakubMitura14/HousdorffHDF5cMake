/*
becouse we need a lot of the additional memory spaces to minimize memory consumption allocations will be postponed after first kernel run enabling
*/
#pragma once
template <typename ZZR>
inline void allocateMemoryAfterBoolKernel(ForBoolKernelArgs<ZZR> gpuArgs, ForFullBoolPrepArgs<ZZR> cpuArgs, 
    uint32_t*& resultListPointerMeta,uint16_t*& resultListPointerLocal,uint16_t*& resultListPointerIterNumb) {
    //copy on cpu
    size_t size = sizeof(unsigned int) * 20;
    cudaMemcpy(cpuArgs.metaData.minMaxes, gpuArgs.metaData.minMaxes, size, cudaMemcpyDeviceToHost);

    unsigned int fpPlusFn = cpuArgs.metaData.minMaxes[7] + cpuArgs.metaData.minMaxes[8];

    size = sizeof(uint32_t)* fpPlusFn + 1;
    cudaMallocAsync(&resultListPointerMeta, size, 0);

    size = sizeof(uint16_t) * fpPlusFn + 1;
    cudaMallocAsync(&resultListPointerLocal, size, 0);
    cudaMallocAsync(&resultListPointerIterNumb, size, 0);

   // metaData.resultList = resultListPointer;


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
    unsigned int mainArrSectionLength = (mainArrXLength * 6) + 19;
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


    uint32_t* mainArrCPU = (uint32_t*)calloc(metaData.totalMetaLength * metaData.mainArrSectionLength, sizeof(uint32_t));

    //cudaMallocAsync(&mainArr, sizeB, 0);
    cudaMallocAsync(&mainArr, sizeB, 0);
    cudaMemcpy(mainArr, mainArrCPU, sizeB, cudaMemcpyHostToDevice);
    free(mainArrCPU);
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
    size_t size = sizeof(uint32_t) * metaData.totalMetaLength * metaData.mainArrSectionLength;
    //size_t size = sizeof(uint32_t) * metaData.totalMetaLength * metaData.mainArrSectionLength;
    uint32_t* mainArrCPU = (uint32_t*)calloc(metaData.totalMetaLength * metaData.mainArrSectionLength, sizeof(uint32_t));
    cudaMemcpy(mainArrCPU, mainArr, size, cudaMemcpyDeviceToHost);

    uint32_t column = mainArrCPU[33];
    printf("column\n ");
    std::cout<<column;
    //in kernel x 33 y 1 z 71 linearLocal 33 linIdexMeta 0
    //    in kernel x 75 y 20 z 70 linearLocal 267 linIdexMeta 3

    for (int linIdexMeta = 0; linIdexMeta < metaData.totalMetaLength; linIdexMeta++) {
        uint8_t xMeta = linIdexMeta % metaData.metaXLength;
        uint8_t zMeta = floor((float)(linIdexMeta / (metaData.metaXLength * metaData.MetaYLength)));
        uint8_t yMeta = floor((float)((linIdexMeta - ((zMeta * metaData.metaXLength * metaData.MetaYLength) + xMeta)) / metaData.metaXLength));

        for (int threadIdxX = 0; threadIdxX < 32; threadIdxX++) {
         
                    for (int threadIdxY = 0; threadIdxY < 18; threadIdxY++) {

                        uint8_t xLoc = threadIdxX;
                        uint16_t x = (xMeta + metaData.minX) * gpuArgs.dbXLength + xLoc;//absolute position
                            uint8_t yLoc = threadIdxY;
                                uint16_t  y = (yMeta + metaData.minY) * gpuArgs.dbYLength + yLoc;//absolute position

                                uint32_t columnGold = mainArrCPU[linIdexMeta * metaData.mainArrSectionLength + gpuArgs.dbXLength * threadIdxY + threadIdxX];
                                if(columnGold >0){
                                    printf("found set at x %d y%d columnGold %d \n", x, y,  columnGold);
                                        for (uint8_t bitPos = 0; bitPos < 32; bitPos++) {
                                            uint16_t z = (zMeta + metaData.minZ) * gpuArgs.dbZLength + bitPos;//absolute position
                                            //if any bit here is set it means it should be added to result list 
                                            if (isBitAtCPU(columnGold, bitPos)) {
                                               // printf("found set at x %d y%d z %d  \n",x,y,z);

                                            
                                        }
                                    }

                                }


            }
        }
    }

        
        
        
        
        
        
        //gold pass
        
        //segm pass
       // mainArr[linIdexMeta * metaData.mainArrSectionLength + metaData.mainArrXLength]



    }



