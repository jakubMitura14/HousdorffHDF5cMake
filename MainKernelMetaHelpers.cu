

#include "cuda_runtime.h"
#include "MetaData.cu"

#include "ExceptionManagUtils.cu"
#include "Structs.cu"


#pragma once
template <typename EEY>
array3dWithDimsCPU<EEY>  get3dArrCPU(EEY* arrP, int Nx, int Ny, int Nz) {
    array3dWithDimsCPU<EEY> res;
    res.Nx = Nx;
    res.Ny = Ny;
    res.Nz = Nz;
    res.arrP = arrP;

    return res;
}



template <typename T >
array3dWithDimsGPU<T> allocateMainArray(T*& gpuArrPointer, T*& cpuArrPointer, const int WIDTH, const int HEIGHT, const int DEPTH, cudaStream_t stream) {
    size_t sizeMainArr = (sizeof(T) * WIDTH * HEIGHT * DEPTH);
    array3dWithDimsGPU<T> res;

    cudaMallocAsync(&gpuArrPointer, sizeMainArr, stream);
    cudaMemcpyAsync(gpuArrPointer, cpuArrPointer, sizeMainArr, cudaMemcpyHostToDevice, stream);
    res.arrP = gpuArrPointer;
    res.Nx = WIDTH;
    res.Ny = HEIGHT;
    res.Nz = DEPTH;
    return res;
}




/*
given appropriate cudaPitchedPtr and ForFullBoolPrepArgs will return ForBoolKernelArgs
*/
#pragma once
template <typename TCC>
inline ForBoolKernelArgs<TCC> getArgsForKernel(ForFullBoolPrepArgs<TCC>& mainFunArgs
    , int& warpsNumbForMainPass, int& blockForMainPass
    , const int xLen, const int yLen, const int zLen, cudaStream_t stream
) {

    //main arrays allocations
    TCC* goldArrPointer;
    TCC* segmArrPointer;
    //size_t sizeMainArr = (sizeof(T) * WIDTH * HEIGHT * DEPTH);
    size_t sizeMainArr = (sizeof(TCC) * xLen * yLen * zLen);
    array3dWithDimsGPU<TCC> goldArr = allocateMainArray(goldArrPointer, mainFunArgs.goldArr.arrP, xLen, yLen, zLen, stream);
    array3dWithDimsGPU<TCC> segmArr = allocateMainArray(segmArrPointer, mainFunArgs.segmArr.arrP, xLen, yLen, zLen, stream);
    unsigned int* minMaxes;
    size_t sizeminMaxes = sizeof(unsigned int) * 20;
    cudaMallocAsync(&minMaxes, sizeminMaxes, stream);
    ForBoolKernelArgs<TCC> res;
    res.metaData = allocateMetaDataOnGPU(mainFunArgs.metaData, minMaxes);
    res.metaData.minMaxes = minMaxes;
    res.minMaxes = minMaxes;
    res.numberToLookFor = mainFunArgs.numberToLookFor;
    res.dbXLength = 32;
    res.dbYLength = warpsNumbForMainPass;
    res.dbZLength = 32;

    //printf("in setting bool args ylen %d dbYlen %d calculated meta %d  \n ", yLen, res.dbYLength, int(ceil(yLen / res.dbYLength)));
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
    res.goldArr = goldArr;
    res.segmArr = segmArr;


    return res;
}







#pragma once
template <typename ZZR>
inline MetaDataGPU allocateMemoryAfterMinMaxesKernel(ForBoolKernelArgs<ZZR>& gpuArgs, ForFullBoolPrepArgs<ZZR>& cpuArgs, cudaStream_t stream) {
    ////reduced arrays
    uint32_t* origArr;
    uint32_t* metaDataArr;
    uint32_t* workQueue;
    //copy on cpu
    size_t size = sizeof(unsigned int) * 20;
    cudaMemcpyAsync(cpuArgs.metaData.minMaxes, gpuArgs.minMaxes, size, cudaMemcpyDeviceToHost, stream);

    //read an modify
    //1)maxX 2)minX 3)maxY 4) minY 5) maxZ 6) minZ
    //7)global FP count; 8)global FN count
    unsigned int xRange = cpuArgs.metaData.minMaxes[1] - cpuArgs.metaData.minMaxes[2] + 1;
    unsigned int yRange = cpuArgs.metaData.minMaxes[3] - cpuArgs.metaData.minMaxes[4] + 1;
    unsigned int zRange = cpuArgs.metaData.minMaxes[5] - cpuArgs.metaData.minMaxes[6] + 1;
    unsigned int totalMetaLength = (xRange) * (yRange) * (zRange);
    //updating size informations
    gpuArgs.metaData.metaXLength = xRange;
    gpuArgs.metaData.MetaYLength = yRange;
    gpuArgs.metaData.MetaZLength = zRange;
    gpuArgs.metaData.totalMetaLength = totalMetaLength;
    //saving min maxes
    gpuArgs.metaData.maxX = cpuArgs.metaData.minMaxes[1];
    gpuArgs.metaData.minX = cpuArgs.metaData.minMaxes[2];
    gpuArgs.metaData.maxY = cpuArgs.metaData.minMaxes[3];
    gpuArgs.metaData.minY = cpuArgs.metaData.minMaxes[4];
    gpuArgs.metaData.maxZ = cpuArgs.metaData.minMaxes[5];
    gpuArgs.metaData.minZ = cpuArgs.metaData.minMaxes[6];

    //allocating needed memory
    // main array
    unsigned int mainArrXLength = gpuArgs.dbXLength * gpuArgs.dbYLength;
    unsigned int mainArrSectionLength = (mainArrXLength * 2);
    gpuArgs.metaData.mainArrXLength = mainArrXLength;
    gpuArgs.metaData.mainArrSectionLength = mainArrSectionLength;

    size_t sizeB = totalMetaLength * mainArrSectionLength * sizeof(uint32_t);
    //cudaMallocAsync(&mainArr, sizeB, 0);
    size_t sizeorigArr = totalMetaLength * (mainArrXLength * 2) * sizeof(uint32_t);
    cudaMallocAsync(&origArr, sizeorigArr, stream);
    size_t sizemetaDataArr = totalMetaLength * (20) * sizeof(uint32_t) + 100;
    cudaMallocAsync(&metaDataArr, sizemetaDataArr, stream);
    size_t sizeC = (totalMetaLength * 2 * sizeof(uint32_t) + 50);
    cudaMallocAsync(&workQueue, sizeC, stream);
    gpuArgs.origArrsPointer = origArr;
    gpuArgs.metaDataArrPointer = metaDataArr;
    gpuArgs.workQueuePointer = workQueue;
    return gpuArgs.metaData;
};




/*
becouse we need a lot of the additional memory spaces to minimize memory consumption allocations will be postponed after first kernel run enabling
*/
#pragma once
template <typename ZZR>
inline int allocateMemoryAfterBoolKernel(ForBoolKernelArgs<ZZR>& gpuArgs, ForFullBoolPrepArgs<ZZR>& cpuArgs, cudaStream_t stream) {

    uint32_t* resultListPointerMeta;
    uint32_t* resultListPointerLocal;
    uint32_t* resultListPointerIterNumb;
    uint32_t* mainArrAPointer;
    uint32_t* mainArrBPointer;
    //free no longer needed arrays
    cudaFreeAsync(gpuArgs.goldArr.arrP, stream);
    cudaFreeAsync(gpuArgs.segmArr.arrP, stream);

    //copy on cpu
    size_t size = sizeof(unsigned int) * 20;
    cudaMemcpyAsync(cpuArgs.metaData.minMaxes, gpuArgs.metaData.minMaxes, size, cudaMemcpyDeviceToHost, stream);

    unsigned int fpPlusFn = cpuArgs.metaData.minMaxes[7] + cpuArgs.metaData.minMaxes[8];
    size = sizeof(uint32_t) * (fpPlusFn + 50);


    cudaMallocAsync(&resultListPointerLocal, size, stream);
    cudaMallocAsync(&resultListPointerIterNumb, size, stream);
    cudaMallocAsync(&resultListPointerMeta, size, stream);

    auto xRange = gpuArgs.metaData.metaXLength;
    auto yRange = gpuArgs.metaData.MetaYLength;
    auto zRange = gpuArgs.metaData.MetaZLength;


    size_t sizeB = gpuArgs.metaData.totalMetaLength * gpuArgs.metaData.mainArrSectionLength * sizeof(uint32_t);

    //printf("size of reduced main arr %d total meta len %d mainArrSectionLen %d  \n", sizeB, metaData.totalMetaLength, metaData.mainArrSectionLength);

    cudaMallocAsync(&mainArrAPointer, sizeB, 0);
    cudaMemcpyAsync(mainArrAPointer, gpuArgs.origArrsPointer, sizeB, cudaMemcpyDeviceToDevice, stream);


    cudaMallocAsync(&mainArrBPointer, sizeB, 0);
    cudaMemcpyAsync(mainArrBPointer, gpuArgs.origArrsPointer, sizeB, cudaMemcpyDeviceToDevice, stream);

    //just in order set it to 0
    uint32_t* resultListPointerMetaCPU = (uint32_t*)calloc(fpPlusFn + 50, sizeof(uint32_t));
    cudaMemcpyAsync(resultListPointerMeta, resultListPointerMetaCPU, size, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(resultListPointerIterNumb, resultListPointerMetaCPU, size, cudaMemcpyHostToDevice, stream);
    free(resultListPointerMetaCPU);

    gpuArgs.resultListPointerMeta = resultListPointerMeta;
    gpuArgs.resultListPointerLocal = resultListPointerLocal;
    gpuArgs.resultListPointerIterNumb = resultListPointerIterNumb;

    //fbArgs.origArrsPointer = origArrsPointer;
    gpuArgs.mainArrAPointer = mainArrAPointer;
    gpuArgs.mainArrBPointer = mainArrBPointer;


    return fpPlusFn;
};








#pragma once
template <typename T>
inline void  copyResultstoCPU(ForBoolKernelArgs<T>& gpuArgs, ForFullBoolPrepArgs<T>& cpuArgs, cudaStream_t stream) {


    ////copy on cpu
    size_t size = sizeof(unsigned int) * 20;
    cudaMemcpyAsync(cpuArgs.metaData.minMaxes, gpuArgs.metaData.minMaxes, size, cudaMemcpyDeviceToHost, stream);
    unsigned int fpPlusFn = cpuArgs.metaData.minMaxes[7] + cpuArgs.metaData.minMaxes[8];
    size = sizeof(uint32_t) * (fpPlusFn + 50);

    //uint32_t* resultListPointerMeta = (uint32_t*)calloc(fpPlusFn + 50, sizeof(uint32_t));
    //uint32_t* resultListPointerLocal = (uint32_t*)calloc(fpPlusFn + 50, sizeof(uint32_t));
    //uint32_t* resultListPointerIterNumb = (uint32_t*)calloc(fpPlusFn + 50, sizeof(uint32_t));

    cpuArgs.resultListPointerMeta = (uint32_t*)calloc(fpPlusFn + 50, sizeof(uint32_t));;
    cpuArgs.resultListPointerLocalCPU = (uint32_t*)calloc(fpPlusFn + 50, sizeof(uint32_t));;
    cpuArgs.resultListPointerIterNumb = (uint32_t*)calloc(fpPlusFn + 50, sizeof(uint32_t));;

    cudaMemcpyAsync(cpuArgs.resultListPointerMeta, gpuArgs.resultListPointerMeta, size, cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(cpuArgs.resultListPointerLocalCPU, gpuArgs.resultListPointerLocal, size, cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(cpuArgs.resultListPointerIterNumb, gpuArgs.resultListPointerIterNumb, size, cudaMemcpyDeviceToHost, stream);


};
