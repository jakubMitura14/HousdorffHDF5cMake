

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
array3dWithDimsGPU<T> allocateMainArray(T*& gpuArrPointer, T*& cpuArrPointer, const int WIDTH, const int HEIGHT, const int DEPTH) {
    size_t sizeMainArr = (sizeof(T) * WIDTH * HEIGHT * DEPTH);
    array3dWithDimsGPU<T> res;

    cudaMallocAsync(&gpuArrPointer, sizeMainArr, 0);
    cudaMemcpyAsync(gpuArrPointer, cpuArrPointer, sizeMainArr, cudaMemcpyHostToDevice, 0);
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
    , const int xLen, const int yLen, const int zLen
) {

    //main arrays allocations
    TCC* goldArrPointer;
    TCC* segmArrPointer;

    //size_t sizeMainArr = (sizeof(T) * WIDTH * HEIGHT * DEPTH);
    size_t sizeMainArr = (sizeof(TCC) * xLen * yLen * zLen);


    array3dWithDimsGPU<TCC> goldArr = allocateMainArray(goldArrPointer, mainFunArgs.goldArr.arrP, xLen, yLen, zLen);
    array3dWithDimsGPU<TCC> segmArr = allocateMainArray(segmArrPointer, mainFunArgs.segmArr.arrP, xLen, yLen, zLen);


    unsigned int* minMaxes;
    size_t sizeminMaxes = sizeof(unsigned int) * 20;
    cudaMallocAsync(&minMaxes, sizeminMaxes, 0);

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
inline MetaDataGPU allocateMemoryAfterMinMaxesKernel(ForBoolKernelArgs<ZZR>& gpuArgs, ForFullBoolPrepArgs<ZZR>& cpuArgs) {
    ////reduced arrays


    uint32_t* origArr;

    uint32_t* metaDataArr;

    uint32_t* workQueue;




    //copy on cpu
    size_t size = sizeof(unsigned int) * 20;
    cudaMemcpy(cpuArgs.metaData.minMaxes, gpuArgs.minMaxes, size, cudaMemcpyDeviceToHost);

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
    cudaMallocAsync(&origArr, sizeorigArr, 0);
    size_t sizemetaDataArr = totalMetaLength * (20) * sizeof(uint32_t) + 100;
    cudaMallocAsync(&metaDataArr, sizemetaDataArr, 0);


    size_t sizeC = (totalMetaLength * 2 * sizeof(uint32_t) + 50);
    cudaMallocAsync(&workQueue, sizeC, 0);


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
inline int allocateMemoryAfterBoolKernel(ForBoolKernelArgs<ZZR>& gpuArgs, ForFullBoolPrepArgs<ZZR>& cpuArgs) {


    uint32_t* resultListPointerMeta;
    uint32_t* resultListPointerLocal;
    uint32_t* resultListPointerIterNumb;

    uint32_t* mainArrAPointer;
    uint32_t* mainArrBPointer;

    //free no longer needed arrays
    cudaFreeAsync(gpuArgs.goldArr.arrP, 0);
    cudaFreeAsync(gpuArgs.segmArr.arrP, 0);

    //copy on cpu
    size_t size = sizeof(unsigned int) * 20;
    cudaMemcpy(cpuArgs.metaData.minMaxes, gpuArgs.metaData.minMaxes, size, cudaMemcpyDeviceToHost);

    unsigned int fpPlusFn = cpuArgs.metaData.minMaxes[7] + cpuArgs.metaData.minMaxes[8];


    size = sizeof(uint32_t) * (fpPlusFn + 50);


    cudaMallocAsync(&resultListPointerLocal, size, 0);
    cudaMallocAsync(&resultListPointerIterNumb, size, 0);
    cudaMallocAsync(&resultListPointerMeta, size, 0);


    auto xRange = gpuArgs.metaData.metaXLength;
    auto yRange = gpuArgs.metaData.MetaYLength;
    auto zRange = gpuArgs.metaData.MetaZLength;




    size_t sizeB = gpuArgs.metaData.totalMetaLength * gpuArgs.metaData.mainArrSectionLength * sizeof(uint32_t);

    //printf("size of reduced main arr %d total meta len %d mainArrSectionLen %d  \n", sizeB, metaData.totalMetaLength, metaData.mainArrSectionLength);

    cudaMallocAsync(&mainArrAPointer, sizeB, 0);
    cudaMemcpyAsync(mainArrAPointer, gpuArgs.origArrsPointer, sizeB, cudaMemcpyDeviceToDevice, 0);


    cudaMallocAsync(&mainArrBPointer, sizeB, 0);
    cudaMemcpyAsync(mainArrBPointer, gpuArgs.origArrsPointer, sizeB, cudaMemcpyDeviceToDevice, 0);

    //just in order set it to 0
    uint32_t* resultListPointerMetaCPU = (uint32_t*)calloc(fpPlusFn + 50, sizeof(uint32_t));
    cudaMemcpyAsync(resultListPointerMeta, resultListPointerMetaCPU, size, cudaMemcpyHostToDevice, 0);
    free(resultListPointerMetaCPU);

    gpuArgs.resultListPointerMeta = resultListPointerMeta;
    gpuArgs.resultListPointerLocal = resultListPointerLocal;
    gpuArgs.resultListPointerIterNumb = resultListPointerIterNumb;

    //fbArgs.origArrsPointer = origArrsPointer;
    gpuArgs.mainArrAPointer = mainArrAPointer;
    gpuArgs.mainArrBPointer = mainArrBPointer;


    return fpPlusFn;
};































////////////////// with pipeline ofr barrier

/*
initial cleaning  and initializations of dilatation kernel

*/
#pragma once
inline __device__  void dilBlockInitialClean(thread_block_tile<32>& tile,
    const  bool isPaddingPass, int(&iterationNumb)[1],
    unsigned int(&localWorkQueueCounter)[1], unsigned int(&blockFpConter)[1],
    unsigned int(&blockFnConter)[1], unsigned int(&localFpConter)[1],
    unsigned int(&localFnConter)[1], bool(&isBlockFull)[2],
    unsigned int(&fpFnLocCounter)[1],
    unsigned int(&localTotalLenthOfWorkQueue)[1], unsigned int(&globalWorkQueueOffset)[1]
    , unsigned int(&worQueueStep)[1], unsigned int*& minMaxes, unsigned int(&localMinMaxes)[5], uint32_t(&lastI)[1])
{

    if (tile.thread_rank() == 7 && tile.meta_group_rank() == 0 && !isPaddingPass) {
        iterationNumb[0] += 1;
    };

    if (tile.thread_rank() == 6 && tile.meta_group_rank() == 0) {
        localWorkQueueCounter[0] = 0;
    };

    if (tile.thread_rank() == 1 && tile.meta_group_rank() == 0) {
        blockFpConter[0] = 0;
    };
    if (tile.thread_rank() == 2 && tile.meta_group_rank() == 0) {
        blockFnConter[0] = 0;
    };
    if (tile.thread_rank() == 3 && tile.meta_group_rank() == 0) {
        localFpConter[0] = 0;
    };
    if (tile.thread_rank() == 4 && tile.meta_group_rank() == 0) {
        localFnConter[0] = 0;
    };
    if (tile.thread_rank() == 9 && tile.meta_group_rank() == 0) {
        isBlockFull[0] = true;
    };
    if (tile.thread_rank() == 9 && tile.meta_group_rank() == 1) {
        isBlockFull[1] = true;
    };

    if (tile.thread_rank() == 10 && tile.meta_group_rank() == 0) {
        fpFnLocCounter[0] = 0;
    };


    if (tile.thread_rank() == 10 && tile.meta_group_rank() == 2) {// this is how it is encoded wheather it is gold or segm block

        lastI[0] = UINT32_MAX;
    };


    if (tile.thread_rank() == 0 && tile.meta_group_rank() == 0) {
        localTotalLenthOfWorkQueue[0] = minMaxes[9];
        globalWorkQueueOffset[0] = floor((float)(localTotalLenthOfWorkQueue[0] / gridDim.x)) + 1;
        worQueueStep[0] = min(localWorkQueLength, globalWorkQueueOffset[0]);
    };
    /* will be used to store all of the minMaxes varibles from global memory (from 7 to 11)
0 : global FP count;
1 : global FN count;
2 : workQueueCounter
3 : resultFP globalCounter
4 : resultFn globalCounter
*/
    if (tile.meta_group_rank() == 1) {
        cooperative_groups::memcpy_async(tile, (&localMinMaxes[0]), (&minMaxes[7]), cuda::aligned_size_t<4>(sizeof(unsigned int) * 5));
    }
}



/*
load work que from global memory
*/
#pragma once
inline __device__  void loadWorkQueue(thread_block& cta, uint32_t(&mainShmem)[lengthOfMainShmem], uint32_t*& workQueue
    , bool(&isGoldForLocQueue)[localWorkQueLength], uint32_t& bigloop, unsigned int(&worQueueStep)[1]) {

    //to do change into barrier

    //cuda::memcpy_async(cta, (&mainShmem[startOfLocalWorkQ]), (&workQueue[bigloop])
    //    , cuda::aligned_size_t<4>(sizeof(uint32_t) * worQueueStep[0]), pipeline);

    for (uint16_t ii = cta.thread_rank(); ii < worQueueStep[0]; ii += cta.size()) {
        mainShmem[startOfLocalWorkQ + ii] = workQueue[bigloop + ii];
        isGoldForLocQueue[ii] = (mainShmem[startOfLocalWorkQ + ii] >= isGoldOffset);
        mainShmem[startOfLocalWorkQ + ii] = mainShmem[startOfLocalWorkQ + ii] - isGoldOffset * isGoldForLocQueue[ii];

    }
}


/*
loads metadata of given block to meta data
*/
#pragma once
inline __device__  void loadMetaDataToShmem(thread_block& cta, uint32_t(&localBlockMetaData)[40]
    , uint32_t(&mainShmem)[lengthOfMainShmem], cuda::pipeline<cuda::thread_scope_block>& pipeline
    , uint32_t*& metaDataArr, MetaDataGPU& metaData, const uint8_t toAdd, uint32_t& ii) {

    //cuda::memcpy_async(cta, (&localBlockMetaData[(ii&1)*20]),
    //    (&metaDataArr[(mainShmem[startOfLocalWorkQ + toAdd+ii])
    //        * metaData.metaDataSectionLength])
    //    , cuda::aligned_size_t<4>(sizeof(uint32_t) * 20), pipeline);

    cuda::memcpy_async(cta, (&localBlockMetaData[((ii + 1) & 1) * 20]),
        (&metaDataArr[(mainShmem[startOfLocalWorkQ + toAdd + ii])
            * metaData.metaDataSectionLength])
        , cuda::aligned_size_t<4>(sizeof(uint32_t) * 20), pipeline);


}





////////////////////MAin
/*
loading data about this block to shmem
*/
#pragma once
template <typename TXPI>
inline __device__  void loadMain(ForBoolKernelArgs<TXPI>& fbArgs, thread_block& cta, uint32_t(&localBlockMetaData)[40]
    , uint32_t(&mainShmem)[lengthOfMainShmem], cuda::pipeline<cuda::thread_scope_block>& pipeline
    , uint32_t*& metaDataArr, MetaDataGPU& metaData, uint32_t& i, thread_block_tile<32>& tile
    , bool(&isGoldForLocQueue)[localWorkQueLength], int(&iterationNumb)[1]) {

    pipeline.producer_acquire();
    //auto inMainLineMeta = mainShmem[startOfLocalWorkQ + i] ;
    //auto inMainFullIndex = inMainLineMeta * metaData.mainArrSectionLength + metaData.mainArrXLength * (1 - isGoldForLocQueue[i]);
    //printf("inMain load full index %d \n ", inMainFullIndex);

    //cuda::memcpy_async(cta, &mainShmem[begSourceShmem], &getSourceReduced(fbArgs, iterationNumb)[inMainFullIndex],
    //    cuda::aligned_size_t<128>(sizeof(uint32_t) * metaData.mainArrXLength), pipeline);
    //pipeline.producer_commit();


    cuda::memcpy_async(cta, &mainShmem[begSourceShmem], &getSourceReduced(fbArgs, iterationNumb)[
        mainShmem[startOfLocalWorkQ + i] * metaData.mainArrSectionLength + metaData.mainArrXLength * (1 - isGoldForLocQueue[i])],
        cuda::aligned_size_t<128>(sizeof(uint32_t) * metaData.mainArrXLength), pipeline);
    pipeline.producer_commit();


}

/*
process data about this block
*/
#pragma once
template <typename TXPI>
inline __device__  void processMain(ForBoolKernelArgs<TXPI>& fbArgs, thread_block& cta, uint32_t(&localBlockMetaData)[40]
    , uint32_t(&mainShmem)[lengthOfMainShmem], cuda::pipeline<cuda::thread_scope_block>& pipeline
    , uint32_t*& metaDataArr, MetaDataGPU& metaData, uint32_t& i, thread_block_tile<32>& tile
    , bool(&isGoldForLocQueue)[localWorkQueLength], int(&iterationNumb)[1], bool(&isBlockFull)[2]) {


    pipeline.consumer_wait();

    if (__popc(mainShmem[begSourceShmem + threadIdx.x + threadIdx.y * 32]) < 32) {
        isBlockFull[i & 1] = false;
    }

    mainShmem[begResShmem + threadIdx.x + threadIdx.y * 32] = bitDilatate(mainShmem[begSourceShmem + threadIdx.x + threadIdx.y * 32]);
    //marking weather block is already full and no more dilatations are possible 


    pipeline.consumer_release();


}

////////////////TOP
/*
loading data about block above to shmem
*/
#pragma once
template <typename TXPI>
inline __device__  void loadTop(ForBoolKernelArgs<TXPI>& fbArgs, thread_block& cta, uint32_t(&localBlockMetaData)[40]
    , uint32_t(&mainShmem)[lengthOfMainShmem], cuda::pipeline<cuda::thread_scope_block>& pipeline
    , uint32_t*& metaDataArr, MetaDataGPU& metaData, uint32_t& i, thread_block_tile<32>& tile
    , bool(&isGoldForLocQueue)[localWorkQueLength], int(&iterationNumb)[1]) {

    pipeline.producer_acquire();
    if (localBlockMetaData[(i & 1) * 20 + 13] < isGoldOffset) {
        cuda::memcpy_async(cta, (&mainShmem[begfirstRegShmem]),
            &getSourceReduced(fbArgs, iterationNumb)[localBlockMetaData[(i & 1) * 20 + 13]
            * metaData.mainArrSectionLength + metaData.mainArrXLength * (1 - isGoldForLocQueue[i])],
            cuda::aligned_size_t<128>(sizeof(uint32_t) * metaData.mainArrXLength)
            , pipeline);
    }
    pipeline.producer_commit();

}


/*
loading data about block above to shmem
*/
#pragma once
template <typename TXPI>
inline __device__  void processTop(ForBoolKernelArgs<TXPI>& fbArgs, thread_block& cta, uint32_t(&localBlockMetaData)[40]
    , uint32_t(&mainShmem)[lengthOfMainShmem], cuda::pipeline<cuda::thread_scope_block>& pipeline
    , uint32_t*& metaDataArr, MetaDataGPU& metaData, uint32_t& i, thread_block_tile<32>& tile
    , bool(&isGoldForLocQueue)[localWorkQueLength], int(&iterationNumb)[1], bool(&isAnythingInPadding)[6]) {

    pipeline.consumer_wait();

    dilatateHelperTopDown(0, mainShmem, isAnythingInPadding, localBlockMetaData, 13
        , 31, 0
        , begfirstRegShmem, i);

    pipeline.consumer_release();

}

/////BOTTOM
#pragma once
template <typename TXPI>
inline __device__  void loadBottom(ForBoolKernelArgs<TXPI>& fbArgs, thread_block& cta, uint32_t(&localBlockMetaData)[40]
    , uint32_t(&mainShmem)[lengthOfMainShmem], cuda::pipeline<cuda::thread_scope_block>& pipeline
    , uint32_t*& metaDataArr, MetaDataGPU& metaData, uint32_t& i, thread_block_tile<32>& tile
    , bool(&isGoldForLocQueue)[localWorkQueLength], int(&iterationNumb)[1], bool(&isAnythingInPadding)[6]) {

    pipeline.producer_acquire();
    if (localBlockMetaData[(i & 1) * 20 + 14] < isGoldOffset) {
        cuda::memcpy_async(cta, (&mainShmem[begSecRegShmem]),
            &getSourceReduced(fbArgs, iterationNumb)[localBlockMetaData[(i & 1) * 20 + 14]
            * metaData.mainArrSectionLength + metaData.mainArrXLength * (1 - isGoldForLocQueue[i])],
            cuda::aligned_size_t<128>(sizeof(uint32_t) * metaData.mainArrXLength)
            , pipeline);
    }
    pipeline.producer_commit();

}
#pragma once
template <typename TXPI>
inline __device__  void processBottom(ForBoolKernelArgs<TXPI>& fbArgs, thread_block& cta, uint32_t(&localBlockMetaData)[40]
    , uint32_t(&mainShmem)[lengthOfMainShmem], cuda::pipeline<cuda::thread_scope_block>& pipeline
    , uint32_t*& metaDataArr, MetaDataGPU& metaData, uint32_t& i, thread_block_tile<32>& tile
    , bool(&isGoldForLocQueue)[localWorkQueLength], int(&iterationNumb)[1], bool(&isAnythingInPadding)[6]) {

    pipeline.consumer_wait();

    dilatateHelperTopDown(1, mainShmem, isAnythingInPadding, localBlockMetaData, 14
        , 0, 31
        , begSecRegShmem, i);

    pipeline.consumer_release();

}






///////////// right
#pragma once
template <typename TXPI>
inline __device__  void loadRight(ForBoolKernelArgs<TXPI>& fbArgs, thread_block& cta, uint32_t(&localBlockMetaData)[40]
    , uint32_t(&mainShmem)[lengthOfMainShmem], cuda::pipeline<cuda::thread_scope_block>& pipeline
    , uint32_t*& metaDataArr, MetaDataGPU& metaData, uint32_t& i, thread_block_tile<32>& tile
    , bool(&isGoldForLocQueue)[localWorkQueLength], int(&iterationNumb)[1], bool(&isAnythingInPadding)[6]) {



    pipeline.producer_acquire();
    if (localBlockMetaData[(i & 1) * 20 + 16] < isGoldOffset) {
        cuda::memcpy_async(cta, (&mainShmem[begfirstRegShmem]),
            &getSourceReduced(fbArgs, iterationNumb)[localBlockMetaData[(i & 1) * 20 + 16] * metaData.mainArrSectionLength + metaData.mainArrXLength * (1 - isGoldForLocQueue[i])],
            cuda::aligned_size_t<128>(sizeof(uint32_t) * metaData.mainArrXLength)
            , pipeline);
    }
    pipeline.producer_commit();
}

#pragma once
template <typename TXPI>
inline __device__  void processRight(ForBoolKernelArgs<TXPI>& fbArgs, thread_block& cta, uint32_t(&localBlockMetaData)[40]
    , uint32_t(&mainShmem)[lengthOfMainShmem], cuda::pipeline<cuda::thread_scope_block>& pipeline
    , uint32_t*& metaDataArr, MetaDataGPU& metaData, uint32_t& i, thread_block_tile<32>& tile
    , bool(&isGoldForLocQueue)[localWorkQueLength], int(&iterationNumb)[1], bool(&isAnythingInPadding)[6]) {


    pipeline.consumer_wait();

    dilatateHelperForTransverse(fbArgs, (threadIdx.x == (fbArgs.dbXLength - 1)),
        3, (1), (0), mainShmem, isAnythingInPadding
        , threadIdx.y, 0
        , 16, begfirstRegShmem, localBlockMetaData, i, isGoldForLocQueue);

    pipeline.consumer_release();
}



///////////// left
#pragma once
template <typename TXPI>
inline __device__  void loadLeft(ForBoolKernelArgs<TXPI>& fbArgs, thread_block& cta, uint32_t(&localBlockMetaData)[40]
    , uint32_t(&mainShmem)[lengthOfMainShmem], cuda::pipeline<cuda::thread_scope_block>& pipeline
    , uint32_t*& metaDataArr, MetaDataGPU& metaData, uint32_t& i, thread_block_tile<32>& tile
    , bool(&isGoldForLocQueue)[localWorkQueLength], int(&iterationNumb)[1], bool(&isAnythingInPadding)[6]) {



    pipeline.producer_acquire();
    if (mainShmem[startOfLocalWorkQ + i] > 0) {
        cuda::memcpy_async(cta, (&mainShmem[begSecRegShmem]),
            &getSourceReduced(fbArgs, iterationNumb)[(mainShmem[startOfLocalWorkQ + i] - 1) * metaData.mainArrSectionLength + metaData.mainArrXLength * (1 - isGoldForLocQueue[i])],
            cuda::aligned_size_t<128>(sizeof(uint32_t) * metaData.mainArrXLength)
            , pipeline);
    }
    pipeline.producer_commit();
}

#pragma once
template <typename TXPI>
inline __device__  void processLeft(ForBoolKernelArgs<TXPI>& fbArgs, thread_block& cta, uint32_t(&localBlockMetaData)[40]
    , uint32_t(&mainShmem)[lengthOfMainShmem], cuda::pipeline<cuda::thread_scope_block>& pipeline
    , uint32_t*& metaDataArr, MetaDataGPU& metaData, uint32_t& i, thread_block_tile<32>& tile
    , bool(&isGoldForLocQueue)[localWorkQueLength], int(&iterationNumb)[1], bool(&isAnythingInPadding)[6]) {


    pipeline.consumer_wait();

    dilatateHelperForTransverse(fbArgs, (threadIdx.x == 0),
        2, (-1), (0), mainShmem, isAnythingInPadding
        , threadIdx.y, 31
        , 15, begSecRegShmem, localBlockMetaData, i, isGoldForLocQueue);

    pipeline.consumer_release();
}

///////////// anterior
#pragma once
template <typename TXPI>
inline __device__  void loadAnterior(ForBoolKernelArgs<TXPI>& fbArgs, thread_block& cta, uint32_t(&localBlockMetaData)[40]
    , uint32_t(&mainShmem)[lengthOfMainShmem], cuda::pipeline<cuda::thread_scope_block>& pipeline
    , uint32_t*& metaDataArr, MetaDataGPU& metaData, uint32_t& i, thread_block_tile<32>& tile
    , bool(&isGoldForLocQueue)[localWorkQueLength], int(&iterationNumb)[1], bool(&isAnythingInPadding)[6]) {

    pipeline.producer_acquire();
    if (localBlockMetaData[(i & 1) * 20 + 17] < isGoldOffset) {

        cuda::memcpy_async(cta, (&mainShmem[begfirstRegShmem]),
            &getSourceReduced(fbArgs, iterationNumb)[localBlockMetaData[(i & 1) * 20 + 17] * metaData.mainArrSectionLength + metaData.mainArrXLength * (1 - isGoldForLocQueue[i])],
            cuda::aligned_size_t<128>(sizeof(uint32_t) * metaData.mainArrXLength)
            , pipeline);
    }
    pipeline.producer_commit();
}

#pragma once
template <typename TXPI>
inline __device__  void processAnterior(ForBoolKernelArgs<TXPI>& fbArgs, thread_block& cta, uint32_t(&localBlockMetaData)[40]
    , uint32_t(&mainShmem)[lengthOfMainShmem], cuda::pipeline<cuda::thread_scope_block>& pipeline
    , uint32_t*& metaDataArr, MetaDataGPU& metaData, uint32_t& i, thread_block_tile<32>& tile
    , bool(&isGoldForLocQueue)[localWorkQueLength], int(&iterationNumb)[1], bool(&isAnythingInPadding)[6]) {

    pipeline.consumer_wait();

    dilatateHelperForTransverse(fbArgs, (threadIdx.y == (fbArgs.dbYLength - 1)), 4
        , (0), (1), mainShmem, isAnythingInPadding
        , 0, threadIdx.x
        , 17, begfirstRegShmem, localBlockMetaData, i, isGoldForLocQueue);
    pipeline.consumer_release();
}

///////////// posterior
#pragma once
template <typename TXPI>
inline __device__  void loadPosterior(ForBoolKernelArgs<TXPI>& fbArgs, thread_block& cta, uint32_t(&localBlockMetaData)[40]
    , uint32_t(&mainShmem)[lengthOfMainShmem], cuda::pipeline<cuda::thread_scope_block>& pipeline
    , uint32_t*& metaDataArr, MetaDataGPU& metaData, uint32_t& i, thread_block_tile<32>& tile
    , bool(&isGoldForLocQueue)[localWorkQueLength], int(&iterationNumb)[1], bool(&isAnythingInPadding)[6]) {

    pipeline.producer_acquire();
    if (localBlockMetaData[(i & 1) * 20 + 18] < isGoldOffset) {


        cuda::memcpy_async(cta, (&mainShmem[begSecRegShmem]),
            &getSourceReduced(fbArgs, iterationNumb)[localBlockMetaData[(i & 1) * 20 + 18] * metaData.mainArrSectionLength + metaData.mainArrXLength * (1 - isGoldForLocQueue[i])],
            cuda::aligned_size_t<128>(sizeof(uint32_t) * metaData.mainArrXLength)
            , pipeline);
    }
    pipeline.producer_commit();
}





//////////// last load 

/*
load reference if needed or data for next iteration if there is such
*/
#pragma once
template <typename TXPI>
inline __device__  void lastLoad(ForBoolKernelArgs<TXPI>& fbArgs, thread_block& cta, uint32_t(&localBlockMetaData)[40]
    , uint32_t(&mainShmem)[lengthOfMainShmem], cuda::pipeline<cuda::thread_scope_block>& pipeline
    , uint32_t*& metaDataArr, MetaDataGPU& metaData, uint32_t& i, thread_block_tile<32>& tile
    , bool(&isGoldForLocQueue)[localWorkQueLength], int(&iterationNumb)[1], bool(&isAnythingInPadding)[6]
    , uint32_t*& origArrs, unsigned int(&worQueueStep)[1]) {

    pipeline.producer_acquire();

    //if block should be validated we load data for validation
    if (localBlockMetaData[(i & 1) * 20 + ((1 - isGoldForLocQueue[i]) + 1)] //fp for gold and fn count for not gold
    > localBlockMetaData[(i & 1) * 20 + ((1 - isGoldForLocQueue[i]) + 3)]) {// so count is bigger than counter so we should validate
        cuda::memcpy_async(cta, (&mainShmem[begfirstRegShmem]),
            &origArrs[mainShmem[startOfLocalWorkQ + i] * metaData.mainArrSectionLength + metaData.mainArrXLength * (isGoldForLocQueue[i])], //we look for 
            cuda::aligned_size_t<128>(sizeof(uint32_t) * metaData.mainArrXLength)
            , pipeline);

    }
    else {//if we are not validating we immidiately start loading data for next loop
        if (i + 1 < worQueueStep[0]) {
            cuda::memcpy_async(cta, (&localBlockMetaData[((i + 1) & 1) * 20]),
                (&metaDataArr[(mainShmem[startOfLocalWorkQ + 1 + i])
                    * metaData.metaDataSectionLength])
                , cuda::aligned_size_t<4>(sizeof(uint32_t) * 20), pipeline);


        }
    }


    pipeline.producer_commit();
}
#pragma once
template <typename TXPI>
inline __device__  void processPosteriorAndSaveResShmem(ForBoolKernelArgs<TXPI>& fbArgs, thread_block& cta
    , uint32_t(&localBlockMetaData)[40]
    , uint32_t(&mainShmem)[lengthOfMainShmem], cuda::pipeline<cuda::thread_scope_block>& pipeline
    , uint32_t*& metaDataArr, MetaDataGPU& metaData, uint32_t& i, thread_block_tile<32>& tile
    , bool(&isGoldForLocQueue)[localWorkQueLength], int(&iterationNumb)[1], bool(&isAnythingInPadding)[6],
    bool(&isBlockFull)[2]) {

    pipeline.consumer_wait();
    //dilatate posterior 
    dilatateHelperForTransverse(fbArgs, (threadIdx.y == 0), 5
        , (0), (-1), mainShmem, isAnythingInPadding
        , fbArgs.dbYLength - 1, threadIdx.x // we add offset depending on y dimension
        , 18, begSecRegShmem, localBlockMetaData, i, isGoldForLocQueue);
    //now all data should be properly dilatated we save it to global memory
    //try save target reduced via mempcy async ...

    getTargetReduced(fbArgs, iterationNumb)[mainShmem[startOfLocalWorkQ + i] * metaData.mainArrSectionLength + metaData.mainArrXLength * (1 - isGoldForLocQueue[i])
        + threadIdx.x + threadIdx.y * 32]
        = mainShmem[begResShmem + threadIdx.x + threadIdx.y * 32];



    pipeline.consumer_release();
}



//////////// validation
#pragma once
template <typename TXPI>
inline __device__  void validate(ForBoolKernelArgs<TXPI>& fbArgs, thread_block& cta, uint32_t(&localBlockMetaData)[40]
    , uint32_t(&mainShmem)[lengthOfMainShmem], cuda::pipeline<cuda::thread_scope_block>& pipeline
    , uint32_t*& metaDataArr, MetaDataGPU& metaData, uint32_t& i, thread_block_tile<32>& tile
    , bool(&isGoldForLocQueue)[localWorkQueLength], int(&iterationNumb)[1], bool(&isAnythingInPadding)[6]
    , bool(&isBlockFull)[2]
    , unsigned int(&localFpConter)[1], unsigned int(&localFnConter)[1]
    , uint32_t*& resultListPointerMeta, uint32_t*& resultListPointerLocal, uint32_t*& resultListPointerIterNumb

) {

    if (localBlockMetaData[(i & 1) * 20 + ((1 - isGoldForLocQueue[i]) + 1)] //fp for gold and fn count for not gold
        > localBlockMetaData[(i & 1) * 20 + ((1 - isGoldForLocQueue[i]) + 3)]) {// so count is bigger than counter so we should validate
            //mainShmem[begSourceShmem + threadIdx.x + threadIdx.y * 32] = 
            //    ((~mainShmem[begSourceShmem + threadIdx.x + threadIdx.y * 32]) 
            //        & mainShmem[begResShmem + threadIdx.x + threadIdx.y * 32]);



            //we now look for bits prasent in both reference arrays and current one
           // mainShmem[begSourceShmem + threadIdx.x + threadIdx.y * 32] = ((mainShmem[begSourceShmem + threadIdx.x + threadIdx.y * 32]) & mainShmem[begfirstRegShmem + threadIdx.x + threadIdx.y * 32]);

            // now we look through bits and when some is set we call it a result 
#pragma unroll
        for (uint8_t bitPos = 0; bitPos < 32; bitPos++) {
            //if any bit here is set it means it should be added to result list 
            if (isBitAt(mainShmem[begResShmem + threadIdx.x + threadIdx.y * 32], bitPos)
                && !isBitAt(mainShmem[begSourceShmem + threadIdx.x + threadIdx.y * 32], bitPos)
                && isBitAt(mainShmem[begfirstRegShmem + threadIdx.x + threadIdx.y * 32], bitPos)
                ) {

                //just re
                mainShmem[begSecRegShmem + threadIdx.x + threadIdx.y * 32] = 0;
                ////// IMPORTANT for some reason in order to make it work resultfnOffset and resultfnOffset swith places
                if (isGoldForLocQueue[i]) {
                    mainShmem[begSecRegShmem + threadIdx.x + threadIdx.y * 32] = uint32_t(atomicAdd_block(&(localFpConter[0]), 1) + localBlockMetaData[(i & 1) * 20 + 6] + localBlockMetaData[(i & 1) * 20 + 3]);
                }
                else {
                    mainShmem[begSecRegShmem + threadIdx.x + threadIdx.y * 32] = uint32_t(atomicAdd_block(&(localFnConter[0]), 1) + localBlockMetaData[(i & 1) * 20 + 5] + localBlockMetaData[(i & 1) * 20 + 4]);
                    //    printf("local fn counter add \n");

                };
                //   add results to global memory    
                //we add one gere jjust to distinguish it from empty result
                resultListPointerMeta[mainShmem[begSecRegShmem + threadIdx.x + threadIdx.y * 32]] = uint32_t(mainShmem[startOfLocalWorkQ + i] + (isGoldOffset * isGoldForLocQueue[i]) + 1);
                resultListPointerLocal[mainShmem[begSecRegShmem + threadIdx.x + threadIdx.y * 32]] = uint32_t((fbArgs.dbYLength * 32 * bitPos) + (threadIdx.y * 32) + (threadIdx.x));
                resultListPointerIterNumb[mainShmem[begSecRegShmem + threadIdx.x + threadIdx.y * 32]] = uint32_t(iterationNumb[0]);



            }

        };

    }
}


