#include "cuda_runtime.h"
#include <cstdint>

#pragma once
constexpr auto localWorkQueLength = 32;
constexpr auto localWorkQueLengthDiv32 = 1;
// includes localWorkQueLength and source and res shmem
constexpr auto totalCombinedShmemWorkQueue = (8 * 32) + localWorkQueLength;

/**
In order to be able to use cuda malloc 3d we will implemnt it as a series
of 3d arrays
*/



#pragma once
template <typename TFPP>
struct array3dWithDimsCPU {
    TFPP*** arrP;
    int Nx;
    int Ny;
    int Nz;
};


#pragma once
struct array3dWithDimsGPU {
    cudaPitchedPtr arrPStr;
    int Nx;
    int Ny;
    int Nz;
};


#pragma once
extern "C" struct MetaDataCPU {
    int metaXLength;
    int MetaYLength;
    int MetaZLength;
    int totalMetaLength;


    //1)maxX 2)minX 3)maxY 4) minY 5) maxZ 6) minZ - minimal and maximum coordinates of blocks with some entries of intrest
    //7)global FP count; 8)global FN count  9) workQueueCounter 10)resultFP globalCounter 11) resultFn globalCounter 
     //12) global FPandFn offset 13)globalIterationNumb
    array3dWithDimsCPU<unsigned int> minMaxes;
    ////// counts of false positive and false negatives in given metadata blocks

    array3dWithDimsCPU<unsigned int> fpCount;
    array3dWithDimsCPU<unsigned int> fnCount;
    //variables needed to add result to correct spot and keep information about it
    //counts how many fps or fns had been already covered in this data block
    array3dWithDimsCPU<unsigned int> fpCounter;
    array3dWithDimsCPU<unsigned int> fnCounter;
    //tells  what is the offset in result list where space for this data block is given
    array3dWithDimsCPU<unsigned int> fpOffset;
    array3dWithDimsCPU<unsigned int> fnOffset;

    // variables neded to establish is block should be put into workqueue
    array3dWithDimsCPU<bool> isActiveGold;
    array3dWithDimsCPU<bool> isFullGold;

    array3dWithDimsCPU<bool> isActiveSegm;
    array3dWithDimsCPU<bool> isFullSegm;

    array3dWithDimsCPU<bool> isToBeActivatedGold;
    array3dWithDimsCPU<bool> isToBeActivatedSegm;


    array3dWithDimsCPU<bool> isToBeValidatedFp;
    array3dWithDimsCPU<bool> isToBeValidatedFn;

    ///// sizes of array below will be established on the basis of fp and fn values known after boolKernel finished execution

    //work queue -  workqueue counter already present in minMaxes as entry 9 
    //in practice it is matrix of length the same as FP+FN global count +1 and width of 4 
        //1) xMeta; 2)yMeta 3)zMeta 4)isGold
    array3dWithDimsCPU<uint16_t> workQueue;
    //in practice it is matrix of length the same as FP+FN global count +1 and width of 5
         //1) xMeta; 2)yMeta 3)zMeta 4)isGold 5)iteration number  
    //we use one single long rewsult list - in order to avoid overwriting each block each block has established offset where it would write it's results 
    uint16_t* resultList;

};

#pragma once
extern "C" struct MetaDataGPU {
    int metaXLength;
    int MetaYLength;
    int MetaZLength;
    int totalMetaLength;

    //1)maxX 2)minX 3)maxY 4) minY 5) maxZ 6) minZ - minimal and maximum coordinates of blocks with some entries of intrest
    //7)global FP count; 8)global FN count 9) workQueueCounter 10)resultFP globalCounter 11) resultFn globalCounter
    //12) global FPandFn offset 13)globalIterationNumb

    array3dWithDimsGPU minMaxes;

    array3dWithDimsGPU fpCount;
    array3dWithDimsGPU fnCount;

    array3dWithDimsGPU fpCounter;
    array3dWithDimsGPU fnCounter;

    array3dWithDimsGPU fpOffset;
    array3dWithDimsGPU fnOffset;


    array3dWithDimsGPU isActiveGold;
    array3dWithDimsGPU isFullGold;
    array3dWithDimsGPU isActiveSegm;
    array3dWithDimsGPU isFullSegm;

    array3dWithDimsGPU isToBeActivatedGold;
    array3dWithDimsGPU isToBeActivatedSegm;


    array3dWithDimsGPU isToBeValidatedFp;
    array3dWithDimsGPU isToBeValidatedFn;

    array3dWithDimsGPU workQueue;
    uint16_t* resultList;


};


/*
* Basically holding the arguments for master functions controlling full preparation to get all for Housedorff kernel
*/
#pragma once
template <typename TFF>
struct ForFullBoolPrepArgs {
    //pointer to reduced arrays holders will be used for dilatation
    array3dWithDimsCPU<uint32_t> reducedGold;
    array3dWithDimsCPU<uint32_t> reducedSegm;
    //will be used as reference - will not be dilatated
    array3dWithDimsCPU<uint32_t> reducedGoldRef;
    array3dWithDimsCPU<uint32_t> reducedSegmRef;
    // space in global memory where one can store padding information
    array3dWithDimsCPU<uint32_t> reducedGoldPrev;
    array3dWithDimsCPU<uint32_t> reducedSegmPrev;
    int reducedArrsZdim;// x and y dimensions are like normal arrays but z dimension gets reduced
    //metadata struct
    MetaDataCPU metaData;
    //pointer to the array used to debug
    array3dWithDimsCPU<int> forDebugArr;
    // dimensions of data block
    int dbXLength;
    int dbYLength;
    int dbZLength;
    // gold standard and segmentation output array
    array3dWithDimsCPU<TFF> goldArr;
    array3dWithDimsCPU<TFF> segmArr;
    TFF numberToLookFor;// what we will look for in arrays

    //number and dimensionality of threads and blocks required to lounch bool kernel
    dim3 threads;
    int blocks;
    //threads and blocks for first metadata pass kernel
    int threadsFirstMetaDataPass;
    int blocksFirstMetaDataPass;
    //threads and blocks for main pass 
    dim3 threadsMainPass;
    int blocksMainPass;
    //threads and blocks for padding pass 
    dim3 threadsPaddingPass;
    int blocksPaddingPass;
    //threads and blocks for non first metadata passes 
    int threadsOtherMetaDataPasses;
    int blocksOtherMetaDataPasses;
    // will establish how many points we want to include in dilatation and how many we can ignore so typically set to 95% - so we will ignore only 5% most distant
    float robustnessPercent = 0.95;

};





/*
* Basically holding the arguments for main kernel in the FullBoolPrep
*/
#pragma once
template <typename TFB>
struct ForBoolKernelArgs {
    //matadata struct
    MetaDataGPU metaData;
    //pointer to the array used to debug
    array3dWithDimsGPU forDebugArr;

    // dimensions of data block
    int dbXLength;
    int dbYLength;
    int dbZLength;
    // gold standard and segmentation output array
    array3dWithDimsGPU goldArr;
    array3dWithDimsGPU segmArr;
    TFB numberToLookFor;

    //pointer to reduced arrays holders
    array3dWithDimsGPU reducedGold;
    array3dWithDimsGPU reducedSegm;

    array3dWithDimsGPU reducedGoldRef;
    array3dWithDimsGPU reducedSegmRef;

    array3dWithDimsGPU reducedGoldPrev;
    array3dWithDimsGPU reducedSegmPrev;
    float robustnessPercent = 0.95;

};





//just utility for unit testing - set some data bout points
#pragma once
extern "C"  struct forTestPointStruct {
    int x;
    int y;
    int z;

    bool isGold;
    bool isGoldAndSegm;

    int xMeta;
    int yMeta;
    int zMeta;



    bool shouldBeInResAfterOneDil;
    bool shouldBeInResAfterTwoDil;


};


#pragma once
extern "C" struct forTestMetaDataStruct {

    int xMeta;
    int yMeta;
    int zMeta;

    int requiredspaceInFpResultList;
    int requiredspaceInFnResultList;

    bool isToBeActiveAtStart;
    bool isToBeActiveAfterOneIter;
    bool isToBeActiveAfterTwoIter;

    bool isToBeValidatedFpAfterOneIter;
    bool isToBeValidatedFpAfterTwoIter;

    bool isToBeValidatedFnAfterOneIter;
    bool isToBeValidatedFnAfterTwoIter;


    bool isToBeFullAfterOneIter;
    bool isToBeFullAfterTwoIter;

    int fpCount;
    int fnCount;

    int fpConterAfterOneDil;
    int fpConterAfterTwoDil;

    int fnConterAfterOneDil;
    int fnConterAfterTwoDil;


};