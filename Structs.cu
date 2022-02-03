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
    //array3dWithDimsCPU<unsigned int> minMaxes;
    unsigned int* minMaxes;

    ////// counts of false positive and false negatives in given metadata blocks

    ///// sizes of array below will be established on the basis of fp and fn values known after boolKernel finished execution

    //work queue -  workqueue counter already present in minMaxes as entry 9 
    uint32_t* workQueue;
    //in practice it is matrix of length the same as FP+FN global count +1 and width of 5
         //1) xMeta; 2)yMeta 3)zMeta 4)isGold 5)iteration number  
    //we use one single long rewsult list - in order to avoid overwriting each block each block has established offset where it would write it's results 
    uint32_t* resultList;

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

    unsigned int* minMaxes;

    uint32_t* workQueue;
    uint32_t* resultList;

    //represents x from description of main Arr
    unsigned int mainArrXLength;
    //have length 6x+18
    unsigned int mainArrSectionLength;
    //have length 6x 
    unsigned int metaDataOffset;
    // now we will store here also calculated by min maxes kernel values of minimum and maximumvalues 
        //1)maxX 2)minX 3)maxY 4) minY 5) maxZ 6) minZ 
    unsigned int maxX;
    unsigned int minX;
    unsigned int maxY;
    unsigned int minY;
    unsigned int maxZ;
    unsigned int minZ;
};


/*
* Basically holding the arguments for master functions controlling full preparation to get all for Housedorff kernel
*/
#pragma once
template <typename TFF>
struct ForFullBoolPrepArgs {



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


    /*
main array with all required data  organized in sections for each metadata block
x-  is block dimx times block dimy
now what occupies what positions
0-x : reducedGold
(x+1) - 2x : reducedSegm
(2x+1) - 3x : reducedGoldRef
(3x+1) - 4x : reducedSegmRef
(4x+1) - 5x : reducedGoldPrev
(5x+1) - 6x : reducedSegmPrev
6x+1 :fpCount
6x+2 :fnCount
6x+3 :fpCounter
6x+4 :fnCounter
6x+5 :fpOffset
6x+6 :fnOffset
6x+7 :isActiveGold
6x+8 :isFullGold
6x+9 :isActiveSegm
6x+10 :isFullSegm
6x+11 :isToBeActivatedGold
6x+12 :isToBeActivatedSegm
6x+12 :isToBeActivatedSegm
//now linear indexes of the blocks in all sides - if there is no block in given direction it will equal UINT32_MAX
6x+13 : top
6x+14 : bottom
6x+15 : left
6x+16 : right
6x+17 : anterior
6x+18 : posterior
*/
    uint32_t* mainArr;





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