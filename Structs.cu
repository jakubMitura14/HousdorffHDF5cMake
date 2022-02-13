#include "cuda_runtime.h"
#include <cstdint>

#pragma once

//constants describing the meaning of main shared memory spaces
constexpr uint32_t localWorkQueLength = 300;
constexpr uint32_t startOfLocalWorkQ = 4160;
constexpr uint32_t lengthOfMainShmem = 4600;//4460;
constexpr uint32_t begResShmem = 1088;
constexpr uint32_t begfirstRegShmem = 2112;
constexpr uint32_t begSecRegShmem = 3136;
constexpr uint32_t begSMallRegShmemA = 0;
constexpr uint32_t begSMallRegShmemB = 1056;
constexpr uint32_t begSourceShmem = 32;

//as the pipeline is so asynchronous we need some additional storage for saving data related to the operations
//will define how big is the  amount of space in order to be able to use fast version of modulo operator it should be power of 2 ... 
constexpr uint32_t modForPipelineVars = 16;

constexpr uint32_t fpLocCountBeg = 4460;




// other
//added to linear index meta in order to  mark weather block is of type gold or not 
constexpr uint32_t  isGoldOffset = (UINT16_MAX * 10);


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
    //have length 4x
    unsigned int mainArrSectionLength;

    unsigned int metaDataSectionLength = 20;


   // int metaDataOffset = 0;


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

    uint32_t* resultListPointerMeta;
    uint32_t* resultListPointerLocal;
    uint32_t* resultListPointerIterNumb;

    uint32_t* origArrsPointer;
uint32_t* mainArrAPointer;
    uint32_t* mainArrBPointer;
    uint32_t* metaDataArrPointer;

    uint32_t* workQueuePointer;
    unsigned int* minMaxes;


    /*
main array with all required data  organized in sections for each metadata block
x-  is block dimx times block dimy
now what occupies what positions
##mainArrA
(0) - x-1 : reducedGoldRef
(x) - 2x-1 : reducedSegmRef
##mainArrB
() - 3x-1 : reducedGoldPrev
(x) - 4x-1 : reducedSegmPrev

##metaDataArr
0: empty
1 :fpCount
2 :fnCount
3 :fpCounter
4 :fnCounter
5 :fpOffset
6 :fnOffset
7 :isActiveGold
8 :isFullGold
9 :isActiveSegm
10 :isFullSegm
11 :isToBeActivatedGold
12 :isToBeActivatedSegm
//now linear indexes of the blocks in all sides - if there is no block in given direction it will equal UINT32_MAX
13 : top
14 : bottom
15 : left
16 : right
17 : anterior
18 : posterior
19 : empty
20 : empty


###origArrs
0-x : reducedGold
(x+1) - 2x : reducedSegm

*/
 //   uint32_t* mainArr;




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
