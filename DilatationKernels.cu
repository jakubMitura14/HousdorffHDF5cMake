#include "CPUAllocations.cu"
#include "MetaData.cu"
#include "IterationUtils.cu"
#include "ExceptionManagUtils.cu"
#include "CooperativeGroupsUtils.cu"
#include "ForBoolKernel.cu"
#include "FirstMetaPass.cu"
#include "MainPassFunctions.cu"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include "UnitTestUtils.cu"
#include "MetaDataOtherPasses.cu"
#include <cooperative_groups/memcpy_async.h>
#include <cuda/pipeline>
using namespace cooperative_groups;




template <typename TKKI>
inline __device__ void mainDilatation(bool isPaddingPass, ForBoolKernelArgs<TKKI> fbArgs, uint32_t* mainArrAPointer,
    uint32_t* mainArrBPointer, MetaDataGPU metaData
    , unsigned int* minMaxes, uint32_t* workQueue
    , uint32_t* resultListPointerMeta, uint32_t* resultListPointerLocal, uint32_t* resultListPointerIterNumb,
    thread_block cta, thread_block_tile<32> tile, grid_group grid, uint32_t mainShmem[lengthOfMainShmem]
    , bool isAnythingInPadding[6]  , bool isBlockFull[1], int iterationNumb[1], unsigned int globalWorkQueueOffset[1],
    unsigned int globalWorkQueueCounter[1], unsigned int localWorkQueueCounter[1],
    unsigned int localTotalLenthOfWorkQueue[1], unsigned int localFpConter[1],
    unsigned int localFnConter[1], unsigned int blockFpConter[1],
    unsigned int blockFnConter[1], unsigned int resultfpOffset[1],
    unsigned int resultfnOffset[1], unsigned int worQueueStep[1],
    uint32_t isGold[1], uint32_t currLinIndM[1], unsigned int localMinMaxes[5]
    , uint32_t localBlockMetaData[20], unsigned int fpFnLocCounter[1]
    , bool isGoldPassToContinue[1], bool isSegmPassToContinue[1]
    , uint32_t* origArrs, uint32_t* metaDataArr, uint32_t oldIsGold[1], uint32_t oldLinIndM[1], uint32_t localBlockMetaDataOld[6]
) {




    auto pipeline = cuda::make_pipeline();
    auto bigShape = cuda::aligned_size_t<128>(sizeof(uint32_t) * (metaData.mainArrXLength));
    auto thirdRegShape = cuda::aligned_size_t<128>(sizeof(uint32_t) * (32));


    if (tile.thread_rank() == 7 && tile.meta_group_rank() == 0  && !isPaddingPass) {
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
        isBlockFull[0] =true;
    };
    if (tile.thread_rank() == 10 && tile.meta_group_rank() == 0) {
        fpFnLocCounter[0] = 0;
    };

    if (tile.thread_rank() == 10 && tile.meta_group_rank() == 0) {
        // if it will be still of such value it mean that no block was processed
        currLinIndM[0] = UINT16_MAX;
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




    sync(cta);
    /// load work QueueData into shared memory 
    for (uint32_t bigloop = blockIdx.x * globalWorkQueueOffset[0]; bigloop < ((blockIdx.x + 1) * globalWorkQueueOffset[0]); bigloop += worQueueStep[0]) {
        // grid stride loop - sadly most of threads will be idle 
        /////////// loading to work queue
        
        cuda::memcpy_async(cta, (&mainShmem[startOfLocalWorkQ]), (&workQueue[bigloop]), cuda::aligned_size_t<4>(sizeof(uint32_t) * worQueueStep[0]));
        //to do change into barrier
        sync(cta);
        //setting main indexes to make later work easier
        if (tile.thread_rank() == 0 && tile.meta_group_rank() == 0) {// this is how it is encoded wheather it is gold or segm block
            isGold[0] = (mainShmem[startOfLocalWorkQ] >= isGoldOffset);

        }
        if (tile.thread_rank() == 1 && tile.meta_group_rank() == 0) {// this is how it is encoded wheather it is gold or segm block

            currLinIndM[0] = mainShmem[startOfLocalWorkQ] - isGoldOffset * (mainShmem[startOfLocalWorkQ] >= isGoldOffset);
            //  printf("in pipeline  linear index meta %d block %d \n ", currLinIndM[0], blockIdx.x);
        }        
        sync(cta);
        //now all of the threads in the block needs to have the same i value so we will increment by 1
        // we are preloading to the pipeline block metaData
////##### pipeline Step 0
        pipeline.producer_acquire();

        cuda::memcpy_async(cta, (&localBlockMetaData[0]),
            (&metaDataArr[(mainShmem[startOfLocalWorkQ] - isGoldOffset * (mainShmem[startOfLocalWorkQ] >= isGoldOffset))
                * metaData.metaDataSectionLength])
            , cuda::aligned_size_t<4>(sizeof(uint32_t) * 20), pipeline);

        pipeline.producer_commit();
        
        for (uint32_t i = 0; i < worQueueStep[0]; i += 1) {
            if (((bigloop + i) < localTotalLenthOfWorkQueue[0]) && ((bigloop + i) < ((blockIdx.x + 1) * globalWorkQueueOffset[0]))) {
                ///#### pipeline step 1) now we load data for next step (to mainly sourceshmem and left-right if apply) and process data loaded in previous step
                pipeline.producer_acquire();

                //load data of intrest from block to the right
                if (currLinIndM[0]<(metaData.totalMetaLength-1)) {
                    cuda::memcpy_async(tile, (&mainShmem[begSMallRegShmemB + tile.meta_group_rank()]),
                        &getSourceReduced(fbArgs, iterationNumb)[
                            (currLinIndM[0] + 1) * metaData.mainArrSectionLength + metaData.mainArrXLength * (1 - isGold[0])
                            + tile.meta_group_rank() * 32], //we look for indicies 0,32,64... up to metaData.mainArrXLength
                        cuda::aligned_size_t<4>(sizeof(uint32_t))
                        , pipeline);
                }
                //load data of interst form block to the left
                if (currLinIndM[0] > 0) {
                    cuda::memcpy_async(tile, (&mainShmem[begSMallRegShmemA + tile.meta_group_rank()]),
                        &getSourceReduced(fbArgs, iterationNumb)[
                            (currLinIndM[0] - 1) * metaData.mainArrSectionLength + metaData.mainArrXLength * (1 - isGold[0])
                                //we look for indicies 31,63... up to metaData.mainArrXLength
                            + (tile.meta_group_rank() * 32) + 31])
                        , cuda::aligned_size_t<4>(sizeof(uint32_t)), pipeline);
                }

                //load data about this block
                cuda::memcpy_async(cta, &mainShmem[begSourceShmem],   &getSourceReduced(fbArgs, iterationNumb)[
                        currLinIndM[0] *metaData.mainArrSectionLength+metaData.mainArrXLength * (1 - isGold[0])],
                    , bigShape   , pipeline);


                pipeline.producer_commit();


                //        ////we need to do the cleaning after previous block .. compute first we load data about calculated linear index meta and information is it gold iteration ...
                pipeline.consumer_wait();

           
                afterBlockClean(cta, worQueueStep, localBlockMetaDataOld, mainShmem, i,
                    metaData, tile, localFpConter, localFnConter
                    , blockFpConter, blockFnConter
                    , metaDataArr, oldLinIndM, oldIsGold
                    , isAnythingInPadding, isBlockFull, isPaddingPass);



                pipeline.consumer_release();

 ////////#### pipeline step 2)  load block from top and process center that is in source shmem; and both smallRegShmems
                //load for next step - so we load block to the top
                if (localBlockMetaData[13] < UINT16_MAX) {
                    pipeline.producer_acquire();

                    cuda::memcpy_async(cta, (&mainShmem[begfirstRegShmem]),
                        &getSourceReduced(fbArgs, iterationNumb)[getIndexForNeighbourForShmem(metaData, mainShmem, iterationNumb, isGold, currLinIndM, localBlockMetaData, 13)], //we look for indicies 0,32,64... up to metaData.mainArrXLength
                        cuda::aligned_size_t<128>(sizeof(uint32_t) * metaData.mainArrXLength)
                        , pipeline);

                    pipeline.producer_commit();
                }

                //compute - now we have data in source shmem about this block and left and right padding and we need to process it 
                pipeline.consumer_wait();
                // first we perform up and down dilatations inside the block
                
                mainShmem[begResShmem + threadIdx.x + threadIdx.y * 32] = bitDilatate(mainShmem[begSourceShmem + threadIdx.x + threadIdx.y * 32]);
                
                
                //we also do the left and right dilatations
                ////left
                
                //dilatateHelperForTransverse((threadIdx.x == 0),
                //    2, (-1), (0), mainShmem, isAnythingInPadding
                //    , 0, threadIdx.y + (32 - fbArgs.dbYLength) // we add offset depending on y dimension
                //    , 15, begSMallRegShmemA, localBlockMetaData);

                ////right
                //dilatateHelperForTransverse((threadIdx.x == (fbArgs.dbXLength - 1)),
                //    3, (1), (0), mainShmem, isAnythingInPadding
                //    , 0, threadIdx.y
                //    , 16, begSMallRegShmemB, localBlockMetaData);





                //dilatateHelperForTransverse((threadIdx.x == (fbArgs.dbXLength - 1)),
                //    3, (1), (0), mainShmem, isAnythingInPadding
                //    , 0,threadIdx.y
                //    , 16, begSMallRegShmemA, localBlockMetaData);

                ///////////saving old
                //additionally we save previous copies of data so refreshing will keep easier
                if (tile.thread_rank() < 6 && tile.meta_group_rank() == 0) {
                    localBlockMetaDataOld[tile.thread_rank()] = localBlockMetaData[13 + tile.thread_rank()];
                }
                if (tile.thread_rank() == 6 && tile.meta_group_rank() == 0) {
                    oldIsGold[0] == isGold[0];
                }
                if (tile.thread_rank() == 7 && tile.meta_group_rank() == 0) {
                    oldLinIndM[0] == currLinIndM[0];
                }

                pipeline.consumer_release();
////////#### pipeline step 3) we load bottom, anterior and posterior and we process top
                      //load anterior and posterior and bottom
                pipeline.producer_acquire();
                //block to anterior 
                if (localBlockMetaData[17] < UINT16_MAX && tile.meta_group_rank() == 0) {

                    cuda::memcpy_async(tile, &mainShmem[begSMallRegShmemA], &getSourceReduced(fbArgs, iterationNumb)[
                        (localBlockMetaData[17]) * metaData.mainArrSectionLength + metaData.mainArrXLength * (1 - isGold[0])],
                        , thirdRegShape, pipeline);

                }
                // block to posterior
                if (localBlockMetaData[18] < UINT16_MAX && tile.meta_group_rank() == 1) {
                    cuda::memcpy_async(tile, &mainShmem[begSMallRegShmemB], &getSourceReduced(fbArgs, iterationNumb)[
                        (localBlockMetaData[18]) * metaData.mainArrSectionLength + metaData.mainArrXLength * (1 - isGold[0])
                        +(blockDim.y-1)*32// we need last 32 length entry of the posterior block 
                    ],
                        , thirdRegShape, pipeline);
                }


                pipeline.producer_commit();


                //    compute - now we have data in source shmem about block to the top
                pipeline.consumer_wait();
                dilatateHelperTopDown(0, mainShmem, isAnythingInPadding, localBlockMetaData, 13
                    , 1// represent a uint32 number that has a bit of intrest in this block set and all others 0 here first bit is set
                    , 2147483648
                    , begfirstRegShmem);
                pipeline.consumer_release();
////////#### pipeline step 5) if block is to be validated we load reference data and we process bottom, left and right
                //load reference data if block is to be validated otherwise if it is not the last step in the loop we load data for next loop
                pipeline.producer_acquire();
                if (localBlockMetaData[((1 - isGold[0]) + 1)] //fp for gold and fn count for not gold
                > localBlockMetaData[((1 - isGold[0]) + 1)]) {// so count is bigger than counter so we should validate
            //now we load data from reference arrays 
                    cuda::memcpy_async(cta, (&mainShmem[begfirstRegShmem]),
                        &origArrs[metaData.mainArrXLength * (isGold[0])], //we look for 
                        cuda::aligned_size_t<128>(sizeof(uint32_t) * metaData.mainArrXLength)
                        , pipeline);

                }
                else {//if we are not validating we immidiately start loading data for next loop
                    if (i + 1 <= worQueueStep[0]) {
                        cuda::memcpy_async(cta, (&localBlockMetaData[0]),
                            (&metaDataArr[(mainShmem[startOfLocalWorkQ + i] - isGoldOffset * (mainShmem[startOfLocalWorkQ + i] >= isGoldOffset))
                                * metaData.metaDataSectionLength])
                            , cuda::aligned_size_t<4>(sizeof(uint32_t) * 20), pipeline);
                    }
                }
                //    compute - now we have data in source shmem about block to the bottom, left and right

                pipeline.producer_commit();
                pipeline.consumer_wait();
                //bottom
                dilatateHelperTopDown(1, mainShmem, isAnythingInPadding, localBlockMetaData, 14
                    , 2147483648// represent a uint32 number that has a bit of intrest in this block set and all others 0 here last bit is set
                    , 1
                    , begSecRegShmem);
                //posterior
                
                //dilatateHelperForTransverse((threadIdx.y == 0), 5
                //    , (0), (-1), mainShmem, isAnythingInPadding
                //    , 0, threadIdx.x // we add offset depending on y dimension
                //    , 18, begSMallRegShmemB, localBlockMetaData);
                ////anterior
                //dilatateHelperForTransverse((threadIdx.y == (fbArgs.dbYLength - 1)), 4
                //    , (0), (1), mainShmem, isAnythingInPadding
                //    , 0, threadIdx.x
                //    , 17, begSMallRegShmemA, localBlockMetaData);
                
               // now all of the data is processed we need to save it into global memory
               // TODO try to use mempcy async here
                getTargetReduced(fbArgs, iterationNumb)[getIndexForSaveResShmem(metaData, mainShmem, iterationNumb, isGold, currLinIndM, localBlockMetaData) + threadIdx.x + threadIdx.y * 32] = mainShmem[begResShmem + threadIdx.x + threadIdx.y * 32];
               // setting information about is block full


                if (mainShmem[begResShmem + threadIdx.x + threadIdx.y * 32] != UINT32_MAX) {
                    isBlockFull[0] = false;
                }
                //setting indexes for next iteration
                if (tile.thread_rank() == 10 && tile.meta_group_rank() == 0 && i + 1 <= worQueueStep[0]) {// this is how it is encoded wheather it is gold or segm block
                    isGold[0] = (mainShmem[startOfLocalWorkQ + i + 1] >= isGoldOffset);

                }
                if (tile.thread_rank() == 11 && tile.meta_group_rank() == 0 && i + 1 <= worQueueStep[0]) {// this is how it is encoded wheather it is gold or segm block

                    currLinIndM[0] = mainShmem[startOfLocalWorkQ + i + 1] - isGoldOffset * (mainShmem[startOfLocalWorkQ + i] >= isGoldOffset);
                }


                pipeline.consumer_release();


//////////#### pipeline step 6) if block is to be validated we process the res and reference data and start loading data for begining of the next loop

                if (localBlockMetaData[((1 - isGold[0]) + 1)] //fp for gold and fn count for not gold
                > localBlockMetaData[((1 - isGold[0]) + 1)]) {// so count is bigger than counter so we should validate
    ////load data for next iteration
        pipeline.producer_acquire();
            if (i + 1 <= worQueueStep[0]) {
                cuda::memcpy_async(cta, (&localBlockMetaData[0]),
                    (&metaDataArr[(mainShmem[startOfLocalWorkQ+i] - isGoldOffset * (mainShmem[startOfLocalWorkQ+i] >= isGoldOffset))
                        * metaData.metaDataSectionLength])
                    , cuda::aligned_size_t<4>(sizeof(uint32_t) * 20), pipeline);
            }
            
        pipeline.producer_commit();

        //process check is there any new result (we have reference in begfirstRegShmem)
                 //now first we need to check for bits that are true now after dilatation but were not in source we will save it in res shmem becouse we will no longer need it
        pipeline.consumer_wait();
            mainShmem[begSourceShmem + threadIdx.x + threadIdx.y * 32] = ((~mainShmem[begSourceShmem + threadIdx.x + threadIdx.y * 32]) & mainShmem[begResShmem + threadIdx.x + threadIdx.y * 32]);


            //we now look for bits prasent in both reference arrays and current one
            mainShmem[begSourceShmem + threadIdx.x + threadIdx.y * 32] = ((mainShmem[begSourceShmem + threadIdx.x + threadIdx.y * 32]) & mainShmem[begfirstRegShmem + threadIdx.x + threadIdx.y * 32]);

            // now we look through bits and when some is set we call it a result 
            #pragma unroll
            for (uint8_t bitPos = 0; bitPos < 32; bitPos++) {
                //if any bit here is set it means it should be added to result list 
                if (isBitAt(mainShmem[begSourceShmem + threadIdx.x + threadIdx.y * 32], bitPos)) {
                    //first we add to the resList
                    //TODO consider first passing it into shared memory and then async mempcy ...
                    //we use offset plus number of results already added (we got earlier count from global memory now we just atomically add locally)
                    unsigned int old = 0;
                    ////// IMPORTANT for some reason in order to make it work resultfnOffset and resultfnOffset swith places
                    if (isGold[0]) { old = atomicAdd_block(&(localFpConter[0]), 1) + localBlockMetaData[6]; }
                    else { old = atomicAdd_block(&(localFnConter[0]), 1)-1 + localBlockMetaData[5]; };
                    //   add results to global memory    
                    resultListPointerMeta[old] = oldLinIndM[0] + isGoldOffset * oldIsGold[0];
                    resultListPointerLocal[old] = (fbArgs.dbYLength * 32 * bitPos + threadIdx.y * 32 + threadIdx.x);
                    resultListPointerIterNumb[old] = iterationNumb[0];
                }
            }
         pipeline.consumer_release();

                };
            }
  }  
    }

    //here we are after all of the blocks planned to be processed by this block are
    sync(cta);

    //updating local counters of last local block (normally it is done at the bagining of the next block)
    //but we need to check weather any block was processed at all
    if (currLinIndM[0]!= UINT16_MAX) {
        afterBlockClean(cta, worQueueStep, localBlockMetaDataOld, mainShmem,2,//2 is completely arbitrary important it is bigger than 0
            metaData, tile, localFpConter, localFnConter
            , blockFpConter, blockFnConter
            , metaDataArr, oldLinIndM, oldIsGold
            , isAnythingInPadding, isBlockFull, isPaddingPass);


    }



    //     updating global counters
    if (tile.thread_rank() == 0 && tile.meta_group_rank() == 0) {
        if (blockFpConter[0] > 0) {
            atomicAdd(&(minMaxes[10]), (blockFpConter[0]));
        }
    };
    if (tile.thread_rank() == 1 && tile.meta_group_rank() == 0) {
        if (blockFnConter[0] > 0) {
            atomicAdd(&(minMaxes[11]), (blockFnConter[0]));
        }
    };
    // in first thread block we zero work queue counter
    if (tile.thread_rank() == 2 && tile.meta_group_rank() == 0) {
        if (blockIdx.x==0) {
            minMaxes[9] = 0;
        }
    };


}
