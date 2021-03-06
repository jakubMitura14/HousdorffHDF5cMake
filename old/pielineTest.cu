////#include "CPUAllocations.cu"
////#include "MetaData.cu"
//// 
////#include "ExceptionManagUtils.cu"
////#include "CooperativeGroupsUtils.cu"
////#include "ForBoolKernel.cu"
////#include "FirstMetaPass.cu"
////#include "MainPassFunctions.cu"
////#include <cooperative_groups.h>
////#include <cooperative_groups/reduce.h>
////#include "UnitTestUtils.cu"
////#include "MetaDataOtherPasses.cu"
////#include <cooperative_groups/memcpy_async.h>
////#include <cuda/pipeline>
////using namespace cooperative_groups;
////
////
////
////
////template <typename TKKI>
////inline __device__ void mainDilatation(bool isPaddingPass, ForBoolKernelArgs<TKKI> fbArgs, uint32_t* mainArrAPointer,
////    uint32_t* mainArrBPointer, MetaDataGPU metaData
////    , unsigned int* minMaxes, uint32_t* workQueue
////    , uint32_t* resultListPointerMeta, uint32_t* resultListPointerLocal, uint32_t* resultListPointerIterNumb,
////    thread_block cta, thread_block_tile<32> tile, grid_group grid, uint32_t mainShmem[lengthOfMainShmem]
////    , bool isAnythingInPadding[6], bool isBlockFull[1], int iterationNumb[1], unsigned int globalWorkQueueOffset[1],
////    unsigned int globalWorkQueueCounter[1], unsigned int localWorkQueueCounter[1],
////    unsigned int localTotalLenthOfWorkQueue[1], unsigned int localFpConter[1],
////    unsigned int localFnConter[1], unsigned int blockFpConter[1],
////    unsigned int blockFnConter[1], unsigned int resultfpOffset[1],
////    unsigned int resultfnOffset[1], unsigned int worQueueStep[1],
////    uint32_t isGold[1], uint32_t currLinIndM[1], unsigned int localMinMaxes[5]
////    , uint32_t localBlockMetaData[(i & 1) * 20+20], unsigned int fpFnLocCounter[1]
////    , bool isGoldPassToContinue[1], bool isSegmPassToContinue[1]
////    , uint32_t* origArrs, uint32_t* metaDataArr, uint32_t oldIsGold[1], uint32_t oldLinIndM[1], uint32_t localBlockMetaDataOld[20],
////    bool isGoldForLocQueue[localWorkQueLength], bool isBlockToBeValidated[1]
////) {
////
////
////
////
////    auto pipeline = cuda::make_pipeline();
////    auto bigShape = cuda::aligned_size_t<128>(sizeof(uint32_t) * (metaData.mainArrXLength));
////    auto thirdRegShape = cuda::aligned_size_t<128>(sizeof(uint32_t) * (32));
////
////
////    if (tile.thread_rank() == 7 && tile.meta_group_rank() == 0 && !isPaddingPass) {
////        iterationNumb[0] += 1;
////    };
////
////    if (tile.thread_rank() == 6 && tile.meta_group_rank() == 0) {
////        localWorkQueueCounter[0] = 0;
////    };
////
////    if (tile.thread_rank() == 1 && tile.meta_group_rank() == 0) {
////        blockFpConter[0] = 0;
////    };
////    if (tile.thread_rank() == 2 && tile.meta_group_rank() == 0) {
////        blockFnConter[0] = 0;
////    };
////    if (tile.thread_rank() == 3 && tile.meta_group_rank() == 0) {
////        localFpConter[0] = 0;
////    };
////    if (tile.thread_rank() == 4 && tile.meta_group_rank() == 0) {
////        localFnConter[0] = 0;
////    };
////    if (tile.thread_rank() == 9 && tile.meta_group_rank() == 0) {
////        isBlockFull[0] = true;
////    };
////    if (tile.thread_rank() == 10 && tile.meta_group_rank() == 0) {
////        fpFnLocCounter[0] = 0;
////    };
////
////    if (tile.thread_rank() == 10 && tile.meta_group_rank() == 0) {
////        // if it will be still of such value it mean that no block was processed
////        oldLinIndM[0] = isGoldOffset;
////    };
////
////    if (tile.thread_rank() == 0 && tile.meta_group_rank() == 0) {
////        localTotalLenthOfWorkQueue[0] = minMaxes[9];
////        globalWorkQueueOffset[0] = floor((float)(localTotalLenthOfWorkQueue[0] / gridDim.x)) + 1;
////        worQueueStep[0] = min(localWorkQueLength, globalWorkQueueOffset[0]);
////    };
////    /* will be used to store all of the minMaxes varibles from global memory (from 7 to 11)
////0 : global FP count;
////1 : global FN count;
////2 : workQueueCounter
////3 : resultFP globalCounter
////4 : resultFn globalCounter
////*/
////    if (tile.meta_group_rank() == 1) {
////        cooperative_groups::memcpy_async(tile, (&localMinMaxes[0]), (&minMaxes[7]), cuda::aligned_size_t<4>(sizeof(unsigned int) * 5));
////    }
////
////
////
////
////    sync(cta);
////    /// load work QueueData into shared memory 
////    for (uint32_t bigloop = blockIdx.x * globalWorkQueueOffset[0]; bigloop < ((blockIdx.x + 1) * globalWorkQueueOffset[0]); bigloop += worQueueStep[0]) {
////        // grid stride loop - sadly most of threads will be idle 
////        /////////// loading to work queue
////
////        //cuda::memcpy_async(cta, (&mainShmem[startOfLocalWorkQ]), (&workQueue[bigloop])
////        //    , cuda::aligned_size_t<4>(sizeof(uint32_t) * worQueueStep[0]), pipeline);
////
////
////        for (uint16_t ii = 0; ii < worQueueStep[0]; ii++) {
////            mainShmem[startOfLocalWorkQ + ii] = workQueue[bigloop + ii];
////            isGoldForLocQueue[ii] = (mainShmem[startOfLocalWorkQ + ii] >= isGoldOffset);
////            mainShmem[startOfLocalWorkQ + ii] = mainShmem[startOfLocalWorkQ + ii] - isGoldOffset * isGoldForLocQueue[ii];
////
////        }
////
////
////        //to do change into barrier
////
////        //now all of the threads in the block needs to have the same i value so we will increment by 1
////        // we are preloading to the pipeline block metaData
////////##### pipeline Step 0
////
////        pipeline.producer_acquire();
////
////        cuda::memcpy_async(cta, (&localBlockMetaData[(i & 1) * 20+0]),
////            (&metaDataArr[mainShmem[startOfLocalWorkQ] * metaData.metaDataSectionLength])
////            , cuda::aligned_size_t<4>(sizeof(uint32_t) * 20), pipeline);
////
////        pipeline.producer_commit();
////        sync(cta);
////
////        for (uint32_t i = 0; i < worQueueStep[0]; i += 1) {
////            if (((bigloop + i) < localTotalLenthOfWorkQueue[0]) && ((bigloop + i) < ((blockIdx.x + 1) * globalWorkQueueOffset[0]))) {
////                ///#### pipeline step 1) now we load data for next step (to mainly sourceshmem and left-right if apply) and process data loaded in previous step
////
////
////                pipeline.producer_acquire();
////
////                cuda::memcpy_async(cta, &mainShmem[begSourceShmem], &getSourceReduced(fbArgs, iterationNumb)[
////                    mainShmem[startOfLocalWorkQ + i] * metaData.mainArrSectionLength + metaData.mainArrXLength * (1 - isGoldForLocQueue[i])],
////                    bigShape, pipeline);
////
////
////                if (mainShmem[startOfLocalWorkQ + i] < (metaData.totalMetaLength - 1)) {
////                    cooperative_groups::memcpy_async(tile, (&mainShmem[begSMallRegShmemB]),
////                        &getSourceReduced(fbArgs, iterationNumb)[
////                            (mainShmem[startOfLocalWorkQ + i] + 1) * metaData.mainArrSectionLength + metaData.mainArrXLength * (1 - isGoldForLocQueue[i])
////                                + tile.meta_group_rank() * 32], //we look for indicies 0,32,64... up to metaData.mainArrXLength
////                        cuda::aligned_size_t<4>(sizeof(uint32_t))
////                                );
////
////                }
////
////                //load data of interst form block to the left
////                if (mainShmem[startOfLocalWorkQ + i] > 0) {
////                    cuda::memcpy_async(tile, (&mainShmem[begSMallRegShmemA + tile.meta_group_rank()]),
////                        &getSourceReduced(fbArgs, iterationNumb)[
////                            (mainShmem[startOfLocalWorkQ + i] - 1) * metaData.mainArrSectionLength + metaData.mainArrXLength * (1 - isGoldForLocQueue[i])
////                                //we look for indicies 31,63... up to metaData.mainArrXLength
////                                + (tile.meta_group_rank() * 32) + 31]
////                        , cuda::aligned_size_t<4>(sizeof(uint32_t)), pipeline);
////
////                }
////                pipeline.producer_commit();
////
////
////
////                // we need to do the cleaning after previous block .. compute first we load data about calculated linear index meta and information is it gold iteration ...
////                pipeline.consumer_wait();
////
////
////                afterBlockClean(cta, worQueueStep, localBlockMetaDataOld, mainShmem, i,
////                    metaData, tile, localFpConter, localFnConter
////                    , blockFpConter, blockFnConter
////                    , metaDataArr, oldLinIndM, oldIsGold
////                    , isAnythingInPadding, isBlockFull, isPaddingPass);
////
////
////
////
////
////                pipeline.consumer_release();
////
////                ////////#### pipeline step 2)  load block from top and process center that is in source shmem; and both smallRegShmems
////                               //load for next step - so we load block to the top
////                pipeline.producer_acquire();
////
////                if (localBlockMetaData[(i & 1) * 20+13] < isGoldOffset) {
////
////                    cuda::memcpy_async(cta, (&mainShmem[begfirstRegShmem]),
////                        &getSourceReduced(fbArgs, iterationNumb)[localBlockMetaData[(i & 1) * 20+13] * metaData.mainArrSectionLength + metaData.mainArrXLength * (1 - isGoldForLocQueue[i])], //we look for indicies 0,32,64... up to metaData.mainArrXLength
////                        cuda::aligned_size_t<128>(sizeof(uint32_t) * metaData.mainArrXLength)
////                        , pipeline);
////
////                }
////                pipeline.producer_commit();
////
////                //compute - now we have data in source shmem about this block and left and right padding and we need to process it 
////                pipeline.consumer_wait();
////                // first we perform up and down dilatations inside the block
////            //if (mainShmem[begSourceShmem + threadIdx.x + threadIdx.y * 32]>0) {
////            //    printf("source shmem linLocalInd %d  linMeta %d \n",(threadIdx.x + threadIdx.y * 32), mainShmem[startOfLocalWorkQ + ii] );
////            //}
////
////                mainShmem[begResShmem + threadIdx.x + threadIdx.y * 32] = bitDilatate(mainShmem[begSourceShmem + threadIdx.x + threadIdx.y * 32]);
////
////
////                //we also do the left and right dilatations
////                ////left
////                dilatateHelperForTransverse((threadIdx.x == 0),
////                    2, (-1), (0), mainShmem, isAnythingInPadding
////                    , 0, threadIdx.y
////                    , 15, begSMallRegShmemA, localBlockMetaData);
////
////                //right
////                dilatateHelperForTransverse((threadIdx.x == (fbArgs.dbXLength - 1)),
////                    3, (1), (0), mainShmem, isAnythingInPadding
////                    , 0, threadIdx.y
////                    , 16, begSMallRegShmemB, localBlockMetaData);
////                ///////////saving old
////                //additionally we save previous copies of data so refreshing will keep easier
////                if (threadIdx.x < 20 && threadIdx.y == 0) {
////                    localBlockMetaDataOld[tile.thread_rank()] = localBlockMetaData[(i & 1) * 20+tile.thread_rank()];
////                }
////                if (threadIdx.x == 0 && threadIdx.y == 1) {
////                    oldIsGold[0] = isGoldForLocQueue[i];
////                }
////                if (threadIdx.x == 1 && threadIdx.y == 1) {
////                    oldLinIndM[0] = mainShmem[startOfLocalWorkQ + i];
////
////                }
////                if (threadIdx.x == 2 && threadIdx.y == 1) {
////                    isBlockToBeValidated[0] = ((localBlockMetaData[(i & 1) * 20+2 - isGoldForLocQueue[i]]) > localBlockMetaData[(i & 1) * 20+(4 - isGoldForLocQueue[i])]);
////
////                }
////
////
////
////                pipeline.consumer_release();
////                ////////#### pipeline step 3) we load bottom, anterior and posterior and we process top
////                                      //load anterior and posterior and bottom
////                pipeline.producer_acquire();
////                //block to anterior 
////                if (localBlockMetaData[(i & 1) * 20+17] < isGoldOffset && tile.meta_group_rank() == 0) {
////
////                    cuda::memcpy_async(tile, &mainShmem[begSMallRegShmemA], &getSourceReduced(fbArgs, iterationNumb)[
////                        (localBlockMetaData[(i & 1) * 20+17]) * metaData.mainArrSectionLength + metaData.mainArrXLength * (1 - isGoldForLocQueue[i])],
////                        thirdRegShape, pipeline);
////
////                }
////                // block to posterior
////                if (localBlockMetaData[(i & 1) * 20+18] < isGoldOffset && tile.meta_group_rank() == 1) {
////                    cuda::memcpy_async(tile, &mainShmem[begSMallRegShmemB], &getSourceReduced(fbArgs, iterationNumb)[
////                        (localBlockMetaData[(i & 1) * 20+18]) * metaData.mainArrSectionLength + metaData.mainArrXLength * (1 - isGoldForLocQueue[i])
////                            + (blockDim.y - 1) * 32// we need last 32 length entry of the posterior block 
////                    ], thirdRegShape, pipeline);
////
////                }
////
////                //bottom  block
////                if (localBlockMetaData[(i & 1) * 20+14] < isGoldOffset) {
////                    cuda::memcpy_async(cta, (&mainShmem[begSecRegShmem]),
////                        &getSourceReduced(fbArgs, iterationNumb)[
////                            localBlockMetaData[(i & 1) * 20+14] * metaData.mainArrSectionLength + metaData.mainArrXLength * (1 - isGoldForLocQueue[i])], //we look for indicies 0,32,64... up to metaData.mainArrXLength
////                        cuda::aligned_size_t<128>(sizeof(uint32_t) * metaData.mainArrXLength)
////                                , pipeline);
////
////                }
////                pipeline.producer_commit();
////
////
////                //    compute - now we have data in source shmem about block to the top
////                pipeline.consumer_wait();
////                dilatateHelperTopDown(0, mainShmem, isAnythingInPadding, localBlockMetaData, 13
////                    , 31// represent a uint32 number that has a bit of intrest in this block set and all others 0 here first bit is set
////                    , 0
////                    , begfirstRegShmem);
////                pipeline.consumer_release();
////                ////////#### pipeline step 5) if block is to be validated we load reference data and we process bottom, left and right
////                                //load reference data if block is to be validated otherwise if it is not the last step in the loop we load data for next loop
////                pipeline.producer_acquire();
////                if (isBlockToBeValidated[0]) {// so count is bigger than counter so we should validate
////            //now we load data from reference arrays 
////                    cuda::memcpy_async(cta, (&mainShmem[begfirstRegShmem]),
////                        &origArrs[mainShmem[startOfLocalWorkQ + i] * metaData.mainArrSectionLength + metaData.mainArrXLength * (isGoldForLocQueue[i])], //we look for 
////                        cuda::aligned_size_t<128>(sizeof(uint32_t) * metaData.mainArrXLength)
////                        , pipeline);
////
////                }
////                else {//if we are not validating we immidiately start loading data for next loop
////                    if (i + 1 <= worQueueStep[0]) {
////                        cuda::memcpy_async(cta, (&localBlockMetaData[(i & 1) * 20+0]),
////                            (&metaDataArr[(mainShmem[startOfLocalWorkQ + 1])
////                                * metaData.metaDataSectionLength])
////                            , cuda::aligned_size_t<4>(sizeof(uint32_t) * 20), pipeline);
////                    }
////                }
////                //    compute - now we have data in source shmem about block to the bottom, left and right
////
////                pipeline.producer_commit();
////                pipeline.consumer_wait();
////                //bottom
////                dilatateHelperTopDown(1, mainShmem, isAnythingInPadding, localBlockMetaData, 14
////                    , 0// represent a uint32 number that has a bit of intrest in this block set and all others 0 here last bit is set
////                    , 31
////                    , begSecRegShmem);
////                //posterior
////                dilatateHelperForTransverse((threadIdx.y == 0), 5
////                    , (0), (-1), mainShmem, isAnythingInPadding
////                    , 0, threadIdx.x // we add offset depending on y dimension
////                    , 18, begSMallRegShmemB, localBlockMetaData);
////                //anterior
////                dilatateHelperForTransverse((threadIdx.y == (fbArgs.dbYLength - 1)), 4
////                    , (0), (1), mainShmem, isAnythingInPadding
////                    , 0, threadIdx.x
////                    , 17, begSMallRegShmemA, localBlockMetaData);
////
////                // now all of the data is processed we need to save it into global memory
////                // TODO try to use mempcy async here
////                 //if (mainShmem[begResShmem + threadIdx.x + threadIdx.y * 32]>0) {
////                getTargetReduced(fbArgs, iterationNumb)[mainShmem[startOfLocalWorkQ + i] * metaData.mainArrSectionLength + metaData.mainArrXLength * (1 - isGoldForLocQueue[i])
////                    + threadIdx.x + threadIdx.y * 32]
////                    = mainShmem[begResShmem + threadIdx.x + threadIdx.y * 32];
////                //}
////                // setting information about is block full
////
////
////                if (mainShmem[begResShmem + threadIdx.x + threadIdx.y * 32] != UINT32_MAX) {
////                    isBlockFull[0] = false;
////                }
////
////
////                pipeline.consumer_release();
////
////
////                //////////#### pipeline step 6) if block is to be validated we process the res and reference data and start loading data for begining of the next loop
////
////
////
////                                                                                    ////load data for next iteration
////                pipeline.producer_acquire();
////                if (localBlockMetaDataOld[((1 - isGoldForLocQueue[i]) + 1)] //fp for gold and fn count for not gold
////                > localBlockMetaDataOld[((1 - isGoldForLocQueue[i]) + 3)]) {// so count is bigger than counter so we should validate
////                    if (i + 1 <= worQueueStep[0]) {
////                        cuda::memcpy_async(cta, (&localBlockMetaData[(i & 1) * 20+0]),
////                            (&metaDataArr[(mainShmem[startOfLocalWorkQ + 1])
////                                * metaData.metaDataSectionLength])
////                            , cuda::aligned_size_t<4>(sizeof(uint32_t) * 20), pipeline);
////                    }
////                }
////                pipeline.producer_commit();
////
////                //process check is there any new result (we have reference in begfirstRegShmem)
////                         //now first we need to check for bits that are true now after dilatation but were not in source we will save it in res shmem becouse we will no longer need it
////                pipeline.consumer_wait();
////                if (isBlockToBeValidated[0]) {// so count is bigger than counter so we should validate
////
////                    mainShmem[begSourceShmem + threadIdx.x + threadIdx.y * 32] = ((~mainShmem[begSourceShmem + threadIdx.x + threadIdx.y * 32]) & mainShmem[begResShmem + threadIdx.x + threadIdx.y * 32]);
////
////
////
////                    //we now look for bits prasent in both reference arrays and current one
////                    mainShmem[begSourceShmem + threadIdx.x + threadIdx.y * 32] = ((mainShmem[begSourceShmem + threadIdx.x + threadIdx.y * 32]) & mainShmem[begfirstRegShmem + threadIdx.x + threadIdx.y * 32]);
////
////
////                    // now we look through bits and when some is set we call it a result 
////#pragma unroll
////                    for (uint8_t bitPos = 0; bitPos < 32; bitPos++) {
////                        //if any bit here is set it means it should be added to result list 
////                        if (isBitAt(mainShmem[begSourceShmem + threadIdx.x + threadIdx.y * 32], bitPos)) {
////                            //first we add to the resList
////                            //TODO consider first passing it into shared memory and then async mempcy ...
////                            //we use offset plus number of results already added (we got earlier count from global memory now we just atomically add locally)
////                            unsigned int old = 0;
////                            ////// IMPORTANT for some reason in order to make it work resultfnOffset and resultfnOffset swith places
////                            if (isGoldForLocQueue[i]) {
////                                old = atomicAdd_block(&(localFpConter[0]), 1) + localBlockMetaDataOld[5];
////                            }
////                            else {
////                                old = atomicAdd_block(&(localFnConter[0]), 1) + localBlockMetaDataOld[6];
////                            };
////                            //   add results to global memory    
////                            resultListPointerMeta[old] = uint32_t(oldLinIndM[0] + isGoldOffset * oldIsGold[0]);
////                            resultListPointerLocal[old] = uint16_t(fbArgs.dbYLength * 32 * bitPos + threadIdx.y * 32 + threadIdx.x);
////                            resultListPointerIterNumb[old] = uint32_t(iterationNumb[0]);
////
////                            printf("rrrrresult meta %d isGold %d old %d localFpConter %d localFnConter %d fpOffset %d fnOffset %d linIndUpdated %d  localInd %d\n"
////                                , mainShmem[startOfLocalWorkQ + i]
////                                , isGoldForLocQueue[i]
////                                , old
////                                , localFpConter[0]
////                                , localFnConter[0]
////                                , localBlockMetaDataOld[6]
////                                , localBlockMetaDataOld[7]
////                                , uint32_t(oldLinIndM[0] + isGoldOffset * oldIsGold[0])
////                                , (fbArgs.dbYLength * 32 * bitPos + threadIdx.y * 32 + threadIdx.x)
////                            );
////
////                        }
////                    }
////                    pipeline.consumer_release();
////
////                };
////            }
////        }
////    }
////
////    //here we are after all of the blocks planned to be processed by this block are
////    sync(cta);
////
////    //updating local counters of last local block (normally it is done at the bagining of the next block)
////    //but we need to check weather any block was processed at all
////    if (oldLinIndM[0] != isGoldOffset) {
////        afterBlockClean(cta, worQueueStep, localBlockMetaDataOld, mainShmem, 2,//2 is completely arbitrary important it is bigger than 0
////            metaData, tile, localFpConter, localFnConter
////            , blockFpConter, blockFnConter
////            , metaDataArr, oldLinIndM, oldIsGold
////            , isAnythingInPadding, isBlockFull, isPaddingPass);
////    }
////
////    //     updating global counters
////    if (tile.thread_rank() == 0 && tile.meta_group_rank() == 0) {
////        if (blockFpConter[0] > 0) {
////            atomicAdd(&(minMaxes[10]), (blockFpConter[0]));
////        }
////    };
////    if (tile.thread_rank() == 1 && tile.meta_group_rank() == 0) {
////        if (blockFnConter[0] > 0) {
////            atomicAdd(&(minMaxes[11]), (blockFnConter[0]));
////        }
////    };
////    // in first thread block we zero work queue counter
////    if (tile.thread_rank() == 2 && tile.meta_group_rank() == 0) {
////        if (blockIdx.x == 0) {
////            minMaxes[9] = 0;
////        }
////    };
////
////
////}
//
//
//
//
//template <typename TKKI>
//inline __global__ void testKernel(ForBoolKernelArgs<TKKI> fbArgs, unsigned int* minMaxes, uint32_t* mainArr, MetaDataGPU metaData, uint32_t* workQueue, uint32_t* origArr) {
//    thread_block cta = this_thread_block();
//
//    //work queue !!
//    //if ((threadIdx.x == 0) && (threadIdx.y == 0)) {
//    //    for (uint32_t ii = blockIdx.x; ii < 7; ii += gridDim.x) {
//    //        if (workQueue[ii] > 0) {
//    //            if (workQueue[ii] > (isGoldOffset-1)) {
//    //                printf("in gold workqueue elment %d  \n", (workQueue[ii] - isGoldOffset));
//    //            }
//    //            else {
//    //                printf("in segm workqueue elment %d  \n", (workQueue[ii]));
//
//    //            }
//
//    //        }
//
//    //    }
//    //}
//    // 
//        //results  !!
//    if ((threadIdx.x == 0) && (threadIdx.y == 0)) {
//        for (uint32_t ii = blockIdx.x; ii < 10; ii += gridDim.x) {
//            if (fbArgs.resultListPointerMeta[ii] > 0) {
//                printf("in TEST kernel  result lin meta %d ii  \n", fbArgs.resultListPointerMeta[ii]);
//
//            }
//
//        }
//    }
//
//
//
//    sync(cta);
//    char* tensorslice;
//
//
//    for (uint32_t linIdexMeta = blockIdx.x; linIdexMeta < metaData.totalMetaLength; linIdexMeta += gridDim.x) {
//        //we get from linear index  the coordinates of the metadata block of intrest
//        uint8_t xMeta = linIdexMeta % metaData.metaXLength;
//        uint8_t zMeta = floor((float)(linIdexMeta / (metaData.metaXLength * metaData.MetaYLength)));
//        uint8_t yMeta = floor((float)((linIdexMeta - ((zMeta * metaData.metaXLength * metaData.MetaYLength) + xMeta)) / metaData.metaXLength));
//
//        for (uint8_t xLoc = threadIdx.x; xLoc < 32; xLoc += blockDim.x) {
//            uint32_t x = (xMeta + metaData.minX) * 32 + xLoc;//absolute position
//            for (uint8_t yLoc = threadIdx.y; yLoc < fbArgs.dbYLength; yLoc += blockDim.y) {
//                uint32_t  y = (yMeta + metaData.minY) * fbArgs.dbYLength + yLoc;//absolute position
//                for (uint8_t zLoc = 0; zLoc < 32; zLoc++) {
//
//                    uint32_t z = (zMeta + metaData.minZ) * fbArgs.dbZLength + zLoc;//absolute position
//                    uint8_t ww = 0;
//                    //uint32_t column = mainArr[linIdexMeta * metaData.mainArrSectionLength + (threadIdx.x + threadIdx.y * fbArgs.dbXLength) + (metaData.mainArrXLength)*ww];//
//                    uint32_t column = mainArr[linIdexMeta * metaData.mainArrSectionLength + (xLoc + yLoc * 32) + (metaData.mainArrXLength) * ww];//
//                    //uint32_t column = mainArr[linIdexMeta * metaData.mainArrSectionLength + (threadIdx.x + threadIdx.y * fbArgs.dbXLength)];
//
//
//
//
//                    //rrrrresult meta 1 isGold 1 old 0 localFpConter 1 localFnConter 0 fpOffset 0 fnOffset 0 linIndUpdated 655351  localInd 24544
//
//                    //if (linIdexMeta== 1 ) {
//                    //    if (  (fbArgs.dbYLength * 32 * zLoc + yLoc * 32 + xLoc) == 24544) {
//                    //            printf("in TEST kernel x %d y%d z %d linearLocal %d linIdexMeta  \n"
//                    //  ,  x, y, z, (xLoc + yLoc * fbArgs.dbXLength), linIdexMeta);
//
//                    //    }
//                    //
//                    //}
//                    ////    rrrrresult meta 2 isGold 1 old 1 localFpConter 1 localFnConter 0 fpOffset 0 fnOffset 0 linIndUpdated 655352  localInd 23839
//
//                    //if (linIdexMeta == 2) {
//                    //    if ((fbArgs.dbYLength * 32 * zLoc + yLoc * 32 + xLoc) == 23839) {
//                    //        printf("in TEST kernel x %d y%d z %d linearLocal %d linIdexMeta  \n"
//                    //            , x, y, z, (xLoc + yLoc * fbArgs.dbXLength), linIdexMeta);
//
//                    //    }
//
//                    //}
//                    ////    rrrrresult meta 4 isGold 1 old 2 localFpConter 1 localFnConter 0 fpOffset 0 fnOffset 0 linIndUpdated 655354  localInd 767
//
//                    //if (linIdexMeta == 4) {
//                    //    if ((fbArgs.dbYLength * 32 * zLoc + yLoc * 32 + xLoc) == 767) {
//                    //        printf("in TEST kernel x %d y%d z %d linearLocal %d linIdexMeta  \n"
//                    //            , x, y, z, (xLoc + yLoc * fbArgs.dbXLength), linIdexMeta);
//
//                    //    }
//
//                    //}
//                    ////    rrrrresult meta 0 isGold 0 old 3 localFpConter 0 localFnConter 1 fpOffset 3 fnOffset 1 linIndUpdated 0  localInd 24575
//
//                    //if (linIdexMeta == 0) {
//                    //    if ((fbArgs.dbYLength * 32 * zLoc + yLoc * 32 + xLoc) == 24575) {
//                    //        printf("in TEST kernel x %d y%d z %d linearLocal %d linIdexMeta  \n"
//                    //            , x, y, z, (xLoc + yLoc * fbArgs.dbXLength), linIdexMeta);
//
//                    //    }
//
//                    //}
//
//
//
//                 //if (x==33 && y==1 && z==71) {
//                 //    printf("in 33 1 71 TEST kernel Metax %d yMeta %d zMeta %d x %d y%d z %d linearLocal %d linIdexMeta %d column %d looking in %d \n"
//                 //        , xMeta, yMeta, zMeta, x, y, z, (xLoc + yLoc * fbArgs.dbXLength), linIdexMeta
//                 //        , column, linIdexMeta * metaData.mainArrSectionLength + (threadIdx.x + threadIdx.y * fbArgs.dbXLength) + (metaData.mainArrXLength) * ww);
//                 //}
//
//
//
//
//                    if (isBitAt(column, zLoc) && column > 0) {
//
//
//                        printf("in TEST kernel Metax %d yMeta %d zMeta %d x %d y%d z %d linearLocal %d linIdexMeta %d looking in %d    \n"
//                            , xMeta, yMeta, zMeta, x, y, z, (xLoc + yLoc * 32), linIdexMeta
//                            , column, linIdexMeta * metaData.mainArrSectionLength + (xLoc + yLoc * fbArgs.dbXLength) + (metaData.mainArrXLength) * ww, fbArgs.dbYLength);
//                    }
//
//                    ww = 1;
//                    // uint32_t column = mainArr[linIdexMeta * metaData.mainArrSectionLength + (threadIdx.x + threadIdx.y * fbArgs.dbXLength) + (metaData.mainArrXLength) * ww];//
//                    column = mainArr[linIdexMeta * metaData.mainArrSectionLength + (xLoc + yLoc * fbArgs.dbXLength) + (metaData.mainArrXLength) * ww];//
//
//
//                    //if (x == 33 && y == 1 && z == 71) {
//                    //    printf("in 33 1 71 TEST kernel Metax %d yMeta %d zMeta %d x %d y%d z %d linearLocal %d linIdexMeta %d column %d looking in %d \n"
//                    //        , xMeta, yMeta, zMeta, x, y, z, (xLoc + yLoc * fbArgs.dbXLength), linIdexMeta
//                    //        , column, linIdexMeta * metaData.mainArrSectionLength + (threadIdx.x + threadIdx.y * fbArgs.dbXLength) + (metaData.mainArrXLength) * ww);
//                    //}
//
//                    if (isBitAt(column, zLoc) && column > 0) {
//
//                        printf("in TEST kernel Metax %d yMeta %d zMeta %d x %d y%d z %d linearLocal %d linIdexMeta %d looking in %d   \n"
//                            , xMeta, yMeta, zMeta, x, y, z, (xLoc + yLoc * fbArgs.dbXLength), linIdexMeta
//                            , column, linIdexMeta * metaData.mainArrSectionLength + (xLoc + yLoc * fbArgs.dbXLength) + (metaData.mainArrXLength) * ww, fbArgs.dbYLength);
//                    }
//
//                }
//            }
//        }
//
//        //if ((threadIdx.x == 0) && (threadIdx.y == 0)) {
//        //    auto count = fbArgs.metaDataArrPointer[linIdexMeta * metaData.metaDataSectionLength + 1];
//        //    if (count > 0) {
//        //        printf("in TEST kernel looking fp count  xMeta %d yMeta %d zMeta %d linIdexMeta %d count %d counter %d \n"
//        //            , xMeta, yMeta, zMeta, linIdexMeta, count, fbArgs.metaDataArrPointer[linIdexMeta * metaData.metaDataSectionLength + 3]);
//        //    }
//        //}
//        //if ((threadIdx.x == 1) && (threadIdx.y == 0)) {
//        //    auto count = fbArgs.metaDataArrPointer[linIdexMeta * metaData.metaDataSectionLength + 2];
//        //    if (count > 0) {
//        //        printf("in TEST kernel looking fn count   xMeta %d yMeta %d zMeta %d linIdexMeta %d count %d counter %d \n"
//        //            , xMeta, yMeta, zMeta, linIdexMeta, count, fbArgs.metaDataArrPointer[linIdexMeta * metaData.metaDataSectionLength + 4]);
//        //    }
//        //}
//
//
//
//
//
//        //if ((threadIdx.x == 0) && (threadIdx.y == 0)) {
//        //    auto count = mainArr[linIdexMeta * metaData.mainArrSectionLength+ metaData.metaDataOffset + 7];
//        //    if (count ==1) {
//        //        printf("in TEST kernel looking active gold  xMeta %d yMeta %d zMeta %d linIdexMeta %d count %d \n"
//        //            , xMeta, yMeta, zMeta, linIdexMeta, count);
//        //    }
//        //}
//        //if ((threadIdx.x == 0) && (threadIdx.y == 0)) {
//        //    auto count = mainArr[linIdexMeta * metaData.mainArrSectionLength + metaData.metaDataOffset + 9];
//        //    if (count == 1) {
//        //        printf("in TEST kernel looking active segm  xMeta %d yMeta %d zMeta %d linIdexMeta %d count %d \n"
//        //            , xMeta, yMeta, zMeta, linIdexMeta, count);
//        //    }
//        //}
//        ///// testing  calculation of surrounding blocks linear indicies
//        // block 1,1,1
//        //if ((threadIdx.x == 0) && (threadIdx.y == 0)) {
//        //    if (xMeta==1 && yMeta==1 && zMeta==1) {
//        //        printf("linear indicies from metadata  top %d bottom %d left %d right %d anterior %d posterior %d  linIdexMeta current %d \n    "
//        //            ,mainArr[linIdexMeta * metaData.mainArrSectionLength + metaData.metaDataOffset + 13]
//        //            , mainArr[linIdexMeta * metaData.mainArrSectionLength + metaData.metaDataOffset + 14]
//
//        //            , mainArr[linIdexMeta * metaData.mainArrSectionLength + metaData.metaDataOffset + 15]
//        //            , mainArr[linIdexMeta * metaData.mainArrSectionLength + metaData.metaDataOffset + 16]
//
//        //            , mainArr[linIdexMeta * metaData.mainArrSectionLength + metaData.metaDataOffset + 17]
//        //            , mainArr[linIdexMeta * metaData.mainArrSectionLength + metaData.metaDataOffset + 18]
//        //            , linIdexMeta
//        //        );
//        //    }
//        //    if (xMeta ==  1&& yMeta == 1 && zMeta == 0) {
//        //        printf("linear index top linIdexMeta %d \n    ", linIdexMeta);
//        //    
//        //    }
//        //    if (xMeta ==1 && yMeta == 1 && zMeta == 2) {
//        //        printf("linear index bottom linIdexMeta %d \n    ", linIdexMeta);
//
//        //    }
//        //    if (xMeta == 1&& yMeta == 2 && zMeta == 1) {
//        //        printf("linear index anterior linIdexMeta %d \n    ", linIdexMeta);
//
//        //    }
//        //    if (xMeta == 1&& yMeta == 0 && zMeta == 1) {
//        //        printf("linear index posterior linIdexMeta %d \n    ", linIdexMeta);
//
//        //    }
//
//        //    if (xMeta ==2 && yMeta == 1 && zMeta == 1) {
//        //        printf("linear index right linIdexMeta %d \n    ", linIdexMeta);
//
//        //    }
//        //    if (xMeta == 0&& yMeta == 1 && zMeta == 1) {
//        //        printf("linear index left linIdexMeta %d \n    ", linIdexMeta);
//
//        //    }
//
//        //}
//
////// checking weather on edges it shows UINT32_MAX
//        //   if ((threadIdx.x == 0) && (threadIdx.y == 0)) {
//
//        //    if (xMeta ==  1&& yMeta == 1 && zMeta == 0) {
//        //        printf("linear index top linIdexMeta %d  and max is %d \n    ", mainArr[linIdexMeta * metaData.mainArrSectionLength + metaData.metaDataOffset + 13], UINT32_MAX);
//        //    
//        //    }
//        //    if (xMeta ==1 && yMeta == 1 && zMeta == 3) {
//        //        printf("linear index bottom linIdexMeta %d \n    ", mainArr[linIdexMeta * metaData.mainArrSectionLength + metaData.metaDataOffset + 14]);
//
//        //    }
//        //    if (xMeta == 1&& yMeta == 5 && zMeta == 1) {
//        //        printf("linear index anterior linIdexMeta %d \n    ", mainArr[linIdexMeta * metaData.mainArrSectionLength + metaData.metaDataOffset + 17]);
//
//        //    }
//        //    if (xMeta == 1&& yMeta == 0 && zMeta == 1) {
//        //        printf("linear index posterior linIdexMeta %d \n    ", mainArr[linIdexMeta * metaData.mainArrSectionLength + metaData.metaDataOffset + 18]);
//
//        //    }
//
//        //    if (xMeta ==2 && yMeta == 1 && zMeta == 1) {
//        //        printf("linear index right linIdexMeta %d \n    ", mainArr[linIdexMeta * metaData.mainArrSectionLength + metaData.metaDataOffset + 16]);
//
//        //    }
//        //    if (xMeta == 0&& yMeta == 1 && zMeta == 1) {
//        //        printf("linear index left linIdexMeta %d \n    ", mainArr[linIdexMeta * metaData.mainArrSectionLength + metaData.metaDataOffset + 15]);
//
//        //    }
//
//        //}
//
//        //if ((threadIdx.x == 0) && (threadIdx.y == 0)) {
//        //    auto count = mainArr[linIdexMeta * metaData.mainArrSectionLength + metaData.metaDataOffset + 5];
//        //    if (count >0) {
//        //        printf("in TEST kernel offset fp  xMeta %d yMeta %d zMeta %d linIdexMeta %d count %d \n"
//        //            , xMeta, yMeta, zMeta, linIdexMeta, count);
//        //    }
//        //}
//        //if ((threadIdx.x == 0) && (threadIdx.y == 0)) {
//        //    auto count = mainArr[linIdexMeta * metaData.mainArrSectionLength + metaData.metaDataOffset + 6];
//        //    if (count > 0) {
//        //        printf("in TEST kernel offset fn  xMeta %d yMeta %d zMeta %d linIdexMeta %d count %d \n"
//        //            , xMeta, yMeta, zMeta, linIdexMeta, count);
//        //    }
//        //}
//
//    }
//
//
//
//
//    //for (uint32_t linIdexMeta = blockIdx.x * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x; linIdexMeta < 80; linIdexMeta += blockDim.x * blockDim.y * gridDim.x) {
//
//
//    // /*   if (fbArgs.metaData.resultList[linIdexMeta * 5 + 4] != 131 && fbArgs.metaData.resultList[linIdexMeta * 5] > 0) {
//
//    //        printf("\n in kernel saving result x %d y %d z %d isGold %d iteration %d spotToUpdate %d \n ",
//    //            fbArgs.metaData.resultList[linIdexMeta * 5]
//    //            , fbArgs.metaData.resultList[linIdexMeta * 5 + 1]
//    //            , fbArgs.metaData.resultList[linIdexMeta * 5 + 2]
//    //            , fbArgs.metaData.resultList[linIdexMeta * 5 + 3]
//    //            , fbArgs.metaData.resultList[linIdexMeta * 5 + 4]
//    //            , linIdexMeta
//
//
//    //        );
//    //    }
//    //    else {
//    //        printf(" *** ");
//    //        atomicAdd(&(getTensorRow<unsigned int>(tensorslice, fbArgs.metaData.minMaxes, 1, 0, 0)[17]), 1);
//
//    //    }*/
//    //}
//}
