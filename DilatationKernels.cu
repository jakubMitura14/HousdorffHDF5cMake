//#include "CPUAllocations.cu"
//#include "MetaData.cu"
// 
//#include "ExceptionManagUtils.cu"
//#include "CooperativeGroupsUtils.cu"
//#include "ForBoolKernel.cu"
//#include "FirstMetaPass.cu"
//#include "MainPassFunctions.cu"
//#include <cooperative_groups.h>
//#include <cooperative_groups/reduce.h>
//#include "UnitTestUtils.cu"
//#include "MetaDataOtherPasses.cu"
//#include <cooperative_groups/memcpy_async.h>
//#include <cuda/pipeline>
//using namespace cooperative_groups;
//
//
//
//
////template <typename TKKI, typename forPipeline >
//template <typename TKKI >
//inline __device__ void mainDilatation(const bool isPaddingPass, ForBoolKernelArgs<TKKI>& fbArgs, uint32_t*& mainArrAPointer,
//    uint32_t*& mainArrBPointer, MetaDataGPU& metaData
//    , unsigned int*& minMaxes, uint32_t*& workQueue
//    , uint32_t*& resultListPointerMeta, uint32_t*& resultListPointerLocal, uint32_t*& resultListPointerIterNumb,
//    thread_block& cta, thread_block_tile<32>& tile, grid_group& grid, uint32_t (&mainShmem)[lengthOfMainShmem]
//    , bool(&isAnythingInPadding)[6], bool (&isBlockFull)[1], int(&iterationNumb)[1], unsigned int(&globalWorkQueueOffset)[1]
//    ,unsigned int(&globalWorkQueueCounter)[1]
//    , unsigned int(&localWorkQueueCounter)[1],unsigned int(&localTotalLenthOfWorkQueue)[1]
//    , unsigned int(&localFpConter)[1]
//    ,unsigned int(&localFnConter)[1], unsigned int(&blockFpConter)[1]
//   , unsigned int(&blockFnConter)[1], unsigned int(&resultfpOffset)[1]
//    ,unsigned int(&resultfnOffset)[1], unsigned int(&worQueueStep)[1]
//,unsigned int(&localMinMaxes)[5]
//    , uint32_t(&localBlockMetaData)[40]
//    , unsigned int(&fpFnLocCounter)[1]
//    , bool(&isGoldPassToContinue)[1], bool(&isSegmPassToContinue)[1]
//    , uint32_t*& origArrs, uint32_t*& metaDataArr, bool (&isGoldForLocQueue)[localWorkQueLength]
//     , uint32_t(&lastI)[1]
//    , cuda::pipeline<cuda::thread_scope_block>& pipeline
//) {
//
//
//    //initial cleaning  and initializations include loading min maxes
//    dilBlockInitialClean(tile, isPaddingPass, iterationNumb, localWorkQueueCounter, blockFpConter,
//        blockFnConter, localFpConter, localFnConter, isBlockFull
//        , fpFnLocCounter,
//        localTotalLenthOfWorkQueue, globalWorkQueueOffset
//        , worQueueStep, minMaxes, localMinMaxes, lastI);
//    sync(cta);
//
//    /// load work QueueData into shared memory 
//    for (uint32_t bigloop = blockIdx.x * globalWorkQueueOffset[0]; bigloop < ((blockIdx.x + 1) * globalWorkQueueOffset[0]); bigloop += worQueueStep[0]) {
//        // grid stride loop - sadly most of threads will be idle 
//        /////////// loading to work queue
//        loadWorkQueue(mainShmem, workQueue, isGoldForLocQueue, bigloop, worQueueStep);
//
//        //now all of the threads in the block needs to have the same i value so we will increment by 1 we are preloading to the pipeline block metaData
//        ////##### pipeline Step 0
//
//
//
//        sync(cta);
//        //loading metadata
//        pipeline.producer_acquire();
//
//        //cuda::memcpy_async(cta, (&localBlockMetaData[20]),
//        //    (&metaDataArr[(mainShmem[startOfLocalWorkQ ])
//        //        * metaData.metaDataSectionLength])
//        //    , cuda::aligned_size_t<4>(sizeof(uint32_t) * 20), pipeline);
//
//        cuda::memcpy_async(cta, (&localBlockMetaData[0]),
//            (&metaDataArr[(mainShmem[startOfLocalWorkQ])
//                * metaData.metaDataSectionLength])
//            , cuda::aligned_size_t<4>(sizeof(uint32_t) * 20), pipeline);
//
//        //loadMetaDataToShmem(cta, localBlockMetaData, mainShmem, pipeline, metaDataArr, metaData, 0, 0);
//
//        pipeline.producer_commit();
//
//
//
//        for (uint32_t i = 0; i < worQueueStep[0]; i += 1) {
//            if (((bigloop + i) < localTotalLenthOfWorkQueue[0]) && ((bigloop + i) < ((blockIdx.x + 1) * globalWorkQueueOffset[0]))) {
//            
//                //if (tile.thread_rank() == 0 && tile.meta_group_rank() == 0) {                      
//                //    printf("\n linMeta beg %d is gold %d is padding pass %d\n ", mainShmem[startOfLocalWorkQ + i], isGoldForLocQueue[i], isPaddingPass);
//                //};
//
//                // if (tile.thread_rank() == 0 && tile.meta_group_rank() == 0 && isGoldForLocQueue[i]==0 ) {
//                //    printf("\n linMeta beg %d is gold %d is padding pass %d\n ", mainShmem[startOfLocalWorkQ + i], isGoldForLocQueue[i], isPaddingPass);
//                //};
//
////////////////// step 0  load main data and final processing of previous block
//               //loading main data for first dilatation
//                //IMPORTANT we need to keep a lot of variables constant here like is Anuthing in padding of fp count .. as the represent processing of previous block  - so do not modify them here ...
//                loadMain( fbArgs  , cta , localBlockMetaData , mainShmem, pipeline, metaDataArr, metaData , i , tile, isGoldForLocQueue, iterationNumb
//                );
//                                
//                pipeline.consumer_wait();
//                afterBlockClean(cta, worQueueStep, localBlockMetaData, mainShmem, i-1,
//                        metaData, tile, localFpConter, localFnConter
//                        , blockFpConter, blockFnConter
//                        , metaDataArr, isAnythingInPadding, isBlockFull, isPaddingPass, isGoldForLocQueue, lastI);
//                //needed for after block metadata update
//                if (tile.thread_rank() == 0 && tile.meta_group_rank() == 3) {
//                    lastI[0] = i;
//                }
//
//                pipeline.consumer_release();
//
/////////// step 1 load top and process main data 
//               //load top 
//                loadTop(fbArgs, cta, localBlockMetaData, mainShmem, pipeline, metaDataArr, metaData, i, tile, isGoldForLocQueue, iterationNumb);
//                //process main
//                processMain(fbArgs, cta, localBlockMetaData, mainShmem, pipeline, metaDataArr, metaData, i, tile, isGoldForLocQueue, iterationNumb, isBlockFull);
/////////// step 2 load bottom and process top 
//                loadBottom(fbArgs, cta, localBlockMetaData, mainShmem, pipeline, metaDataArr, metaData, i, tile, isGoldForLocQueue, iterationNumb, isAnythingInPadding);
//                //process top
//                processTop(fbArgs, cta, localBlockMetaData, mainShmem, pipeline, metaDataArr, metaData, i, tile, isGoldForLocQueue, iterationNumb, isAnythingInPadding);                     
///////////// step 3 load right  process bottom  
//                loadRight(fbArgs, cta, localBlockMetaData, mainShmem, pipeline, metaDataArr, metaData, i, tile, isGoldForLocQueue, iterationNumb, isAnythingInPadding);
//                //process bototm
//                processBottom(fbArgs, cta, localBlockMetaData, mainShmem, pipeline, metaDataArr, metaData, i, tile, isGoldForLocQueue, iterationNumb, isAnythingInPadding);
///////////// step 4 load left process right  
//               
//                loadLeft(fbArgs, cta, localBlockMetaData, mainShmem, pipeline, metaDataArr, metaData, i, tile, isGoldForLocQueue, iterationNumb, isAnythingInPadding);
//                processRight(fbArgs, cta, localBlockMetaData, mainShmem, pipeline, metaDataArr, metaData, i, tile, isGoldForLocQueue, iterationNumb, isAnythingInPadding);
///////// step 5 load anterior process left 
//                loadAnterior(fbArgs, cta, localBlockMetaData, mainShmem, pipeline, metaDataArr, metaData, i, tile, isGoldForLocQueue, iterationNumb, isAnythingInPadding);
//                processLeft(fbArgs, cta, localBlockMetaData, mainShmem, pipeline, metaDataArr, metaData, i, tile, isGoldForLocQueue, iterationNumb, isAnythingInPadding);
///////// step 6 load posterior process anterior 
//                loadPosterior(fbArgs, cta, localBlockMetaData, mainShmem, pipeline, metaDataArr, metaData, i, tile, isGoldForLocQueue, iterationNumb, isAnythingInPadding);
//                processAnterior(fbArgs, cta, localBlockMetaData, mainShmem, pipeline, metaDataArr, metaData, i, tile, isGoldForLocQueue, iterationNumb, isAnythingInPadding);
///////// step 7 
//// 
//                
//            //    sync(cta);
//
//                //load reference if needed or data for next iteration if there is such 
//                //process posterior, save data from res shmem to global memory also we mark weather block is full
//                lastLoad(fbArgs, cta, localBlockMetaData, mainShmem, pipeline, metaDataArr, metaData, i, tile, isGoldForLocQueue, iterationNumb, isAnythingInPadding, origArrs, worQueueStep);
//                processPosteriorAndSaveResShmem(fbArgs, cta, localBlockMetaData, mainShmem, pipeline, metaDataArr, metaData, i, tile, isGoldForLocQueue, iterationNumb, isAnythingInPadding, isBlockFull);
//                sync(cta);
//
// //////// step 8 basically in order to complete here anyting the count need to be bigger than counter
//               // loading for next block if block is not to be validated it was already done earlier
//                pipeline.producer_acquire();
//                if (localBlockMetaData[(i & 1) * 20 + ((1 - isGoldForLocQueue[i]) + 1)] //fp for gold and fn count for not gold
//                  > localBlockMetaData[(i & 1) * 20 + ((1 - isGoldForLocQueue[i]) + 3)]) {// so count is bigger than counter so we should validate
//                    if (i + 1 <= worQueueStep[0]) {
//                        loadMetaDataToShmem(cta, localBlockMetaData, mainShmem, pipeline, metaDataArr, metaData, 1, i);
//                    }
//                }
//                pipeline.producer_commit();
//                
//
//                //validation - so looking for newly covered voxel for opposite array so new fps or new fns
//                pipeline.consumer_wait();
//
//                validate(fbArgs, cta, localBlockMetaData, mainShmem, pipeline, metaDataArr, metaData, i, tile, isGoldForLocQueue, iterationNumb, isAnythingInPadding, isBlockFull, localFpConter, localFnConter, resultListPointerMeta, resultListPointerLocal, resultListPointerIterNumb);
//                /////////
//                pipeline.consumer_release();
//
//              //  sync(cta);
//
//                //pipeline.producer_acquire();
//
//                //pipeline.producer_commit();
//
//                //pipeline.consumer_wait();
//
//                //getTargetReduced(fbArgs, iterationNumb)[mainShmem[startOfLocalWorkQ + i] * metaData.mainArrSectionLength + metaData.mainArrXLength * (1 - isGoldForLocQueue[i])
//                //    + threadIdx.x + threadIdx.y * 32]
//                //    = mainShmem[begResShmem + threadIdx.x + threadIdx.y * 32];
//
//                //pipeline.consumer_release();
//
//           }
//       }
//
//        //here we are after all of the blocks planned to be processed by this block are
//
////updating local counters of last local block (normally it is done at the bagining of the next block)
////but we need to check weather any block was processed at all
//        pipeline.consumer_wait();
//
//        if (lastI[0] != UINT32_MAX) {
//            afterBlockClean(cta, worQueueStep, localBlockMetaData, mainShmem, lastI[0],
//                metaData, tile, localFpConter, localFnConter
//                , blockFpConter, blockFnConter
//                , metaDataArr, isAnythingInPadding, isBlockFull, isPaddingPass, isGoldForLocQueue, lastI);
//        
//         if (tile.thread_rank() == 0 && tile.meta_group_rank() == 3) {// this is how it is encoded wheather it is gold or segm block
//                 lastI[0] = UINT32_MAX;
//            }
//        }
//        pipeline.consumer_release();
//
//    }
//
//
//
//    sync(cta);
//
//    //     updating global counters
//    if (tile.thread_rank() == 0 && tile.meta_group_rank() == 0) {
//        if (blockFpConter[0] > 0) {
//            atomicAdd(&(minMaxes[10]), (blockFpConter[0]));
//        }
//    };
//    if (tile.thread_rank() == 1 && tile.meta_group_rank() == 0) {
//        if (blockFnConter[0] > 0) {
//            atomicAdd(&(minMaxes[11]), (blockFnConter[0]));
//        }
//    };
//    // in first thread block we zero work queue counter
//    if (threadIdx.x == 2 && threadIdx.y == 0) {
//        if (blockIdx.x == 0) {
//       
//            minMaxes[9] = 0;
//        }
//    };
//
//
//}
