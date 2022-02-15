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




//template <typename TKKI, typename forPipeline >
template <typename TKKI >
inline __device__ void mainDilatation(bool isPaddingPass, ForBoolKernelArgs<TKKI>& fbArgs, uint32_t* mainArrAPointer,
    uint32_t* mainArrBPointer, MetaDataGPU& metaData
    , unsigned int* minMaxes, uint32_t* workQueue
    , uint32_t* resultListPointerMeta, uint32_t* resultListPointerLocal, uint32_t* resultListPointerIterNumb,
    thread_block& cta, thread_block_tile<32>& tile, grid_group& grid, uint32_t mainShmem[lengthOfMainShmem]
    , bool isAnythingInPadding[6], bool isBlockFull[], int iterationNumb[1], unsigned int globalWorkQueueOffset[1],
    unsigned int globalWorkQueueCounter[1], unsigned int localWorkQueueCounter[1],
    unsigned int localTotalLenthOfWorkQueue[1], unsigned int localFpConter[1],
    unsigned int localFnConter[1], unsigned int blockFpConter[1],
    unsigned int blockFnConter[1], unsigned int resultfpOffset[1],
    unsigned int resultfnOffset[1], unsigned int worQueueStep[1],
    uint32_t isGold[1], uint32_t currLinIndM[1], unsigned int localMinMaxes[5]
    , uint32_t localBlockMetaData[40], unsigned int fpFnLocCounter[1]
    , bool isGoldPassToContinue[1], bool isSegmPassToContinue[1]
    , uint32_t* origArrs, uint32_t* metaDataArr, bool iasAnyProcessed[1],
    bool isGoldForLocQueue[localWorkQueLength], bool isBlockToBeValidated[1]
    , cuda::pipeline<cuda::thread_scope_block>& pipeline, cuda::aligned_size_t<128Ui64>& bigShape
    , cuda::aligned_size_t<128Ui64>& thirdRegShape
) {

    //initial cleaning  and initializations include loading min maxes
    dilBlockInitialClean(tile, isPaddingPass, iterationNumb, localWorkQueueCounter, blockFpConter,
        blockFnConter, localFpConter, localFnConter, isBlockFull, fpFnLocCounter,
        iasAnyProcessed, localTotalLenthOfWorkQueue, globalWorkQueueOffset
        , worQueueStep, minMaxes, localMinMaxes);
    sync(cta);
    /// load work QueueData into shared memory 
    for (uint32_t bigloop = blockIdx.x * globalWorkQueueOffset[0]; bigloop < ((blockIdx.x + 1) * globalWorkQueueOffset[0]); bigloop += worQueueStep[0]) {
        // grid stride loop - sadly most of threads will be idle 
        /////////// loading to work queue
        loadWorkQueue(mainShmem, workQueue, isGoldForLocQueue, bigloop, worQueueStep);

        //now all of the threads in the block needs to have the same i value so we will increment by 1 we are preloading to the pipeline block metaData
////##### pipeline Step 0



        sync(cta);
        //loading metadata
        pipeline.producer_acquire();

        cuda::memcpy_async(cta, (&localBlockMetaData[20]),
            (&metaDataArr[(mainShmem[startOfLocalWorkQ ])
                * metaData.metaDataSectionLength])
            , cuda::aligned_size_t<4>(sizeof(uint32_t) * 20), pipeline);

        cuda::memcpy_async(cta, (&localBlockMetaData[0]),
            (&metaDataArr[(mainShmem[startOfLocalWorkQ])
                * metaData.metaDataSectionLength])
            , cuda::aligned_size_t<4>(sizeof(uint32_t) * 20), pipeline);

        //loadMetaDataToShmem(cta, localBlockMetaData, mainShmem, pipeline, metaDataArr, metaData, 0, 0);

        pipeline.producer_commit();


        //if (tile.thread_rank() < 20 && tile.meta_group_rank() == 2) {
        //    localBlockMetaData[(i & 1) * 20+tile.thread_rank()] =
        //        metaDataArr[(mainShmem[startOfLocalWorkQ ])
        //        * metaData.metaDataSectionLength + tile.thread_rank()];
        //};




        for (uint32_t i = 0; i < worQueueStep[0]; i += 1) {
            if (((bigloop + i) < localTotalLenthOfWorkQueue[0]) && ((bigloop + i) < ((blockIdx.x + 1) * globalWorkQueueOffset[0]))) {
                //if (tile.thread_rank() == 0 && tile.meta_group_rank() == 0) {                      
                //    printf("linMeta beg %d is gold %d\n ", mainShmem[startOfLocalWorkQ + i], isGoldForLocQueue[i]);
                //};

//////////////// step 0  load main data and final processing of previous block
               //loading main data for first dilatation
                loadMain(fbArgs, cta, localBlockMetaData, mainShmem, pipeline, metaDataArr, metaData, i, tile, isGoldForLocQueue, iterationNumb);
                
                
                pipeline.consumer_wait();
               // mainShmem[begResShmem + threadIdx.x + threadIdx.y * 32] = bitDilatate(mainShmem[begSourceShmem + threadIdx.x + threadIdx.y * 32]);



                pipeline.consumer_release();
///////// step 1 load top and process main data 
               //load top 
                loadTop(fbArgs, cta, localBlockMetaData, mainShmem, pipeline, metaDataArr, metaData, i, tile, isGoldForLocQueue, iterationNumb);
                //process main
                processMain(fbArgs, cta, localBlockMetaData, mainShmem, pipeline, metaDataArr, metaData, i, tile, isGoldForLocQueue, iterationNumb);                
///////// step 2 load bottom and process top 
                loadBottom(fbArgs, cta, localBlockMetaData, mainShmem, pipeline, metaDataArr, metaData, i, tile, isGoldForLocQueue, iterationNumb, isAnythingInPadding);
                //process top
                processTop(fbArgs, cta, localBlockMetaData, mainShmem, pipeline, metaDataArr, metaData, i, tile, isGoldForLocQueue, iterationNumb, isAnythingInPadding);                     
///////// step 3 load right  process bottom  
                loadRight(fbArgs, cta, localBlockMetaData, mainShmem, pipeline, metaDataArr, metaData, i, tile, isGoldForLocQueue, iterationNumb, isAnythingInPadding);
                //process bototm
                processBottom(fbArgs, cta, localBlockMetaData, mainShmem, pipeline, metaDataArr, metaData, i, tile, isGoldForLocQueue, iterationNumb, isAnythingInPadding);
///////// step 4 load left process right  
                loadLeft(fbArgs, cta, localBlockMetaData, mainShmem, pipeline, metaDataArr, metaData, i, tile, isGoldForLocQueue, iterationNumb, isAnythingInPadding);
                processRight(fbArgs, cta, localBlockMetaData, mainShmem, pipeline, metaDataArr, metaData, i, tile, isGoldForLocQueue, iterationNumb, isAnythingInPadding);
///////// step 5 load anterior process left 
                loadAnterior(fbArgs, cta, localBlockMetaData, mainShmem, pipeline, metaDataArr, metaData, i, tile, isGoldForLocQueue, iterationNumb, isAnythingInPadding);
                processLeft(fbArgs, cta, localBlockMetaData, mainShmem, pipeline, metaDataArr, metaData, i, tile, isGoldForLocQueue, iterationNumb, isAnythingInPadding);
///////// step 6 load posterior process anterior 
                loadPosterior(fbArgs, cta, localBlockMetaData, mainShmem, pipeline, metaDataArr, metaData, i, tile, isGoldForLocQueue, iterationNumb, isAnythingInPadding);
                processAnterior(fbArgs, cta, localBlockMetaData, mainShmem, pipeline, metaDataArr, metaData, i, tile, isGoldForLocQueue, iterationNumb, isAnythingInPadding);
///////// step 7 
                //load reference if needed or data for next iteration if there is such 
                //process posterior, save data from res shmem to global memory also we mark weather block is full
                lastLoad(fbArgs, cta, localBlockMetaData, mainShmem, pipeline, metaDataArr, metaData, i, tile, isGoldForLocQueue, iterationNumb, isAnythingInPadding, origArrs, worQueueStep);
                processPosteriorAndSaveResShmem(fbArgs, cta, localBlockMetaData, mainShmem, pipeline, metaDataArr, metaData, i, tile, isGoldForLocQueue, iterationNumb, isAnythingInPadding, isBlockFull);

 ////////// step 8 basically in order to complete here anyting the count need to be bigger than counter
                //loading for next block if block is not to be validated it was already done earlier
                pipeline.producer_acquire();
                if (localBlockMetaData[(i & 1) * 20 + ((1 - isGoldForLocQueue[i]) + 1)] //fp for gold and fn count for not gold
                  > localBlockMetaData[(i & 1) * 20 + ((1 - isGoldForLocQueue[i]) + 3)]) {// so count is bigger than counter so we should validate
                    if (i + 1 <= worQueueStep[0]) {
                        loadMetaDataToShmem(cta, localBlockMetaData, mainShmem, pipeline, metaDataArr, metaData, 1, i);
                    }
                }
                pipeline.producer_commit();
                


                pipeline.consumer_wait();


                //dilatateHelperForTransverse((threadIdx.y == 0), 5
                //    , (0), (-1), mainShmem, isAnythingInPadding
                //    , 0, threadIdx.x // we add offset depending on y dimension
                //    , 18, begSecRegShmem, localBlockMetaData);

                // //TODO remove
                // getTargetReduced(fbArgs, iterationNumb)[mainShmem[startOfLocalWorkQ + i] * metaData.mainArrSectionLength + metaData.mainArrXLength * (1 - isGoldForLocQueue[i])
                //     + threadIdx.x + threadIdx.y * 32]
                //     = mainShmem[begResShmem + threadIdx.x + threadIdx.y * 32];

                pipeline.consumer_release();
//////////

      



              //  ////////#### pipeline step 6) if block is to be validated we process the res and reference data and start loading data for begining of the next loop
              // sync(cta);
              //  ////load data for next iteration
              //  if (i + 1 <= worQueueStep[0]) {
              //      pipeline.producer_acquire();
              //      cuda::memcpy_async(cta, &mainShmem[begSourceShmem], &getSourceReduced(fbArgs, iterationNumb)[
              //          mainShmem[startOfLocalWorkQ + i+1] * metaData.mainArrSectionLength + metaData.mainArrXLength * (1 - isGoldForLocQueue[i+1])],
              //          bigShape, pipeline);
              //      pipeline.producer_commit();
              //  
              //   }


              //  // now all of the data is processed we need to save it into global memory
              //  // TODO try to use mempcy async here
              //  //if (mainShmem[begResShmem + threadIdx.x + threadIdx.y * 32]>0) {
              //  getTargetReduced(fbArgs, iterationNumb)[mainShmem[startOfLocalWorkQ + i] * metaData.mainArrSectionLength + metaData.mainArrXLength * (1 - isGoldForLocQueue[i])
              //      + threadIdx.x + threadIdx.y * 32]
              //      = mainShmem[begResShmem + threadIdx.x + threadIdx.y * 32];
              //  //}


              //  if (localBlockMetaData[(i & 1) * 20+((1 - isGoldForLocQueue[i]) + 1)] //fp for gold and fn count for not gold
              //          > localBlockMetaData[(i & 1) * 20+((1 - isGoldForLocQueue[i]) + 3)]) {// so count is bigger than counter so we should validate
              //      mainShmem[begSourceShmem + threadIdx.x + threadIdx.y * 32] = ((~mainShmem[begSourceShmem + threadIdx.x + threadIdx.y * 32]) & mainShmem[begResShmem + threadIdx.x + threadIdx.y * 32]);



              //      //we now look for bits prasent in both reference arrays and current one
              //      mainShmem[begSourceShmem + threadIdx.x + threadIdx.y * 32] = ((mainShmem[begSourceShmem + threadIdx.x + threadIdx.y * 32]) & mainShmem[begfirstRegShmem + threadIdx.x + threadIdx.y * 32]);


              //      // now we look through bits and when some is set we call it a result 
              //      #pragma unroll
              //      for (uint8_t bitPos = 0; bitPos < 32; bitPos++) {
              //          //if any bit here is set it means it should be added to result list 
              //          if (isBitAt(mainShmem[begSourceShmem + threadIdx.x + threadIdx.y * 32], bitPos)) {
              //              //first we add to the resList
              //              //TODO consider first passing it into shared memory and then async mempcy ...
              //              //we use offset plus number of results already added (we got earlier count from global memory now we just atomically add locally)
              //              unsigned int old = 0;
              //              ////// IMPORTANT for some reason in order to make it work resultfnOffset and resultfnOffset swith places
              //              if (isGoldForLocQueue[i]) {
              //                  old = atomicAdd_block(&(localFpConter[0]), 1) + localBlockMetaData[(i & 1) * 20+6] + localBlockMetaData[(i & 1) * 20+4];
              //              }
              //              else {
              //                  old = atomicAdd_block(&(localFnConter[0]), 1) + localBlockMetaData[(i & 1) * 20+5] + localBlockMetaData[(i & 1) * 20+3];
              //              };
              //              //   add results to global memory    
              //              //we add one gere jjust to distinguish it from empty result
              //              resultListPointerMeta[old] = uint32_t(mainShmem[startOfLocalWorkQ + i] +(isGoldOffset * isGoldForLocQueue[i])+1);
              //              resultListPointerLocal[old] = uint32_t((fbArgs.dbYLength * 32 * bitPos) + (threadIdx.y * 32) + (threadIdx.x) );
              //              resultListPointerIterNumb[old] = uint32_t(iterationNumb[0]);

              //           /*   printf("rrrrresult i %d  meta %d isGold %d old %d localFpConter %d localFnConter %d fpOffset %d fnOffset %d linIndUpdated %d  localInd %d  xLoc %d yLoc %d zLoc %d \n"
              //                  ,i
              //                  ,mainShmem[startOfLocalWorkQ + i]
              //                  , isGoldForLocQueue[i]
              //                  , old
              //                  , localFpConter[0]
              //                  , localFnConter[0]
              //                  , localBlockMetaData[(i & 1) * 20+ 5]
              //                  , localBlockMetaData[(i & 1) * 20+6]
              //                  , uint32_t(mainShmem[startOfLocalWorkQ + i] + isGoldOffset * isGoldForLocQueue[i])
              //                  , uint32_t((fbArgs.dbYLength * 32 * bitPos) + (threadIdx.y * 32) + (threadIdx.x))
              //                  , threadIdx.x
              //                  , threadIdx.y
              //                  , bitPos
              //              );*/

              //          }

              //      };
              //sync(cta);
              //  }
              //      //loading metadaa for next loop 
              //      if (i + 1 <= worQueueStep[0]) {
              //          if (tile.thread_rank() < 20 && tile.meta_group_rank() == 2) {
              //                  localBlockMetaData[(i & 1) * 20+tile.thread_rank()] = 
              //                  metaDataArr[(mainShmem[startOfLocalWorkQ + i + 1])
              //                      * metaData.metaDataSectionLength + tile.thread_rank()];
              //          };
              //      }

              //      //finilizing
              //      afterBlockClean(cta, worQueueStep, localBlockMetaData, mainShmem, i,
              //          metaData, tile, localFpConter, localFnConter
              //          , blockFpConter, blockFnConter
              //          , metaDataArr, isAnythingInPadding, isBlockFull, isPaddingPass, isGoldForLocQueue);


              //  sync(cta);


            }
        }
    }

    //here we are after all of the blocks planned to be processed by this block are
    sync(cta);

    //updating local counters of last local block (normally it is done at the bagining of the next block)
    //but we need to check weather any block was processed at all
    //if (iasAnyProcessed[0]) {
    //    afterBlockClean(cta, worQueueStep, localBlockMetaData, mainShmem, 1,
    //        metaData, tile, localFpConter, localFnConter
    //        , blockFpConter, blockFnConter
    //        , metaDataArr, isAnythingInPadding, isBlockFull, isPaddingPass, isGoldForLocQueue);
    //}

    ////     updating global counters
    //if (tile.thread_rank() == 0 && tile.meta_group_rank() == 0) {
    //    if (blockFpConter[0] > 0) {
    //        atomicAdd(&(minMaxes[10]), (blockFpConter[0]));
    //    }
    //};
    //if (tile.thread_rank() == 1 && tile.meta_group_rank() == 0) {
    //    if (blockFnConter[0] > 0) {
    //        atomicAdd(&(minMaxes[11]), (blockFnConter[0]));
    //    }
    //};
    //// in first thread block we zero work queue counter
    //if (tile.thread_rank() == 2 && tile.meta_group_rank() == 0) {
    //    if (blockIdx.x == 0) {
    //        minMaxes[9] = 0;
    //    }
    //};


}
