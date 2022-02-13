#include "cuda_runtime.h"
#include <cmath>

#include "device_launch_parameters.h"

// includes, system
#include <iostream>     // std::cout
#include <algorithm>    // std::min
//#include <helper_cuda.h>
#include <cmath>
//#include "Structs.cu"
#include <math.h>
//#include "MemoryTransfers.cu"
#include <cstdint>
#include <assert.h>
#include <numeric>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cooperative_groups/memcpy_async.h>
#include <cuda/pipeline>

//#include "BoolKernelTests.cu"
#include "testAll.cu"
#include "CooperativeGroupsUtils.cu"
using namespace cooperative_groups;

#include <iostream>
#include <string>
#include <vector>
#include <H5Cpp.h>
using namespace H5;


#ifdef OLD_HEADER_FILENAME
#include <iostream.h>
#else
#include <iostream>
#endif



using std::cout;
using std::endl;

#ifdef OLD_HEADER_FILENAME
#include <iostream.h>
#else
#include <iostream>
#endif
using std::cout;
using std::endl;
#include <string>
#include "H5Cpp.h"
using namespace H5;
const H5std_string FILE_NAME("C:\\Users\\1\\PycharmProjects\\pythonProject3\\mytestfile.hdf5");
const H5std_string DATASET_NAME("onlyLungs");
//const int    NX_SUB = 3;    // hyperslab dimensions
//const int    NY_SUB = 4;
//const int    NX = 7;        // output buffer dimensions
//const int    NY = 7;
//const int    NZ = 3;
//const int    RANK_OUT = 3;





//  pipeline_producer_commit(pipeline, barrier);


__global__ void with_staging(uint32_t* global_out, uint32_t* global_inA, uint32_t* globalOutGPUB, float* globalDummyGPU) {
    cooperative_groups::thread_block block = cooperative_groups::this_thread_block();
    thread_block_tile<32> tile = tiled_partition<32>(block);

    __shared__ uint32_t shmem[200];
    __shared__ uint32_t currBatch[1];
    __shared__ float dummmy[1];

    cuda::pipeline<cuda::thread_scope_thread> pipeline = cuda::make_pipeline();

    // for simplicity ignored Initializing first pipeline stage of submitting `memcpy_async` 
    //pipeline.producer_acquire();
    //...
    //pipeline.producer_commit();
    
    
    for (size_t batch = 1; batch < 10; ++batch) {
        ///////step 1
        // load
        pipeline.producer_acquire();
            if (tile.meta_group_rank() == 0) {
                cuda::memcpy_async(tile, &shmem[0], &global_inA[batch * 64], cuda::aligned_size_t<64>(sizeof(uint32_t) * 16), pipeline);
                pipeline.producer_commit();
            }
            if (tile.meta_group_rank() == 1) {
                cuda::memcpy_async(tile, &shmem[16], &global_inA[batch * 64 +16], cuda::aligned_size_t<64>(sizeof(uint32_t) * 16), pipeline);
                pipeline.producer_commit();
            }

        //compute data loaded in step 2 of previous iteration
        cuda::pipeline_consumer_wait_prior<0>(pipeline);
         
        //this works correctly
        if (tile.meta_group_rank() == 0) {
            global_out[batch * 64 + 32 + tile.thread_rank()] = shmem[32 + tile.thread_rank()];
        }

        if (tile.thread_rank() == batch && tile.meta_group_rank() == 0) {
            float w = 326;
            for (int j = 0; j < 5000; j++) {
                w += w / j;
            };
            globalDummyGPU[0] += w;
            currBatch[0] = batch;
        };

        pipeline.consumer_release();
        ///// step 2 
        //load
        pipeline.producer_acquire();
        if (tile.meta_group_rank() == 0) {
            cuda::memcpy_async(tile, &shmem[32], &global_inA[(batch +1)* 64+32], cuda::aligned_size_t<64>(sizeof(uint32_t) * 16), pipeline);
            pipeline.producer_commit();
        }
        if (tile.meta_group_rank() == 1) {
            cuda::memcpy_async(tile, &shmem[32 + 16], &global_inA[(batch +1)* 64 +32+ 16], cuda::aligned_size_t<64>(sizeof(uint32_t) * 16), pipeline);
            pipeline.producer_commit();
        }
        //compute data loaded in  step 1
        cuda::pipeline_consumer_wait_prior<0>(pipeline);
        
        //this works correctly
        if (tile.meta_group_rank() == 0) {
           global_out[batch * 64 + tile.thread_rank()] = shmem[tile.thread_rank()];//correct
        }

        if (tile.thread_rank() == (batch+1) && tile.meta_group_rank() == 1) {
            float w = 326;
            for (int j = 0; j < 5000; j++) {
                w += w / j;
            };
            globalOutGPUB[batch]=currBatch[0];
            globalDummyGPU[0] += w;
        };

        pipeline.consumer_release();

    }
    //  for simplicity ignored Computing the data fetch by the last iteration
    //cuda::pipeline_consumer_wait_prior<0>(pipeline);
    ////last computatons .. here omitted
    //pipeline.consumer_release();

    }


/*
results 

val 68 in 1
val 2 in 2
val 196 in 3
val 4 in 4
val 324 in 5
val 6 in 6
val 8 in 7
val 9 in 8
val 580 in 9


*/




int main(void){



    testMainPasswes();
//
//    cudaError_t syncErr;
//    cudaError_t asyncErr;
////    creating test data for pipeline concept
//    uint32_t* globalInGPUA;
//    int sizeOfArr = 6400;
//    int sizeOfArrB = 20;
//    uint32_t* globalOutGPU;
//    uint32_t* globalOutCPU;
//    float* globalDummyGPU;
//    uint32_t* globalOutGPUB;
//    size_t sizeC = (sizeOfArr * sizeof(uint32_t));
//    size_t sizeD = (sizeOfArrB * sizeof(uint32_t));
//    size_t sizeE = (sizeOfArrB * sizeof(float));
//    uint32_t* globalInCPUA = (uint32_t*)calloc(sizeOfArr, sizeof(uint32_t));
//
//
//    //populating with data 
//    for (int i = 0; i < sizeOfArr; i++) {
//        globalInCPUA[i] = i;
//    };
//
//
//    uint32_t* globalOUTCPU = (uint32_t*)calloc(sizeOfArr, sizeof(uint32_t));
//    uint32_t* globalOUTCPB = (uint32_t*)calloc(sizeOfArrB, sizeof(uint32_t));
//
//
//    //cudaMallocAsync(&mainArr, sizeB, 0);
//    cudaMalloc(&globalInGPUA, sizeC);
//    cudaMemcpy(globalInGPUA, globalInCPUA, sizeC, cudaMemcpyHostToDevice);
//
//    cudaMalloc(&globalOutGPU, sizeC);
//    cudaMemcpy(globalOutGPU, globalOUTCPU, sizeC, cudaMemcpyHostToDevice);
//
//
//
//    float* globalDummyCPU = (float*)calloc(sizeOfArrB, sizeof(float));
//    cudaMalloc(&globalDummyGPU, sizeE);
//
//    cudaMalloc(&globalOutGPUB, sizeD);
//    cudaMemcpy(globalOutGPUB, globalOUTCPB, sizeD, cudaMemcpyHostToDevice);
//
//    with_staging << <1,64 >> > (globalOutGPU, globalInGPUA, globalOutGPUB, globalDummyGPU);
//
//    //this works correctly
//    cudaDeviceSynchronize();    
//
//
//
//
//    cudaMemcpy(globalOUTCPU, globalOutGPU, sizeC, cudaMemcpyDeviceToHost);
//    for (int i = 130; i < 500; i++) {
//        if (globalOUTCPU[i]!= i) {
//            printf("val %d in %d \n", globalOUTCPU[i], i);
//        }
//    };
//
//    cudaMemcpy(globalOUTCPB, globalOutGPUB, sizeD, cudaMemcpyDeviceToHost);
//
//    for (int i = 0; i < 10; i++) {
//            printf("val %d in %d \n", globalOUTCPB[i], i);
//      
//    };
//    cudaMemcpy(globalDummyCPU, globalDummyGPU, sizeE, cudaMemcpyDeviceToHost);
//    printf("duppy %f \n", globalDummyCPU[0]);
//
//
//
//    syncErr = cudaGetLastError();
//    asyncErr = cudaDeviceSynchronize();
//    if (syncErr != cudaSuccess) printf("Error in syncErr: %s\n", cudaGetErrorString(syncErr));
//    if (asyncErr != cudaSuccess) printf("Error in asyncErr: %s\n", cudaGetErrorString(asyncErr));
//

    //workqueue








  




   // testMainPasswes();


        ///*
        // * Open the specified file and the specified dataset in the file.
        // */
        //H5File file(FILE_NAME, H5F_ACC_RDONLY);
        //DataSet dataset = file.openDataSet(DATASET_NAME);
        ///*
        // * Get the class of the datatype that is used by the dataset.
        // */
        //H5T_class_t type_class = dataset.getTypeClass();
        //DataSpace dataspace = dataset.getSpace();
        //int rank = dataspace.getSimpleExtentNdims();
        ///*
        // * Get the dimension size of each dimension in the dataspace and
        // * display them.
        // */
        //hsize_t dims_out[3];
        //int ndims = dataspace.getSimpleExtentDims(dims_out, NULL);
        //cout << "rank " << rank << ", dimensions " <<
        //    (unsigned long)(dims_out[0]) << " x " <<
        //    (unsigned long)(dims_out[1]) << 
        //    (unsigned long)(dims_out[2]) << endl;


        ///*
        // * Get class of datatype and print message if it's an integer.
        // */
        //if (type_class == H5T_INTEGER)
        //{
        //    cout << "Data set has INTEGER type" << endl;
        //    /*
        // * Get the integer datatype
        //     */
        //    IntType intype = dataset.getIntType();
        //    /*
        //     * Get order of datatype and print message if it's a little endian.
        //     */
        //    H5std_string order_string;
        //    H5T_order_t order = intype.getOrder(order_string);
        //    cout << order_string << endl;
        //    /*
        //     * Get size of the data element stored in file and print it.
        //     */
        //    size_t size = intype.getSize();
        //    cout << "Data size is " << size << endl;
        //}








        //hsize_t memdim = dims_out[0] * dims_out[1] * dims_out[2];;

        //std::vector<float> data_out(memdim);






        //use the same layout for file and memory
        //dataset.read(data_out.data(), PredType::NATIVE_INT64, dataspace, dataspace);


        ///*
        // * Define hyperslab in the dataset; implicitly giving strike and
        // * block NULL.
        // */
        //hsize_t      offset[2];   // hyperslab offset in the file
        //hsize_t      count[2];    // size of the hyperslab in the file
        //offset[0] = 1;
        //offset[1] = 2;
        //count[0] = NX_SUB;
        //count[1] = NY_SUB;
        //dataspace.selectHyperslab(H5S_SELECT_SET, count, offset);
        ///*
        // * Define the memory dataspace.
        // */
        //hsize_t     dimsm[3];              /* memory space dimensions */
        //dimsm[0] = NX;
        //dimsm[1] = NY;
        //dimsm[2] = NZ;
        //DataSpace memspace(RANK_OUT, dimsm);
        ///*
        // * Define memory hyperslab.
        // */
        //hsize_t      offset_out[3];   // hyperslab offset in memory
        //hsize_t      count_out[3];    // size of the hyperslab in memory
        //offset_out[0] = 3;
        //offset_out[1] = 0;
        //offset_out[2] = 0;
        //count_out[0] = NX_SUB;
        //count_out[1] = NY_SUB;
        //count_out[2] = 1;
        //memspace.selectHyperslab(H5S_SELECT_SET, count_out, offset_out);
        ///*
        // * Read data from hyperslab in the file into the hyperslab in
        // * memory and display the data.
        // */
        //dataset.read(data_out, PredType::NATIVE_INT, memspace, dataspace);
        //for (j = 0; j < NX; j++)
        //{
        //    for (i = 0; i < NY; i++)
        //        cout << data_out[j][i][0] << " ";
        //    cout << endl;
        //}
        /*
         * 0 0 0 0 0 0 0
         * 0 0 0 0 0 0 0
         * 0 0 0 0 0 0 0
         * 3 4 5 6 0 0 0
         * 4 5 6 7 0 0 0
         * 5 6 7 8 0 0 0
         * 0 0 0 0 0 0 0
         */
 



    return 0;  // successfully terminated
}