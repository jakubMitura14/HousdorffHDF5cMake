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



__device__ void computeA(uint32_t* global_out, uint32_t const* shared_in) {
    for (uint16_t linIdexMeta = blockIdx.x * blockDim.x + threadIdx.x; linIdexMeta < 32; linIdexMeta += blockDim.x * gridDim.x) {
        
     //   printf("  ***  ");
       global_out[linIdexMeta] = shared_in[linIdexMeta] +1;   }

};

__device__ void computeB(uint32_t* global_out, uint32_t const* shared_in) {
    for (uint16_t linIdexMeta = blockIdx.x * blockDim.x + threadIdx.x; linIdexMeta < 32; linIdexMeta += blockDim.x * gridDim.x) {

        //   printf("  ***  ");
        global_out[linIdexMeta] = shared_in[linIdexMeta] + 2;
    }

};

__device__ void computeC(uint32_t* global_out, uint32_t const* shared_in) {
    for (uint16_t linIdexMeta = blockIdx.x * blockDim.x + threadIdx.x; linIdexMeta < 32; linIdexMeta += blockDim.x * gridDim.x) {

        //   printf("  ***  ");
        global_out[linIdexMeta] = shared_in[linIdexMeta] + 3;
    }

};


__global__ void with_staging(uint32_t* global_out, uint32_t* global_inA,  uint32_t* global_inB,  uint32_t* global_inC) {
    auto grid = cooperative_groups::this_grid();
    cooperative_groups::thread_block block = cooperative_groups::this_thread_block();
    constexpr size_t stages_count = 2; // Pipeline with two stages
   
                                       
    bool isBlockFull = true;// usefull to establish do we have block completely filled and no more dilatations possible
    /*
    * according to https://forums.developer.nvidia.com/t/find-the-limit-of-shared-memory-that-can-be-used-per-block/48556 it is good to keep shared memory below 16kb kilo bytes
    main shared memory spaces
    0-1023 : sourceShmem
    1024-2047 : resShmem
    2048-3071 : first register space
    3072-4095 : second register space
    4096-4468 (372 length) : place for local work queue in dilatation kernels
    */
    __shared__ uint32_t shmem[100];
    // holding data about paddings 


    // holding data weather we have anything in padding 0)top  1)bottom, 2)left 3)right, 4)anterior, 5)posterior,
    __shared__ bool isAnythingInPadding[6];

    __shared__ unsigned int localBlockMetaData[19];


   cuda::pipeline<cuda::thread_scope_thread> pipeline = cuda::make_pipeline();

    size_t shared_offset[stages_count] = { 0, block.size() }; // Offsets to each

    // Initialize first pipeline stage by submitting a `memcpy_async` to fetch a whole batch for the block 


   // cuda::memcpy_async(block, &shmem[0], &global_in[0], cuda::aligned_size_t <alignof(uint32_t)>(sizeof(uint32_t) * 32), pipeline);
    //pipeline.producer_commit();

    // get first data into pipeline so from global in to first half in shmem
    pipeline.producer_acquire();
    cuda::memcpy_async(block, &shmem[0], &global_inA[0], cuda::aligned_size_t<4>(sizeof(uint32_t) * 32), pipeline);
    pipeline.producer_commit();

    
    // loadIntoShmem(pipeline, block, shmem, global_inA,  0, 0, 32);

    // Pipelined copy/compute:
    for (size_t batch = 1; batch < 3; ++batch) {
        //here we load data for compute step that will be in next loop iteration
        pipeline.producer_acquire();
        cuda::memcpy_async(block, &shmem[(batch & 1) *32], &global_inA[batch*32], cuda::aligned_size_t <alignof(uint32_t)>(sizeof(uint32_t) * 32), pipeline);
        pipeline.producer_commit();

        //so here we wait for previous data load - in case it is fist loop we wait for data that was scheduled before loop started
        pipeline.consumer_wait();
        computeA(&global_out[(batch-1)*32] , &shmem[((batch-1) & 1) * 32]);
        // Collectively release the stage resources
        pipeline.consumer_release();
    }
    // Compute the data fetch by the last iteration

    pipeline.consumer_wait();
    computeA(&global_out[2 * 32] , &shmem[(2 & 1) * 32]);
    pipeline.consumer_release();

    }







int main(void){
    //creating test data for pipeline concept
    uint32_t* globalInGPUA;
    uint32_t* globalInGPUB;
    uint32_t* globalInGPUC;


    uint32_t* globalOutGPU;
    size_t sizeC = (320 * sizeof(uint32_t));
    uint32_t* globalInCPUA = (uint32_t*)calloc(320 , sizeof(uint32_t));
    uint32_t* globalInCPUB = (uint32_t*)calloc(320 , sizeof(uint32_t));
    uint32_t* globalInCPUC = (uint32_t*)calloc(320 , sizeof(uint32_t));

    //populating to ones
    for (int i = 0; i < 96; i++) {
        globalInCPUA[i] = 10;
    };

    //populating to ones
    for (int i = 0; i < 96; i++) {
        globalInCPUB[i] = 100;
    };


    //populating to ones
    for (int i = 0; i < 96; i++) {
        globalInCPUC[i] = 1000;
    };

    uint32_t* globalOUTCPU = (uint32_t*)calloc(320, sizeof(uint32_t));


    //cudaMallocAsync(&mainArr, sizeB, 0);
    cudaMalloc(&globalInGPUA, sizeC);
    cudaMemcpy(globalInGPUA, globalInCPUA, sizeC, cudaMemcpyHostToDevice);

    cudaMalloc(&globalInGPUB, sizeC);
    cudaMemcpy(globalInGPUB, globalInCPUB, sizeC, cudaMemcpyHostToDevice);

    cudaMalloc(&globalInGPUC, sizeC);
    cudaMemcpy(globalInGPUC, globalInCPUC, sizeC, cudaMemcpyHostToDevice);


    cudaMalloc(&globalOutGPU, sizeC);
    cudaMemcpy(globalOutGPU, globalOUTCPU, sizeC, cudaMemcpyHostToDevice);

    with_staging << <1,32 >> > (globalOutGPU, globalInGPUA, globalInGPUB, globalInGPUC);


    checkCuda(cudaDeviceSynchronize(), "just after copy device to host");
    
    cudaMemcpy(globalOUTCPU, globalOutGPU, sizeC, cudaMemcpyDeviceToHost);

    for (int i = 0; i < 96; i++) {
       printf("val %d in %d \n", globalOUTCPU[i],i);
    };


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