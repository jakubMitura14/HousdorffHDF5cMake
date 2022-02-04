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



__device__ void compute(int* global_out, int const* shared_in) {



};
__global__ void with_staging(int* global_out, int const* global_in, size_t size,
    size_t batch_sz) {
    auto grid = cooperative_groups::this_grid();
    auto block = cooperative_groups::this_thread_block();
    assert(size == batch_sz * grid.size()); // Assume input size fits batch_sz *  grid_size
        constexpr size_t stages_count = 2; // Pipeline with two stages
        // Two batches must fit in shared memory:
    extern __shared__ int shared[]; // stages_count * block.size() * sizeof(int)     bytes
        size_t shared_offset[stages_count] = { 0, block.size() }; // Offsets to each    batch
        // Allocate shared storage for a two-stage cuda::pipeline:
        __shared__ cuda::pipeline_shared_state<
        cuda::thread_scope::thread_scope_block,
        stages_count
        > shared_state;
    auto pipeline = cuda::make_pipeline(block, &shared_state);
    // Each thread processes `batch_sz` elements.
    // Compute offset of the batch `batch` of this thread block in global memory:
    auto block_batch = [&](size_t batch) -> int {
        return block.group_index().x * block.size() + grid.size() * batch;
    };
    // Initialize first pipeline stage by submitting a `memcpy_async` to fetch a
    //whole batch for the block :
    if (batch_sz == 0) return;
    pipeline.producer_acquire();
    cuda::memcpy_async(block, shared + shared_offset[0], global_in +   block_batch(0), sizeof(int) * block.size(), pipeline);
    pipeline.producer_commit();
    // Pipelined copy/compute:
    for (size_t batch = 1; batch < batch_sz; ++batch) {
        // Stage indices for the compute and copy stages:
        size_t compute_stage_idx = (batch - 1) % 2;
        size_t copy_stage_idx = batch % 2;
        size_t global_idx = block_batch(batch);
        // Collectively acquire the pipeline head stage from all producer threads:
        pipeline.producer_acquire();
        // Submit async copies to the pipeline's head stage to be
        // computed in the next loop iteration
        cuda::memcpy_async(block, shared + shared_offset[copy_stage_idx], global_in     + global_idx, sizeof(int) * block.size(), pipeline);
        // Collectively commit (advance) the pipeline's head stage
        pipeline.producer_commit();
        // Collectively wait for the operations commited to the
        // previous `compute` stage to complete:
        pipeline.consumer_wait();
        // Computation overlapped with the memcpy_async of the "copy" stage:
        compute(global_out + global_idx, shared + shared_offset[compute_stage_idx]);
        // Collectively release the stage resources
        pipeline.consumer_release();
    }
    // Compute the data fetch by the last iteration
    pipeline.consumer_wait();
    compute(global_out + block_batch(batch_sz - 1), shared + shared_offset[(batch_sz -
        1) % 2]);
    pipeline.consumer_release();
}






int main(void){

  




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