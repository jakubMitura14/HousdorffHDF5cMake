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
int main(void){

  
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
 

    testMainPasswes();


    return 0;  // successfully terminated
}