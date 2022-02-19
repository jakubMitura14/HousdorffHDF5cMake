#include "cuda_runtime.h"
#include <cmath>

#include "device_launch_parameters.h"

#include <algorithm>    // std::min
#include <cmath>
#include <math.h>
#include <cstdint>
#include <assert.h>
#include <numeric>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cooperative_groups/memcpy_async.h>
#include <cuda/pipeline>
#include <chrono>

#include <iostream>
#include <string>
#include <vector>
#include <H5Cpp.h>
using namespace H5;


using std::cout;
using std::endl;
#include <string>
#include "H5Cpp.h"
#include "Volume.h"
#include "HausdorffDistance.cuh"
#include "HausdorffDistance.cu"



using namespace H5;
const H5std_string FILE_NAME("C:\\Users\\1\\PycharmProjects\\pythonProject3\\mytestfile.hdf5");
const H5std_string DATASET_NAME("onlyLungsBoolFlat");



void loadHDF() {
    /*
     * Open the specified file and the specified dataset in the file.
     */
    H5File file(FILE_NAME, H5F_ACC_RDONLY);
    DataSet dset = file.openDataSet(DATASET_NAME);
    /*
     * Get the class of the datatype that is used by the dataset.
     */
    H5T_class_t type_class = dset.getTypeClass();
    DataSpace dspace = dset.getSpace();
    int rank = dspace.getSimpleExtentNdims();
    
    
    hsize_t dims[2];
    rank = dspace.getSimpleExtentDims(dims, NULL); // rank = 1
    cout << "Datasize: " << dims[0] << endl; // this is the correct number of values

    // Define the memory dataspace
    hsize_t dimsm[1];
    dimsm[0] = dims[0];
    DataSpace memspace(1, dimsm);


    // create a vector the same size as the dataset
    bool* data = (bool*)calloc(dims[0], sizeof(bool));
    




    dset.read(data, PredType::NATIVE_HBOOL, memspace, dspace); 


    int sum = 0;
    for (int i = 0; i < dims[0]; i++) {
        sum += data[i];
    }
    printf("suuum %d \n  ", sum);


    file.close();
    /*
     * Get the dimension size of each dimension in the dataspace and
     * display them.
     */
 //   hsize_t dims_out[1];
 //   int ndims = dataspace.getSimpleExtentDims(dims_out, NULL);
 //   cout << "rank " << rank << ", dimensions " <<
 //       (unsigned long)(dims_out[0]) << " x " <<
 //endl;









    /* Close the dataset and dataspace */
   // dataset.
   //auto  status = H5Dclose(dataset);

   // status = H5Sclose(dataspace);



    /// <summary>
    /// ////////////original morphological hausdorff
    /// </summary>

    //bool* img1B = img1I.getBooleanLinearImage(0);
    //bool* img2B = img2I.getBooleanLinearImage(0);

    //auto begin = std::chrono::high_resolution_clock::now();

    ////img1.setVoxelValue(1, 0, 0, 0);
    ////img2.setVoxelValue(1, 255, 0, 0);

    //HausdorffDistance* hd = new HausdorffDistance();
    //int dist = (*hd).computeDistance(&img1, &img2);



    //auto end = std::chrono::high_resolution_clock::now();

    //std::cout << "Total elapsed time: ";
    //std::cout << (double)(::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count() / (double)1000000000) << "s" << std::endl;

    //printf("HD: %d \n", dist);








}