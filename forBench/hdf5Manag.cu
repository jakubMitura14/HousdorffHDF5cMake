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








void loadHDFIntoBoolArr(H5std_string FILE_NAME, H5std_string DATASET_NAME, bool*& data) {
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


    
   data = (bool*)calloc(dims[0], sizeof(bool));




    dset.read(data, PredType::NATIVE_HBOOL, memspace, dspace); 


    //int sum = 0;
    //for (int i = 0; i < dims[0]; i++) {
    //    sum += data[i];
    //}
    //printf("suuum %d \n  ", sum);


    file.close();

}



/*
benchmark for original code from  https://github.com/Oyatsumi/HausdorffDistanceComparison
*/
void benchmarkOliviera(bool* onlyBladderBoolFlat, bool* onlyLungsBoolFlat, const int WIDTH, const int HEIGHT, const int DEPTH) {
    Volume img1 = Volume(WIDTH, HEIGHT, DEPTH), img2 = Volume(WIDTH, HEIGHT, DEPTH);

    for (int x = 0; x < WIDTH; x++) {
        for (int y = 0; y < HEIGHT; y++) {
            for (int z = 0; z < DEPTH; z++) {
                img1.setVoxelValue(onlyLungsBoolFlat[x + y * WIDTH + z * WIDTH * HEIGHT], x, y, z);
                img2.setVoxelValue(onlyBladderBoolFlat[x + y * WIDTH + z * WIDTH * HEIGHT], x, y, z);
            }
        }
    }



    auto begin = std::chrono::high_resolution_clock::now();


    HausdorffDistance* hd = new HausdorffDistance();
    int dist = (*hd).computeDistance(&img1, &img2);



    auto end = std::chrono::high_resolution_clock::now();

    std::cout << "Total elapsed time: ";
    std::cout << (double)(std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count() / (double)1000000000) << "s" << std::endl;

    printf("HD: %d \n", dist);

    //freeing memory
    img1.dispose(); img2.dispose();

//Datasize: 216530944
//Datasize : 216530944
//Total elapsed time : 2.62191s
//HD : 234

}





void benchmarkMitura(bool* onlyBladderBoolFlat, bool* onlyLungsBoolFlat, const int WIDTH, const int HEIGHT, const int DEPTH) {
    
    //// some preparations and configuring
    MetaDataCPU metaData;
    size_t size = sizeof(unsigned int) * 20;
    unsigned int* minMaxesCPU = (unsigned int*)malloc(size);
    metaData.minMaxes = minMaxesCPU;

    ForFullBoolPrepArgs<bool> forFullBoolPrepArgs;
    forFullBoolPrepArgs.metaData = metaData;
    forFullBoolPrepArgs.numberToLookFor = true;
    forFullBoolPrepArgs.goldArr = get3dArrCPU(onlyBladderBoolFlat, WIDTH, HEIGHT, DEPTH);
    forFullBoolPrepArgs.segmArr = get3dArrCPU(onlyLungsBoolFlat, WIDTH, HEIGHT, DEPTH);
    /// for debugging
    uint32_t* resultListPointerMetaCPU;
    uint32_t* resultListPointerLocalCPU;
    uint32_t* resultListPointerIterNumbCPU;
    uint32_t* metaDataArrPointerCPU;
    uint32_t* workQueuePointerCPU;
    uint32_t* reducedResCPU;
    uint32_t* origArrsCPU;


    //function invocation
    auto begin = std::chrono::high_resolution_clock::now();

    ForBoolKernelArgs<bool> fbArgs = mainKernelsRun(forFullBoolPrepArgs, reducedResCPU, resultListPointerMetaCPU
        , resultListPointerLocalCPU, resultListPointerIterNumbCPU
        , metaDataArrPointerCPU, workQueuePointerCPU, origArrsCPU, WIDTH, HEIGHT, DEPTH
    );
    auto end = std::chrono::high_resolution_clock::now();

    std::cout << "Total elapsed time: ";
    std::cout << (double)(std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count() / (double)1000000000) << "s" << std::endl;


    size_t sizeMinMax = sizeof(unsigned int) * 20;
    cudaMemcpy(minMaxesCPU, fbArgs.metaData.minMaxes, sizeMinMax, cudaMemcpyDeviceToHost);

    printf("HD: %d \n", minMaxesCPU[13]);


    // freeee
    free(onlyBladderBoolFlat);
    free(onlyLungsBoolFlat);


    free(resultListPointerMetaCPU);
    free(resultListPointerLocalCPU);
    free(resultListPointerIterNumbCPU);
    free(metaDataArrPointerCPU);
    free(workQueuePointerCPU);

    free(reducedResCPU);
    free(origArrsCPU);

}



void loadHDF() {
    const int WIDTH = 512;
    const int HEIGHT = 512;
    const int DEPTH = 826;

    const H5std_string FILE_NAMEonlyLungsBoolFlat("C:\\Users\\1\\PycharmProjects\\pythonProject3\\mytestfile.hdf5");
    const H5std_string DATASET_NAMEonlyLungsBoolFlat("onlyLungsBoolFlat");
    // create a vector the same size as the dataset
    bool* onlyLungsBoolFlat;
    loadHDFIntoBoolArr(FILE_NAMEonlyLungsBoolFlat, DATASET_NAMEonlyLungsBoolFlat, onlyLungsBoolFlat);

    const H5std_string FILE_NAMEonlyBladderBoolFlat("C:\\Users\\1\\PycharmProjects\\pythonProject3\\mytestfile.hdf5");
    const H5std_string DATASET_NAMEonlyBladderBoolFlat("onlyBladderBoolFlat");
    // create a vector the same size as the dataset
    bool* onlyBladderBoolFlat;
    loadHDFIntoBoolArr(FILE_NAMEonlyBladderBoolFlat, DATASET_NAMEonlyBladderBoolFlat, onlyBladderBoolFlat);

  //  benchmarkOliviera(onlyBladderBoolFlat, onlyLungsBoolFlat, WIDTH, HEIGHT, DEPTH);
    benchmarkMitura(onlyBladderBoolFlat, onlyLungsBoolFlat, WIDTH, HEIGHT, DEPTH);





}


