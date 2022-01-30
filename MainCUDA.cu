//
//
//#include "cuda_runtime.h"
//#include <cmath>
//
//#include "device_launch_parameters.h"
//#include "simpletest.h"
//
//// includes, system
//#include <iostream>     // std::cout
//#include <algorithm>    // std::min
////#include <helper_cuda.h>
//#include <cmath>
////#include "Structs.cu"
//#include <math.h>
////#include "MemoryTransfers.cu"
//#include <cstdint>
//#include <assert.h>
//#include <numeric>
//#include <cooperative_groups.h>
//#include <cooperative_groups/reduce.h>
////#include "BoolKernelTests.cu"
//#include "testAll.cu"
//using namespace cooperative_groups;
//
//#include <iostream>
//#include <string>
//#include <vector>
//#include <H5Cpp.h>
//using namespace H5;
//
//
//const H5std_string FILE_NAME("SDS.h5");
//const H5std_string DATASET_NAME("IntArray");
//const int    NX_SUB = 3;    // hyperslab dimensions
//const int    NY_SUB = 4;
//const int    NX = 7;        // output buffer dimensions
//const int    NY = 7;
//const int    NZ = 3;
//const int    RANK_OUT = 3;
//
//
//
//int main()
//{
//    /*
//   * Output buffer initialization.
//   */
//    int i, j, k;
//    int         data_out[NX][NY][NZ]; /* output buffer */
//    for (j = 0; j < NX; j++)
//    {
//        for (i = 0; i < NY; i++)
//        {
//            for (k = 0; k < NZ; k++)
//                data_out[j][i][k] = 0;
//        }
//    }
//    /*
//     * Try block to detect exceptions raised by any of the calls inside it
//     */
//    try
//    {
//        /*
//         * Turn off the auto-printing when failure occurs so that we can
//         * handle the errors appropriately
//         */
//        Exception::dontPrint();
//        /*
//         * Open the specified file and the specified dataset in the file.
//         */
//        H5File file(FILE_NAME, H5F_ACC_RDONLY);
//        DataSet dataset = file.openDataSet(DATASET_NAME);
//        /*
//         * Get the class of the datatype that is used by the dataset.
//         */
//        H5T_class_t type_class = dataset.getTypeClass();
//        /*
//         * Get class of datatype and print message if it's an integer.
//         */
//        if (type_class == H5T_INTEGER)
//        {
//            //cout << "Data set has INTEGER type" << endl;
//            /*
//         * Get the integer datatype
//             */
//            IntType intype = dataset.getIntType();
//            /*
//             * Get order of datatype and print message if it's a little endian.
//             */
//            H5std_string order_string;
//            H5T_order_t order = intype.getOrder(order_string);
//          //  cout << order_string << endl;
//            /*
//             * Get size of the data element stored in file and print it.
//             */
//            size_t size = intype.getSize();
//           // cout << "Data size is " << size << endl;
//        }
//        /*
//         * Get dataspace of the dataset.
//         */
//        DataSpace dataspace = dataset.getSpace();
//        /*
//         * Get the number of dimensions in the dataspace.
//         */
//        int rank = dataspace.getSimpleExtentNdims();
//        /*
//         * Get the dimension size of each dimension in the dataspace and
//         * display them.
//         */
//        hsize_t dims_out[2];
//        int ndims = dataspace.getSimpleExtentDims(dims_out, NULL);
//       /* cout << "rank " << rank << ", dimensions " <<
//            (unsigned long)(dims_out[0]) << " x " <<
//            (unsigned long)(dims_out[1]) << endl;*/
//        /*
//         * Define hyperslab in the dataset; implicitly giving strike and
//         * block NULL.
//         */
//        hsize_t      offset[2];   // hyperslab offset in the file
//        hsize_t      count[2];    // size of the hyperslab in the file
//        offset[0] = 1;
//        offset[1] = 2;
//        count[0] = NX_SUB;
//        count[1] = NY_SUB;
//        dataspace.selectHyperslab(H5S_SELECT_SET, count, offset);
//        /*
//         * Define the memory dataspace.
//         */
//        hsize_t     dimsm[3];              /* memory space dimensions */
//        dimsm[0] = NX;
//        dimsm[1] = NY;
//        dimsm[2] = NZ;
//        DataSpace memspace(RANK_OUT, dimsm);
//        /*
//         * Define memory hyperslab.
//         */
//        hsize_t      offset_out[3];   // hyperslab offset in memory
//        hsize_t      count_out[3];    // size of the hyperslab in memory
//        offset_out[0] = 3;
//        offset_out[1] = 0;
//        offset_out[2] = 0;
//        count_out[0] = NX_SUB;
//        count_out[1] = NY_SUB;
//        count_out[2] = 1;
//        memspace.selectHyperslab(H5S_SELECT_SET, count_out, offset_out);
//        /*
//         * Read data from hyperslab in the file into the hyperslab in
//         * memory and display the data.
//         */
//        dataset.read(data_out, PredType::NATIVE_INT, memspace, dataspace);
//        //for (j = 0; j < NX; j++)
//        //{
//        //    for (i = 0; i < NY; i++)
//        //    //    cout << data_out[j][i][0] << " ";
//        //  //  cout << endl;
//        //}
//        /*
//         * 0 0 0 0 0 0 0
//         * 0 0 0 0 0 0 0
//         * 0 0 0 0 0 0 0
//         * 3 4 5 6 0 0 0
//         * 4 5 6 7 0 0 0
//         * 5 6 7 8 0 0 0
//         * 0 0 0 0 0 0 0
//         */
//    }  // end of try block
//    // catch failure caused by the H5File operations
//    catch (FileIException error)
//    {
//      //  error.printError();
//        return -1;
//    }
//    // catch failure caused by the DataSet operations
//    catch (DataSetIException error)
//    {
//      //  error.printError();
//        return -1;
//    }
//    // catch failure caused by the DataSpace operations
//    catch (DataSpaceIException error)
//    {
//      //  error.printError();
//        return -1;
//    }
//    // catch failure caused by the DataSpace operations
//    catch (DataTypeIException error)
//    {
//      //  error.printError();
//        return -1;
//    }
//    return 0;  // successfully terminated
//
//
//
//
//
//
////    ///tst
////    int localTotalLenthOfWorkQueue[1];
////    int globalWorkQueueOffset[1];
////    int worQueueStep[1];
////
////
////    localTotalLenthOfWorkQueue[0] =177;
////    int gridDimX = 2;
////    int blockDimX = 32;
////
////    globalWorkQueueOffset[0] = floor((float)(localTotalLenthOfWorkQueue[0] / gridDimX))+1;
////    worQueueStep[0] = std::min(30, globalWorkQueueOffset[0]);
////
////    int debugArr[1000];
////
////    for (int j = 0; j < 1000; j++) {
////        debugArr[j]=0;
////    };
////
////    for (int blockIdxX = 0; blockIdxX < gridDimX; blockIdxX++) {
////
////        for (int bigloop = blockIdxX * globalWorkQueueOffset[0]; bigloop < ((blockIdxX+1) * globalWorkQueueOffset[0])
////            ; bigloop += worQueueStep[0]) {
////
////            for (int threadidX = 0; threadidX < blockDimX; threadidX++){
////
////                for (int i = threadidX ; i < worQueueStep[0]; i += blockDimX) {
////
////                    if (((bigloop + i)< localTotalLenthOfWorkQueue[0]) && ((bigloop + i) <  ((blockIdxX + 1) * globalWorkQueueOffset[0]) )) {
////                        printf("%d    blockIdxX %d  bigloop %d  threadidX %d i %d \n", bigloop + i, blockIdxX, bigloop, threadidX, i);
////
////                        debugArr[bigloop + i] += 1;
////
////                    }
////                };
////        };
////    };
////    };
////
////
////    int oo = 0;
////for (int j = 0; j < 1000; j++) {
////    oo += debugArr[j];
////};
////printf("worQueueStep %d  globalWorkQueueOffset %d  \n", worQueueStep[0], globalWorkQueueOffset[0]);
////
////printf("summ %d  localTotalLenthOfWorkQueue %d  \n", oo , localTotalLenthOfWorkQueue[0]);
////for (int j = 0; j < 178; j++) {
////    printf("%d in %d \n ",j, debugArr[j]);
////
////};
//
//
//
//  testMainPasswes();
//
//
//
//
//
//
//
//
//
//
//
//
//
//
//    //uint32_t numb = 0;
//    //int pos1 = 2;
//    //int pos2 = 8;
//    //int pos3 = 22;
//
//    //std::cout << "pre" << std::endl;
//    //std::cout << (numb & (1 << (pos1))) << std::endl;
//    //std::cout << (numb & (1 << (pos2))) << std::endl;
//    //std::cout << (numb & (1 << (pos3))) << std::endl;
//
//
//    //numb |= 1 << pos1;
//    //numb |= 1 << pos2;
//    //numb |= 1 << pos3;
//
//
//    //std::cout << "post" << std::endl;
//    //std::cout << ((numb & (1 << (pos1))) > 0) << std::endl;
//    //std::cout << ((numb & (1 << (pos2))) > 0) << std::endl;
//    //std::cout << ((numb & (1 << (pos3))) > 0) << std::endl;
//
//    //std::cout << ((numb & (1 << (3))) > 0) << std::endl;
//    //std::cout << ((numb & (1 << (7))) > 0) << std::endl;
//    //std::cout << ((numb & (1 << (30))) > 0) << std::endl;
//
//
//
//
//
//
//
//
//    return 0;
//}
//
