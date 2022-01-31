#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
//#include <helper_cuda.h>
#include <cmath>
#include "Structs.cu"
#include <cstdint>



#pragma once
template <typename TADD>
inline void copyDeviceToHost3d(array3dWithDimsGPU arrGPU, array3dWithDimsCPU<TADD> arrCPU) {
    cudaMemcpy3DParms cpyB = { 0 };
    cpyB.srcPtr = arrGPU.arrPStr;
    cpyB.dstPtr = make_cudaPitchedPtr(arrCPU.arrP[0][0], arrCPU.Nx * sizeof(TADD), arrCPU.Nx, arrCPU.Ny);
    cpyB.extent = make_cudaExtent(arrCPU.Nx * sizeof(TADD), arrCPU.Ny, arrCPU.Nz);
    cpyB.kind = cudaMemcpyDeviceToHost;
    cudaMemcpy3DAsync(&cpyB);
};

//
//#pragma once
//template <typename TADY>
//inline void copyDeviceToDevice(array3dWithDimsGPU arrGPUSource, array3dWithDimsGPU arrGPUTarget) {
//    cudaMemcpy3DParms cpyB = { 0 };
//    cpyB.srcPtr = arrGPUSource.arrPStr;
//    cpyB.dstPtr = arrGPUTarget.arrPStr;
//    cpyB.extent = make_cudaExtent(arrGPUSource.Nx * sizeof(TADY), arrGPUSource.Ny, arrGPUSource.Nz);
//    cpyB.kind = cudaMemcpyDeviceToDevice;
//    cudaMemcpy3DAsync(&cpyB);
//};

#pragma once
template <typename TAL>
inline void copyHostToDevice(array3dWithDimsGPU arrGPU, array3dWithDimsCPU<TAL> arrCPU) {
    cudaMemcpy3DParms cpy = { 0 };
    cpy.srcPtr = make_cudaPitchedPtr(arrCPU.arrP[0][0], arrCPU.Nx * sizeof(TAL), arrCPU.Nx, arrCPU.Ny);
    cpy.dstPtr = arrGPU.arrPStr;
    cpy.extent = make_cudaExtent(arrCPU.Nx * sizeof(TAL), arrCPU.Ny, arrCPU.Nz);
    cpy.kind = cudaMemcpyHostToDevice;

    cudaMemcpy3DAsync(&cpy);
};


#pragma once
template <typename TAL>
inline array3dWithDimsGPU allocate3dInGPU(array3dWithDimsCPU<TAL> arrCPU) {
    array3dWithDimsGPU res;
    struct cudaPitchedPtr resStrPointer;
    cudaMalloc3D(&resStrPointer, make_cudaExtent(arrCPU.Nx * sizeof(TAL), arrCPU.Ny, arrCPU.Nz));
    //cudaMalloc3D(&resStrPointer, make_cudaExtent(8 * 4, 9, 10));
    res.arrPStr = resStrPointer;
    //!!!!!!!!!!!!!!! intentionally swithing x and z dimensions to make iterations possible ...
    res.Nz = arrCPU.Nx;
    res.Ny = arrCPU.Ny;
    res.Nx = arrCPU.Nz;


    copyHostToDevice(res, arrCPU);
    //cudaMemcpy3DParms cpy = { 0 };
    //cpy.srcPtr = make_cudaPitchedPtr(arrCPU.arrP[0][0], arrCPU.Nx * sizeof(TAL), arrCPU.Nx, arrCPU.Ny);
    //cpy.dstPtr = res.arrPStr;
    //cpy.extent = make_cudaExtent(arrCPU.Nx * sizeof(TAL), arrCPU.Ny, arrCPU.Nz);
    //cpy.kind = cudaMemcpyHostToDevice;

    //cudaMemcpy3DAsync(&cpy);


    //array3dWithDimsGPU res;
    //struct cudaPitchedPtr resStrPointer;
    //cudaMalloc3D(&resStrPointer, make_cudaExtent(arrCPU.Nx * sizeof(TAL), arrCPU.Ny, arrCPU.Nz));
    ////cudaMalloc3D(&resStrPointer, make_cudaExtent(8 * 4, 9, 10));
    //res.arrPStr = resStrPointer;
    //res.Nx = arrCPU.Nx;
    //res.Ny = arrCPU.Ny;
    //res.Nz = arrCPU.Nz;  
    //
    //cudaMemcpy3DParms cpy = { 0 };
    //cpy.srcPtr = make_cudaPitchedPtr(arrCPU.arrP[0][0], arrCPU.Nx * sizeof(TAL), arrCPU.Ny, arrCPU.Nz);
    //cpy.dstPtr = resStrPointer;
    //cpy.extent = make_cudaExtent(arrCPU.Nx * sizeof(TAL), arrCPU.Ny, arrCPU.Nz);
    //cpy.kind = cudaMemcpyHostToDevice;
    //cudaMemcpy3D(&cpy);


    return res;
};



template <typename TALGG>
inline cudaPitchedPtr allocate3dInGPUSimple(TALGG*** cpuArr, int Nx, int Ny, int Nz) {
    struct cudaPitchedPtr res;
    cudaMalloc3D(&res, make_cudaExtent(Nx * sizeof(TALGG), Ny, Nz));
    copyDeviceToHost3dSimple(cpuArr, res, Nx, Ny, Nz);
    return res;
};

template <typename TADHDF>
inline void copyDeviceToHost3dSimple(TADHDF*** hostTensor, cudaPitchedPtr deviceTarget, int Nx, int Ny, int Nz) {
    cudaMemcpy3DParms cpy = { 0 };
    cpy.srcPtr = make_cudaPitchedPtr(hostTensor[0][0], Nx * sizeof(TADHDF), Nx, Ny);
    cpy.dstPtr = deviceTarget;
    cpy.extent = make_cudaExtent(Nx * sizeof(TADHDF), Ny, Nz);
    cpy.kind = cudaMemcpyHostToDevice;

    cudaMemcpy3DAsync(&cpy);
};


#pragma once
template <typename ZZ>
inline void setArrCPU(array3dWithDimsCPU<ZZ> arrCPU, int x, int y, int z, ZZ value, bool toPrint = true) {
    if (toPrint) {
      //  printf(" set imn meta gold %d  %d  %d \n", x, y, z);
    }
    arrCPU.arrP[z][y][x] = value;
};




