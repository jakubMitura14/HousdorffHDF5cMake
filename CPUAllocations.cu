#include "cuda_runtime.h"
#include <cstdint>
#include "Structs.cu"

/*
* from https://stackoverflow.com/questions/23310520/using-cudamemcpy3d-to-transfer-pointer
*/
#pragma once
template <typename T>
inline T*** alloc_tensor(int Nx, int Ny, int Nz) {
    int i, j;
    T*** tensor;

    tensor = (T***)malloc((size_t)(Nz * sizeof(T**)));
    tensor[0] = (T**)malloc((size_t)(Nz * Ny * sizeof(T*)));
    tensor[0][0] = (T*)malloc((size_t)(Nz * Ny * Nx * sizeof(T)));

    for (j = 1; j < Ny; j++)
        tensor[0][j] = tensor[0][j - 1] + Nx;
    for (i = 1; i < Nz; i++) {
        tensor[i] = tensor[i - 1] + Ny;
        tensor[i][0] = tensor[i - 1][0] + Ny * Nx;
        for (j = 1; j < Ny; j++)
            tensor[i][j] = tensor[i][j - 1] + Nx;
    }

    return tensor;

    /*    int i, j;
    T*** tensor;

    tensor = (T***)malloc((size_t)(Nx * sizeof(T**)));
    tensor[0] = (T**)malloc((size_t)(Nx * Ny * sizeof(T*)));
    tensor[0][0] = (T*)malloc((size_t)(Nx * Ny * Nz * sizeof(T)));

    for (j = 1; j < Ny; j++)
        tensor[0][j] = tensor[0][j - 1] + Nz;
    for (i = 1; i < Nx; i++) {
        tensor[i] = tensor[i - 1] + Ny;
        tensor[i][0] = tensor[i - 1][0] + Ny * Nz;
        for (j = 1; j < Ny; j++)
            tensor[i][j] = tensor[i][j - 1] + Nz;
    }

    return tensor;*/
}



/*
* from https://stackoverflow.com/questions/23310520/using-cudamemcpy3d-to-transfer-pointer
*/
#pragma once
template <typename TC>
inline TC*** alloc_tensorToZeros(int Nx, int Ny, int Nz) {
    int i, j;
    TC*** tensor;



    tensor = (TC***)calloc(Nz, sizeof(TC**));
    tensor[0] = (TC**)calloc(Nz * Ny, sizeof(TC*));
    tensor[0][0] = (TC*)calloc(Nz * Ny * Nx, sizeof(TC));


    //tensor = (TC***)calloc((size_t)(Nz , sizeof(TC**)));
    //tensor[0] = (TC**)calloc((size_t)(Nz * Ny , sizeof(TC*)));
    //tensor[0][0] = (TC*)calloc((size_t)(Nz * Ny * Nx , sizeof(TC)));

    for (j = 1; j < Ny; j++)
        tensor[0][j] = tensor[0][j - 1] + Nx;
    for (i = 1; i < Nz; i++) {
        tensor[i] = tensor[i - 1] + Ny;
        tensor[i][0] = tensor[i - 1][0] + Ny * Nx;
        for (j = 1; j < Ny; j++)
            tensor[i][j] = tensor[i][j - 1] + Nx;
    }

    return tensor;
    //int i, j;
    //TC*** tensor;

    //tensor = (TC***)calloc(Nx, sizeof(TC**));
    //tensor[0] = (TC**)calloc(Nx * Ny, sizeof(TC*));
    //tensor[0][0] = (TC*)calloc(Nx * Ny * Nz, sizeof(TC));

    //for (j = 1; j < Ny; j++)
    //    tensor[0][j] = tensor[0][j - 1] + Nz;
    //for (i = 1; i < Nx; i++) {
    //    tensor[i] = tensor[i - 1] + Ny;
    //    tensor[i][0] = tensor[i - 1][0] + Ny * Nz;
    //    for (j = 1; j < Ny; j++)
    //        tensor[i][j] = tensor[i][j - 1] + Nz;
    //}

    //return tensor;
}



#pragma once
template <typename EEY>
array3dWithDimsCPU<EEY>  get3dArrCPU(EEY*** arrP, int Nx, int Ny, int Nz) {
    array3dWithDimsCPU<EEY> res;
    res.Nx = Nx;
    res.Ny = Ny;
    res.Nz = Nz;
    res.arrP = arrP;

    return res;
}
