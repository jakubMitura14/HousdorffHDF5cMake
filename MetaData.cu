#include "cuda_runtime.h"
#include <cstdint>
#include "Structs.cu"
#include "MemoryTransfers.cu"

/*
copy from host to device
*/
#pragma once
inline MetaDataGPU allocateMetaDataOnGPU(unsigned int Nx, unsigned int Ny, unsigned int Nz) {
	MetaDataGPU res;
	//!! x and z intentionally mixed !!
	res.fpCount = getArrGpu<unsigned int>(Nx, Ny, Nz);
	res.fnCount = getArrGpu<unsigned int>(Nx, Ny, Nz);
	//res.minMaxes = allocate3dInGPU(metaDataCPU.minMaxes);

	res.fpCount = getArrGpu<unsigned int>(Nx, Ny, Nz);
	res.fnCount = getArrGpu<unsigned int>(Nx, Ny, Nz);
	res.fpCounter = getArrGpu<unsigned int>(Nx, Ny, Nz);
	res.fnCounter = getArrGpu<unsigned int>(Nx, Ny, Nz);
	res.fpOffset = getArrGpu<unsigned int>(Nx, Ny, Nz);
	res.fnOffset = getArrGpu<unsigned int>(Nx, Ny, Nz);

	res.isActiveGold = getArrGpu<bool>(Nx, Ny, Nz);
	res.isFullGold = getArrGpu<bool>(Nx, Ny, Nz);
	res.isActiveSegm = getArrGpu<bool>(Nx, Ny, Nz);
	res.isFullSegm = getArrGpu<bool>(Nx, Ny, Nz);

	res.isToBeActivatedGold = getArrGpu<bool>(Nx, Ny, Nz);
	res.isToBeActivatedSegm = getArrGpu<bool>(Nx, Ny, Nz);


	uint16_t* workQueue;
	size_t size = (Nx * Ny * Nz) * 4 + 5;
	cudaMallocAsync(&workQueue, size, 0);
	//res.workQueue = workQueue;

	//res.isToBeValidatedFp = allocate3dInGPU(metaDataCPU.isToBeValidatedFp);
	//res.isToBeValidatedFn = allocate3dInGPU(metaDataCPU.isToBeValidatedFn);

	//res.workQueue = allocate3dInGPU(metaDataCPU.workQueue);
	//res.resultList = allocate3dInGPU(metaDataCPU.resultList);

	res.metaXLength = res.fpCount.Nx;
	res.MetaYLength = res.fpCount.Ny;
	res.MetaZLength = res.fpCount.Nz;

	res.totalMetaLength = (Nx * Ny * Nz);
	//allocating on GPU and copying  cpu data onto GPU

	return res;

}

/*
copy from device to host
*/
#pragma once
inline void copyMetaDataToCPU(MetaDataCPU metaDataCPU, MetaDataGPU metaDataGPU) {
	copyDeviceToHost3d(metaDataGPU.fpCount, metaDataCPU.fpCount);
	copyDeviceToHost3d(metaDataGPU.fnCount, metaDataCPU.fnCount);

	copyDeviceToHost3d(metaDataGPU.minMaxes, metaDataCPU.minMaxes);

	copyDeviceToHost3d(metaDataGPU.fpCounter, metaDataCPU.fpCounter);
	copyDeviceToHost3d(metaDataGPU.fnCounter, metaDataCPU.fnCounter);
	copyDeviceToHost3d(metaDataGPU.fpOffset, metaDataCPU.fpOffset);
	copyDeviceToHost3d(metaDataGPU.fnOffset, metaDataCPU.fnOffset);

	copyDeviceToHost3d(metaDataGPU.isActiveGold, metaDataCPU.isActiveGold);
	copyDeviceToHost3d(metaDataGPU.isFullGold, metaDataCPU.isFullGold);
	copyDeviceToHost3d(metaDataGPU.isActiveSegm, metaDataCPU.isActiveSegm);
	copyDeviceToHost3d(metaDataGPU.isFullSegm, metaDataCPU.isFullSegm);

	copyDeviceToHost3d(metaDataGPU.isToBeActivatedGold, metaDataCPU.isToBeActivatedGold);
	copyDeviceToHost3d(metaDataGPU.isToBeActivatedSegm, metaDataCPU.isToBeActivatedSegm);

	copyDeviceToHost3d(metaDataGPU.workQueue, metaDataCPU.workQueue);
	//copyDeviceToHost3d(metaDataGPU.resultList, metaDataCPU.resultList);

	copyDeviceToHost3d(metaDataGPU.isToBeValidatedFp, metaDataCPU.isToBeValidatedFp);
	copyDeviceToHost3d(metaDataGPU.isToBeValidatedFn, metaDataCPU.isToBeValidatedFn);




}


/*
free metadata
*/
#pragma once
inline void freeMetaDataGPU(MetaDataGPU metaDataGPU) {
	cudaFree(metaDataGPU.fpCount.arrPStr.ptr);
	cudaFree(metaDataGPU.fnCount.arrPStr.ptr);
	cudaFree(metaDataGPU.minMaxes.arrPStr.ptr);

	cudaFree(metaDataGPU.fpCounter.arrPStr.ptr);
	cudaFree(metaDataGPU.fnCounter.arrPStr.ptr);
	cudaFree(metaDataGPU.fpOffset.arrPStr.ptr);
	cudaFree(metaDataGPU.fnOffset.arrPStr.ptr);

	cudaFree(metaDataGPU.isActiveGold.arrPStr.ptr);
	cudaFree(metaDataGPU.isFullGold.arrPStr.ptr);
	cudaFree(metaDataGPU.isActiveSegm.arrPStr.ptr);
	cudaFree(metaDataGPU.isFullSegm.arrPStr.ptr);

	cudaFree(metaDataGPU.isToBeActivatedGold.arrPStr.ptr);
	cudaFree(metaDataGPU.isToBeActivatedSegm.arrPStr.ptr);

	cudaFree(metaDataGPU.workQueue.arrPStr.ptr);
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD


=======
=======
>>>>>>> parent of ebdf6ce (up not working min maxes for some reason)
=======
>>>>>>> parent of ebdf6ce (up not working min maxes for some reason)
	
	
>>>>>>> parent of ebdf6ce (up not working min maxes for some reason)
	//cudaFree(metaDataGPU.resultList);

	//cudaFreeAsync(metaDataGPU.resultList,0);
	//cudaFree(metaDataGPU.resultList.arrPStr.ptr);

	cudaFree(metaDataGPU.isToBeValidatedFp.arrPStr.ptr);
	cudaFree(metaDataGPU.isToBeValidatedFn.arrPStr.ptr);




}


///*
//free metadata
//*/
//inline void freeMetaDataCPU(MetaDataCPU metaDataCPU) {
//	free(metaDataCPU.fpCount);
//	free(metaDataCPU.fnCount);
//}