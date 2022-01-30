#include "cuda_runtime.h"
#include <cstdint>
#include "Structs.cu"
#include "MemoryTransfers.cu"

/*
copy from host to device
*/
#pragma once
inline MetaDataGPU allocateMetaDataOnGPU(MetaDataCPU metaDataCPU) {
	MetaDataGPU res;
	setArrCPU<unsigned int>(metaDataCPU.minMaxes, 1, 0, 0, 0, false);
	setArrCPU<unsigned int>(metaDataCPU.minMaxes, 2, 0, 0, 1000, false);
	setArrCPU<unsigned int>(metaDataCPU.minMaxes, 3, 0, 0, 0, false);
	setArrCPU<unsigned int>(metaDataCPU.minMaxes, 4, 0, 0, 1000, false);
	setArrCPU<unsigned int>(metaDataCPU.minMaxes, 5, 0, 0, 0, false);
	setArrCPU<unsigned int>(metaDataCPU.minMaxes, 6, 0, 0, 1000, false);
	setArrCPU<unsigned int>(metaDataCPU.minMaxes, 7, 0, 0, 0, false);
	setArrCPU<unsigned int>(metaDataCPU.minMaxes, 8, 0, 0, 0, false);
	setArrCPU<unsigned int>(metaDataCPU.minMaxes, 9, 0, 0, 0, false);
	setArrCPU<unsigned int>(metaDataCPU.minMaxes, 10, 0, 0, 0, false);
	setArrCPU<unsigned int>(metaDataCPU.minMaxes, 11, 0, 0, 0, false);
	setArrCPU<unsigned int>(metaDataCPU.minMaxes, 12, 0, 0, 1, false);
	setArrCPU<unsigned int>(metaDataCPU.minMaxes, 13, 0, 0, 0, false);
	setArrCPU<unsigned int>(metaDataCPU.minMaxes, 14, 0, 0, 0, false);
	setArrCPU<unsigned int>(metaDataCPU.minMaxes, 15, 0, 0, 0, false);




	//!! x and z intentionally mixed !!
	res.fpCount = allocate3dInGPU(metaDataCPU.fpCount);
	res.fnCount = allocate3dInGPU(metaDataCPU.fnCount);
	res.minMaxes = allocate3dInGPU(metaDataCPU.minMaxes);

	res.fpCount = allocate3dInGPU(metaDataCPU.fpCount);
	res.fnCount = allocate3dInGPU(metaDataCPU.fnCount);
	res.fpCounter = allocate3dInGPU(metaDataCPU.fpCounter);
	res.fnCounter = allocate3dInGPU(metaDataCPU.fnCounter);
	res.fpOffset = allocate3dInGPU(metaDataCPU.fpOffset);
	res.fnOffset = allocate3dInGPU(metaDataCPU.fnOffset);

	res.isActiveGold = allocate3dInGPU(metaDataCPU.isActiveGold);
	res.isFullGold = allocate3dInGPU(metaDataCPU.isFullGold);
	res.isActiveSegm = allocate3dInGPU(metaDataCPU.isActiveSegm);
	res.isFullSegm = allocate3dInGPU(metaDataCPU.isFullSegm);

	res.isToBeActivatedGold = allocate3dInGPU(metaDataCPU.isToBeActivatedGold);
	res.isToBeActivatedSegm = allocate3dInGPU(metaDataCPU.isToBeActivatedSegm);



	res.isToBeValidatedFp = allocate3dInGPU(metaDataCPU.isToBeValidatedFp);
	res.isToBeValidatedFn = allocate3dInGPU(metaDataCPU.isToBeValidatedFn);

	res.workQueue = allocate3dInGPU(metaDataCPU.workQueue);
	res.resultList = allocate3dInGPU(metaDataCPU.resultList);

	res.metaXLength = res.fpCount.Nx;
	res.MetaYLength = res.fpCount.Ny;
	res.MetaZLength = res.fpCount.Nz;

	res.totalMetaLength = metaDataCPU.totalMetaLength;
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
	copyDeviceToHost3d(metaDataGPU.resultList, metaDataCPU.resultList);

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

	cudaFree(metaDataGPU.resultList.arrPStr.ptr);

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