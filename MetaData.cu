#include "cuda_runtime.h"
#include <cstdint>
#include "Structs.cu"

/*
copy from host to device
*/
#pragma once
inline MetaDataGPU allocateMetaDataOnGPU(MetaDataCPU metaDataCPU, unsigned int*& minMaxes) {
	MetaDataGPU res;

	metaDataCPU.minMaxes[1] = 0;
	metaDataCPU.minMaxes[2] = 1000;
	metaDataCPU.minMaxes[3] = 0;
	metaDataCPU.minMaxes[4] = 1000;
	metaDataCPU.minMaxes[5] = 0;
	metaDataCPU.minMaxes[6] = 1000;
	metaDataCPU.minMaxes[7] = 0;
	metaDataCPU.minMaxes[8] = 0;
	metaDataCPU.minMaxes[9] = 0;
	metaDataCPU.minMaxes[10] = 0;
	metaDataCPU.minMaxes[11] = 0;
	metaDataCPU.minMaxes[12] = 0;
	metaDataCPU.minMaxes[13] = 0;
	metaDataCPU.minMaxes[14] = 0;
	metaDataCPU.minMaxes[15] = 0;
	metaDataCPU.minMaxes[16] = 0;
	metaDataCPU.minMaxes[17] = 0;
	metaDataCPU.minMaxes[18] = 0;
	metaDataCPU.minMaxes[19] = 0;
	metaDataCPU.minMaxes[20] = 0;

	size_t size = sizeof(unsigned int) * 20;
	cudaMemcpy(minMaxes, metaDataCPU.minMaxes, size, cudaMemcpyHostToDevice);

	//res.resultList = allocate3dInGPU(metaDataCPU.resultList);

	//res.metaXLength = metaDataCPU.metaXLength;
	//res.MetaYLength = metaDataCPU.MetaYLength;
	//res.MetaZLength = metaDataCPU.MetaZLength;

	//res.totalMetaLength = metaDataCPU.totalMetaLength;
	//allocating on GPU and copying  cpu data onto GPU

	return res;

}

/*
copy from device to host
*/
#pragma once
inline void copyMetaDataToCPU(MetaDataCPU metaDataCPU, MetaDataGPU metaDataGPU) {
	//copyDeviceToHost3d(metaDataGPU.fpCount, metaDataCPU.fpCount);
	//copyDeviceToHost3d(metaDataGPU.fnCount, metaDataCPU.fnCount);
	size_t size = sizeof(unsigned int) * 20;

	cudaMemcpy(metaDataCPU.minMaxes, metaDataGPU.minMaxes, size, cudaMemcpyDeviceToHost);







}


///*
//free metadata
//*/
//#pragma once
//inline void freeMetaDataGPU(MetaDataGPU metaDataGPU) {
//	cudaFree(metaDataGPU.fpCount.arrPStr.ptr);
//	cudaFree(metaDataGPU.fnCount.arrPStr.ptr);
//	cudaFree(metaDataGPU.minMaxes);
//
//	cudaFree(metaDataGPU.fpCounter.arrPStr.ptr);
//	cudaFree(metaDataGPU.fnCounter.arrPStr.ptr);
//	cudaFree(metaDataGPU.fpOffset.arrPStr.ptr);
//	cudaFree(metaDataGPU.fnOffset.arrPStr.ptr);
//
//	cudaFree(metaDataGPU.isActiveGold.arrPStr.ptr);
//	cudaFree(metaDataGPU.isFullGold.arrPStr.ptr);
//	cudaFree(metaDataGPU.isActiveSegm.arrPStr.ptr);
//	cudaFree(metaDataGPU.isFullSegm.arrPStr.ptr);
//
//	cudaFree(metaDataGPU.isToBeActivatedGold.arrPStr.ptr);
//	cudaFree(metaDataGPU.isToBeActivatedSegm.arrPStr.ptr);
//
//	cudaFree(metaDataGPU.workQueue.arrPStr.ptr);
//
//
//	//cudaFree(metaDataGPU.resultList);
//
//	//cudaFreeAsync(metaDataGPU.resultList,0);
//	//cudaFree(metaDataGPU.resultList.arrPStr.ptr);
//
//	cudaFree(metaDataGPU.isToBeValidatedFp.arrPStr.ptr);
//	cudaFree(metaDataGPU.isToBeValidatedFn.arrPStr.ptr);
//
//
//
//
//}


///*
//free metadata
//*/
//inline void freeMetaDataCPU(MetaDataCPU metaDataCPU) {
//	free(metaDataCPU.fpCount);
//	free(metaDataCPU.fnCount);
//}