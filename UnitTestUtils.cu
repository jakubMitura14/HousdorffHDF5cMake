#include "Structs.cu"
#include "cuda_runtime.h"
#include <iostream>     // std::cout
#pragma once
void printFromReduced(ForBoolKernelArgs<int> fbArgs, uint32_t* arrsCPU) {
	for (uint32_t linIdexMeta = 0; linIdexMeta < fbArgs.metaData.totalMetaLength; linIdexMeta += 1) {
		//we get from linear index  the coordinates of the metadata block of intrest
		uint8_t xMeta = linIdexMeta % fbArgs.metaData.metaXLength;
		uint8_t zMeta = floor((float)(linIdexMeta / (fbArgs.metaData.metaXLength * fbArgs.metaData.MetaYLength)));
		uint8_t yMeta = floor((float)((linIdexMeta - ((zMeta * fbArgs.metaData.metaXLength * fbArgs.metaData.MetaYLength) + xMeta)) / fbArgs.metaData.metaXLength));

		for (int locPos = 0; locPos < 32 * fbArgs.dbYLength; locPos++) {
			auto col = arrsCPU[linIdexMeta * fbArgs.metaData.mainArrSectionLength + locPos];
			if (col > 0) {
				for (uint8_t bitPos = 0; bitPos < 32; bitPos++) {
					if (isBitAtCPU(col, bitPos)) {
						printf("point gold set at x %d y %d z %d  \n"
							, locPos % 32 + xMeta * fbArgs.dbXLength
							, int(floor((float)(locPos / 32)) + yMeta * fbArgs.dbYLength)
							, bitPos + zMeta * fbArgs.dbZLength
						);
					}
				}
			}
		}


		for (int locPos = 32 * fbArgs.dbYLength; locPos < 32 * 2 * fbArgs.dbYLength; locPos++) {
			auto col = arrsCPU[linIdexMeta * fbArgs.metaData.mainArrSectionLength + locPos];
			if (col > 0) {
				for (uint8_t bitPos = 0; bitPos < 32; bitPos++) {
					if (isBitAtCPU(col, bitPos)) {
						int locPosB = locPos - 32 * fbArgs.dbYLength;
						printf("point segm  set at x %d y %d z %d  \n"
							, locPosB % 32 + xMeta * fbArgs.dbXLength
							, int(floor((float)(locPosB / 32)) + yMeta * fbArgs.dbYLength)
							, bitPos + zMeta * fbArgs.dbZLength
						);
					}
				}
			}
		}
	}
}















#pragma once
inline forTestPointStruct getTestPoint(int x, int y, int z,
	bool isGold, int xMeta, int yMeta, int zMeta
	, int dbXLength, int dbYLength, int  dbZLength
	, int& pointsNumberRef
	, bool isGoldAndSegm = false
	, bool shouldBeInResAfterOneDil = false
	, bool shouldBeInResAfterTwoDil = false
) {
	pointsNumberRef += 1;
	forTestPointStruct res;

	res.x = xMeta * dbXLength + x;
	res.y = yMeta * dbYLength + y;
	res.z = zMeta * dbZLength + z;
	res.isGoldAndSegm = isGoldAndSegm;
	res.isGold = isGold;

	res.xMeta = xMeta;
	res.yMeta = yMeta;
	res.zMeta = zMeta;


	res.shouldBeInResAfterOneDil = shouldBeInResAfterOneDil;
	res.shouldBeInResAfterTwoDil = shouldBeInResAfterTwoDil;

	return res;

}




#pragma once
inline forTestMetaDataStruct getMetdataTestStruct(
	int& metasNumberRef,
	int xMeta,
	int yMeta,
	int zMeta,

	int fpCount = 0,
	int fnCount = 0,

	bool isToBeActiveAtStart = true,
	bool isToBeActiveAfterOneIter = true,
	bool isToBeActiveAfterTwoIter = true,

	bool isToBeFullAfterOneIter = false,
	bool isToBeFullAfterTwoIter = false,

	bool isToBeValidatedFpAfterOneIter = false,
	bool isToBeValidatedFpAfterTwoIter = false,

	bool isToBeValidatedFnAfterOneIter = false,
	bool isToBeValidatedFnAfterTwoIter = false,

	int fpConterAfterOneDil = 0,
	int fpConterAfterTwoDil = 0,

	int fnConterAfterOneDil = 0,
	int fnConterAfterTwoDil = 0) {


	forTestMetaDataStruct res;
	metasNumberRef += 1;
	res.xMeta = xMeta;
	res.yMeta = yMeta;
	res.zMeta = zMeta;


	res.isToBeActiveAtStart = (fpCount+ fnCount)>0;
	res.isToBeActiveAfterOneIter = isToBeActiveAfterOneIter;
	res.isToBeActiveAfterTwoIter = isToBeActiveAfterTwoIter;

	res.isToBeFullAfterOneIter = isToBeFullAfterOneIter;
	res.isToBeFullAfterTwoIter = isToBeFullAfterTwoIter;

	res.fpCount = fpCount;
	res.fnCount = fnCount;

	res.requiredspaceInFpResultList = fpCount;
	res.requiredspaceInFnResultList = fnCount;

	res.isToBeValidatedFpAfterOneIter = fpCount > 0;
	res.isToBeValidatedFpAfterTwoIter = fpCount > 0;

	res.isToBeValidatedFnAfterOneIter = fnCount > 0;
	res.isToBeValidatedFnAfterTwoIter = fnCount > 0;


	res.fpConterAfterOneDil = fpConterAfterOneDil;
	res.fpConterAfterTwoDil = fpConterAfterTwoDil;

	res.fnConterAfterOneDil = fnConterAfterOneDil;
	res.fnConterAfterTwoDil = fnConterAfterTwoDil;
	return res;
}