//#include "ForBoolKernel.cu"
//#include "Structs.cu"
//
//
//
////testing loopMeta function in order to execute test unhash proper function in loopMeta
//#pragma once
//extern "C" inline void testDataTransfer() {
//	const int blocks = 1;
//	const int xThreadDim = 32;
//	const int yThreadDim = 11;
//	const dim3 threads = dim3(xThreadDim, yThreadDim);
//
//	//metadata
//	const int metaXLength =10;
//	const int MetaYLength = 10;
//	const int MetaZLength = 10;
//
//
//	const int totalLength = metaXLength * MetaYLength * MetaZLength;
//	const int loopMetaTimes = floor(totalLength / blocks);
//
//	/*   int*** h_tensor;
//	   h_tensor = alloc_tensorToZeros<int>(metaXLength, MetaYLength, MetaZLength);*/
//
//
//	int*** forDebugArr;
//
//	const int dXLength = 4;
//	const int dYLength = 4;
//	const int dZLength = 4;
//
//	//datablock dimensions
//	const int dbXLength = xThreadDim;
//	const int dbYLength = yThreadDim;
//	const int dbZLength = 32;
//
//	const int mainXLength = dbXLength * metaXLength;
//	const int mainYLength = dbYLength * MetaYLength;
//	const int mainZLength = dbZLength * MetaZLength;
//
//
//	//main data arrays
//	int*** goldArr = alloc_tensorToZeros<int>(mainXLength, mainYLength, mainZLength);
//
//	int*** segmArr;
//	segmArr = alloc_tensorToZeros<int>(mainXLength, mainYLength, mainZLength);
//	MetaDataCPU metaData;
//	metaData.metaXLength = metaXLength;
//	metaData.MetaYLength = MetaYLength;
//	metaData.MetaZLength = MetaZLength;
//	metaData.totalMetaLength = totalLength;
//	auto fpCPointer = alloc_tensorToZeros<unsigned int>(metaXLength, MetaYLength, MetaZLength);
//	auto fnCPointer = alloc_tensorToZeros<unsigned int>(metaXLength, MetaYLength, MetaZLength);
//	auto minMaxesPointer = alloc_tensorToZeros<int>(7, 1, 1);
//	auto isActivePointer = alloc_tensorToZeros<bool>(metaXLength, MetaYLength, MetaZLength);
//
//	auto fpC = get3dArrCPU(fpCPointer, metaXLength, MetaYLength, MetaZLength);
//	auto fnC = get3dArrCPU(fnCPointer, metaXLength, MetaYLength, MetaZLength);
//	auto minMaxes = get3dArrCPU(minMaxesPointer, 9, 1, 1);
//	auto isActive = get3dArrCPU(isActivePointer, metaXLength, MetaYLength, MetaZLength);
//
//	metaData.fpCount = fpC;
//	metaData.fnCount = fnC;
//	metaData.minMaxes = minMaxes;
//	forDebugArr = alloc_tensorToZeros<int>(dXLength, dYLength, dZLength);
//
//
//	uint32_t*** reducedGold = alloc_tensorToZeros<uint32_t>(mainXLength, mainYLength, mainZLength);
//	uint32_t*** reducedSegm = alloc_tensorToZeros<uint32_t>(mainXLength, mainYLength, mainZLength);
//
//
//	// arguments to pass
//	ForFullBoolPrepArgs<int> forFullBoolPrepArgs;
//	forFullBoolPrepArgs.metaData = metaData;
//	forFullBoolPrepArgs.numberToLookFor = 2;
//	forFullBoolPrepArgs.forDebugArr = get3dArrCPU(forDebugArr, dXLength, dYLength, dZLength);
//	forFullBoolPrepArgs.dbXLength = dbXLength;
//	forFullBoolPrepArgs.dbYLength = dbYLength;
//	forFullBoolPrepArgs.dbZLength = dbZLength;
//	forFullBoolPrepArgs.goldArr = get3dArrCPU(goldArr, mainXLength, mainYLength, mainZLength);
//	forFullBoolPrepArgs.segmArr = get3dArrCPU(segmArr, mainXLength, mainYLength, mainZLength);
//
//	forFullBoolPrepArgs.reducedGold = get3dArrCPU(reducedGold, mainXLength, mainYLength, MetaZLength);
//	forFullBoolPrepArgs.reducedSegm = get3dArrCPU(reducedSegm, mainXLength, mainYLength, MetaZLength);
//
//	forFullBoolPrepArgs.reducedArrsZdim = mainZLength;
//
//	forFullBoolPrepArgs.threads = threads;
//	forFullBoolPrepArgs.blocks = blocks;
//
//	//populate segm  and gold Arr
//
//
//	auto arrGoldObj = forFullBoolPrepArgs.goldArr;
//	auto arrSegmObj = forFullBoolPrepArgs.segmArr;
//	//setArrCPU(arrGoldObj,2, 3,4,2);
//	setArrCPU(arrGoldObj, dbXLength + 2, dbYLength + 1, dbZLength * 3 + 1, 2);
//	setArrCPU(arrGoldObj, dbXLength + 2, dbYLength + 2, dbZLength * 3 + 1, 2);
//	setArrCPU(arrGoldObj, dbXLength + 2+1, dbYLength + 2, dbZLength * 3 + 1, 2);
//	setArrCPU(arrGoldObj, dbXLength * 2 + 2, dbYLength * 2 + 3, dbZLength * 2 + 4, 2);
//	setArrCPU(arrGoldObj, dbXLength * 2 + 3, dbYLength * 2 + 3, dbZLength * 2 + 4, 2);
//	setArrCPU(arrGoldObj, dbXLength * 3 + 2, dbYLength + 2, dbZLength * 2 + 5, 2);
//	setArrCPU(arrGoldObj, dbXLength * 4 + 9, dbYLength * 2, dbZLength * 2 + 1, 2);
//	
//	
//	setArrCPU(arrSegmObj, dbXLength * 6 + 1, dbYLength * 2, dbZLength * 2 + 1, 2);
//	setArrCPU(arrSegmObj, dbXLength * 6 + 5, dbYLength * 2, dbZLength * 2 + 1, 2);
//	setArrCPU(arrSegmObj, dbXLength * 7 + 4, dbYLength * 2, dbZLength * 2 + 1, 2);
//
//
//	//printf("mainXLength %d mainYLength %d mainZLength %d \n", mainXLength, mainYLength, mainZLength);
//	boolPrepare(forFullBoolPrepArgs);
//
//
//	//int i, j, k, value = 0;
//	//for (i = 0; i < mainXLength; i++) {
//	//	for (j = 0; j < mainYLength; j++) {
//	//		for (k = 0; k < mainZLength; k++) {
//	//			//goldArr[k][j][i] = 1;
//	//			if (goldArr[k][j][i] > 0){
//	//				printf("segmArr[%d][%d][%d] = %d\n", i, j, k, goldArr[k][j][i]);
//	//		}
//	//		}
//	//	}
//	//}
//
//
//
//	int i, j, k, value = 0;
//	for (i = 0; i < mainXLength; i++) {
//		for (j = 0; j < mainYLength; j++) {
//			for (k = 0; k < MetaZLength; k++) {
//				//goldArr[k][j][i] = 1;
//				if (reducedGold[k][j][i] > 0) {
//					for (int tt = 0; tt < 32; tt++) {
//						if ((reducedGold[k][j][i] & (1 << (tt)))) {
//							printf("found in reduced fp  [%d][%d][%d]\n", i, j, k * 32 + tt);
//
//						}
//					}
//
//
//				}
//			}
//		}
//	}
//
//	for (i = 0; i < mainXLength; i++) {
//		for (j = 0; j < mainYLength; j++) {
//			for (k = 0; k < MetaZLength; k++) {
//				//goldArr[k][j][i] = 1;
//				if (forFullBoolPrepArgs.reducedSegm.arrP[k][j][i] > 0) {
//					for (int tt = 0; tt < 32; tt++) {
//						if ((forFullBoolPrepArgs.reducedSegm.arrP[k][j][i] & (1 << (tt)))) {
//							printf("found in reduced fn [%d][%d][%d]\n", i, j, k * 32 + tt);
//
//						}
//					}
//
//
//				}
//			}
//		}
//	}
//
//
//
//	i, j, k, value = 0;
//	for (i = 0; i < metaXLength; i++) {
//		for (j = 0; j < MetaYLength; j++) {
//			for (k = 0; k < MetaZLength; k++) {
//				//goldArr[k][j][i] = 1;
//				if (isActive.arrP[k][j][i]) {
//
//					printf("found as Active [%d][%d][%d]\n", i, j, k);
//
//
//				}
//
//
//			}
//		}
//
//	};
//
//
//
//	i, j, k, value = 0;
//	for (i = 0; i < metaXLength; i++) {
//		for (j = 0; j < MetaYLength; j++) {
//			for (k = 0; k < MetaZLength; k++) {
//				//goldArr[k][j][i] = 1;
//				if (fpC.arrP[k][j][i]>0) {
//					printf("found Fp %d  [%d][%d][%d]\n", fpC.arrP[k][j][i], i, j, k);
//
//				}
//
//
//			}
//		}
//
//	};
//
//
//	for (i = 0; i < metaXLength; i++) {
//		for (j = 0; j < MetaYLength; j++) {
//			for (k = 0; k < MetaZLength; k++) {
//				//goldArr[k][j][i] = 1;
//				if (fnC.arrP[k][j][i]>0) {
//					printf("found Fn %d  [%d][%d][%d]\n", fnC.arrP[k][j][i], i, j, k);
//
//				}
//
//
//			}
//		}
//
//	};
//
//
//
//
//	i, j, k, value = 0;
//	for (i = 1; i < 9; i++) {
//		for (j = 0; j < 1; j++) {
//			for (k = 0; k < 1; k++) {
//				//goldArr[k][j][i] = 1;
//
//
//				printf("in minMaxes %d  [%d][%d][%d]\n", minMaxes.arrP[k][j][i], i, j, k);
//
//
//
//
//
//			}
//		}
//
//	};
//
//
//
//
//
//
//	//int i, j, k, value = 0;
//	//for (i = 0; i < mainXLength; i++) {
//	//	for (j = 0; j < mainYLength; j++) {
//	//		for (k = 0; k < mainZLength; k++) {
//	//			//goldArr[k][j][i] = 1;
//	//			if (goldArr[k][j][i] > 0) {
//	//				printf("segmArr[%d][%d][%d] = %d\n", i, j, k, goldArr[k][j][i]);
//	//			}
//	//		}
//	//	}
//	//}
//
//
//
//
//
//	free(metaData.fpCount.arrP);
//	free(metaData.fnCount.arrP);
//	free(metaData.isActive.arrP);
//	free(forDebugArr);
//	free(goldArr);
//	free(segmArr);
//	free(reducedSegm);
//	free(reducedGold);
//
//
//
//	//std::cout << longInts[3] << std::endl;
//
//}
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
