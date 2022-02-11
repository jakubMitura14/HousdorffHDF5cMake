//#include "FirstMetaPass.cu"
//#include "Structs.cu"
//
//
//
////testing loopMeta function in order to execute test unhash proper function in loopMeta
//#pragma once
//extern "C" inline void testFirstMetaPass() {
//	// threads and blocks for bool kernel
//	const int blocks = 10;
//	const int xThreadDim = 32;
//	const int yThreadDim = 8;
//	const dim3 threads = dim3(xThreadDim, yThreadDim);
//	// threads and blocks for first metadata pass
//	int threadsFirstMetaDataPass = 32;
//	int blocksFirstMetaDataPass = 10;
//
//	//metadata
//	const int metaXLength = 14;
//	const int MetaYLength = 14;
//	const int MetaZLength = 17;
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
//	const int dXLength = metaXLength;
//	const int dYLength = MetaYLength;
//	const int dZLength = MetaZLength;
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
//	
//	auto fpCounterPointer = alloc_tensorToZeros<unsigned int>(metaXLength, MetaYLength, MetaZLength);
//	auto fnCounterPointer = alloc_tensorToZeros<unsigned int>(metaXLength, MetaYLength, MetaZLength);
//
//	auto fpOffsetPointer = alloc_tensorToZeros<unsigned int>(metaXLength, MetaYLength, MetaZLength);
//	auto fnOffsetPointer = alloc_tensorToZeros<unsigned int>(metaXLength, MetaYLength, MetaZLength);
//
//
//	auto minMaxesPointer = alloc_tensorToZeros<int>(17, 1, 1);
//	
//
//
//
//	auto fpC = get3dArrCPU(fpCPointer, metaXLength, MetaYLength, MetaZLength);
//	auto fnC = get3dArrCPU(fnCPointer, metaXLength, MetaYLength, MetaZLength);
//	auto minMaxes = get3dArrCPU(minMaxesPointer, 9, 1, 1);
//
//	metaData.fpCount = fpC;
//	metaData.fnCount = fnC;
//	metaData.minMaxes = minMaxes;
//
//	metaData.fpCounter = get3dArrCPU(fpCounterPointer, metaXLength, MetaYLength, MetaZLength);;
//	metaData.fnCounter = get3dArrCPU(fnCounterPointer, metaXLength, MetaYLength, MetaZLength);;
//	metaData.fpOffset = get3dArrCPU(fpOffsetPointer, metaXLength, MetaYLength, MetaZLength);;
//	metaData.fnOffset = get3dArrCPU(fnOffsetPointer, metaXLength, MetaYLength, MetaZLength);;
//
//	int workQueueAndRLLength = 200;
//	int workQueueWidth = 4;
//	int resultListWidth = 5;
//	//allocating to semiarbitrrary size 
//	auto workQueuePointer = alloc_tensorToZeros<uint32_t>(workQueueAndRLLength, workQueueWidth, 1);
//	auto resultListPointer = alloc_tensorToZeros<uint32_t>(workQueueAndRLLength, resultListWidth, 1);
//	metaData.workQueue = get3dArrCPU(workQueuePointer, workQueueAndRLLength, workQueueWidth, 1);
//	metaData.resultList = get3dArrCPU(resultListPointer, workQueueAndRLLength, resultListWidth, 1);
//
//
//
//
//	forDebugArr = alloc_tensorToZeros<int>(dXLength, dYLength, dZLength);
//
//
//
//
//
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
//	forFullBoolPrepArgs.threadsFirstMetaDataPass = threadsFirstMetaDataPass;
//	forFullBoolPrepArgs.blocksFirstMetaDataPass = blocksFirstMetaDataPass;
//
//	//populate segm  and gold Arr
//
//
//	auto arrGoldObj = forFullBoolPrepArgs.goldArr;
//	auto arrSegmObj = forFullBoolPrepArgs.segmArr;
//	//setArrCPU(arrGoldObj,2, 3,4,2);
//	setArrCPU(arrGoldObj, dbXLength + 2, dbYLength + 1, dbZLength * 3 + 1, 2);
//	setArrCPU(arrGoldObj, dbXLength + 2, dbYLength + 2, dbZLength * 3 + 1, 2);
//	setArrCPU(arrGoldObj, dbXLength + 2 + 1, dbYLength + 2, dbZLength * 3 + 1, 2);
//	setArrCPU(arrGoldObj, dbXLength * 2 + 2, dbYLength * 2 + 3, dbZLength * 2 + 4, 2);
//	setArrCPU(arrGoldObj, dbXLength * 2 + 3, dbYLength * 2 + 3, dbZLength * 2 + 4, 2);
//	setArrCPU(arrGoldObj, dbXLength * 3 + 2, dbYLength + 2, dbZLength * 2 + 5, 2);
//	setArrCPU(arrGoldObj, dbXLength * 4 + 9, dbYLength * 2, dbZLength * 2 + 1, 2);
//
//	//setArrCPU(arrGoldObj, dbXLength * 2 + 2, dbYLength * 2 + 3, dbZLength  + 4, 2);
//	//setArrCPU(arrSegmObj, dbXLength * 2 + 3, dbYLength * 2 + 3, dbZLength * 4 + 4, 2);
//	//setArrCPU(arrSegmObj, dbXLength * 3 + 2, dbYLength + 2, dbZLength * 5 + 5, 2);
//	//setArrCPU(arrGoldObj, dbXLength * 4 + 9, dbYLength * 2, dbZLength * 6 + 1, 2);
//
//
//
//
//	setArrCPU(arrSegmObj, dbXLength * 6 + 1, dbYLength * 2, dbZLength * 2 + 1, 2);
//	setArrCPU(arrSegmObj, dbXLength * 6 + 5, dbYLength * 2, dbZLength * 2 + 1, 2);
//	setArrCPU(arrSegmObj, dbXLength * 7 + 4, dbYLength * 2, dbZLength * 2 + 1, 2);
//
//
//	//printf("mainXLength %d mainYLength %d mainZLength %d \n", mainXLength, mainYLength, mainZLength);
//	firstMetaAndBoolRun(forFullBoolPrepArgs);
//
//
//
//
//
//	int i, j, k, value = 0;
//	//for (i = 0; i < mainXLength; i++) {
//	//	for (j = 0; j < mainYLength; j++) {
//	//		for (k = 0; k < MetaZLength; k++) {
//	//			//goldArr[k][j][i] = 1;
//	//			if (reducedGold[k][j][i] > 0) {
//	//				for (int tt = 0; tt < 32; tt++) {
//	//					if ((reducedGold[k][j][i] & (1 << (tt)))) {
//	//						printf("found in reduced fp  [%d][%d][%d]\n", i, j, k * 32 + tt);
//
//	//					}
//	//				}
//
//
//	//			}
//	//		}
//	//	}
//	//}
//
//	//for (i = 0; i < mainXLength; i++) {
//	//	for (j = 0; j < mainYLength; j++) {
//	//		for (k = 0; k < MetaZLength; k++) {
//	//			//goldArr[k][j][i] = 1;
//	//			if (forFullBoolPrepArgs.reducedSegm.arrP[k][j][i] > 0) {
//	//				for (int tt = 0; tt < 32; tt++) {
//	//					if ((forFullBoolPrepArgs.reducedSegm.arrP[k][j][i] & (1 << (tt)))) {
//	//						printf("found in reduced fn [%d][%d][%d]\n", i, j, k * 32 + tt);
//
//	//					}
//	//				}
//
//
//	//			}
//	//		}
//	//	}
//	//}
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
//				if (fpC.arrP[k][j][i] > 0) {
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
//				if (fnC.arrP[k][j][i] > 0) {
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
//
//
//
//
//	//i, j, k, value = 0;
//	//for (i = 1; i < 9; i++) {
//	//	for (j = 0; j < 1; j++) {
//	//		for (k = 0; k < 1; k++) {
//	//			//goldArr[k][j][i] = 1;
//
//
//	//			printf("in minMaxes %d  [%d][%d][%d]\n", minMaxes.arrP[k][j][i], i, j, k);
//
//
//
//
//
//	//		}
//	//	}
//
//	//};
//
//
//
//
//
//	for (i = 0; i < mainXLength; i++) {
//		for (j = 0; j < mainYLength; j++) {
//			for (k = 0; k < mainZLength; k++) {
//				//goldArr[k][j][i] = 1;
//				if (goldArr[k][j][i] > 0) {
//					printf("segmArr[%d][%d][%d] = %d\n", i, j, k, goldArr[k][j][i]);
//				}
//			}
//		}
//	}
//
//
//	printf("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! firstMetaPass!!!!!!!!!!!!!!!!!!!!!\n\n");
//
//	i, j, k, value = 0;
//	for (i = 0; i < metaXLength; i++) {
//		for (j = 0; j < MetaYLength; j++) {
//			for (k = 0; k < MetaZLength; k++) {
//				//goldArr[k][j][i] = 1;
//				if (metaData.fpOffset.arrP[k][j][i] > 0) {
//					printf("Offsets Fp %d  [%d][%d][%d]\n", metaData.fpOffset.arrP[k][j][i], i, j, k);
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
//				if (metaData.fnOffset.arrP[k][j][i] > 0) {
//					printf("Offsets Fn %d  [%d][%d][%d]\n", metaData.fnOffset.arrP[k][j][i], i, j, k);
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
//
//	for (i = 0; i < workQueueAndRLLength; i++) {
//	
//				//goldArr[k][j][i] = 1;
//				if (workQueuePointer[0][0][i] > 0) {
//					printf("work queue [%d][%d][%d] = [%d][%d][%d][%d]\n"
//						, 0, 0, i
//						, workQueuePointer[0][0][i]
//						, workQueuePointer[0][1][i]
//						, workQueuePointer[0][2][i]
//						, workQueuePointer[0][3][i]
//						);
//				}
//
//	}
//
//
//
//	//	for (i = 0; i < dXLength; i++) {
//	//	for (j = 0; j < dYLength; j++) {
//	//		for (k = 0; k < dZLength; k++) {
//	//			//goldArr[k][j][i] = 1;
//	//			
//	//				printf("found in forDebugArr %d  [%d][%d][%d]\n", forDebugArr[k][j][i], i, j, k);
//
//
//	//		}
//	//	}
//	//}
//
//
//
//
//
//	free(metaData.minMaxes.arrP);
//	free(metaData.fpCount.arrP);
//	free(metaData.fnCount.arrP);
//	free(metaData.fpCounter.arrP);
//	free(metaData.fnCounter.arrP);
//	free(metaData.fpOffset.arrP);
//	free(metaData.fnOffset.arrP);
//	free(metaData.isActive.arrP);
//	free(metaData.isToBeActivated.arrP);
//	free(workQueuePointer);
//	free(resultListPointer);
//	free(metaData.isFull.arrP);
//
//
//
//
//
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
