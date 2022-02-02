#include "MainPassesKernels.cu"
//#include "Structs.cu"
#include "UnitTestUtils.cu"







//testing loopMeta function in order to execute test unhash proper function in loopMeta
#pragma once
extern "C" inline void testMainPasswes() {
	// threads and blocks for bool kernel
	const int blocks = 17;
	const int xThreadDim = 32;
	const int yThreadDim = 12;
	const dim3 threads = dim3(xThreadDim, yThreadDim);
	// threads and blocks for first metadata pass
	int threadsFirstMetaDataPass = 32;
	int blocksFirstMetaDataPass = 10;



	//datablock dimensions
	const int dbXLength = xThreadDim;
	const int dbYLength = yThreadDim;
	const int dbZLength = 32;



	//threads and blocks for main pass 
	dim3 threadsMainPass = dim3(dbXLength, dbYLength);
	int blocksMainPass = 7;
	//threads and blocks for padding pass 
	dim3 threadsPaddingPass = dim3(32, 11);
	int blocksPaddingPass = 13;
	//threads and blocks for non first metadata passes 
	int threadsOtherMetaDataPasses = 32;
	int blocksOtherMetaDataPasses = 7;


	int minMaxesLength = 17;



	//metadata
	const int metaXLength = 6;
	const int MetaYLength = 6;
	const int MetaZLength = 10;


	const int totalLength = metaXLength * MetaYLength * MetaZLength;
	const int loopMetaTimes = floor(totalLength / blocks);

	/*   int*** h_tensor;
	   h_tensor = alloc_tensorToZeros<int>(metaXLength, MetaYLength, MetaZLength);*/

	int i, j, k, value = 0;
	int*** forDebugArr;

	const int dXLength = 8;
	const int dYLength = 1;
	const int dZLength = 1;


	const int mainXLength = dbXLength * metaXLength;
	const int mainYLength = dbYLength * MetaYLength;
	const int mainZLength = dbZLength * MetaZLength;


	//main data arrays
	int*** goldArr = alloc_tensorToZeros<int>(mainXLength, mainYLength, mainZLength);

	int*** segmArr;
	segmArr = alloc_tensorToZeros<int>(mainXLength, mainYLength, mainZLength);
	MetaDataCPU metaData;
	metaData.metaXLength = metaXLength;
	metaData.MetaYLength = MetaYLength;
	metaData.MetaZLength = MetaZLength;
	metaData.totalMetaLength = totalLength;
	auto fpCPointer = alloc_tensorToZeros<unsigned int>(metaXLength, MetaYLength, MetaZLength);
	auto fnCPointer = alloc_tensorToZeros<unsigned int>(metaXLength, MetaYLength, MetaZLength);

	auto fpCounterPointer = alloc_tensorToZeros<unsigned int>(metaXLength, MetaYLength, MetaZLength);
	auto fnCounterPointer = alloc_tensorToZeros<unsigned int>(metaXLength, MetaYLength, MetaZLength);

	auto fpOffsetPointer = alloc_tensorToZeros<unsigned int>(metaXLength, MetaYLength, MetaZLength);
	auto fnOffsetPointer = alloc_tensorToZeros<unsigned int>(metaXLength, MetaYLength, MetaZLength);


	auto minMaxesPointer = alloc_tensorToZeros<unsigned int>(minMaxesLength, 1, 1);

	auto isActiveGoldPointer = alloc_tensorToZeros<bool>(metaXLength, MetaYLength, MetaZLength);
	auto isFullGoldPointer = alloc_tensorToZeros<bool>(metaXLength, MetaYLength, MetaZLength);
	auto isActiveSegmPointer = alloc_tensorToZeros<bool>(metaXLength, MetaYLength, MetaZLength);
	auto isFullSegmPointer = alloc_tensorToZeros<bool>(metaXLength, MetaYLength, MetaZLength);

	auto isToBeActivatedGoldPointer = alloc_tensorToZeros<bool>(metaXLength, MetaYLength, MetaZLength);
	auto isToBeActivatedSegmPointer = alloc_tensorToZeros<bool>(metaXLength, MetaYLength, MetaZLength);



	auto isToBeValidatedFpPointer = alloc_tensorToZeros<bool>(metaXLength, MetaYLength, MetaZLength);
	auto isToBeValidatedFnPointer = alloc_tensorToZeros<bool>(metaXLength, MetaYLength, MetaZLength);



	auto fpC = get3dArrCPU(fpCPointer, metaXLength, MetaYLength, MetaZLength);
	auto fnC = get3dArrCPU(fnCPointer, metaXLength, MetaYLength, MetaZLength);
	auto minMaxes = get3dArrCPU(minMaxesPointer, minMaxesLength, 1, 1);

	auto isToBeValidatedFp = get3dArrCPU(isToBeValidatedFpPointer, metaXLength, MetaYLength, MetaZLength);
	auto isToBeValidatedFn = get3dArrCPU(isToBeValidatedFnPointer, metaXLength, MetaYLength, MetaZLength);


	size_t size = sizeof(unsigned int) * 20;
	unsigned int* minMaxesCPU = (unsigned int*)malloc(size);
	metaData.minMaxes = minMaxesCPU;

	//metaData.fpCount = fpC;
	//metaData.fnCount = fnC;
	////metaData.minMaxes = minMaxes;

	//metaData.fpCounter = get3dArrCPU(fpCounterPointer, metaXLength, MetaYLength, MetaZLength);;
	//metaData.fnCounter = get3dArrCPU(fnCounterPointer, metaXLength, MetaYLength, MetaZLength);;
	//metaData.fpOffset = get3dArrCPU(fpOffsetPointer, metaXLength, MetaYLength, MetaZLength);;
	//metaData.fnOffset = get3dArrCPU(fnOffsetPointer, metaXLength, MetaYLength, MetaZLength);;

	//metaData.isActiveGold = get3dArrCPU(isActiveGoldPointer, metaXLength, MetaYLength, MetaZLength);;
	//metaData.isFullGold = get3dArrCPU(isFullGoldPointer, metaXLength, MetaYLength, MetaZLength);;
	//metaData.isActiveSegm = get3dArrCPU(isActiveSegmPointer, metaXLength, MetaYLength, MetaZLength);;
	//metaData.isFullSegm = get3dArrCPU(isFullSegmPointer, metaXLength, MetaYLength, MetaZLength);;

	//metaData.isToBeActivatedGold = get3dArrCPU(isToBeActivatedGoldPointer, metaXLength, MetaYLength, MetaZLength);;
	//metaData.isToBeActivatedSegm = get3dArrCPU(isToBeActivatedSegmPointer, metaXLength, MetaYLength, MetaZLength);;


	//metaData.isToBeValidatedFp = isToBeValidatedFp;
	//metaData.isToBeValidatedFn = isToBeValidatedFn;


	//int paddingStoreX = metaXLength * 32;
	//int paddingStoreY = MetaYLength * 32;
	//int paddingStoreZ = MetaZLength;

	//auto paddingsStoreGoldPointer = alloc_tensorToZeros<uint8_t>(paddingStoreX, paddingStoreY, paddingStoreZ);
	//auto paddingsStoreSegmPointer = alloc_tensorToZeros<uint8_t>(paddingStoreX, paddingStoreY, paddingStoreZ);

	int workQueueAndRLLength = 200;
	int workQueueWidth = 4;
	int resultListWidth = 5;
	//allocating to semiarbitrrary size 
	auto workQueuePointer = alloc_tensorToZeros<uint16_t>(workQueueAndRLLength, workQueueWidth, 1);
	//auto resultListPointer = alloc_tensorToZeros<int>(workQueueAndRLLength, resultListWidth, 1);
	metaData.workQueue = get3dArrCPU(workQueuePointer, workQueueAndRLLength, workQueueWidth, 1);
	//metaData.resultList = get3dArrCPU(resultListPointer, workQueueAndRLLength, resultListWidth, 1);


	forDebugArr = alloc_tensorToZeros<int>(dXLength, dYLength, dZLength);

	uint32_t*** reducedGold = alloc_tensorToZeros<uint32_t>(mainXLength, mainYLength, mainZLength);
	uint32_t*** reducedSegm = alloc_tensorToZeros<uint32_t>(mainXLength, mainYLength, mainZLength);

	uint32_t*** reducedGoldRef = alloc_tensorToZeros<uint32_t>(mainXLength, mainYLength, mainZLength);
	uint32_t*** reducedSegmRef = alloc_tensorToZeros<uint32_t>(mainXLength, mainYLength, mainZLength);

	uint32_t*** reducedGoldPrevPointer = alloc_tensorToZeros<uint32_t>(mainXLength, mainYLength, mainZLength);
	uint32_t*** reducedSegmPrevPointer = alloc_tensorToZeros<uint32_t>(mainXLength, mainYLength, mainZLength);

	// arguments to pass
	ForFullBoolPrepArgs<int> forFullBoolPrepArgs;
	forFullBoolPrepArgs.metaData = metaData;
	forFullBoolPrepArgs.numberToLookFor = 2;
	forFullBoolPrepArgs.forDebugArr = get3dArrCPU(forDebugArr, dXLength, dYLength, dZLength);
	forFullBoolPrepArgs.dbXLength = dbXLength;
	forFullBoolPrepArgs.dbYLength = dbYLength;
	forFullBoolPrepArgs.dbZLength = dbZLength;
	forFullBoolPrepArgs.goldArr = get3dArrCPU(goldArr, mainXLength, mainYLength, mainZLength);
	forFullBoolPrepArgs.segmArr = get3dArrCPU(segmArr, mainXLength, mainYLength, mainZLength);

	forFullBoolPrepArgs.reducedGold = get3dArrCPU(reducedGold, mainXLength, mainYLength, MetaZLength);
	forFullBoolPrepArgs.reducedSegm = get3dArrCPU(reducedSegm, mainXLength, mainYLength, MetaZLength);

	forFullBoolPrepArgs.reducedGoldRef = get3dArrCPU(reducedGoldRef, mainXLength, mainYLength, MetaZLength);
	forFullBoolPrepArgs.reducedSegmRef = get3dArrCPU(reducedSegmRef, mainXLength, mainYLength, MetaZLength);

	forFullBoolPrepArgs.reducedGoldPrev = get3dArrCPU(reducedGoldPrevPointer, mainXLength, mainYLength, MetaZLength);
	forFullBoolPrepArgs.reducedSegmPrev = get3dArrCPU(reducedSegmPrevPointer, mainXLength, mainYLength, MetaZLength);


	forFullBoolPrepArgs.threads = threads;
	forFullBoolPrepArgs.blocks = blocks;

	forFullBoolPrepArgs.threadsFirstMetaDataPass = threadsFirstMetaDataPass;
	forFullBoolPrepArgs.blocksFirstMetaDataPass = blocksFirstMetaDataPass;

	forFullBoolPrepArgs.threadsMainPass = threadsMainPass;
	forFullBoolPrepArgs.blocksMainPass = blocksMainPass;

	forFullBoolPrepArgs.threadsPaddingPass = threadsPaddingPass;
	forFullBoolPrepArgs.blocksPaddingPass = blocksPaddingPass;

	forFullBoolPrepArgs.threadsOtherMetaDataPasses = threadsOtherMetaDataPasses;
	forFullBoolPrepArgs.blocksOtherMetaDataPasses = blocksOtherMetaDataPasses;

	//populate segm  and gold Arr


	auto arrGoldObj = forFullBoolPrepArgs.goldArr;
	auto arrSegmObj = forFullBoolPrepArgs.segmArr;

	// 2 planes with distance 7 relative to each other
	//for (int x = 33; x < 71; x++) {
	//	for (int y = 1; y < 40; y++) {

	//		setArrCPU(arrGoldObj, x, y,70 ,2);

	//		setArrCPU(arrSegmObj,x, y,100 , 2);

	//	}
	//
	//}

	setArrCPU(arrGoldObj, 33, 1, 71, 2);
	setArrCPU(arrGoldObj, 75, 20, 70, 2);

	setArrCPU(arrGoldObj, 33, 1, 110, 2);
	setArrCPU(arrGoldObj, 75, 20, 161, 2);

	/*
	maxX 2  [1]
minX 1  [2]
maxY 1  [3]
minY 0  [4]
maxZ 5  [5]
minZ 2  [6]
	*/


	printf("\n aaa \n");
	mainKernelsRun(forFullBoolPrepArgs);



	i = 1;
	printf("maxX %d  [%d]\n", minMaxesCPU[i], i);
	i = 2;
	printf("minX %d  [%d]\n", minMaxesCPU[i], i);
	i = 3;
	printf("maxY %d  [%d]\n", minMaxesCPU[i], i);
	i = 4;
	printf("minY %d  [%d]\n", minMaxesCPU[i], i);
	i = 5;
	printf("maxZ %d  [%d]\n", minMaxesCPU[i], i);
	i = 6;
	printf("minZ %d  [%d]\n", minMaxesCPU[i], i);

	int ii = 7;
	printf("global FP count %d  [%d]\n", minMaxesCPU[ii], ii);
	ii = 8;
	printf("global FN count %d  [%d]\n", minMaxesCPU[ii], ii);
	ii = 9;
	printf("workQueueCounter %d  [%d]\n", minMaxesCPU[ii], ii);
	ii = 10;
	printf("resultFP globalCounter %d  [%d]\n", minMaxesCPU[ii], ii);
	ii = 11;
	printf("resultFn globalCounter %d  [%d]\n", minMaxesCPU[ii], ii);
	ii = 12;
	printf("global offset counter %d  [%d]\n", minMaxesCPU[ii], ii);

	ii = 13;
	printf("globalIterationNumb %d  [%d]\n", minMaxesCPU[ii], ii);
	ii = 17;
	printf("suum debug %d  [%d]\n", minMaxesCPU[ii], ii);





	i, j, k, value = 0;
	i = 31;
	j = 12;
	for (k = 0; k < MetaZLength; k++) {
		//goldArr[k][j][i] = 1;
		if (reducedGold[k][j][i] > 0) {
			for (int tt = 0; tt < 32; tt++) {
				if ((reducedGold[k][j][i] & (1 << (tt)))) {
					printf("found in reduced fp  [%d]\n", k * 32 + tt);

				}
			}

		}
	}


	//		i, j, k, value = 0;
	//for (i = 0; i < mainXLength; i++) {
	//	for (j = 0; j < mainYLength; j++) {
	//		for (k = 0; k < MetaZLength; k++) {
	//			//goldArr[k][j][i] = 1;
	//			if (reducedGold[k][j][i] > 0) {
	//				for (int tt = 0; tt < 32; tt++) {
	//					if ((reducedGold[k][j][i] & (1 << (tt)))) {
	//						printf("found in reduced fp  [%d][%d][%d]\n", i, j, k * 32 + tt);

	//					}
	//				}

	//			}
	//		}
	//	}
	//}






	//minMaxes.arrP[0][0][10] + minMaxes.arrP[0][0][11]

	//int sumDebug = 0;
	//for (int ji = 0; ji < 8000; ji++) {
	//	if (forDebugArr[0][0][ji]==1) {
	//		sumDebug += forDebugArr[0][0][ji];
	//		//printf("for debug %d i %d \n", forDebugArr[0][0][ji],ji);
	//	}
	//}
	//printf("\n sumDebug %d \n", sumDebug);


//
//
//	//	for (int ji = 0; ji < minMaxes.arrP[0][0][10] + minMaxes.arrP[0][0][11]; ji++) {
//		for (int ji = 0; ji < 10; ji++) {
//    if (forFullBoolPrepArgs.metaData.resultList.arrP[0][2][ji] + forFullBoolPrepArgs.metaData.resultList.arrP[0][1][ji]  > 0) {
//   	 int x = forFullBoolPrepArgs.metaData.resultList.arrP[0][0][ji];
//	 int y = forFullBoolPrepArgs.metaData.resultList.arrP[0][1][ji];
//	 int z = forFullBoolPrepArgs.metaData.resultList.arrP[0][2][ji];
//	 int isGold = forFullBoolPrepArgs.metaData.resultList.arrP[0][3][ji];
//	 int iternumb = forFullBoolPrepArgs.metaData.resultList.arrP[0][4][ji];
//
//	 //uint16_t x = forFullBoolPrepArgs.metaData.resultList.arrP[ji][0][0];
//	 //uint16_t y = forFullBoolPrepArgs.metaData.resultList.arrP[ji][1][0];
//	 //uint16_t z = forFullBoolPrepArgs.metaData.resultList.arrP[ji][2][0];
//	 //uint16_t isGold = forFullBoolPrepArgs.metaData.resultList.arrP[ji][3][0];
//	 //uint16_t iternumb = forFullBoolPrepArgs.metaData.resultList.arrP[ji][4][0];
//
//
//   	 if (iternumb!=9) {
//   		 printf("result  in point  %d %d %d isGold %d iteration %d \n "
//   			 , x
//   			 , y
//   			 , z
//   			 , isGold
//   			 , iternumb);
//   	 }
//   	 else {
//   		 printf("**");
//   	 }
//
//    }
//}





	 //for (int i = 0; i < workQueueAndRLLength; i++) {

		// if (workQueuePointer[0][2][i] > 0) {
		//	 printf("work queue [%d][%d][%d] = [%d][%d][%d][%d]\n"
		//		 , 0, 0, i
		//		 , workQueuePointer[0][0][i]
		//		 , workQueuePointer[0][1][i]
		//		 , workQueuePointer[0][2][i]
		//		 , workQueuePointer[0][3][i]
		//	 );
		// }

	 //}






	printf("cleaaning");

	//free(isToBeValidatedFpPointer);
	//free(isToBeValidatedFnPointer);
	/*free(metaData.minMaxes);
	free(metaData.fpCount.arrP);
	free(metaData.fnCount.arrP);
	free(metaData.fpCounter.arrP);
	free(metaData.fnCounter.arrP);
	free(metaData.fpOffset.arrP);
	free(metaData.fnOffset.arrP);

	free(metaData.isActiveGold.arrP);
	free(metaData.isFullGold.arrP);

	free(metaData.isActiveSegm.arrP);
	free(metaData.isFullSegm.arrP);*/

	free(workQueuePointer);
	//	free(resultListPointer);

	//free(isToBeActivatedGoldPointer);
	//free(isToBeActivatedSegmPointer);


	free(forDebugArr);
	free(goldArr);
	free(segmArr);
	free(reducedSegm);
	free(reducedGold);
	free(reducedGoldPrevPointer);
	free(reducedSegmPrevPointer);
	free(reducedGoldRef);
	free(reducedSegmRef);



}













