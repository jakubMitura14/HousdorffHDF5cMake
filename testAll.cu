#include "MainPassesKernels.cu"
//#include "Structs.cu"
#include "UnitTestUtils.cu"
#include "testData.cu"







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


	size_t size = sizeof(unsigned int) * 20;
	unsigned int* minMaxesCPU = (unsigned int*)malloc(size);
	metaData.minMaxes = minMaxesCPU;

	int workQueueAndRLLength = 200;
	int workQueueWidth = 4;
	int resultListWidth = 5;
	//allocating to semiarbitrrary size 
	auto workQueuePointer = alloc_tensorToZeros<uint32_t>(workQueueAndRLLength, workQueueWidth, 1);


	forDebugArr = alloc_tensorToZeros<int>(dXLength, dYLength, dZLength);


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




	//setArrCPU(arrGoldObj, 0, 0, 0, 2);//
	//setArrCPU(arrGoldObj, 8, 8, 6, 2);//

	////setArrCPU(arrSegmObj, 8, 8, 5, 2);//
	//
	//
	//
	//
	//setArrCPU(arrGoldObj, 32, 20, 32, 2);//
	//setArrCPU(arrSegmObj, 31, 20, 32, 2);//
	//setArrCPU(arrSegmObj, 32, 19, 32, 2);//
	//setArrCPU(arrSegmObj, 32, 20, 31, 2);//

	////setArrCPU(arrSegmObj, 38, 38, 35, 2);//
	////setArrCPU(arrGoldObj, 38, 38, 36, 2);//
	////setArrCPU(arrSegmObj, 38, 38, 37, 2);//





	setArrCPU(arrGoldObj, 0, 0, 0, 2);//
//	setArrCPU(arrSegmObj, 0, 0, 0, 2);//

	setArrCPU(arrGoldObj, 39, 5, 5, 2);//

	setArrCPU(arrGoldObj, 31, 5, 5, 2);//

	setArrCPU(arrGoldObj, 32, 7, 7, 2);//


	//setArrCPU(arrGoldObj, 3, 3, 32, 2);//

	//setArrCPU(arrGoldObj, 35, 38, 31, 2);//
	//setArrCPU(arrGoldObj, 35, 38, 39, 2);//

	//setArrCPU(arrSegmObj, 16, 16, 16, 2);//
	//setArrCPU(arrGoldObj, 38, 38, 36, 2);//

	//setArrCPU(arrSegmObj, 8, 8, 5, 2);//




	//setArrCPU(arrGoldObj, 32, 20, 32, 2);//
	//setArrCPU(arrSegmObj, 31, 20, 32, 2);//
	//setArrCPU(arrSegmObj, 32, 19, 32, 2);//
	//setArrCPU(arrSegmObj, 32, 20, 31, 2);//






	/*
	maxX 2  [1]
minX 1  [2]
maxY 1  [3]
minY 0  [4]
maxZ 5  [5]
minZ 2  [6]
	*/


	printf("\n aaa \n");

	uint32_t* resultListPointerMetaCPU;
	uint32_t* resultListPointerLocalCPU;
	uint32_t* resultListPointerIterNumbCPU;
	uint32_t* metaDataArrPointerCPU;
	uint32_t* workQueuePointerCPU;

	uint32_t* reducedResCPU;
	uint32_t* origArrsCPU;
	ForBoolKernelArgs<int> fbArgs= mainKernelsRun(forFullBoolPrepArgs, reducedResCPU, resultListPointerMetaCPU
		, resultListPointerLocalCPU, resultListPointerIterNumbCPU
		, metaDataArrPointerCPU, workQueuePointerCPU, origArrsCPU
		);

	printFromReduced(fbArgs, reducedResCPU);


	for (int i = 0; i < 5;i++) {
		if (resultListPointerLocalCPU[i]>0 || resultListPointerMetaCPU[i]>0) {
			uint32_t linIdexMeta = resultListPointerMetaCPU[i] - (isGoldOffset * (resultListPointerMetaCPU[i] >= isGoldOffset))-1;
			uint32_t xMeta = linIdexMeta % fbArgs.metaData.metaXLength;
			uint32_t zMeta = uint32_t(floor((float)(linIdexMeta / (fbArgs.metaData.metaXLength * fbArgs.metaData.MetaYLength))));
			uint32_t yMeta = uint32_t(floor((float)((linIdexMeta - ((zMeta * fbArgs.metaData.metaXLength * fbArgs.metaData.MetaYLength) + xMeta)) / fbArgs.metaData.metaXLength)));
			
			uint32_t linLocal = resultListPointerLocalCPU[i];
			uint32_t xLoc = linLocal % fbArgs.dbXLength;
			uint32_t zLoc = uint32_t(floor((float)(linLocal / (32 * fbArgs.dbYLength))));
			uint32_t yLoc = uint32_t(floor((float)((linLocal - ((zLoc * 32 * fbArgs.dbYLength) + xLoc)) / 32)));


			uint32_t x = xMeta * 32 + xLoc;
			uint32_t y= yMeta * fbArgs.dbYLength + yLoc;
			uint32_t z = zMeta * 32 + zLoc;

			printf("resullt linIdexMeta %d x %d y %d z %d  xMeta %d yMeta %d zMeta %d xLoc %d yLoc %d zLoc %d linLocal %d   \n"
				,linIdexMeta
				,x,y,z
				,xMeta,yMeta, zMeta
				,xLoc,yLoc,zLoc
				, linLocal


			);


		
		}
	}





	printf("\n **************************************** \n");

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





	//i, j, k, value = 0;
	//i = 31;
	//j = 12;
	//for (k = 0; k < MetaZLength; k++) {
	//	goldArr[k][j][i] = 1;
	//	if (reducedGold[k][j][i] > 0) {
	//		for (int tt = 0; tt < 32; tt++) {
	//			if ((reducedGold[k][j][i] & (1 << (tt)))) {
	//				printf("found in reduced fp  [%d]\n", k * 32 + tt);

	//			}
	//		}

	//	}
	//}


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
//	 //uint32_t x = forFullBoolPrepArgs.metaData.resultList.arrP[ji][0][0];
//	 //uint32_t y = forFullBoolPrepArgs.metaData.resultList.arrP[ji][1][0];
//	 //uint32_t z = forFullBoolPrepArgs.metaData.resultList.arrP[ji][2][0];
//	 //uint32_t isGold = forFullBoolPrepArgs.metaData.resultList.arrP[ji][3][0];
//	 //uint32_t iternumb = forFullBoolPrepArgs.metaData.resultList.arrP[ji][4][0];
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


	free(resultListPointerMetaCPU);
	free(resultListPointerLocalCPU);
	free(resultListPointerIterNumbCPU);
	free(metaDataArrPointerCPU);
	free(workQueuePointerCPU);

	free(reducedResCPU);
	free(origArrsCPU);

}













