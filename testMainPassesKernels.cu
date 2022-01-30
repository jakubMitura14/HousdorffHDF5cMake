//#include "MainPassesKernels.cu"
////#include "Structs.cu"
//#include "UnitTestUtils.cu"
//
//
//
//
//
//
//
////testing loopMeta function in order to execute test unhash proper function in loopMeta
//#pragma once
//extern "C" inline void testMainPasswes() {
//	// threads and blocks for bool kernel
//	const int blocks = 17;
//	const int xThreadDim = 32;
//	const int yThreadDim = 12;
//	const dim3 threads = dim3(xThreadDim, yThreadDim);
//	// threads and blocks for first metadata pass
//	int threadsFirstMetaDataPass = 32;
//	int blocksFirstMetaDataPass = 10;
//	
//	
//
//	//datablock dimensions
//	const int dbXLength = xThreadDim;
//	const int dbYLength = yThreadDim;
//	const int dbZLength = 32;
//	
//	
//	
//	//threads and blocks for main pass 
//	dim3 threadsMainPass= dim3(dbXLength, dbYLength);
//	int blocksMainPass =7;
//	//threads and blocks for padding pass 
//	dim3 threadsPaddingPass = dim3(32, 11);
//	int blocksPaddingPass=13;
//	//threads and blocks for non first metadata passes 
//	int threadsOtherMetaDataPasses=32;
//	int blocksOtherMetaDataPasses=7;
//
//
//	int minMaxesLength = 17;
//
//
//
//	//metadata
//	const int metaXLength = 13;
//	const int MetaYLength = 13;
//	const int MetaZLength = 13;
//
//
//	const int totalLength = metaXLength * MetaYLength * MetaZLength;
//	const int loopMetaTimes = floor(totalLength / blocks);
//
//	/*   int*** h_tensor;
//	   h_tensor = alloc_tensorToZeros<int>(metaXLength, MetaYLength, MetaZLength);*/
//
//	int i, j, k, value = 0;
//	int*** forDebugArr;
//
//	const int dXLength = metaXLength;
//	const int dYLength = MetaYLength;
//	const int dZLength = MetaZLength;
//
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
//	auto minMaxesPointer = alloc_tensorToZeros<unsigned int>(minMaxesLength, 1, 1);
//
//	auto isActiveGoldPointer = alloc_tensorToZeros<bool>(metaXLength, MetaYLength, MetaZLength);
//	auto isFullGoldPointer = alloc_tensorToZeros<bool>(metaXLength, MetaYLength, MetaZLength);
//	auto isActiveSegmPointer = alloc_tensorToZeros<bool>(metaXLength, MetaYLength, MetaZLength);
//	auto isFullSegmPointer = alloc_tensorToZeros<bool>(metaXLength, MetaYLength, MetaZLength);
//
//	auto isToBeActivatedGoldPointer = alloc_tensorToZeros<bool>(metaXLength, MetaYLength, MetaZLength);
//	auto isToBeActivatedSegmPointer = alloc_tensorToZeros<bool>(metaXLength, MetaYLength, MetaZLength);
//
//
//
//	auto isToBeValidatedFpPointer = alloc_tensorToZeros<bool>(metaXLength, MetaYLength, MetaZLength);
//	auto isToBeValidatedFnPointer = alloc_tensorToZeros<bool>(metaXLength, MetaYLength, MetaZLength);
//
//
//
//	auto fpC = get3dArrCPU(fpCPointer, metaXLength, MetaYLength, MetaZLength);
//	auto fnC = get3dArrCPU(fnCPointer, metaXLength, MetaYLength, MetaZLength);
//	auto minMaxes = get3dArrCPU(minMaxesPointer, minMaxesLength, 1, 1);
//
//	auto isToBeValidatedFp = get3dArrCPU(isToBeValidatedFpPointer, metaXLength, MetaYLength, MetaZLength);
//	auto isToBeValidatedFn = get3dArrCPU(isToBeValidatedFnPointer, metaXLength, MetaYLength, MetaZLength);
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
//	metaData.isActiveGold = get3dArrCPU(isActiveGoldPointer, metaXLength, MetaYLength, MetaZLength);;
//	metaData.isFullGold = get3dArrCPU(isFullGoldPointer, metaXLength, MetaYLength, MetaZLength);;
//	metaData.isActiveSegm = get3dArrCPU(isActiveSegmPointer, metaXLength, MetaYLength, MetaZLength);;
//	metaData.isFullSegm = get3dArrCPU(isFullSegmPointer, metaXLength, MetaYLength, MetaZLength);;
//
//	metaData.isToBeActivatedGold = get3dArrCPU(isToBeActivatedGoldPointer, metaXLength, MetaYLength, MetaZLength);;
//	metaData.isToBeActivatedSegm = get3dArrCPU(isToBeActivatedSegmPointer, metaXLength, MetaYLength, MetaZLength);;
//
//
//	metaData.isToBeValidatedFp = isToBeValidatedFp;
//	metaData.isToBeValidatedFn = isToBeValidatedFn;
//
//
//	//int paddingStoreX = metaXLength * 32;
//	//int paddingStoreY = MetaYLength * 32;
//	//int paddingStoreZ = MetaZLength;
//
//	//auto paddingsStoreGoldPointer = alloc_tensorToZeros<uint8_t>(paddingStoreX, paddingStoreY, paddingStoreZ);
//	//auto paddingsStoreSegmPointer = alloc_tensorToZeros<uint8_t>(paddingStoreX, paddingStoreY, paddingStoreZ);
//
//	int workQueueAndRLLength = 200;
//	int workQueueWidth = 4;
//	int resultListWidth = 5;
//	//allocating to semiarbitrrary size 
//	auto workQueuePointer = alloc_tensorToZeros<uint16_t>(workQueueAndRLLength, workQueueWidth, 1);
//	auto resultListPointer = alloc_tensorToZeros<uint16_t>(workQueueAndRLLength, resultListWidth, 1);
//	metaData.workQueue = get3dArrCPU(workQueuePointer, workQueueAndRLLength, workQueueWidth, 1);
//	metaData.resultList = get3dArrCPU(resultListPointer, workQueueAndRLLength, resultListWidth, 1);
//	
//
//	forDebugArr = alloc_tensorToZeros<int>(dXLength, dYLength, dZLength);
//
//	uint32_t*** reducedGold = alloc_tensorToZeros<uint32_t>(mainXLength, mainYLength, mainZLength);
//	uint32_t*** reducedSegm = alloc_tensorToZeros<uint32_t>(mainXLength, mainYLength, mainZLength);
//
//	uint32_t*** reducedGoldRef = alloc_tensorToZeros<uint32_t>(mainXLength, mainYLength, mainZLength);
//	uint32_t*** reducedSegmRef = alloc_tensorToZeros<uint32_t>(mainXLength, mainYLength, mainZLength);
//
//	uint32_t*** reducedGoldPrevPointer = alloc_tensorToZeros<uint32_t>(mainXLength, mainYLength, mainZLength);
//	uint32_t*** reducedSegmPrevPointer = alloc_tensorToZeros<uint32_t>(mainXLength, mainYLength, mainZLength);
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
//	forFullBoolPrepArgs.reducedGoldRef = get3dArrCPU(reducedGoldRef, mainXLength, mainYLength, MetaZLength);
//	forFullBoolPrepArgs.reducedSegmRef = get3dArrCPU(reducedSegmRef, mainXLength, mainYLength, MetaZLength);
//
//	forFullBoolPrepArgs.reducedGoldPrev = get3dArrCPU(reducedGoldPrevPointer, mainXLength, mainYLength, MetaZLength);
//	forFullBoolPrepArgs.reducedSegmPrev = get3dArrCPU(reducedSegmPrevPointer, mainXLength, mainYLength, MetaZLength);
//
//
//	forFullBoolPrepArgs.threads = threads;
//	forFullBoolPrepArgs.blocks = blocks;
//
//	forFullBoolPrepArgs.threadsFirstMetaDataPass = threadsFirstMetaDataPass;
//	forFullBoolPrepArgs.blocksFirstMetaDataPass = blocksFirstMetaDataPass;
//
//	forFullBoolPrepArgs.threadsMainPass = threadsMainPass;
//	forFullBoolPrepArgs.blocksMainPass = blocksMainPass;
//
//	forFullBoolPrepArgs.threadsPaddingPass = threadsPaddingPass;
//	forFullBoolPrepArgs.blocksPaddingPass = blocksPaddingPass;
//
//	forFullBoolPrepArgs.threadsOtherMetaDataPasses = threadsOtherMetaDataPasses;
//	forFullBoolPrepArgs.blocksOtherMetaDataPasses = blocksOtherMetaDataPasses;
//
//	//populate segm  and gold Arr
//
//
//	auto arrGoldObj = forFullBoolPrepArgs.goldArr;
//	auto arrSegmObj = forFullBoolPrepArgs.segmArr;
//
//
//
//	//printf("mainXLength %d mainYLength %d mainZLength %d \n", mainXLength, mainYLength, mainZLength);
//
//
//
//
//	//assert(("There are five lights", 2 + 2 == 5));
////
////	int i, j, k, value = 0;
////	for (i = 0; i < mainXLength; i++) {
////		for (j = 0; j < mainYLength; j++) {
////			for (k = 0; k < MetaZLength; k++) {
////				//goldArr[k][j][i] = 1;
////				if (reducedGold[k][j][i] > 0) {
////					for (int tt = 0; tt < 32; tt++) {
////						if ((reducedGold[k][j][i] & (1 << (tt)))) {
////							printf("found in reduced fp  [%d][%d][%d]\n", i, j, k * 32 + tt);
////
////						}
////					}
////
////				}
////			}
////		}
////	}
////
////	for (i = 0; i < mainXLength; i++) {
////		for (j = 0; j < mainYLength; j++) {
////			for (k = 0; k < MetaZLength; k++) {
////				//goldArr[k][j][i] = 1;
////				if (forFullBoolPrepArgs.reducedSegm.arrP[k][j][i] > 0) {
////					for (int tt = 0; tt < 32; tt++) {
////						if ((forFullBoolPrepArgs.reducedSegm.arrP[k][j][i] & (1 << (tt)))) {
////							printf("found in reduced fn [%d][%d][%d]\n", i, j, k * 32 + tt);
////						}
////					}
////				}
////			}
////		}
////	}
////
////	i, j, k, value = 0;
////	for (i = 0; i < metaXLength; i++) {
////		for (j = 0; j < MetaYLength; j++) {
////			for (k = 0; k < MetaZLength; k++) {
////				//goldArr[k][j][i] = 1;
////				if (metaData.isActiveGold.arrP[k][j][i]) {
////					printf("found as Active in gold  [%d][%d][%d]\n", i, j, k);
////				}
////			}
////		}
////	};
////
////	i, j, k, value = 0;
////	for (i = 0; i < metaXLength; i++) {
////		for (j = 0; j < MetaYLength; j++) {
////			for (k = 0; k < MetaZLength; k++) {
////				//goldArr[k][j][i] = 1;
////				if (metaData.isActiveSegm.arrP[k][j][i]) {
////					printf("found as Active in segm  [%d][%d][%d]\n", i, j, k);
////				}
////			}
////		}
////	};
////
////	i, j, k, value = 0;
////	for (i = 0; i < metaXLength; i++) {
////		for (j = 0; j < MetaYLength; j++) {
////			for (k = 0; k < MetaZLength; k++) {
////				//goldArr[k][j][i] = 1;
////				if (fpC.arrP[k][j][i] > 0) {
////					printf("found Fp %d  [%d][%d][%d]\n", fpC.arrP[k][j][i], i, j, k);
////				}
////			}
////		}
////	};
////
////	for (i = 0; i < metaXLength; i++) {
////		for (j = 0; j < MetaYLength; j++) {
////			for (k = 0; k < MetaZLength; k++) {
////				//goldArr[k][j][i] = 1;
////				if (fnC.arrP[k][j][i] > 0) {
////					printf("found Fn %d  [%d][%d][%d]\n", fnC.arrP[k][j][i], i, j, k);
////				}
////			}
////		}
////	};
////
////	i = 1;
////	printf("maxX %d  [%d]\n", minMaxes.arrP[0][0][i], i);
////	i = 2;
////	printf("minX %d  [%d]\n", minMaxes.arrP[0][0][i], i);
////	i = 3;
////	printf("maxY %d  [%d]\n", minMaxes.arrP[0][0][i], i);
////	i = 4;
////	printf("minY %d  [%d]\n", minMaxes.arrP[0][0][i], i);
////	i = 5;
////	printf("maxZ %d  [%d]\n", minMaxes.arrP[0][0][i], i);
////	i = 6;
////	printf("minZ %d  [%d]\n", minMaxes.arrP[0][0][i], i);
////	i = 7;
////	printf("global FP count %d  [%d]\n", minMaxes.arrP[0][0][i], i);
////	i = 8;
////	printf("global FN count %d  [%d]\n", minMaxes.arrP[0][0][i], i);
////	i = 9;
////	printf("workQueueCounter %d  [%d]\n", minMaxes.arrP[0][0][i], i);
////	i = 10;
////	printf("resultFP globalCounter %d  [%d]\n", minMaxes.arrP[0][0][i], i);
////	i = 11;
////	printf("resultFn globalCounter %d  [%d]\n", minMaxes.arrP[0][0][i], i);
////	i = 12;
////	printf("global FPandFn offset %d  [%d]\n", minMaxes.arrP[0][0][i], i);
////	i = 13;
////	printf("globalIterationNumb %d  [%d]\n", minMaxes.arrP[0][0][i], i);
////
////	for (i = 0; i < mainXLength; i++) {
////		for (j = 0; j < mainYLength; j++) {
////			for (k = 0; k < mainZLength; k++) {
////				//goldArr[k][j][i] = 1;
////				if (goldArr[k][j][i] > 0) {
////					printf("segmArr[%d][%d][%d] = %d\n", i, j, k, goldArr[k][j][i]);
////				}
////			}
////		}
////	}
////
////
////	printf("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! firstMetaPass!!!!!!!!!!!!!!!!!!!!!\n\n");
////
////	i, j, k, value = 0;
////	for (i = 0; i < metaXLength; i++) {
////		for (j = 0; j < MetaYLength; j++) {
////			for (k = 0; k < MetaZLength; k++) {
////				//goldArr[k][j][i] = 1;
////				if (metaData.fpOffset.arrP[k][j][i] > 0) {
////					printf("Offsets Fp %d  [%d][%d][%d]\n", metaData.fpOffset.arrP[k][j][i], i, j, k);
////				}
////			}
////		}
////    };
////
////
////	for (i = 0; i < metaXLength; i++) {
////		for (j = 0; j < MetaYLength; j++) {
////			for (k = 0; k < MetaZLength; k++) {
////				//goldArr[k][j][i] = 1;
////				if (metaData.fnOffset.arrP[k][j][i] > 0) {
////					printf("Offsets Fn %d  [%d][%d][%d]\n", metaData.fnOffset.arrP[k][j][i], i, j, k);
////				}
////			}
////		}
////	};
////
////
////
////
////
////	for (i = 0; i < workQueueAndRLLength; i++) {
////
////		goldArr[k][j][i] = 1;
////		if (workQueuePointer[0][0][i] > 0) {
////			printf("work queue [%d][%d][%d] = [%d][%d][%d][%d]\n"
////				, 0, 0, i
////				, workQueuePointer[0][0][i]
////				, workQueuePointer[0][1][i]
////				, workQueuePointer[0][2][i]
////				, workQueuePointer[0][3][i]
////			);
////		}
////
////	}
////
////	printf("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! main pass kernel !!!!!!!!!!!!!!!!!!!!!\n\n");
////	/*
////	need to test up, down , left , right dilatations given it will not get over the edge of data block
////	check is block correctly set as full
////	check do the results are added to the res list
////	check weather fp and fn counters are updated correctly
////	check is prev reducesd are set corrctly
////	
////	*/
////for (i = 0; i < mainXLength; i++) {
////	for (j = 0; j < mainYLength; j++) {
////		for (k = 0; k < MetaZLength; k++) {
////			 k = 5;
////		if (reducedSegm[k][j][i] > 0) {
////			for (int tt = 0; tt < 32; tt++) {
////				if ((reducedSegm[k][j][i] & (1 << (tt)))) {
////					printf("found in reduced segm  [%d][%d][%d]\n", i, j, k * 32 + tt);
////				}
////			}
////		}
////		}
////	}
////}
////
////
////
////
////for (i = 0; i < mainXLength; i++) {
////	for (j = 0; j < mainYLength; j++) {
////		for (k = 0; k < MetaZLength; k++) {
////			if (forFullBoolPrepArgs.reducedSegm.arrP[k][j][i] > 0) {
////				for (int tt = 0; tt < 32; tt++) {
////					if ((forFullBoolPrepArgs.reducedSegm.arrP[k][j][i] & (1 << (tt)))) {
////						printf("found in reduced fn [%d][%d][%d]\n", i, j, k * 32 + tt);
////
////					}
////				}
////
////
////			}
////		}
////	}
////}
////
////
////
////
////
////		for (i = 0; i < dXLength; i++) {
////		for (j = 0; j < dYLength; j++) {
////			for (k = 0; k < dZLength; k++) {
////				//goldArr[k][j][i] = 1;
////				
////					printf("found in forDebugArr %d  [%d][%d][%d]\n", forDebugArr[k][j][i], i, j, k);
////
////
////			}
////		}
////	}
//int pointsNumber = 0;
//int metasNumber =0;
//
//int& pointsNumberRef = pointsNumber;
//int& metasNumberRef = metasNumber;
//	printf("teeests");
//	/////////////
//	/////define Test points 
//	forTestPointStruct allPointsA[]={
//
//
//
//	// inside the block
//	// meta 2,2,2 only gold points not in result after 2 dilataions
//	getTestPoint(
//	2,5,8//x,y,z
//	,true//isGold
//	,2,2,2//xMeta,yMeta,Zmeta
//	,dbXLength,dbYLength,dbZLength,pointsNumberRef)
//	,getTestPoint(
//	3,3,9//x,y,z
//	,true//isGold
//	,2,2,2//xMeta,yMeta,Zmeta
//	,dbXLength,dbYLength,dbZLength,pointsNumberRef)
//	,getTestPoint(
//	1,5,3//x,y,z
//	,true//isGold
//	,2,2,2//xMeta,yMeta,Zmeta
//	,dbXLength,dbYLength,dbZLength,pointsNumberRef)
//	// block 0 corner 0 
//	,getTestPoint(
//	0,0,1//x,y,z
//	,true//isGold
//	,0,0,0//xMeta,yMeta,Zmeta
//	,dbXLength,dbYLength,dbZLength,pointsNumberRef,false, true)
//	//lpwer right corner	
//	,getTestPoint(
//	dbXLength-2,dbYLength-2,dbZLength-2//x,y,z
//	,true//isGold
//	,metaXLength-1,MetaYLength-1,MetaZLength-1//xMeta,yMeta,Zmeta
//	,dbXLength,dbYLength,dbZLength,pointsNumberRef)
//	// block 0 corner 0 
//	,getTestPoint(
//	0,0,0//x,y,z
//	,false//isGold
//	,0,0,0//xMeta,yMeta,Zmeta
//	,dbXLength,dbYLength,dbZLength,pointsNumberRef,false, true)
//	//lpwer right corner	
//	,getTestPoint(
//	dbXLength-2,dbYLength-2,dbZLength-2//x,y,z
//	,false//isGold
//	,metaXLength-1,MetaYLength-1,MetaZLength-1//xMeta,yMeta,Zmeta
//	,dbXLength,dbYLength,dbZLength,pointsNumberRef)
//	//// some overlapping  voxels - should lead to dilatation but not add to fp or fn 	
//	,getTestPoint(
//	5,6,7//x,y,z
//	,false//isGold
//	,3,4,6//xMeta,yMeta,Zmeta
//	,dbXLength,dbYLength,dbZLength,pointsNumberRef,true
//	)
//	,getTestPoint(
//	9,11,7//x,y,z
//	,false//isGold
//	,3,4,6//xMeta,yMeta,Zmeta
//	,dbXLength,dbYLength,dbZLength,pointsNumberRef,true)	,
//		
//
//
//	//now some points that should be covered by first dilatation		
//	getTestPoint(
//	9,11,7//x,y,z
//	,false//isGold
//	,7,4,6//xMeta,yMeta,Zmeta
//	,dbXLength,dbYLength,dbZLength,pointsNumberRef,false, true)	
//
//	,getTestPoint(
//	9,11,8//x,y,z
//	,true//isGold
//	,7,4,6//xMeta,yMeta,Zmeta
//	,dbXLength,dbYLength,dbZLength,pointsNumberRef,false, true)
//		
//		
//	,getTestPoint(
//	9,3,7//x,y,z
//	,false//isGold
//	,7,4,6//xMeta,yMeta,Zmeta
//	,dbXLength,dbYLength,dbZLength,pointsNumberRef,false, true)
//	,getTestPoint(
//	9,2,7//x,y,z
//	,true//isGold
//	,7,4,6//xMeta,yMeta,Zmeta
//	,dbXLength,dbYLength,dbZLength,pointsNumberRef,false, true)
//		
//	,getTestPoint(
//	9,5,7//x,y,z
//	,false//isGold
//	,7,4,6//xMeta,yMeta,Zmeta
//	,dbXLength,dbYLength,dbZLength,pointsNumberRef,false, true)
//	,getTestPoint(
//	9,6,7//x,y,z
//	,true//isGold
//	,7,4,6//xMeta,yMeta,Zmeta
//	,dbXLength,dbYLength,dbZLength,pointsNumberRef,false, true)
//		
//	,getTestPoint(
//	2,3,7//x,y,z
//	,false//isGold
//	,7,4,6//xMeta,yMeta,Zmeta
//	,dbXLength,dbYLength,dbZLength,pointsNumberRef,false, true)
//	,getTestPoint(
//	3,3,7//x,y,z
//	,true//isGold
//	,7,4,6//xMeta,yMeta,Zmeta
//	,dbXLength,dbYLength,dbZLength,pointsNumberRef, false, true)
//		
//		
//		
//	//now some points that should be covered by second dilatation		
//	,getTestPoint(
//	9,11,7//x,y,z
//	,false//isGold
//	,7,2,6//xMeta,yMeta,Zmeta
//	,dbXLength,dbYLength,dbZLength,pointsNumberRef, false, false, true)
//	,getTestPoint(
//	9,11,9//x,y,z
//	,true//isGold
//	,7,2,6//xMeta,yMeta,Zmeta
//	,dbXLength,dbYLength,dbZLength,pointsNumberRef, false, false, true)
//		
//		
//	,getTestPoint(
//	9,3,7//x,y,z
//	,false//isGold
//	,7,2,6//xMeta,yMeta,Zmeta
//	,dbXLength,dbYLength,dbZLength,pointsNumberRef, false, false, true)
//	,getTestPoint(
//	9,1,7//x,y,z
//	,true//isGold
//	,7,2,6//xMeta,yMeta,Zmeta
//	,dbXLength,dbYLength,dbZLength,pointsNumberRef, false, false, true)	,
//		
//		
//		
//	/*now specifically we will get some points on the borders  to establish if they dilatate properly
//
//
//
//
//
//	//top*/
//        getTestPoint(
//	2,2,0//x,y,z
//	,false//isGold
//	,0,0,2//xMeta,yMeta,Zmeta
//	,dbXLength,dbYLength,dbZLength,pointsNumberRef),	
//
//
//			////top*/
//			//getTestPoint(
//			//	2, 2, 1//x,y,z
//			//	, true//isGold
//			//	, 0, 0, 2//xMeta,yMeta,Zmeta
//			//	, dbXLength, dbYLength, dbZLength, pointsNumberRef),
//	
//
//			////getTestPoint(
//			////	2, 2, 9//x,y,z
//			////	, false//isGold
//			////	, 0, 0, 2//xMeta,yMeta,Zmeta
//			////	, dbXLength, dbYLength, dbZLength, pointsNumberRef),
//			//getTestPoint(
//			//	2, 2, 15//x,y,z
//			//	, true//isGold
//			//	, 0, 0, 2//xMeta,yMeta,Zmeta
//			//	, dbXLength, dbYLength, dbZLength, pointsNumberRef),
//
//			//getTestPoint(
//			//	2, 2, 19//x,y,z
//			//	, true//isGold
//			//	, 0, 0, 2//xMeta,yMeta,Zmeta
//			//	, dbXLength, dbYLength, dbZLength, pointsNumberRef),
//
//
//	//bottom	
//        getTestPoint(
//	2,2,dbZLength-1//x,y,z
//	,false//isGold
//	,0,0,4//xMeta,yMeta,Zmeta
//	,dbXLength,dbYLength,dbZLength,pointsNumberRef)	,
//		
//	//left	
//        getTestPoint(
//	0,2,2//x,y,z
//	,false//isGold
//	,8,0,6//xMeta,yMeta,Zmeta
//	,dbXLength,dbYLength,dbZLength,pointsNumberRef)	,		
//		
//	//right	
//        getTestPoint(
//	dbXLength-1,3,7//x,y,z
//	,false//isGold
//	,0,0,8//xMeta,yMeta,Zmeta
//	,dbXLength,dbYLength,dbZLength,pointsNumberRef)	
//		
//		
//	//anterior	
//        ,getTestPoint(
//	9,dbYLength-1,7//x,y,z
//	,false//isGold
//	,0,0,10//xMeta,yMeta,Zmeta
//	,dbXLength,dbYLength,dbZLength,pointsNumberRef)	
//		
//	//posterior	
//        ,getTestPoint(
//	9,0,7//x,y,z
//	,false//isGold
//	,2,2,4//xMeta,yMeta,Zmeta
//	,dbXLength,dbYLength,dbZLength,pointsNumberRef)	
//		
//
//
//	
//
//		
//			////top
//			//getTestPoint(
//			//	2, 2, 1//x,y,z
//			//	, true//isGold
//			//	, 0, 0, 5//xMeta,yMeta,Zmeta
//			//	, dbXLength, dbYLength, dbZLength, pointsNumberRef)
//
//
//
//			//////bottom	
//			////, getTestPoint(
//			////	2, 9, 7//x,y,z
//			////	, true//isGold
//			////	, 0, 0, 11//xMeta,yMeta,Zmeta
//			////	, dbXLength, dbYLength, dbZLength, pointsNumberRef)
//
//			//////left	
//			////, getTestPoint(
//			////	2, 2, 2//x,y,z
//			////	, true//isGold
//			////	, 0, 0, 11//xMeta,yMeta,Zmeta
//			////	, dbXLength, dbYLength, dbZLength, pointsNumberRef)
//
//			//////right	
//			////, getTestPoint(
//			////	2, 3, 7//x,y,z
//			////	, true//isGold
//			////	, 0, 0, 11//xMeta,yMeta,Zmeta
//			////	, dbXLength, dbYLength, dbZLength, pointsNumberRef)
//
//
//			//////anterior	
//			////, getTestPoint(
//			////	9, 2, 7//x,y,z
//			////	, true//isGold
//			////	, 0, 1, 7//xMeta,yMeta,Zmeta
//			////	, dbXLength, dbYLength, dbZLength, pointsNumberRef)
//
//			//////posterior	
//			////, getTestPoint(
//			////	9, 2, 7//x,y,z
//			////	, true//isGold
//			////	, 2, 1, 7//xMeta,yMeta,Zmeta
//			////	, dbXLength, dbYLength, dbZLength, pointsNumberRef)
//
//
//
//
//		
//	//left up anterior corner	
//        ,getTestPoint(
//	0,dbYLength-1,0//x,y,z
//	,false//isGold
//	,2,2,6//xMeta,yMeta,Zmeta
//	,dbXLength,dbYLength,dbZLength,pointsNumberRef)	
//		
//		
//	//right up anterior corner	
//        ,getTestPoint(
//	dbXLength-1,dbYLength-1,0//x,y,z
//	,false//isGold
//	,2,2,8//xMeta,yMeta,Zmeta
// 	,dbXLength,dbYLength,dbZLength,pointsNumberRef)	
//		
// 	//left down anterior corner	
// //        ,getTestPoint(
// //	0,dbYLength-1,dbZLength-1//x,y,z
// //	,false//isGold
// //	,2,2,10//xMeta,yMeta,Zmeta
// //	dbXLength,dbYLength,dbZLength,pointsNumberRef)			
//	//	
// //	//right dow anterior  corner	
// //        ,getTestPoint(
// //	dbXLength-1,dbYLength-1,dbZLength-1//x,y,z
// //	,false//isGold
// //	,4,4,2//xMeta,yMeta,Zmeta
// //	dbXLength,dbYLength,dbZLength,pointsNumberRef)			
//	//	
//
//	//	
//	//	
//	//	
// //	//left up posterior corner	
// //        ,getTestPoint(
// //	0,0,0//x,y,z
// //	,false//isGold
// //	,4,4,4//xMeta,yMeta,Zmeta
// //	dbXLength,dbYLength,dbZLength,pointsNumberRef)	
//	//	
//	//	
// //	//right up posterior corner	
// //        ,getTestPoint(
// //	dbXLength-1,0,0//x,y,z
// //	,false//isGold
// //	,7,2,6//xMeta,yMeta,Zmeta
// //	dbXLength,dbYLength,dbZLength,pointsNumberRef)	
//	//	
// //	//left down posterior corner	
// //        ,getTestPoint(
// //	0,0,dbZLength-1//x,y,z
// //	,false//isGold
// //	,7,2,6//xMeta,yMeta,Zmeta
// //	dbXLength,dbYLength,dbZLength,pointsNumberRef)			
//	//	
//	//right dow posterior  corner	
//        ,getTestPoint(
//	dbXLength-1,0,dbZLength-1//x,y,z
//	,false//isGold
//	,4,4,4//xMeta,yMeta,Zmeta
//	,dbXLength,dbYLength,dbZLength,pointsNumberRef)			
//				
//	//rshould be activated aafter two dilatations
//        ,getTestPoint(
//	1,1,1//x,y,z
//	,false//isGold
//	,4,4,8//xMeta,yMeta,Zmeta
//	,dbXLength,dbYLength,dbZLength,pointsNumberRef)			
//
//
//
//
//
//		
//		};// list holding most points
//	
//
//
//
//
//	
//	//now we neeed additionally to supply some block that will be full
//	// 5,5,5 full at start	
//		for (i = 0; i < dbXLength; i++) {
//			for (j = 0; j < dbYLength; j++) {
//				for (k = 0; k < dbZLength; k++) {
//					setArrCPU(arrSegmObj, dbXLength * 5 + i, dbYLength * 5 + j, dbZLength * 5 + k, 2,false);
//				}
//			}
//		};
//
//	// 5,7,7 full after one dil
//for (i = 1; i < dbXLength; i++) {
//	for (j = 0; j < dbYLength; j++) {
//		for (k = 0; k < dbZLength; k++) {
//			setArrCPU(arrSegmObj, dbXLength * 5 + i, dbYLength * 7 + j, dbZLength * 7 + k, 2, false);
//		}
//	}
//};
//
//			// 5,7,10 full after two dil
//for (i = 2; i < dbXLength - 2; i++) {
//	for (j = 0; j < dbYLength; j++) {
//		for (k = 0; k < dbZLength; k++) {
//			setArrCPU(arrSegmObj, dbXLength * 5 + i, dbYLength * 7 + j, dbZLength * 10 + k, 2, false);
//		}
//	}
//};
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
//		forTestMetaDataStruct a0 = getMetdataTestStruct(metasNumberRef, 0, 0, 0, 1, 1);
//		a0.fpConterAfterOneDil = 1;
//		a0.fnConterAfterOneDil = 1;
//		a0.isToBeValidatedFpAfterOneIter = false;
//
//
//	forTestMetaDataStruct a1 = getMetdataTestStruct(metasNumberRef,3, 4, 6,2);
//	a1.isToBeValidatedFpAfterOneIter = false;
//	a1.isToBeValidatedFpAfterTwoIter = false;
//	a1.isToBeValidatedFnAfterOneIter = false;
//	a1.isToBeValidatedFnAfterTwoIter = false;
//
//	forTestMetaDataStruct a2 = getMetdataTestStruct(metasNumberRef,7, 4, 6);
//	a2.fnCount = 4;
//	a2.fpCount = 4;
//	a2.fpConterAfterOneDil = 4;
//	a2.fnConterAfterOneDil = 4;
//
//	a2.fpConterAfterTwoDil = 4;
//	a2.fnConterAfterTwoDil = 4;
//
//	a2.isToBeValidatedFpAfterOneIter = false;
//	a2.isToBeValidatedFpAfterTwoIter = false;
//	a2.isToBeValidatedFnAfterOneIter = false;
//	a2.isToBeValidatedFnAfterTwoIter = false;
//
//	forTestMetaDataStruct a3 = getMetdataTestStruct(metasNumberRef,7, 2, 6);
//	a3.fnCount = 2;
//	a3.fpCount = 2;
//	a3.fpConterAfterOneDil = 0;
//	a3.fnConterAfterOneDil = 0;
//
//	a3.fpConterAfterTwoDil = 2;
//	a3.fnConterAfterTwoDil = 2;
//
//	a3.isToBeValidatedFpAfterOneIter = true;
//	a3.isToBeValidatedFpAfterTwoIter = true;
//	a3.isToBeValidatedFnAfterOneIter = false;
//	a3.isToBeValidatedFnAfterTwoIter = false;
//
//	forTestMetaDataStruct full1 = getMetdataTestStruct(metasNumberRef,5, 7, 7, (dbXLength - 1) * dbYLength * dbZLength);
//	full1.isToBeFullAfterOneIter = true;
//	forTestMetaDataStruct full2 = getMetdataTestStruct(metasNumberRef,5, 5, 5, (dbXLength )* dbYLength * dbZLength);
//	full2.isToBeFullAfterOneIter = true;
//	forTestMetaDataStruct full3 = getMetdataTestStruct(metasNumberRef,5, 7, 10, (dbXLength - 4) * dbYLength * dbZLength);
//	full3.isToBeFullAfterTwoIter = true;
//	
//	
//
//	forTestMetaDataStruct allMetas[] = {
//	getMetdataTestStruct(metasNumberRef,2,2,2,  0, 3)
//		,a0
//
//		
//		//	,getMetdataTestStruct(metasNumberRef,metaXLength - 1,MetaYLength - 1,MetaZLength - 1 , 1,1)
//		,a1// should not be validated at all
//		,a2//now some points that should be covered by second dilatation after one dilatation no need to validate it
//		,a3, //now some points that should be covered by second dilatation after one dilatation no need to validate it
//
//		getMetdataTestStruct(metasNumberRef,0,0,2, 1)
//		,getMetdataTestStruct(metasNumberRef,0,0,1,0,0,false,true)//just marking it get activated	
//		,getMetdataTestStruct(metasNumberRef,0,0,4, 1)
//		,getMetdataTestStruct(metasNumberRef,0,0,5,0,0,false,true)//just marking it get activated	
//		,getMetdataTestStruct(metasNumberRef,8,0,6, 1)
//		,getMetdataTestStruct(metasNumberRef,7,0,6,0,0,false,true)//just marking it get activated	
//
//		,getMetdataTestStruct(metasNumberRef,0,0,8, 1)
//		,getMetdataTestStruct(metasNumberRef,1,0,8,0,0,false,true)//just marking it get activated	
//		,getMetdataTestStruct(metasNumberRef,0,0,10, 1)
//		,getMetdataTestStruct(metasNumberRef,0,1,10 ,0,0,false,true)//just marking it get activated			
//		,getMetdataTestStruct(metasNumberRef,2,2,4, 1)
//		,getMetdataTestStruct(metasNumberRef,2,1,4,0,0,false,true)//just marking it get activated			
//		
//		,getMetdataTestStruct(metasNumberRef,2,2,6, 1)
//		,getMetdataTestStruct(metasNumberRef,1,2,6,0,0,false,true)//just marking it get activated			
//		,getMetdataTestStruct(metasNumberRef,2,2,5,0,0,false,true)//just marking it get activated			
//		,getMetdataTestStruct(metasNumberRef,2,3,6,0,0,false,true)//just marking it get activated		
//
//				//right dow posterior  corner	
//		,getMetdataTestStruct(metasNumberRef,4,4,4, 1)
//		,getMetdataTestStruct(metasNumberRef,5,4,4,0,0,false,true)//just marking it get activated			
//		,getMetdataTestStruct(metasNumberRef,4,4,5,0,0,false,true)//just marking it get activated			
//		,getMetdataTestStruct(metasNumberRef,4,3,4,0,0,false,true)//just marking it get activated		
//
//		,getMetdataTestStruct(metasNumberRef,4,4,8, 1)
//		,getMetdataTestStruct(metasNumberRef,3,4,8,0,0,false,true)//just marking it get activated			
//		,getMetdataTestStruct(metasNumberRef,4,3,8,0,0,false,true)//just marking it get activated			
//		,getMetdataTestStruct(metasNumberRef,4,4,7,0,0,false,true)//just marking it get activated		
//		,getMetdataTestStruct(metasNumberRef,9,9,9,0,0)//some ampty block
//
//
//
//		
//	,full1, full2, full3
//	};
//	
//
//
//
//	 /// <summary>
//	 /// setting points 
//	 /// </summary>
//	 for (int i = 0; i < pointsNumber; i++) {
//		 forTestPointStruct currPoint = allPointsA[i];
//			 if (currPoint.isGold) {
//				 setArrCPU(arrGoldObj, currPoint.x, currPoint.y, currPoint.z, 2);
//			 }
//			 else {
//				 setArrCPU(arrSegmObj, currPoint.x, currPoint.y, currPoint.z, 2);
//
//		 };
//
//	 }
//
//	 //setArrCPU(arrGoldObj, 671, 263, 735, 2);
//	 //setArrCPU(arrSegmObj, 671, 263, 735, 2);
//
//
//
//	//mainKernelsRun(forFullBoolPrepArgs);
//	//printf("\n aaaaaaaaaaaaaaaaaaaaa\n ");
//
//	//i = 9;
//	//printf("workQueueCounter %d  [%d]\n", minMaxes.arrP[0][0][i], i);
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
//	 mainKernelsTestRun(forFullBoolPrepArgs, allPointsA, allMetas,pointsNumber, metasNumber);
//
//
//	 int ii = 7;
//	 	printf("global FP count %d  [%d]\n", minMaxes.arrP[0][0][ii], ii);
//		ii = 8;
//	 	printf("global FN count %d  [%d]\n", minMaxes.arrP[0][0][ii], ii);
//		ii = 9;
//	 	printf("workQueueCounter %d  [%d]\n", minMaxes.arrP[0][0][ii], ii);
//		ii = 10;
//	 	printf("resultFP globalCounter %d  [%d]\n", minMaxes.arrP[0][0][ii], ii);
//		ii = 11;
//	 	printf("resultFn globalCounter %d  [%d]\n", minMaxes.arrP[0][0][ii], ii);
//		ii = 12;
//		printf("global offset counter %d  [%d]\n", minMaxes.arrP[0][0][ii], ii);
//
//		ii  = 13;
//	 	printf("globalIterationNumb %d  [%d]\n", minMaxes.arrP[0][0][ii], ii);
//
//	 //for (int i = 0; i < workQueueAndRLLength; i++) {
//
//		// if (workQueuePointer[0][2][i] > 0) {
//		//	 printf("work queue [%d][%d][%d] = [%d][%d][%d][%d]\n"
//		//		 , 0, 0, i
//		//		 , workQueuePointer[0][0][i]
//		//		 , workQueuePointer[0][1][i]
//		//		 , workQueuePointer[0][2][i]
//		//		 , workQueuePointer[0][3][i]
//		//	 );
//		// }
//
//	 //}
//
//	 for (int ji = 0; ji < 30; ji++) {
//		 if (forFullBoolPrepArgs.metaData.resultList.arrP[0][2][ji] + forFullBoolPrepArgs.metaData.resultList.arrP[0][1][ji]  > 0) {
//			 printf("result  in point  %d %d %d isGold %d iteration %d \n ", forFullBoolPrepArgs.metaData.resultList.arrP[0][0][ji]
//				 , forFullBoolPrepArgs.metaData.resultList.arrP[0][1][ji]
//				 , forFullBoolPrepArgs.metaData.resultList.arrP[0][2][ji]
//				 , forFullBoolPrepArgs.metaData.resultList.arrP[0][3][ji]
//				 , forFullBoolPrepArgs.metaData.resultList.arrP[0][4][ji]);
//		 }
//	 }
//
//
//
//	// 	i, j, k, value = 0;
// //for (i = 0; i < metaXLength; i++) {
// //	for (j = 0; j < MetaYLength; j++) {
// //		for (k = 0; k < MetaZLength; k++) {
// //			//goldArr[k][j][i] = 1;
// //			if (metaData.isToBeValidatedFp.arrP[k][j][i]) {
// //				printf("found as to be validated fp  [%d][%d][%d]\n", i, j, k);
// //			}
// //		}
// //	}
// //};
//
// //i, j, k, value = 0;
// //for (i = 0; i < metaXLength; i++) {
// //	for (j = 0; j < MetaYLength; j++) {
// //		for (k = 0; k < MetaZLength; k++) {
// //			//goldArr[k][j][i] = 1;
// //			if (metaData.isToBeValidatedFn.arrP[k][j][i]) {
// //				printf("found as  to be validated fn  [%d][%d][%d]\n", i, j, k);
// //			}
// //		}
// //	}
// //};
//
//
//
//
//
//
//
//	 //for (i = 0; i < mainXLength; i++) {
// 	//	for (j = 0; j < mainYLength; j++) {
// 	//		for (k = 0; k < MetaZLength; k++) {
// 	//			//goldArr[k][j][i] = 1;
// 	//			if (reducedGold[k][j][i] > 0) {
// 	//				for (int tt = 0; tt < 32; tt++) {
// 	//					if ((reducedGold[k][j][i] & (1 << (tt)))) {
// 	//						printf("found in reduced fp  [%d][%d][%d]\n", i, j, k * 32 + tt);
// 
// 	//					}
// 	//				}
// 
// 	//			}
// 	//		}
// 	//	}
// 	//}
//
//
//	 //for (i = 0; i < mainXLength; i++) {
//		// for (j = 0; j < mainYLength; j++) {
//		//	 for (k = 0; k < MetaZLength; k++) {
//		//		 //goldArr[k][j][i] = 1;
//		//		 if (reducedSegm[k][j][i] > 0) {
//		//			 for (int tt = 0; tt < 32; tt++) {
//		//				 if ((reducedSegm[k][j][i] & (1 << (tt)))) {
//		//					 printf("found in reduced fn  [%d][%d][%d]\n", i, j, k * 32 + tt);
//
//		//				 }
//		//			 }
//
//		//		 }
//		//	 }
//		// }
//	 //}
//
//
//	 
//	 /*	i = 1;
//	 	printf("maxX %d  [%d]\n", minMaxes.arrP[0][0][i], i);
//	 	i = 2;
//	 	printf("minX %d  [%d]\n", minMaxes.arrP[0][0][i], i);
//	 	i = 3;
//	 	printf("maxY %d  [%d]\n", minMaxes.arrP[0][0][i], i);
//	 	i = 4;
//	 	printf("minY %d  [%d]\n", minMaxes.arrP[0][0][i], i);
//	 	i = 5;
//	 	printf("maxZ %d  [%d]\n", minMaxes.arrP[0][0][i], i);
//	 	i = 6;
//	 	printf("minZ %d  [%d]\n", minMaxes.arrP[0][0][i], i);
//	 	i = 7;
//	 	printf("global FP count %d  [%d]\n", minMaxes.arrP[0][0][i], i);
//	 	i = 8;
//	 	printf("global FN count %d  [%d]\n", minMaxes.arrP[0][0][i], i);
//	 	i = 9;
//	 	printf("workQueueCounter %d  [%d]\n", minMaxes.arrP[0][0][i], i);
//	 	i = 10;
//	 	printf("resultFP globalCounter %d  [%d]\n", minMaxes.arrP[0][0][i], i);
//	 	i = 11;
//	 	printf("resultFn globalCounter %d  [%d]\n", minMaxes.arrP[0][0][i], i);
//	 	i = 12;
//	 	printf("global FPandFn offset %d  [%d]\n", minMaxes.arrP[0][0][i], i);
//	 	i = 13;
//	 	printf("globalIterationNumb %d  [%d]\n", minMaxes.arrP[0][0][i], i);*/
//
//
//	
//	//	setArrCPU(arrSegmObj, dbXLength * 5 + 2, dbYLength * 5+2, dbZLength * 5 + 2, 2);
//
//	//setArrCPU(arrGoldObj, dbXLength  + 2, dbYLength  + 3, dbZLength  + 4, 2);
//	
//	
//	/////define metadata
//	
//	
//	
//	
//	
//	
//	
//	
//	
//	printf("cleaaning");
//
//	free(isToBeValidatedFpPointer);
//	free(isToBeValidatedFnPointer);
//	free(metaData.minMaxes.arrP);
//	free(metaData.fpCount.arrP);
//	free(metaData.fnCount.arrP);
//	free(metaData.fpCounter.arrP);
//	free(metaData.fnCounter.arrP);
//	free(metaData.fpOffset.arrP);
//	free(metaData.fnOffset.arrP);
//
//	free(metaData.isActiveGold.arrP);
//	free(metaData.isFullGold.arrP);
//
//	free(metaData.isActiveSegm.arrP);
//	free(metaData.isFullSegm.arrP);
//
//	free(workQueuePointer);
//	free(resultListPointer);
//
//	free(isToBeActivatedGoldPointer);
//	free(isToBeActivatedSegmPointer);
//
//
//	free(forDebugArr);
//	free(goldArr);
//	free(segmArr);
//	free(reducedSegm);
//	free(reducedGold);
//	free(reducedGoldPrevPointer);
//	free(reducedSegmPrevPointer);
//	free(reducedGoldRef);
//	free(reducedSegmRef);
//
//
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
