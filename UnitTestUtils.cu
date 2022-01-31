//#include "Structs.cu"
//#include "cuda_runtime.h"
//#include <iostream>     // std::cout
//
//
//
//#pragma once
//inline forTestPointStruct getTestPoint(int x, int y, int z,
//	bool isGold, int xMeta, int yMeta, int zMeta
//	, int dbXLength, int dbYLength, int  dbZLength
//	, int& pointsNumberRef
//	, bool isGoldAndSegm = false
//	, bool shouldBeInResAfterOneDil = false
//	, bool shouldBeInResAfterTwoDil = false
//) {
//	pointsNumberRef += 1;
//	forTestPointStruct res;
//
//	res.x = xMeta * dbXLength + x;
//	res.y = yMeta * dbYLength + y;
//	res.z = zMeta * dbZLength + z;
//	res.isGoldAndSegm = isGoldAndSegm;
//	res.isGold = isGold;
//
//	res.xMeta = xMeta;
//	res.yMeta = yMeta;
//	res.zMeta = zMeta;
//
//
//	res.shouldBeInResAfterOneDil = shouldBeInResAfterOneDil;
//	res.shouldBeInResAfterTwoDil = shouldBeInResAfterTwoDil;
//
//	return res;
//
//}
//
//
//
//
//#pragma once
//inline forTestMetaDataStruct getMetdataTestStruct(
//	int& metasNumberRef,
//	int xMeta,
//	int yMeta,
//	int zMeta,
//
//	int fpCount = 0,
//	int fnCount = 0,
//
//	bool isToBeActiveAtStart = true,
//	bool isToBeActiveAfterOneIter = true,
//	bool isToBeActiveAfterTwoIter = true,
//
//	bool isToBeFullAfterOneIter = false,
//	bool isToBeFullAfterTwoIter = false,
//
//	bool isToBeValidatedFpAfterOneIter = false,
//	bool isToBeValidatedFpAfterTwoIter = false,
//
//	bool isToBeValidatedFnAfterOneIter = false,
//	bool isToBeValidatedFnAfterTwoIter = false,
//
//	int fpConterAfterOneDil = 0,
//	int fpConterAfterTwoDil = 0,
//
//	int fnConterAfterOneDil = 0,
//	int fnConterAfterTwoDil = 0) {
//
//
//	forTestMetaDataStruct res;
//	metasNumberRef += 1;
//	res.xMeta = xMeta;
//	res.yMeta = yMeta;
//	res.zMeta = zMeta;
//
//
//	res.isToBeActiveAtStart = (fpCount+ fnCount)>0;
//	res.isToBeActiveAfterOneIter = isToBeActiveAfterOneIter;
//	res.isToBeActiveAfterTwoIter = isToBeActiveAfterTwoIter;
//
//	res.isToBeFullAfterOneIter = isToBeFullAfterOneIter;
//	res.isToBeFullAfterTwoIter = isToBeFullAfterTwoIter;
//
//	res.fpCount = fpCount;
//	res.fnCount = fnCount;
//
//	res.requiredspaceInFpResultList = fpCount;
//	res.requiredspaceInFnResultList = fnCount;
//
//	res.isToBeValidatedFpAfterOneIter = fpCount > 0;
//	res.isToBeValidatedFpAfterTwoIter = fpCount > 0;
//
//	res.isToBeValidatedFnAfterOneIter = fnCount > 0;
//	res.isToBeValidatedFnAfterTwoIter = fnCount > 0;
//
//
//	res.fpConterAfterOneDil = fpConterAfterOneDil;
//	res.fpConterAfterTwoDil = fpConterAfterTwoDil;
//
//	res.fnConterAfterOneDil = fnConterAfterOneDil;
//	res.fnConterAfterTwoDil = fnConterAfterTwoDil;
//	return res;
//}
//
//
//
//
//
//
//
//
//////////// for boolkernel tests
//
////1) is reduced arrs are they should be - all of them - are there entries in correct spots
////2) do number of fp and fn fo the begining works
////3) do we have min and maxes aset correctly
//#pragma once	
//inline void forBoolKernelTestUnitTests(ForFullBoolPrepArgs<int> fbArgs, forTestPointStruct allPointsA[], forTestMetaDataStruct allMetas[], int pointsNumber, int metasNumber
//	,int dbXLength, int dbYLength, int dbZLength) {
//	
//
//	//1) is reduced arrs are they should be - all of them - are there entries in correct spots
//	for (int i = 0; i < pointsNumber; i++) {
//		bool isInReducedRef=false;
//		bool isInReduced = false;
//		bool isInReducedPrev = false;
//
//		forTestPointStruct currPoint = allPointsA[i];
//		int bitPos = currPoint.z - currPoint.zMeta * dbZLength;
//
//		//printf("point %d %d %d \n  ", currPoint.x, currPoint.y, currPoint.z);
//
//		if (currPoint.isGold) {  
//			isInReducedRef = (fbArgs.reducedGoldRef.arrP[currPoint.zMeta][currPoint.y][currPoint.x] & (1 << (bitPos)));
//			isInReduced = (fbArgs.reducedGold.arrP[currPoint.zMeta][currPoint.y][currPoint.x] & (1 << (bitPos)));
//			isInReducedPrev = (fbArgs.reducedGoldPrev.arrP[currPoint.zMeta][currPoint.y][currPoint.x] & (1 << (bitPos)));		
//		}
//		else { 
//			isInReducedRef = (fbArgs.reducedSegmRef.arrP[currPoint.zMeta][currPoint.y][currPoint.x] & (1 << (bitPos)));
//			isInReduced = (fbArgs.reducedSegm.arrP[currPoint.zMeta][currPoint.y][currPoint.x] & (1 << (bitPos)));
//			isInReducedPrev = (fbArgs.reducedSegmPrev.arrP[currPoint.zMeta][currPoint.y][currPoint.x] & (1 << (bitPos)));
//		
//		}
//
//		if (!isInReducedRef) {
//			printf("nnnnnnnnnnn  not found point %d %d %d in referenca reduced \n ", currPoint.x, currPoint.y, currPoint.z);
//		}
//		if (!isInReduced) {
//			printf("nnnnnnnnnnn  not found point %d %d %d in  reduced \n ", currPoint.x, currPoint.y, currPoint.z);
//
//		}
//		if (!isInReducedPrev) {
//			printf("nnnnnnnnnnn  not found point %d %d %d in referenca prev \n ", currPoint.x, currPoint.y, currPoint.z);
//
//		}
//
//
//
//		//else {
//		//	printf("ffffffffffffft found point %d %d %d in referenca reduced \n ", currPoint.x, currPoint.y, currPoint.z);
//
//		//}
//
////		
//	}
//	printf("metasNumber %d \n ", metasNumber);
//	//2) do number of fp and fn fo the begining works
//	for (int i = 0; i < metasNumber; i++) {
//		forTestMetaDataStruct locMeta = allMetas[i];
//
//		//printf("block  %d %d %d fp count %d fncount %d \n ", locMeta.xMeta, locMeta.yMeta, locMeta.zMeta, locMeta.fpCount, locMeta.fnCount);
//
//
//		bool isFpOk = fbArgs.metaData.fpCount.arrP[locMeta.zMeta][locMeta.yMeta][locMeta.xMeta] == locMeta.fpCount;
//		bool isFnOk = fbArgs.metaData.fnCount.arrP[locMeta.zMeta][locMeta.yMeta][locMeta.xMeta] == locMeta.fnCount;
//
//		if (!isFpOk) {
//			printf("nnnnnnnnnnn  not correct fp number in block  %d %d %d is %d should be %d \n ", locMeta.xMeta, locMeta.yMeta, locMeta.zMeta
//				, fbArgs.metaData.fpCount.arrP[locMeta.zMeta][locMeta.yMeta][locMeta.xMeta], locMeta.fpCount);
//		}
//		else {
//		//	printf("tttttt  correct fp number in block  %d %d %d is %d should be %d \n ", locMeta.xMeta, locMeta.yMeta, locMeta.zMeta
//			//	, fbArgs.metaData.fpCount.arrP[locMeta.zMeta][locMeta.yMeta][locMeta.xMeta], locMeta.fpCount);
//		}
//		if (!isFnOk) {
//			printf("nnnnnnnnnnn  not correct fn number in block  %d %d %d is %d should be %d \n ", locMeta.xMeta, locMeta.yMeta, locMeta.zMeta
//				, fbArgs.metaData.fnCount.arrP[locMeta.zMeta][locMeta.yMeta][locMeta.xMeta], locMeta.fnCount);
//		}
//		else {
//		//	printf("tttttt  correct fn number in block  %d %d %d is %d should be %d \n ", locMeta.xMeta, locMeta.yMeta, locMeta.zMeta
//			//	, fbArgs.metaData.fnCount.arrP[locMeta.zMeta][locMeta.yMeta][locMeta.xMeta], locMeta.fnCount);
//		}
//
//
//	}
//
//
//
////	for (i = 0; i < mainXLength; i++) {
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
//
//
//}
//
//
//
//
//
///// first meta pass
////1) do all blocks have enough space defined by offsets	
////2) doues all blocks marked as active are in the work queue
////3) are block that supposed to be actie are 
//#pragma once
//inline void firstMetaPassKernelTestUnitTests(ForFullBoolPrepArgs<int> fbArgs, forTestPointStruct allPointsA[], forTestMetaDataStruct allMetas[], int pointsNumber, int metasNumber
//	, int dbXLength, int dbYLength, int dbZLength) {
//
//	int totalFp = 0;
//	int totalFn = 0;
//
//	bool isSetArr[100000];
//	for (int i = 0; i < metasNumber; i++) {
//		isSetArr[i] = false;
//	}
//
//
//	for (int i = 0; i < metasNumber; i++) {
//		forTestMetaDataStruct locMeta = allMetas[i];
//		int fpOffset = fbArgs.metaData.fpOffset.arrP[locMeta.zMeta][locMeta.yMeta][locMeta.xMeta];
//		int fnOffset = fbArgs.metaData.fnOffset.arrP[locMeta.zMeta][locMeta.yMeta][locMeta.xMeta];
//
//		isSetArr[fpOffset]=true ;
//		isSetArr[fnOffset] =true;
//
//
//		
//	};
//
//	for (int i = 0; i < metasNumber; i++) {
//
//		//printf("block  %d %d %d fp count %d fncount %d \n ", locMeta.xMeta, locMeta.yMeta, locMeta.zMeta, locMeta.fpCount, locMeta.fnCount);
//		//1) do all blocks have enough space defined by offsets	
//		forTestMetaDataStruct locMeta = allMetas[i];
//
//
//		int fpOffset = fbArgs.metaData.fpOffset.arrP[locMeta.zMeta][locMeta.yMeta][locMeta.xMeta];
//		int fnOffset = fbArgs.metaData.fnOffset.arrP[locMeta.zMeta][locMeta.yMeta][locMeta.xMeta];
//		int fpCount = fbArgs.metaData.fpCount.arrP[locMeta.zMeta][locMeta.yMeta][locMeta.xMeta];
//		int fnCount = fbArgs.metaData.fnCount.arrP[locMeta.zMeta][locMeta.yMeta][locMeta.xMeta];
//
//
//		totalFp += fpCount;
//		totalFn+= fnCount;
//
//
//		//printf("block  %d %d %d fp count %d fncount %d   \n ", locMeta.xMeta, locMeta.yMeta, locMeta.zMeta, locMeta.fpCount, locMeta.fnCount);
//
//		if (fpOffset > 0) {
//				for (int jj = 1; jj < 100000; jj++) {
//					if (isSetArr[fpOffset + jj]==true) {
//					//	printf("fpOffset %d  jj %d \n ", fpOffset, jj);
//
//						if (jj<fpCount) {
//							printf("nnnnnnnnnnn  not correct fp offset  in block  %d %d %d is %d should be %d \n ", locMeta.xMeta, locMeta.yMeta, locMeta.zMeta, jj, fpCount);
//
//						}
//						else {
//						//	printf("tttttttt  correct fp offset  in block  %d %d %d is %d should be %d \n ", locMeta.xMeta, locMeta.yMeta, locMeta.zMeta,  jj, fpCount);
//
//						}
//						break;
//					}
//				}
//			};
//
//		if (fnOffset > 0) {
//			for (int jj = 1; jj < 100000; jj++) {
//				if (isSetArr[fnOffset + jj] == true) {
//					//	printf("fpOffset %d  jj %d \n ", fpOffset, jj);
//
//					if (jj < fnCount) {
//						printf("nnnnnnnnnnn  not correct fn offset  in block  %d %d %d is %d should be %d \n ", locMeta.xMeta, locMeta.yMeta, locMeta.zMeta, jj, fnCount);
//
//					}
//					else {
//						//printf("tttttttt  correct fn offset  in block  %d %d %d is %d should be %d \n ", locMeta.xMeta, locMeta.yMeta, locMeta.zMeta, jj, fnCount);
//
//					}
//					break;
//				}
//			}
//		};
//		////checking is block active
//
//		int fpActive = fbArgs.metaData.isActiveGold.arrP[locMeta.zMeta][locMeta.yMeta][locMeta.xMeta];
//		int fnActie = fbArgs.metaData.isActiveSegm.arrP[locMeta.zMeta][locMeta.yMeta][locMeta.xMeta];
//
//
//		////if (fpActive) {
//		////	if (fnCount > 0) {
//		////		//printf("tttt  fp correct is active as should be   in block  %d %d %d fpActive  %d fpCount %d \n ", locMeta.xMeta, locMeta.yMeta, locMeta.zMeta, fpActive, fpCount);
//
//		////	}
//		////	else {
//		////		printf("nnnnnnnnnnn fp is not active and should be   in block  %d %d %d fpActive %d fpCount %d \n ", locMeta.xMeta, locMeta.yMeta, locMeta.zMeta, fpActive, fpCount);
//
//
//		////	};
//		////}
//		////if (fnActie) {
//		////	if (fpCount > 0) {
//		////		//printf("tttt fn correct is active as should be   in block  %d %d %d fnActie %d fnCount%d \n ", locMeta.xMeta, locMeta.yMeta, locMeta.zMeta, fnActie, fnCount);
//
//		////	}
//		////	else {
//		////		printf("nnnnnnnnnnn  fn is not active and should be   in block  %d %d %d fnActie %d  fnCount %d  \n ", locMeta.xMeta, locMeta.yMeta, locMeta.zMeta, fnActie, fnCount);
//
//
//		////	};
//		////}
//
//		/// checking work queue
//		bool isInWorkQueue=false;
//		int numb = 0;
//		for (int ji = 0; ji < 10000;ji++) {
//		bool boolX=	fbArgs.metaData.workQueue.arrP[0][0][ji] == locMeta.xMeta;
//		bool boolY=	fbArgs.metaData.workQueue.arrP[0][1][ji] == locMeta.yMeta;
//		bool boolZ=	fbArgs.metaData.workQueue.arrP[0][2][ji] == locMeta.zMeta;
//		//bool boolIsGold=	fbArgs.metaData.workQueue.arrP[0][3][ji] == fpActive;
//		isInWorkQueue = (boolX && boolY && boolZ );
//			if (isInWorkQueue) {
//				numb = ji;
//				break;
//			}
//
//		//bool booliG=	fbArgs.metaData.workQueue.arrP[0][3][ji] == locMeta.is;
//		}
//
//		if (fpActive || fnActie){
//			if (isInWorkQueue) {
//		//	printf("tttt  correct is in work queue  in block  %d %d %d isGold %d \n ", locMeta.xMeta, locMeta.yMeta, locMeta.zMeta, fbArgs.metaData.workQueue.arrP[0][3][numb]);
//
//			}
//			else {
//				printf("nnnnnnnnnnn   not in work queue and should be  in block  %d %d %d isGold %d \n ", locMeta.xMeta, locMeta.yMeta, locMeta.zMeta, fbArgs.metaData.workQueue.arrP[0][3][numb]);
//
//
//			};
//		}
//		
//
//	}
//
//
//	if (fbArgs.metaData.minMaxes.arrP[0][0][7] == totalFp) {
//			printf("tttt  correct number of fp global  \n ");
//	}
//	else {
//		printf("nnnnnnnnnnn   incorrect fp global  is  %d  should be  %d  \n ", (fbArgs.metaData.minMaxes.arrP[0][0][7]), totalFp);
//
//
//	};
//
//	//totalFp += fpCount;
//	//totalFn += fnCount;
//
//
//	//printf("global FP count %d  [%d]\n", minMaxes.arrP[0][0][7], i);
//	//printf("global FN count %d  [%d]\n", minMaxes.arrP[0][0][8], i);
//
//
//}
//
////////////////////////////////main pass single
//
//inline void isDilatatedSingle(int xChange,int yChange, int zChange, int x, int y, int z, uint32_t*** arr,int dbZLength, int zMeta, int maxZmeta, int maxy, int maxx, int xMeta, int yMeta) {
//		int newZmeta = zMeta;
//		int bitPos = z - zMeta * dbZLength;
//		int newBitPos = bitPos + zChange;
//		//correcting for z fringes
//		if (newBitPos== dbZLength) {
//			newZmeta += 1;
//				newBitPos = 0;
//		
//		}
//		if (newBitPos < 0) {
//			newZmeta -= 1;
//			newBitPos = (dbZLength-1) ;
//
//		}
//		int newX = x+ xChange;
//		int newY = y+ yChange;
//		auto str = "";
//		if (xChange == 1) { str =  "look right "; };
//		if (xChange == -1) { str =  "look left "; };
//		if (yChange == 1) { str =  "look anterior "; };
//		if (yChange == -1) { str =  "look posterior "; };
//		if (zChange == 1) { str =  "look down "; };
//		if (zChange == -1) { str =  "look up "; };
//
//
//
//		if (newZmeta>0 && newX>0 && newY>0
//			&& newZmeta < maxZmeta && newX < maxx && newY <maxy) {
//
//			bool newVal = (arr[newZmeta][newY][newX] & (1 << (newBitPos)));
//			if (newVal) {
//				// printf("ttt  found %s dil orig point %d %d %d new point %d %d %d     xMeta %d yMeta %d zMeta %d  \n ", str, x,y,z, newX,newY, newZmeta* dbZLength + newBitPos, xMeta, yMeta, zMeta);
//
//			}
//			else {
//				printf("ffff not found %s  dil orig point %d %d %d new point %d %d %d      xMeta %d yMeta %d zMeta %d  \n ", str, x, y, z, newX, newY, newZmeta * dbZLength + newBitPos, xMeta, yMeta, zMeta);
//
//			}
//
//		}
//
//
//
//	
//}
//
//
//inline void isDilatatedAll(uint32_t*** arr, int x, int y , int z , int dbZLength, int zMeta, int maxZmeta, int maxy, int maxx, uint32_t*** reference, int xMeta, int yMeta) {
//
//
//	//printf("point %d %d %d   in result here %d and in reference %d \n  ", x, y, z, arr[zMeta][y][x], reference[zMeta][y][x]);
//
//	isDilatatedSingle(1, 0, 0, x, y, z, arr, dbZLength, zMeta, maxZmeta, maxy, maxx, xMeta, yMeta);
//	isDilatatedSingle(-1, 0, 0, x, y, z, arr, dbZLength, zMeta, maxZmeta, maxy, maxx, xMeta, yMeta);
//
//	isDilatatedSingle(0, 1, 0, x, y, z, arr, dbZLength, zMeta, maxZmeta, maxy, maxx, xMeta, yMeta);
//	isDilatatedSingle(0, -1, 0, x, y, z, arr, dbZLength, zMeta, maxZmeta, maxy, maxx, xMeta, yMeta);
//
//	isDilatatedSingle(0, 0, 1, x, y, z, arr, dbZLength, zMeta, maxZmeta, maxy, maxx, xMeta, yMeta);
//	isDilatatedSingle(0, 0, -1, x, y, z, arr, dbZLength, zMeta, maxZmeta, maxy, maxx, xMeta, yMeta);
//}
//
//
////1) do we have a correct dilatation for points inside the block
////2) do we have correct dilatations for points on the fringes of the blocks
////3) did metadatablock counters changed as they should
////4) are results that should be added are
////5) does result counter is ok?
////6) are block that should be marked as full are
//  //7) do blocks that should be marked as to be activated are 
//#pragma once
//inline void mainPassKernelTestUnitTests(ForFullBoolPrepArgs<int> fbArgs, forTestPointStruct allPointsA[], forTestMetaDataStruct allMetas[], int pointsNumber, int metasNumber
//	, int dbXLength, int dbYLength, int dbZLength, int maxZmeta, int maxy, int maxx) {
//
//	//1) do we have a correct dilatation for points inside the block
//	//2) do we have correct dilatations for points on the fringes of the blocks
//	for (int i = 0; i < pointsNumber; i++) {
//
//		forTestPointStruct currPoint = allPointsA[i];
//		int bitPos = currPoint.z - currPoint.zMeta * dbZLength;
//
//		//printf("point %d %d %d \n  ", currPoint.x, currPoint.y, currPoint.z);
//		
//		if (currPoint.isGold) {
//			isDilatatedAll(fbArgs.reducedGold.arrP, currPoint.x, currPoint.y, currPoint.z, dbZLength, currPoint.zMeta, maxZmeta, maxy, maxx, fbArgs.reducedGoldRef.arrP, currPoint.xMeta, currPoint.yMeta);
//		}
//		else {
//			isDilatatedAll(fbArgs.reducedSegm.arrP, currPoint.x, currPoint.y, currPoint.z, dbZLength, currPoint.zMeta, maxZmeta, maxy, maxx, fbArgs.reducedSegmRef.arrP, currPoint.xMeta, currPoint.yMeta);
//
//		}	}
//
//
//
//
//	////4) are results that should be added are
//	//for (int i = 0; i < pointsNumber; i++) {
//
//	//	forTestPointStruct currPoint = allPointsA[i];
//
//	//	if (currPoint.shouldBeInResAfterOneDil){
//	//		//printf("point %d %d %d   \n  ", currPoint.x, currPoint.y, currPoint.z);
//
//	//		bool isInRes = false;
//	//		int numb = 0;
//	//		for (int ji = 0; ji < 10000; ji++) {
//	//			bool boolX = fbArgs.metaData.resultList.arrP[0][0][ji] == currPoint.x;
//	//			bool boolY = fbArgs.metaData.resultList.arrP[0][1][ji] == currPoint.y;
//	//			bool boolZ = fbArgs.metaData.resultList.arrP[0][2][ji] == currPoint.z;
//	//			/*	bool boolG = fbArgs.metaData.resultList.arrP[0][3][ji] == currPoint.isGold;
//	//			bool boolI = fbArgs.metaData.resultList.arrP[0][4][ji] == 0;*/
//	//			//bool boolIsGold=	fbArgs.metaData.resultList.arrP[0][3][ji] == fpActive;
//	//			isInRes = (boolX && boolY && boolZ);// && boolG && boolI
//	//			if (isInRes) {
//	//				numb = ji;
//	//				break;
//	//			}
//
//	//			//bool booliG=	fbArgs.metaData.resultList.arrP[0][3][ji] == locMeta.is;
//	//		}
//	//		if (isInRes) {
//	//			printf("tttt  found correct result  in point  %d %d %d isGold %d iteration %d \n ", fbArgs.metaData.resultList.arrP[0][0][numb]
//	//				, fbArgs.metaData.resultList.arrP[0][1][numb], fbArgs.metaData.resultList.arrP[0][2][numb], fbArgs.metaData.resultList.arrP[0][3][numb], fbArgs.metaData.resultList.arrP[0][4][numb]);
//
//	//		}
//	//		else {
//	//		/*	printf("fffffffff  not found result  in point  %d %d %d isGold %d iteration %d \n ", fbArgs.metaData.resultList.arrP[0][0][numb]
//	//				, fbArgs.metaData.resultList.arrP[0][1][numb], fbArgs.metaData.resultList.arrP[0][2][numb], fbArgs.metaData.resultList.arrP[0][3][numb], fbArgs.metaData.resultList.arrP[0][4][numb]);*/
//
//
//	//		};
//	//	}
//	//
//
//
//	//}
//
//	//5) does result counter is ok?
////6) are block that should be marked as full are
//  //7) do blocks that should be marked as to be activated are 
//	for (int i = 0; i < metasNumber; i++) {
//
//		//printf("block  %d %d %d fp count %d fncount %d \n ", locMeta.xMeta, locMeta.yMeta, locMeta.zMeta, locMeta.fpCount, locMeta.fnCount);
//		forTestMetaDataStruct locMeta = allMetas[i];
//
//		int fpCounter = fbArgs.metaData.fpCounter.arrP[locMeta.zMeta][locMeta.yMeta][locMeta.xMeta];
//		int fnCounter = fbArgs.metaData.fnCounter.arrP[locMeta.zMeta][locMeta.yMeta][locMeta.xMeta];
//
//		if (fpCounter == locMeta.fpConterAfterOneDil) {
//			//printf("tttt correct fp counter in block  %d %d %d is %d should be  %d \n ",
//			//	locMeta.xMeta, locMeta.yMeta, locMeta.zMeta, fpCounter, locMeta.fpConterAfterOneDil);
//
//		}
//		else {
//			printf("fff incorrect fp counter in block  %d %d %d is %d should be  %d \n ",
//				locMeta.xMeta, locMeta.yMeta, locMeta.zMeta, fpCounter, locMeta.fpConterAfterOneDil);
//		};
//
//		if (fnCounter == locMeta.fnConterAfterOneDil) {
//				//printf("tttt correct fn counter in block  %d %d %d is %d should be  %d \n ",
//				//	locMeta.xMeta, locMeta.yMeta, locMeta.zMeta, fnCounter, locMeta.fnConterAfterOneDil);
//
//		}
//		else {
//			printf("fff in correct fn counter in block  %d %d %d is %d should be  %d \n ",
//				locMeta.xMeta, locMeta.yMeta, locMeta.zMeta, fnCounter, locMeta.fnConterAfterOneDil);
//		};
//
//	}
//
//
//	/*for (int i = 0; i < metasNumber; i++) {
//
//		printf("block  %d %d %d fp count %d fncount %d \n ", locMeta.xMeta, locMeta.yMeta, locMeta.zMeta, locMeta.fpCount, locMeta.fnCount);
//		forTestMetaDataStruct locMeta = allMetas[i];
//
//		bool isActivatedGold = fbArgs.metaData.isToBeValidatedFp.arrP[locMeta.zMeta][locMeta.yMeta][locMeta.xMeta];
//		int isActivatedSegm = fbArgs.metaData.isToBeValidatedFn.arrP[locMeta.zMeta][locMeta.yMeta][locMeta.xMeta];
//	
//		
//		if (locMeta.isToBeValidatedFpAfterOneIter) {
//			if (isActivatedGold) {
//				printf("tttt correct gold is to be validated  as should be in block  %d %d %d  \n ",
//					locMeta.xMeta, locMeta.yMeta, locMeta.zMeta);
//
//			}
//			else {
//				printf("fff incorrect gold  is to be validated   in block  %d %d %d \n ",
//					locMeta.xMeta, locMeta.yMeta, locMeta.zMeta);
//			};
//		};
//		if (isActivatedSegm) {
//			if (locMeta.isToBeValidatedFnAfterOneIter) {
//				printf("tttt segm is to be validated  as should be in block  %d %d %d  \n ",
//					locMeta.xMeta, locMeta.yMeta, locMeta.zMeta);
//
//			}
//			else {
//				printf("fff  segm is not   to be validated  as should be in block  %d %d %d\n ",
//					locMeta.xMeta, locMeta.yMeta, locMeta.zMeta);
//			};
//		};
//
//	}*/
//
//
//	//printf("\n result aaa \n ");
//
//	//for (int ji = 0; ji < 30; ji++) {
//	//	if (fbArgs.metaData.resultList.arrP[0][0][ji]>0) {
//	//		printf("result  in point  %d %d %d isGold %d iteration %d \n ", fbArgs.metaData.resultList.arrP[0][0][ji]
//	//			, fbArgs.metaData.resultList.arrP[0][1][ji], fbArgs.metaData.resultList.arrP[0][2][ji], fbArgs.metaData.resultList.arrP[0][3][ji], fbArgs.metaData.resultList.arrP[0][4][ji]);
//	//	}
//	//}
//
//
//
//}
//
//#pragma once
//inline void checkAfterSecondDil(ForFullBoolPrepArgs<int> fbArgs, forTestPointStruct allPointsA[], forTestMetaDataStruct allMetas[], int pointsNumber, int metasNumber
//	, int dbXLength, int dbYLength, int dbZLength) {
//
//
//
//	//4) are results that should be added are
//	for (int i = 0; i < pointsNumber; i++) {
//
//		forTestPointStruct currPoint = allPointsA[i];
//
//		if (currPoint.shouldBeInResAfterTwoDil) {
//			//printf("point %d %d %d   \n  ", currPoint.x, currPoint.y, currPoint.z);
//
//			bool isInRes = false;
//			int numb = 0;
//			for (int ji = 0; ji < 10000; ji++) {
//				bool boolX = fbArgs.metaData.resultList.arrP[0][0][ji] == currPoint.x;
//				bool boolY = fbArgs.metaData.resultList.arrP[0][1][ji] == currPoint.y;
//				bool boolZ = fbArgs.metaData.resultList.arrP[0][2][ji] == currPoint.z;
//				/*	bool boolG = fbArgs.metaData.resultList.arrP[0][3][ji] == currPoint.isGold;
//				bool boolI = fbArgs.metaData.resultList.arrP[0][4][ji] == 0;*/
//				//bool boolIsGold=	fbArgs.metaData.resultList.arrP[0][3][ji] == fpActive;
//				isInRes = (boolX && boolY && boolZ);// && boolG && boolI
//				if (isInRes) {
//					numb = ji;
//					break;
//				}
//
//				//bool booliG=	fbArgs.metaData.resultList.arrP[0][3][ji] == locMeta.is;
//			}
//			if (isInRes) {
//				printf("tttt  found correct result after second dil  in point  %d %d %d isGold %d iteration %d  and in point  x %d y %d z %d \n ", fbArgs.metaData.resultList.arrP[0][0][numb]
//					, fbArgs.metaData.resultList.arrP[0][1][numb], fbArgs.metaData.resultList.arrP[0][2][numb], fbArgs.metaData.resultList.arrP[0][3][numb], fbArgs.metaData.resultList.arrP[0][4][numb]
//				, currPoint.x, currPoint.y, currPoint.z);
//
//			}
//			else {
//				printf("fffffffff  not found result  in point  after second dil   %d %d %d isGold %d iteration %d   and in point  x %d y %d z %d  \n ", fbArgs.metaData.resultList.arrP[0][0][numb]
//					, fbArgs.metaData.resultList.arrP[0][1][numb], fbArgs.metaData.resultList.arrP[0][2][numb], fbArgs.metaData.resultList.arrP[0][3][numb], fbArgs.metaData.resultList.arrP[0][4][numb], currPoint.x, currPoint.y, currPoint.z);
//
//
//			};
//		}
//
//
//
//	}
//
//
//}
//
//
//
//
/////// till the end
//
////1) does final iteration number matches max distance
//
////2) do we have all points that do not match in both arrays in result list and with correct coordinates
//#pragma once
//inline void finalCheckTestUnitTests(ForFullBoolPrepArgs<int> fbArgs, forTestPointStruct allPointsA[], forTestMetaDataStruct allMetas[], int pointsNumber, int metasNumber
//	, int dbXLength, int dbYLength, int dbZLength) {
//
//
//	//for (int i = 0; i < pointsNumber; i++) {
//
//	//	forTestPointStruct currPoint = allPointsA[i];
//	//	int bitPos = currPoint.z - currPoint.zMeta * dbZLength;
//
//	//	//printf("point %d %d %d \n  ", currPoint.x, currPoint.y, currPoint.z);
//
//	//	if (currPoint.isGold) {
//	//		isDilatatedAll(fbArgs.reducedGold.arrP, currPoint.x, currPoint.y, currPoint.z, dbZLength, currPoint.zMeta, maxZmeta, maxy, maxx, fbArgs.reducedGoldRef.arrP, currPoint.xMeta, currPoint.yMeta);
//	//	}
//	//	else {
//	//		isDilatatedAll(fbArgs.reducedSegm.arrP, currPoint.x, currPoint.y, currPoint.z, dbZLength, currPoint.zMeta, maxZmeta, maxy, maxx, fbArgs.reducedSegmRef.arrP, currPoint.xMeta, currPoint.yMeta);
//
//	//	}
//	//}
//
//
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
