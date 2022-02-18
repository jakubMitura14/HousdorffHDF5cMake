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
void printIsBlockActiveEtc(ForBoolKernelArgs<int> fbArgs,  uint32_t* metaDataArr,MetaDataGPU metaData) {
//testing dilatations
	printf("\n********* printIsBlockActiveEtc %d ***************\n", fbArgs.metaData.totalMetaLength);

	for (uint32_t linIdexMeta = 0; linIdexMeta <= fbArgs.metaData.totalMetaLength; linIdexMeta += 1) {
		//we get from linear index  the coordinates of the metadata block of intrest
		uint8_t xMeta = linIdexMeta % fbArgs.metaData.metaXLength;
		uint8_t zMeta = floor((float)(linIdexMeta / (fbArgs.metaData.metaXLength * fbArgs.metaData.MetaYLength)));
		uint8_t yMeta = floor((float)((linIdexMeta - ((zMeta * fbArgs.metaData.metaXLength * fbArgs.metaData.MetaYLength) + xMeta)) / fbArgs.metaData.metaXLength));
// 7 :isActiveGold
// 8 :isFullGold
// 9 :isActiveSegm
// 10 :isFullSegm
// 11 :isToBeActivatedGold
// 12 :isToBeActivatedSegm
	if( metaDataArr[linIdexMeta * metaData.metaDataSectionLength + 7] >0 ){
		printf("is active gold linMeta %d xMeta %d yMeta %d zMeta %d   \n", linIdexMeta, xMeta, yMeta, zMeta);
	}
	if( metaDataArr[linIdexMeta * metaData.metaDataSectionLength + 8] ==1 ){
		printf("is full gold linMeta %d xMeta %d yMeta %d zMeta %d   \n", linIdexMeta, xMeta, yMeta, zMeta);
	}
	if( metaDataArr[linIdexMeta * metaData.metaDataSectionLength + 9] ==1 ){
		printf("is active segm linMeta %d xMeta %d yMeta %d zMeta %d   \n", linIdexMeta, xMeta, yMeta, zMeta);
	}
	if( metaDataArr[linIdexMeta * metaData.metaDataSectionLength + 10] ==1 ){
		printf("is full segm linMeta %d xMeta %d yMeta %d zMeta %d   \n", linIdexMeta, xMeta, yMeta, zMeta);
	}
	if( metaDataArr[linIdexMeta * metaData.metaDataSectionLength + 11] ==1 ){
		printf("is to be activated gold linMeta %d xMeta %d yMeta %d zMeta %d   \n", linIdexMeta, xMeta, yMeta, zMeta);
	}	
	if( metaDataArr[linIdexMeta * metaData.metaDataSectionLength + 12] ==1 ){
		printf("is to be activated segm linMeta %d xMeta %d yMeta %d zMeta %d   \n", linIdexMeta, xMeta, yMeta, zMeta);
	}
	
	}

	printf("\n********* end ***************\n");


}



void testDilsSinglePoint(forTestPointStruct*& points, int x, int y, int z, int& pointsNumberRef, bool isGold) {
	for (int i = 0; i < pointsNumberRef; i++) {
		forTestPointStruct point = points[i];
		if ((point.isGold && isGold) || (!point.isGold && !isGold)) {
			//this point
			if (point.x == (x) && point.y == (y) && point.z == (z)) {
				point.isFoundAndDilatated = true;
			}
			//top
			if (point.x == (x) && point.y == (y) && point.z - 1 == (z)) {
				point.isFoundDilTop = true;
			}
			//bottom
			if (point.x == (x) && point.y == (y) && point.z + 1 == (z)) {
				point.isFoundDilBottom = true;
			}


			//anterior
			if (point.x == (x) && point.y - 1 == (y) && point.z == (z)) {
				point.isFoundDilAnterior = true;
			}
			//posterior
			if (point.x == (x) && point.y + 1 == (y) && point.z == (z)) {
				point.isFoundDilPosterior = true;
			}


			//left
			if (point.x + 1 == (x) && point.y == (y) && point.z == (z)) {
				point.isFoundDilLeft = true;
			}
			//right
			if (point.x - 1 == (x) && point.y == (y) && point.z == (z)) {
				point.isFoundDilRight = true;
			}
		}
	}



}




#pragma once
void testDilatations (ForBoolKernelArgs<int> fbArgs, uint32_t* arrsCPU,forTestPointStruct* pointsList, int& pointsNumberRef) {
//testing dilatations
	
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
						int locPosB = locPos - 32 * fbArgs.dbYLength;

						int x = locPosB % 32 + xMeta * fbArgs.dbXLength;
						int y = int(floor((float)(locPosB / 32)) + yMeta * fbArgs.dbYLength);
						int z = bitPos + zMeta * fbArgs.dbZLength;
							testDilsSinglePoint(pointsList, x, y, z, pointsNumberRef, true);
						
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
						int x = locPosB % 32 + xMeta * fbArgs.dbXLength;
						int y = int(floor((float)(locPosB / 32)) + yMeta * fbArgs.dbYLength);
						int z = bitPos + zMeta * fbArgs.dbZLength;
						testDilsSinglePoint(pointsList, x,y,z,pointsNumberRef,false );
					}
				}
			}
		}
	}
	
	//by now we checked wheather a point is present in dilatation array now we need to also establish is dilatation that we are looking for is in range
	for(int i =0; i<pointsNumberRef;i++ ){
		forTestPointStruct point= pointsList[i];
		//this point
		if( !point.isFoundAndDilatated){
			printf(" point %d %d %d not found at all \n", point.x, point.y, point.z);
			}
		//top
		if(!point.isFoundDilTop  &&  point.z>0 ){
			printf(" point %d %d %d not found TOP dilatation \n", point.x, point.y, point.z);
		}	
		//bottom
		if(!point.isFoundDilBottom && point.z< (fbArgs.metaData.MetaZLength*32 -1) ){
			printf(" point %d %d %d not found BOTTOM dilatation \n", point.x, point.y, point.z);
		}	
		
		
		//anterior
		if(!point.isFoundDilAnterior && point.y <(fbArgs.metaData.MetaYLength*fbArgs.dbYLength -1)  ){
			printf(" point %d %d %d not found ANTERIOR dilatation \n", point.x, point.y, point.z);
		}	
		//posterior
		if(!point.isFoundDilPosterior && point.y>0){
			printf(" point %d %d %d not found POSTERIOR dilatation \n", point.x, point.y, point.z);
		}	
		
				
		//left
		if( !point.isFoundDilLeft && point.x>0 ){
			printf(" point %d %d %d not found LEFT dilatation \n", point.x, point.y, point.z);
		}	
		//right
		if( !point.isFoundDilRight && point.x< (fbArgs.metaData.metaXLength*32 -1)  ){
			printf(" point %d %d %d not found RIGHT dilatation \n", point.x, point.y, point.z);
		}	
	}
		
	
}




/*
so we will look through result list and  given the information about wheather it 
is first or second dilatation we would establish weather all results that should be present here are 
additionally we will collect the fp counts and fncounts and in the end of the function establish weather fp and fn counts 
of the block in metadata are correct
*/
#pragma once
void testResultsAndCounters(ForBoolKernelArgs<int> fbArgs, uint32_t* arrsCPU,forTestPointStruct* points, int& pointsNumberRef
,uint32_t*& resultListPointerMetaCPU, uint32_t*& resultListPointerLocalCPU, uint32_t*& resultListPointerIterNumb
, int numberOfResults, uint32_t* metaDataArrPointerCPU, MetaDataGPU metaData) {
		
		int* fpCounts = (int*)calloc(fbArgs.metaData.totalMetaLength, sizeof(int));
		int* fnCounts = (int*)calloc(fbArgs.metaData.totalMetaLength, sizeof(int));



	
		for(int iRes=0;iRes< numberOfResults; iRes++){
		if(resultListPointerMetaCPU[iRes]>0){
			uint32_t linIdexMeta = resultListPointerMetaCPU[iRes] - (isGoldOffset * (resultListPointerMetaCPU[iRes] > isGoldOffset))-1;
			uint32_t xMeta = linIdexMeta % fbArgs.metaData.metaXLength;
			uint32_t zMeta = uint32_t(floor((float)(linIdexMeta / (fbArgs.metaData.metaXLength * fbArgs.metaData.MetaYLength))));
			uint32_t yMeta = uint32_t(floor((float)((linIdexMeta - ((zMeta * fbArgs.metaData.metaXLength * fbArgs.metaData.MetaYLength) + xMeta)) / fbArgs.metaData.metaXLength)));
			
			uint32_t linLocal = resultListPointerLocalCPU[iRes];
			uint32_t xLoc = linLocal % fbArgs.dbXLength;
			uint32_t zLoc = uint32_t(floor((float)(linLocal / (32 * fbArgs.dbYLength))));
			uint32_t yLoc = uint32_t(floor((float)((linLocal - ((zLoc * 32 * fbArgs.dbYLength) + xLoc)) / 32)));


			uint32_t x = xMeta * 32 + xLoc;
			uint32_t y= yMeta * fbArgs.dbYLength + yLoc;
			uint32_t z = zMeta * 32 + zLoc;
			
			
			uint32_t iterNumb = resultListPointerIterNumb[iRes];
   // bool shouldBeInResAfterOneDil;
    //bool shouldBeInResAfterTwoDil;
			if((resultListPointerMetaCPU[iRes] > isGoldOffset)){
				fpCounts[linIdexMeta]+=1;
			}
			else{
				fnCounts[linIdexMeta]+=1;
			}
			
				for(int i =0; i<pointsNumberRef;i++ ){
						forTestPointStruct point= points[i];
						if( (point.shouldBeInResAfterOneDil && iterNumb==0)  
							||(point.shouldBeInResAfterTwoDil &&  iterNumb==1)      ){
							point.isFoundInResult=true;
						}
					}
			
		
		}
		}
		for (int i = 0; i < pointsNumberRef; i++) {
			forTestPointStruct point = points[i];
			if ((point.shouldBeInResAfterOneDil || point.shouldBeInResAfterTwoDil) && !point.isFoundInResult) {
				printf("fff point %d %d %d not found in result \n", point.x, point.y, point.z);

			}
			else {
				printf("ttt  point %d %d %d found in result \n", point.x, point.y, point.z);

			}

		};
//checking weather counters in metadata match the count that should be present 		
for(int linIdexMeta=0;linIdexMeta< fbArgs.metaData.totalMetaLength; linIdexMeta++){
if(fpCounts[linIdexMeta]>0){
	if(metaDataArrPointerCPU[ linIdexMeta* metaData.metaDataSectionLength + 3] = fpCounts[linIdexMeta] ){
		printf(" correct fp count %d in linMeta %d  "
		,fpCounts[linIdexMeta]
		,linIdexMeta
		);

	}else{
		printf("fff incorrect fp count %d is in meta %d in linMeta %d  "
		,fpCounts[linIdexMeta]
		,metaDataArrPointerCPU[ linIdexMeta* metaData.metaDataSectionLength + 3]
		,linIdexMeta
		);
	}
}
if(fnCounts[linIdexMeta]>0){
	if(metaDataArrPointerCPU[ linIdexMeta* metaData.metaDataSectionLength + 4] = fnCounts[linIdexMeta] ){
		printf(" correct fn count %d in linMeta %d  "
		,fnCounts[linIdexMeta]
		,linIdexMeta
		);

	}else{
		printf("fff incorrect fn count %d is in meta %d in linMeta %d  "
		,fnCounts[linIdexMeta]
		,metaDataArrPointerCPU[ linIdexMeta* metaData.metaDataSectionLength + 4]
		,linIdexMeta
		);
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
