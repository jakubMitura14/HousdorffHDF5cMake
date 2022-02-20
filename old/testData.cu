#include "UnitTestUtils.cu"


forTestPointStruct* getTestPoints(int dbXLength, int dbYLength, int dbZLength, int metaXLength, int MetaYLength, int MetaZLength, int& pointsNumberRef) {
	forTestPointStruct allPointsA[] = {
		// meta 2,2,2 only gold points not in result after 2 dilataions
	getTestPoint(
	2,5,8//x,y,z
	,true//isGold
	,2,2,2//xMeta,yMeta,Zmeta
	,dbXLength,dbYLength,dbZLength,pointsNumberRef)
	,getTestPoint(
	3,3,9//x,y,z
	,true//isGold
	,2,2,2//xMeta,yMeta,Zmeta
	,dbXLength,dbYLength,dbZLength,pointsNumberRef)
	,getTestPoint(
	1,5,3//x,y,z
	,true//isGold
	,2,2,2//xMeta,yMeta,Zmeta
	,dbXLength,dbYLength,dbZLength,pointsNumberRef)
		// block 0 corner 0 
		,getTestPoint(
		0,0,1//x,y,z
		,true//isGold
		,0,0,0//xMeta,yMeta,Zmeta
		,dbXLength,dbYLength,dbZLength,pointsNumberRef,false, true)
		//lpwer right corner	
		,getTestPoint(
		dbXLength - 2,dbYLength - 2,dbZLength - 2//x,y,z
		,true//isGold
		,metaXLength - 1,MetaYLength - 1,MetaZLength - 1//xMeta,yMeta,Zmeta
		,dbXLength,dbYLength,dbZLength,pointsNumberRef)
		// block 0 corner 0 
		,getTestPoint(
		0,0,0//x,y,z
		,false//isGold
		,0,0,0//xMeta,yMeta,Zmeta
		,dbXLength,dbYLength,dbZLength,pointsNumberRef,false, true)
		//lpwer right corner	
		,getTestPoint(
		dbXLength - 2,dbYLength - 2,dbZLength - 2//x,y,z
		,false//isGold
		,metaXLength - 1,MetaYLength - 1,MetaZLength - 1//xMeta,yMeta,Zmeta
		,dbXLength,dbYLength,dbZLength,pointsNumberRef)
		//// some overlapping  voxels - should lead to dilatation but not add to fp or fn 	
		,getTestPoint(
		5,6,7//x,y,z
		,false//isGold
		,3,4,6//xMeta,yMeta,Zmeta
		,dbXLength,dbYLength,dbZLength,pointsNumberRef,true
		)
		,getTestPoint(
		9,11,7//x,y,z
		,false//isGold
		,3,4,6//xMeta,yMeta,Zmeta
		,dbXLength,dbYLength,dbZLength,pointsNumberRef,true)	,



		//now some points that should be covered by first dilatation		
		getTestPoint(
		9,11,7//x,y,z
		,false//isGold
		,7,4,6//xMeta,yMeta,Zmeta
		,dbXLength,dbYLength,dbZLength,pointsNumberRef,false, true)

		,getTestPoint(
		9,11,8//x,y,z
		,true//isGold
		,7,4,6//xMeta,yMeta,Zmeta
		,dbXLength,dbYLength,dbZLength,pointsNumberRef,false, true)


		,getTestPoint(
		9,3,7//x,y,z
		,false//isGold
		,7,4,6//xMeta,yMeta,Zmeta
		,dbXLength,dbYLength,dbZLength,pointsNumberRef,false, true)
		,getTestPoint(
		9,2,7//x,y,z
		,true//isGold
		,7,4,6//xMeta,yMeta,Zmeta
		,dbXLength,dbYLength,dbZLength,pointsNumberRef,false, true)

		,getTestPoint(
		9,5,7//x,y,z
		,false//isGold
		,7,4,6//xMeta,yMeta,Zmeta
		,dbXLength,dbYLength,dbZLength,pointsNumberRef,false, true)
		,getTestPoint(
		9,6,7//x,y,z
		,true//isGold
		,7,4,6//xMeta,yMeta,Zmeta
		,dbXLength,dbYLength,dbZLength,pointsNumberRef,false, true)

		,getTestPoint(
		2,3,7//x,y,z
		,false//isGold
		,7,4,6//xMeta,yMeta,Zmeta
		,dbXLength,dbYLength,dbZLength,pointsNumberRef,false, true)
		,getTestPoint(
		3,3,7//x,y,z
		,true//isGold
		,7,4,6//xMeta,yMeta,Zmeta
		,dbXLength,dbYLength,dbZLength,pointsNumberRef, false, true)



		//now some points that should be covered by second dilatation		
		,getTestPoint(
		9,11,7//x,y,z
		,false//isGold
		,7,2,6//xMeta,yMeta,Zmeta
		,dbXLength,dbYLength,dbZLength,pointsNumberRef, false, false, true)
		,getTestPoint(
		9,11,9//x,y,z
		,true//isGold
		,7,2,6//xMeta,yMeta,Zmeta
		,dbXLength,dbYLength,dbZLength,pointsNumberRef, false, false, true)


		,getTestPoint(
		9,3,7//x,y,z
		,false//isGold
		,7,2,6//xMeta,yMeta,Zmeta
		,dbXLength,dbYLength,dbZLength,pointsNumberRef, false, false, true)
		,getTestPoint(
		9,1,7//x,y,z
		,true//isGold
		,7,2,6//xMeta,yMeta,Zmeta
		,dbXLength,dbYLength,dbZLength,pointsNumberRef, false, false, true)	,



		/*now specifically we will get some points on the borders  to establish if they dilatate properly





		//top*/
			getTestPoint(
		2,2,0//x,y,z
		,false//isGold
		,0,0,2//xMeta,yMeta,Zmeta
		,dbXLength,dbYLength,dbZLength,pointsNumberRef),


		////top*/
		//getTestPoint(
		//	2, 2, 1//x,y,z
		//	, true//isGold
		//	, 0, 0, 2//xMeta,yMeta,Zmeta
		//	, dbXLength, dbYLength, dbZLength, pointsNumberRef),


		////getTestPoint(
		////	2, 2, 9//x,y,z
		////	, false//isGold
		////	, 0, 0, 2//xMeta,yMeta,Zmeta
		////	, dbXLength, dbYLength, dbZLength, pointsNumberRef),
		//getTestPoint(
		//	2, 2, 15//x,y,z
		//	, true//isGold
		//	, 0, 0, 2//xMeta,yMeta,Zmeta
		//	, dbXLength, dbYLength, dbZLength, pointsNumberRef),

		//getTestPoint(
		//	2, 2, 19//x,y,z
		//	, true//isGold
		//	, 0, 0, 2//xMeta,yMeta,Zmeta
		//	, dbXLength, dbYLength, dbZLength, pointsNumberRef),


//bottom	
	getTestPoint(
2,2,dbZLength - 1//x,y,z
,false//isGold
,0,0,4//xMeta,yMeta,Zmeta
,dbXLength,dbYLength,dbZLength,pointsNumberRef)	,

//left	
	getTestPoint(
0,2,2//x,y,z
,false//isGold
,8,0,6//xMeta,yMeta,Zmeta
,dbXLength,dbYLength,dbZLength,pointsNumberRef)	,

//right	
	getTestPoint(
dbXLength - 1,3,7//x,y,z
,false//isGold
,0,0,8//xMeta,yMeta,Zmeta
,dbXLength,dbYLength,dbZLength,pointsNumberRef)


//anterior	
	,getTestPoint(
9,dbYLength - 1,7//x,y,z
,false//isGold
,0,0,10//xMeta,yMeta,Zmeta
,dbXLength,dbYLength,dbZLength,pointsNumberRef)

//posterior	
	,getTestPoint(
9,0,7//x,y,z
,false//isGold
,2,2,4//xMeta,yMeta,Zmeta
,dbXLength,dbYLength,dbZLength,pointsNumberRef)






////top
//getTestPoint(
//	2, 2, 1//x,y,z
//	, true//isGold
//	, 0, 0, 5//xMeta,yMeta,Zmeta
//	, dbXLength, dbYLength, dbZLength, pointsNumberRef)



//////bottom	
////, getTestPoint(
////	2, 9, 7//x,y,z
////	, true//isGold
////	, 0, 0, 11//xMeta,yMeta,Zmeta
////	, dbXLength, dbYLength, dbZLength, pointsNumberRef)

//////left	
////, getTestPoint(
////	2, 2, 2//x,y,z
////	, true//isGold
////	, 0, 0, 11//xMeta,yMeta,Zmeta
////	, dbXLength, dbYLength, dbZLength, pointsNumberRef)

//////right	
////, getTestPoint(
////	2, 3, 7//x,y,z
////	, true//isGold
////	, 0, 0, 11//xMeta,yMeta,Zmeta
////	, dbXLength, dbYLength, dbZLength, pointsNumberRef)


//////anterior	
////, getTestPoint(
////	9, 2, 7//x,y,z
////	, true//isGold
////	, 0, 1, 7//xMeta,yMeta,Zmeta
////	, dbXLength, dbYLength, dbZLength, pointsNumberRef)

//////posterior	
////, getTestPoint(
////	9, 2, 7//x,y,z
////	, true//isGold
////	, 2, 1, 7//xMeta,yMeta,Zmeta
////	, dbXLength, dbYLength, dbZLength, pointsNumberRef)





//left up anterior corner	
	,getTestPoint(
0,dbYLength - 1,0//x,y,z
,false//isGold
,2,2,6//xMeta,yMeta,Zmeta
,dbXLength,dbYLength,dbZLength,pointsNumberRef)


//right up anterior corner	
	,getTestPoint(
dbXLength - 1,dbYLength - 1,0//x,y,z
,false//isGold
,2,2,8//xMeta,yMeta,Zmeta
,dbXLength,dbYLength,dbZLength,pointsNumberRef)

//left down anterior corner	
//        ,getTestPoint(
//	0,dbYLength-1,dbZLength-1//x,y,z
//	,false//isGold
//	,2,2,10//xMeta,yMeta,Zmeta
//	dbXLength,dbYLength,dbZLength,pointsNumberRef)			
   //	
//	//right dow anterior  corner	
//        ,getTestPoint(
//	dbXLength-1,dbYLength-1,dbZLength-1//x,y,z
//	,false//isGold
//	,4,4,2//xMeta,yMeta,Zmeta
//	dbXLength,dbYLength,dbZLength,pointsNumberRef)			
   //	

   //	
   //	
   //	
//	//left up posterior corner	
//        ,getTestPoint(
//	0,0,0//x,y,z
//	,false//isGold
//	,4,4,4//xMeta,yMeta,Zmeta
//	dbXLength,dbYLength,dbZLength,pointsNumberRef)	
   //	
   //	
//	//right up posterior corner	
//        ,getTestPoint(
//	dbXLength-1,0,0//x,y,z
//	,false//isGold
//	,7,2,6//xMeta,yMeta,Zmeta
//	dbXLength,dbYLength,dbZLength,pointsNumberRef)	
   //	
//	//left down posterior corner	
//        ,getTestPoint(
//	0,0,dbZLength-1//x,y,z
//	,false//isGold
//	,7,2,6//xMeta,yMeta,Zmeta
//	dbXLength,dbYLength,dbZLength,pointsNumberRef)			
   //	
   //right dow posterior  corner	
	   ,getTestPoint(
   dbXLength - 1,0,dbZLength - 1//x,y,z
   ,false//isGold
   ,4,4,4//xMeta,yMeta,Zmeta
   ,dbXLength,dbYLength,dbZLength,pointsNumberRef)

		//rshould be activated aafter two dilatations
			,getTestPoint(
		1,1,1//x,y,z
		,false//isGold
		,4,4,8//xMeta,yMeta,Zmeta
		,dbXLength,dbYLength,dbZLength,pointsNumberRef)
	};
	return allPointsA;
}

	//	////now we neeed additionally to supply some block that will be full
	//	// 5,5,5 full at start	
	//	for (i = 0; i < dbXLength; i++) {
	//		for (j = 0; j < dbYLength; j++) {
	//			for (k = 0; k < dbZLength; k++) {
	//				setArrCPU(arrSegmObj, dbXLength * 5 + i, dbYLength * 5 + j, dbZLength * 5 + k, 2);
	//			}
	//		}
	//	};

	//	// 5,7,7 full after one dil
	//	for (i = 1; i < dbXLength; i++) {
	//		for (j = 0; j < dbYLength; j++) {
	//			for (k = 0; k < dbZLength; k++) {
	//				setArrCPU(arrSegmObj, dbXLength * 5 + i, dbYLength * 7 + j, dbZLength * 7 + k, 2);
	//			}
	//		}
	//	};

	//	// 5,7,10 full after two dil
	//	for (i = 2; i < dbXLength - 2; i++) {
	//		for (j = 0; j < dbYLength; j++) {
	//			for (k = 0; k < dbZLength; k++) {
	//				setArrCPU(arrSegmObj, dbXLength * 5 + i, dbYLength * 7 + j, dbZLength * 10 + k, 2);
	//			}
	//		}
	//	};


	//	//getMetdataTestStruct(2, 2, 2, fnCount = 3, isToBeValidatedFnAfterOneIter = true, isToBeValidatedFnAfterTwoIter = true);






	//	forTestMetaDataStruct a1 = getMetdataTestStruct(3, 4, 6);
	//	a1.isToBeValidatedFpAfterOneIter = false;
	//	a1.isToBeValidatedFpAfterTwoIter = false;
	//	a1.isToBeValidatedFnAfterOneIter = false;
	//	a1.isToBeValidatedFnAfterTwoIter = false;

	//	forTestMetaDataStruct a2 = getMetdataTestStruct(7, 4, 6);
	//	a2.fnCount = 4;
	//	a2.fpCount = 4;
	//	a2.fpConterAfterOneDil = 4;
	//	a2.fnConterAfterOneDil = 4;

	//	a2.fpConterAfterTwoDil = 4;
	//	a2.fnConterAfterTwoDil = 4;

	//	a2.isToBeValidatedFpAfterOneIter = false;
	//	a2.isToBeValidatedFpAfterTwoIter = false;
	//	a2.isToBeValidatedFnAfterOneIter = false;
	//	a2.isToBeValidatedFnAfterTwoIter = false;

	//	forTestMetaDataStruct a3 = getMetdataTestStruct(7, 2, 6);
	//	a2.fnCount = 2;
	//	a2.fpCount = 2;
	//	a2.fpConterAfterOneDil = 0;
	//	a2.fnConterAfterOneDil = 0;

	//	a2.fpConterAfterTwoDil = 2;
	//	a2.fnConterAfterTwoDil = 2;

	//	a2.isToBeValidatedFpAfterOneIter = true;
	//	a2.isToBeValidatedFpAfterTwoIter = true;
	//	a2.isToBeValidatedFnAfterOneIter = false;
	//	a2.isToBeValidatedFnAfterTwoIter = false;

	//	forTestMetaDataStruct full1 = getMetdataTestStruct(5, 7, 7);
	//	full1.isToBeFullAfterOneIter = true;

	//	forTestMetaDataStruct full2 = getMetdataTestStruct(5, 5, 5);
	//	full2.isToBeFullAfterOneIter = true;
	//	forTestMetaDataStruct full3 = getMetdataTestStruct(5, 7, 10);
	//	full3.isToBeFullAfterTwoIter = true;



	//	forTestMetaDataStruct allMetas[] = {
	//	getMetdataTestStruct(2,2,2,  0, 3)
	//		,getMetdataTestStruct(0,0,0,1,1)
	//			,getMetdataTestStruct(metaXLength - 1,MetaYLength - 1,MetaZLength - 1 , 1,1)
	//		,a1// should not be validated at all
	//		,a2//now some points that should be covered by second dilatation after one dilatation no need to validate it
	//		,a3 //now some points that should be covered by second dilatation after one dilatation no need to validate it

	//		,getMetdataTestStruct(0,0,2, 1)
	//		,getMetdataTestStruct(0,0,1,0,0,false,true)//just marking it get activated	
	//		,getMetdataTestStruct(0,0,4, 1)
	//		,getMetdataTestStruct(0,0,5,0,0,false,true)//just marking it get activated	
	//		,getMetdataTestStruct(8,0,6, 1)
	//		,getMetdataTestStruct(7,0,6,0,0,false,true)//just marking it get activated	

	//		,getMetdataTestStruct(0,0,8, 1)
	//		,getMetdataTestStruct(1,0,8,0,0,false,true)//just marking it get activated	
	//		,getMetdataTestStruct(0,0,10, 1)
	//		,getMetdataTestStruct(0,1,10 ,0,0,false,true)//just marking it get activated			
	//		,getMetdataTestStruct(2,2,4, 1)
	//		,getMetdataTestStruct(2,1,4,0,0,false,true)//just marking it get activated			
	//		,getMetdataTestStruct(2,2,6, 1)
	//		,getMetdataTestStruct(1,2,6,0,0,false,true)//just marking it get activated			
	//		,getMetdataTestStruct(2,2,5,0,0,false,true)//just marking it get activated			
	//		,getMetdataTestStruct(2,3,6,0,0,false,true)//just marking it get activated		

	//				//right dow posterior  corner	
	//		,getMetdataTestStruct(4,4,4, 1)
	//		,getMetdataTestStruct(5,4,4,0,0,false,true)//just marking it get activated			
	//		,getMetdataTestStruct(4,4,5,0,0,false,true)//just marking it get activated			
	//		,getMetdataTestStruct(4,3,4,0,0,false,true)//just marking it get activated		

	//		,getMetdataTestStruct(4,4,8, 1)
	//		,getMetdataTestStruct(3,4,8,0,0,false,true)//just marking it get activated			
	//		,getMetdataTestStruct(4,3,8,0,0,false,true)//just marking it get activated			
	//		,getMetdataTestStruct(4,4,7,0,0,false,true)//just marking it get activated		

	//		// 5,7,7 full after one dil
	//			// 5,5,5 full at start	
	//					// 5,7,10 full after two dil
	//	,full1, full2, full3
	//	};

	//}