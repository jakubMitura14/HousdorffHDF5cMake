#include "cuda_runtime.h"
#include <cstdint>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
using namespace cooperative_groups;

/*
We will  here return true if the thread is at the moment active - Hovewer sometimes we have couple things to do so we will add additional numb
untlin numb will not execute number of available threads in a group it will be executed on this thread in other case we will use some already used thread ...
*/

#pragma once
inline __device__ bool isToBeExecutedOnActive(coalesced_group group, int numb, int metaNumb=0) {
    return ((threadIdx.x == numb) && (threadIdx.y == metaNumb));
    //if ((numb < group.num_threads()) && (threadIdx.x == numb) && (threadIdx.y == metaNumb) ) {
    //    return true;
    //}
    //else {// defoult is first thread in group
    //    if (threadIdx.y == 0 && (threadIdx.y == metaNumb)) {
    //        return true;
    //    }
    //};
    //return false;


}
