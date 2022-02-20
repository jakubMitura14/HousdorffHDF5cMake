//
//#include "cuda_runtime.h"
//#include <cstdint>
//#include <cooperative_groups.h>
//#include <cooperative_groups/reduce.h>
//namespace cg = cooperative_groups;
///// The following example accepts input in *A and outputs a result into *sum
///// It spreads the data within the block, one element per thread
//template <typename TYU>
//inline __device__ void block_reduce(const int* A, int* sum, thread_block cta, thread_block_tile<32> tile) {
//	__shared__ int reduction_s[32];
//	//cg::thread_block cta = cg::this_thread_block();
//	//cg::thread_block_tile<32> tile = cg::tiled_partition<32>(cta);
//	const int tid = cta.thread_rank();
//	// reduce across the tile
//	// cg::plus<int> allows cg::reduce() to know it can use hardware acceleration
//
//	reduction_s[tid] = cg::reduce(tile, beta, cg::plus<int>());
//	// synchronize the block so all data is ready
//	cg::sync(cta);
//	// single leader accumulates the result
//	if (cta.thread_rank() == 0) {
//		beta = 0;
//		for (int i = 0; i < blocksz; i += tile.num_threads()) {
//			beta += reduction_s[i];
//		}
//
//		sum[blockIdx.x] = beta;
//	}
//}
