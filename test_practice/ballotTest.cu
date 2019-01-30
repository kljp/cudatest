#include <cuda_runtime.h>
#include <stdio.h>

__global__ void ballot_test(){

	int warpID = threadIdx.x / 32;
	int laneID = threadIdx.x % 32;

	printf("%8X from thread %2d of warp %2d\n", __ballot_sync(0xEEEEEEEE, laneID % 2 == 0), laneID, warpID);
}

__global__ void ffs_test(){

	int warpID = threadIdx.x / 32;
	int laneID = threadIdx.x % 32;

	printf("%d from thread %2d of warp %2d\n", __ffs(0x80000000), laneID, warpID);
}

int main(){

	//ballot_test <<<1, 32 * 4>>> ();
	ffs_test <<<1, 32>>> ();

	cudaDeviceReset();

	return 0;
}
