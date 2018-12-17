#define N 96

#include <stdio.h>

__global__ void helloFromGPU(){

	printf("HelloFrom GPU! %d %d\n", blockIdx.x, threadIdx.x);
}

int main()
{
	printf("Hello World from CPU\n");
	helloFromGPU<<<3, 32>>>();
	cudaDeviceReset();

	return 0;
}
