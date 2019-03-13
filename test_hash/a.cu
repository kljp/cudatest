#include <include>
#include <cuda_runtime.h>

__global__ void asd(){

	printf("a\n");
}

int main(){

	int dev = 0;
	cudaSetDevice(dev);
	asd <<<1, 32>>> ():
	cudeReset();
	return 0;
}
