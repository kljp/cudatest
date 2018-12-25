#ifndef CUDA_PRAC_H
#define CUDA_PRAC_H

#define CHECK(call)																			\
{																									\
	const cudaError_t error = call;														\
	if(error != cudaSuccess)																\
	{																								\
		printf("Error: %s: %d, ", __FILE__, __LINE__);								\
		printf("code: %d, reason: %s\n", error, cudaGetErrorString(error));	\
		exit(-10 * error);																	\
	}																								\
}

void checkResult(float *hostRef, float *gpuRef, const int N);
void initialData(float *ip, int size);
double cpuSecond();

#endif

