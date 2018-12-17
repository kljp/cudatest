#define N 1024

#include<stdio.h>
#include<stdlib.h>
#include<iostream>
//#include<curand_kernel.h>

using namespace std;
/*
__device__ int getRand(curandState *s, int a, int b){

	float rand_int = curand_uniform(s);
	rand_int = rand_int * (b - a) + a;

	return rand_int;
}
*/
__global__ void add_array(int *a, int *b, int *c){

	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;

	c[i * N + j] = a[i * N + j] + b[i * N + j];

	printf("%d %d %d     index: %2d, %2d      block: %2d, %2d      thread: %2d, %2d\n", a[i * N + j], b[i * N + j], c[i * N + j], i, j, blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y);

}

__global__ void build_array(int *a, int *b, int *c){

	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
/*
	int k = blockIdx.x * blockDim.y + threadIdx.x;
	unsigned int seed[3];
	seed[0] = i;
	seed[1] = j;
	seed[2] = k;
	curandState s[3];
	for(int v = 0; v < 2; v++)
		curand_init(seed[v], 0, 0, &s[v]);

	a[i * N + j] = getRand(&s[0], 0, 10);
	b[i * N + j] = getRand(&s[1], 0, 10);
	c[i * N + j] = getRand(&s[2], 0, 10);
*/
	a[i * N + j] = 1;
	b[i * N + j] = 2;
	c[i * N + j] = 3;

	printf("%d %d %d		index: %2d, %2d		block: %2d, %2d		thread: %2d, %2d\n", a[i * N + j], b[i * N + j], c[i * N + j], i, j, blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y);
}

void random_ints(int *array, int size){

	int i;
	for(i = 0; i < size; i++)
		array[i] = rand() % 10;
}

int main()
{
	int *a, *b, *c;
	int *gpu_a, *gpu_b, *gpu_c;

	a = (int *) malloc(sizeof(int) * N * N);
	b = (int *) malloc(sizeof(int) * N * N);
	c = (int *) malloc(sizeof(int) * N * N);

	cudaMalloc((void **) &gpu_a, sizeof(int) * N * N);
	cudaMalloc((void **) &gpu_b, sizeof(int) * N * N);
	cudaMalloc((void **) &gpu_c, sizeof(int) * N * N);

	cudaMemcpy(gpu_a, a, sizeof(int) * N * N, cudaMemcpyHostToDevice);
	cudaMemcpy(gpu_b, b, sizeof(int) * N * N, cudaMemcpyHostToDevice);
	cudaMemcpy(gpu_c, c, sizeof(int) * N * N, cudaMemcpyHostToDevice);

	dim3 threadsPerBlock(16, 16);
	dim3 numBlocks(N / threadsPerBlock.x, N / threadsPerBlock.y);
	build_array<<<numBlocks, threadsPerBlock>>>(gpu_a, gpu_b, gpu_c);

	cudaMemcpy(c, gpu_c, sizeof(int) * N * N, cudaMemcpyDeviceToHost);
/*
	for(int i = 0; i < N; i++)
	{
		for(int j = 0; j < N; j++)
			printf("%d		index: %d, %d\n", c[i * N + j], i, j);
	}
*/
	free(a);
	free(b);
	free(c);

	cudaFree(gpu_a);
	cudaFree(gpu_b);
	cudaFree(gpu_c);

	return 0;
}
