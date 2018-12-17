#define N 16

#include<stdio.h>
#include<stdlib.h>
#include<iostream>

using namespace std;

__global__ void add(int *a, int *b, int *c){

	c[blockIdx.x] = a[blockIdx.x] + b[blockIdx.x];
	printf("%d		blockIdx=%d\n", c[blockIdx.x], blockIdx.x);
}

__global__ void build_array(int **a, int **b, int **c){

	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;

	c[i][j] = 3;
	printf("%d		index: %2d, %2d		block: %2d, %2d		thread: %2d, %2d\n", c[i][j], i, j, blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y);
}

void random_ints(int *array, int size){

	int i;
	for(i = 0; i < size; i++)
		array[i] = rand() % 10;
}

int main()
{
	int **a, **b, **c;
	int **gpu_a, **gpu_b, **gpu_c;
	
	a = (int **) malloc(sizeof(int *) * N);
	b = (int **) malloc(sizeof(int *) * N);
	c = (int **) malloc(sizeof(int *) * N);

	cudaMalloc((void **) &gpu_a, sizeof(int *) * N);
	cudaMalloc((void **) &gpu_b, sizeof(int *) * N);
	cudaMalloc((void **) &gpu_c, sizeof(int *) * N);

	for(int i = 0; i < N; i++)
	{
		//a[i] = (int *) malloc(sizeof(int) * N);
		//b[i] = (int *) malloc(sizeof(int) * N);
		//c[i] = (int *) malloc(sizeof(int) * N);

		cudaMalloc((void **) &a[i], sizeof(int) * N);
		cudaMalloc((void **) &b[i], sizeof(int) * N);
		cudaMalloc((void **) &c[i], sizeof(int) * N);
	}

	cudaMemcpy(gpu_b, b, sizeof(int *) * N, cudaMemcpyHostToDevice);
	cudaMemcpy(gpu_c, c, sizeof(int *) * N, cudaMemcpyHostToDevice);
	
	dim3 threadsPerBlock(16, 16);
	dim3 numBlocks(N / threadsPerBlock.x, N / threadsPerBlock.y);
	build_array<<<numBlocks, threadsPerBlock>>>(gpu_a, gpu_b, gpu_c);

	cudaMemcpy(c, gpu_c, sizeof(int *) * N, cudaMemcpyDeviceToHost);

	for(int i = 0; i < N; i++)
	{
		for(int j = 0; j < N; j++)
			cout << c[i][j] << endl;//printf("%d		index: %d, %d\n", c[i][j], i, j);
	}
	
	for(int i = 0; i < N; i++)
	{
		cudaFree(a[i]);
		cudaFree(b[i]);
		cudaFree(c[i]);
	}

	free(a);
	free(b);
	free(c);

	cudaFree(gpu_a);
	cudaFree(gpu_b);
	cudaFree(gpu_c);

	return 0;
}
