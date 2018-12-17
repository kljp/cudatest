#define N 16

#include<stdio.h>
#include<stdlib.h>

__global__ void add(int *a, int *b, int *c){

	c[blockIdx.x] = a[blockIdx.x] + b[blockIdx.x];
	printf("%d		blockIdx=%d\n", c[blockIdx.x], blockIdx.x);
}

void random_ints(int *array, int size){

	int i;
	for(i = 0; i < size; i++)
		array[i] = rand() % 10;
}

int main()
{
	int *a, *b, *c;
	int *d_a, *d_b, *d_c;
	int size = N * sizeof(int);
	int i;

	a = (int *) malloc(size);
	b = (int *) malloc(size);
	c = (int *) malloc(size);

	random_ints(a, N);
	random_ints(b, N);
	random_ints(c, N);

	cudaMalloc((void **) &d_a, size);
	cudaMalloc((void **) &d_b, size);
	cudaMalloc((void **) &d_c, size);

	cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

	add<<<N, 1>>>(d_a, d_b, d_c);

	cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);

	printf("\n\n\n");

	for(i = 0; i < N; i++)
		printf("%d		index=%d\n", c[i], i);

	free(a);
	free(b);
	free(c);

	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);

	return 0;
}
