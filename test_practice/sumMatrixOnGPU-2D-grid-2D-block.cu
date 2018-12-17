#include <cuda_runtime.h>
#include <stdio.h>
#include "cuda_prac.h"

void sumMatrixOnHost(float *A, float *B, float *C, const int nx, const int ny){

	float *ia = A;
	float *ib = B;
	float *ic = C;

	for(int iy = 0; iy < ny; iy++)
	{
		for(int ix = 0; ix < nx; ix++)
				ic[ix] = ia[ix] + ib[ix];

		ia += nx;
		ib += nx;
		ic += nx;
	}
}

__global__ void sumMatrixOnGPU2D(float *MatA, float *MatB, float *MatC, int nx, int ny){

	unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y;
	unsigned int idx = iy * nx + ix;

	if(ix < nx && iy < ny)
		MatC[idx] = MatA[idx] + MatB[idx];
}

int main(int argc, char** argv){

	printf("%s Starting...\n", argv[0]);

	int dev = 0;
	cudaDeviceProp deviceProp;
	CHECK(cudaGetDeviceProperties(&deviceProp, dev));
	printf("Using Device %d: %s\n", dev, deviceProp.name);
	CHECK(cudaSetDevice(dev));
	
	int nx = 2048;
	int ny = 2048;

	int nxy = nx * ny;
	int nBytes = nxy * sizeof(float);
	printf("Matrix size: nx %d ny %d\n", nx, ny);

	float *h_A, *h_B, *hostRef, *gpuRef;
	h_A =			(float *) malloc(nBytes);
	h_B =			(float *) malloc(nBytes);
	hostRef =	(float *) malloc(nBytes);
	gpuRef =		(float *) malloc(nBytes);

	double iStart = cpuSecond();
	initialData(h_A, nxy);
	initialData(h_B, nxy);
	double iElaps = cpuSecond() - iStart;

	memset(hostRef, 0, nBytes);
	memset(gpuRef, 0, nBytes);

	iStart = cpuSecond();
	sumMatrixOnHost(h_A, h_B, hostRef, nx, ny);
	iElaps = cpuSecond() - iStart;
	printf("sumMatrixOnHost elapsed %f sec\n", iElaps);

	float *d_MatA, *d_MatB, *d_MatC;
	cudaMalloc((void **) &d_MatA, nBytes);
	cudaMalloc((void **) &d_MatB, nBytes);
	cudaMalloc((void **) &d_MatC, nBytes);

	cudaMemcpy(d_MatA, h_A, nBytes, cudaMemcpyHostToDevice);
	cudaMemcpy(d_MatB, h_B, nBytes, cudaMemcpyHostToDevice);

	int dimx = 32;
	int dimy = 16;
	dim3 block(dimx, dimy);
	dim3 grid((nx + block.x - 1) /block.x, (ny + block.y - 1) / block.y);

	iStart = cpuSecond();
	sumMatrixOnGPU2D <<<grid, block>>> (d_MatA, d_MatB, d_MatC, nx, ny);
	cudaDeviceSynchronize();
	iElaps = cpuSecond() - iStart;
	printf("sumMatrixOnGPU2D <<< (%d, %d), (%d, %d)>>> elapsed %f sec\n", grid.x, grid.y, block.x, block.y, iElaps);

	cudaMemcpy(gpuRef, d_MatC, nBytes, cudaMemcpyDeviceToHost);

	checkResult(hostRef, gpuRef, nxy);

	cudaFree(d_MatA);
	cudaFree(d_MatB);
	cudaFree(d_MatC);

	free(h_A);
	free(h_B);
	free(hostRef);
	free(gpuRef);

	cudaDeviceReset();

	return 0;
}
