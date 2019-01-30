#include <stdio.h>
#include <cuda_runtime.h>
#include "cuda_prac.h"

int recursiveReduce(int *data, int const size){

	if(size == 1)
		return data[0];

	int const stride = size / 2;

	for(int i = 0; i < stride; i++)
		data[i] += data[i + stride];

	return recursiveReduce(data, stride);
}

__global__ void reduceNeighbored(int *g_idata, int *g_odata, unsigned int n){

	unsigned int tid = threadIdx.x;
	unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

	int *idata = g_idata + blockIdx.x * blockDim.x;

	if(idx >= n)
		return;

	for(int stride = 1; stride < blockDim.x; stride *= 2)
	{
		if((tid % (2 * stride)) == 0)
			idata[tid] += idata[tid + stride];

		__syncthread();
	}

	if(tid == 0)
		g_odata[blockIdx.x] = idata[0];
}

__global__ void reduceNeighboredLess(int *g_idata, int *g_odata, unsigned int n){

	unsigned int tid = threadIdx.x;
	unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

	int *idata = g_idata + blockIdx.x * blockDim.x;

	if(idx >= n)
		return;

	for(int stride = 1; stride < blockDim.x; stride *= 2)
	{
		int index = 2 * stride * tid;

		if(index < blockDim.x)
			idata[index] += idata[index + stride];
		
		__syncthreads();
	}

	if(tid == 0) g_odata[blockIdx.x] = idata[0];
}

__global__ void reduceInterleaved(int *g_idata, int *g_odata, unsigned int n){

	unsigned int tid = threadIdx.x;
	unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

	int *idata = g_idata * blockDim.x + threadIdx.x;

	if(idx >= n)
		return;

	for(int stride = blockDim.x / 2; stride > 0; stride >>= 1)
	{
		if(tid < stride)
			idata[tid] += idata[tid + stride];
		
		__syncthreads();
	}

	if(tid == 0)
		g_odata[blockIdx.x] = idata[0];
}

__global__ void reduceUnrolling2(int *g_idata, int g_odata, unsigned int n){

	unsigned int tid = threadIdx.x;
	unsigned int idx = blockIdx.x * blockDim.x * 2 + threadIdx.x;

	int *idata = g_idata + blockIdx.x * blockDim.x * 2;

	if(idx + blockDim.x < n)
		g_idata[idx] += g_idata[idx + blockDim.x];

	_syncthreads();

	for(int stride = blockDim.x / 2; stride > 0; stride >>= 1)
	{
		if(tid < stride)
			idata[tid] += idata[tid + stride];

		__syncthreads();
	}

	if(tid == 0)
		g_odata[blockIdx.x] = idata[0];
}

__global__ void reduceUnrolling4(int *g_idata, int *g_odata, unsigned int n){

	unsigned int tid = threadIdx.x;
	unsigned int idx = blockIdx.x * blockDim.x * 4 + threadIdx.x;

	int *idata = g_idata + blockIdx.x * blockDim.x * 4;
	////////        incomplete
}













































