#include <stdio.h>
#include <cuda_runtime.h>
#include <stdint.h>
#define MAX_SIZE 64

typedef struct Node{

	int idx;
	Node *hasNext;
} Node;

__device__ Node *ds[MAX_SIZE];

__global__ void wcws(float *a){

	int warpID = threadIdx.x / 32;
	int laneID = threadIdx.x % 32;

	a[laneID] = laneID;
	printf("%f\n", a[laneID]);
}

 __device__ Node *wcws_allocate(int laneID){
	  
	Node *node;
//	cudaMalloc((Node **) &node, sizeof(Node));
	node = (Node *) malloc(sizeof(Node));
	node->idx = laneID;
	node->hasNext = reinterpret_cast<Node *>(__shfl_up_sync(0xFFFFFFFF, reinterpret_cast<uintptr_t>(node), 1));
   printf("%p\n", node);    
	return node;
}

__global__ void wcws_insert(){

	int warpID = threadIdx.x / 32;
	int laneID = threadIdx.x % 32;
	
	ds[laneID] = wcws_allocate(laneID);
	printf("%d\n", ds[laneID]->idx);
}

int main(int argc, char** argv){

	int dev = 0;
	cudaSetDevice(dev);

	int nElem = 32;
	size_t nBytes = nElem * sizeof(float);

	dim3 block(nElem);
	dim3 grid(nElem / block.x);

	wcws_insert <<<grid, block>>> ();

	return 0;
}
