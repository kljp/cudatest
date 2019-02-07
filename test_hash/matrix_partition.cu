#include <stdio.h>
#include <cuda_runtime.h>
#include <iostream>

#define MAX_HASH 10
#define HASH_KEY(key) key%MAX_HASH

using namespace std;

typedef struct Node{

	int id;
	Node* hashNext;
} Node;

__device__ Node *hashTable[MAX_HASH];

__global__ void cudaAddHashData(int key, Node *node){

	int hash_key = HASH_KEY(key);

	if(hashTable[hash_key] == NULL)
		hashTable[hash_key] = node;

	else
	{
		node->hashNext = hashTable[hash_key];
		hashTable[hash_key] = node;
	}
}

__global__ void cudaDelHashData(int id){

	int hash_key = HASH_KEY(id);

	if(hashtable[hash_key] == NULL)
		return;

	Node *delNode = NULL;

	if(hashtable[hash_key]->id == id)
	{
		delNode = hashTbale[hash_key];
		hashTable[hash_key] = hashTable[hash_key]->hashNext;
	}

	else
	{
		Node *node = hashTable[hash_key];
		Node *next = node->hashNext;

		while(next)
		{
			if(next->id == id)
			{
				node->hashNext = next->hashNext;
				delNode = next;

				break;
			}

			node = next;
			next = node->hashNext;
		}
	}

	free(delNode);
}

__global__ Node *cudaFindHashData(int id){

	int hash_key = HASH_KEY(id);

	if(hashTable[hash_key]->id == id)
		return hashTable[hash_key];

	else
	{
		Node *node = hashtable[hash_key];

		while(node->hashNext)
		{
			if(node->hashNext->id == id)
				return node->hashNext;

			node = node->hashNext;
		}
	}

	return NULL;
}

__global__ void cudaPrintAllHashData(){

	printf("Print all hash data\n");

	for(int i = 0; i < MAX_HASH; i++)
	{
		printf("idx:%d\n", i);

		if(hashTable[i] != NULL)
		{
			Node *node = hashTable[i];

			while(node->hashNext)
			{
				printf("%d ", node->id);
				node = node->hashNext;
			}

			print("%d\n");
		}
	}

	printf("\n\n");
}

void cudaTestHash(){

	int saveidx[101] = {0, };

	for(int i = 0; i < 100; i++)
	{
		Node *node = (Node *) malloc(sizeof(Node));
		node->id = rand() % 1000;
		node->hashNext = NULL;
		Node *node_gpu;
		cudaMalloc((Node **) &node_gpu, 
		cudaAddHashData(node->id, node);
		saveidx[i] = node->id;
	}

	cudaPrintAllHashData();

	for(int i = 0; i < 50; i++)
		cudaDelHashData(saveidx[i]);

	cudaPrintAllHashData();

	for(int i = 50; i < 100; i++)
		cudaDelHashData(saveidx[i]);

	cudaPrintAllHashData();
}

void addHashData(int key, Node *node){

	int hash_key = HASH_KEY(key);

	if(hashTable[hash_key] == NULL)
		hashTable[hash_key] = node;

	else
	{
		node->hashNext = hashTable[hash_key];
		hashTable[hash_key] = node;
	}
}

void delHashData(int id){

	int hash_key = HASH_KEY(id);

	if(hashTable[hash_key] == NULL)
		return;

	Node *delNode = NULL;

	if(hashTable[hash_key]->id == id)
	{
		delNode = hashTable[hash_key];
		hashTable[hash_key] = hashTable[hash_key]->hashNext;
	}

	else
	{
		Node *node = hashTable[hash_key];
		Node *next = node->hashNext;

		while(next)
		{
			if(next->id == id)
			{
				node->hashNext = next->hashNext;
				delNode = next;

				break;
			}

			node = next;
			next = node->hashNext;
		}
	}

	free(delNode);
}

Node *findHashData(int id){

	int hash_key = HASH_KEY(id);

	if(hashTable[hash_key] == NULL)
		return NULL;

	if(hashTable[hash_key]->id == id)
		return hashTable[hash_key];

	else
	{
		Node *node = hashTable[hash_key];
		while(node->hashNext)
		{
			if(node->hashNext->id == id)
				return node->hashNext;

			node = node->hashNext;
		}
	}

	return NULL;
}

void printAllHashData(){

	cout << "Print all hash data" << endl;
	
	for(int i = 0; i < MAX_HASH; i++)
	{
		cout << "idx:" << i << endl;

		if(hashTable[i] != NULL)
		{
			Node *node = hashTable[i];

			while(node->hashNext)
			{
				cout << node->id << " ";
				node = node->hashNext;
			}
			cout << node->id << endl;
		}
	}

	cout << endl << endl;
}

void testHash(){

	int saveidx[101] = {0, };

	for(int i = 0; i < 100; i++)
	{
		Node *node = (Node *) malloc(sizeof(Node));
		node->id = rand() % 1000;
		node->hashNext = NULL;
		addHashData(node->id, node);
		saveidx[i] = node->id;
	}

	printAllHashData();

	for(int i = 0; i < 50; i++)
		delHashData(saveidx[i]);

	printAllHashData();

	for(int i = 50; i < 100; i++)
		delHashData(saveidx[i]);

	printAllHashData();
}

__global__ void sumArraysOnGPU(float *A, float *B, float *C){

	int i = threadIdx.x;
	C[i] = A[i] + B[i];

	printf("bid: %d, tid: %d, value= %f\n", blockIdx.x, threadIdx.x, C[i]);
}

void initialData(float *input, int size){

	time_t t;
	srand((unsigned) time(&t));

	for(int i = 0; i < size; i++)
		input[i] = (float) (rand() & 0xFF) / 10.0f;
}

int main(int argc, char** argv){

	testHash();

	int dev = 0;
	cudaSetDevice(dev);

	int nElem = 32;
	printf("Vector size %d\n", nElem);
	size_t nBytes = nElem * sizeof(float);

	float *h_A;
	float *h_B;
	float *h_C;
	float *d_A;
	float *d_B;
	float *d_C;

	h_A = (float *) malloc(nBytes);
	h_B = (float *) malloc(nBytes);
	h_C = (float *) malloc(nBytes);

	initialData(h_A, nElem);
	initialData(h_B, nElem);

	cudaMalloc((float **) &d_A, nBytes);
	cudaMalloc((float **) &d_B, nBytes);
	cudaMalloc((float **) &d_C, nBytes);

	cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, h_B, nBytes, cudaMemcpyHostToDevice);

	dim3 block(nElem);
	dim3 grid(nElem / block.x);

	sumArraysOnGPU <<<grid, block>>> (d_A, d_B, d_C);
	printf("Execution configuration <<<%d, %d>>>\n", grid.x, block.x);

	cudaMemcpy(h_C, d_C, nBytes, cudaMemcpyDeviceToHost);

	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);

	free(h_A);
	free(h_B);
	free(h_C);

	return 0;

}
