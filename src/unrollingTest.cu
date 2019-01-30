#include <stdio.h>
#include "cuda_prac.h"

#define N 1<<30

void buildup(double *a, double *b, double *c){

	for(int i = 0; i < N; i++)
	{
		a[i] = 0.127312;
		b[i] = 0.123712;
	}
}

void warmup(double *a, double *b, double *c){

	for(int i = 0; i < N; i++)
		c[i] = a[i] + b[i];
}

void operAdd(double *a, double *b, double *c){

	for(int i = 0; i < N; i++)
		c[i] = a[i] + b[i];
}

void operAddUnrolling2(double *a, double *b, double *c){

	for(int i = 0; i < N; i += 2)
	{
		c[i] = a[i] + b[i];
		c[i + 1] = a[i + 1] + b[i + 1];
	}
}

void operAddUnrolling4(double *a, double *b, double *c){

	for(int i = 0; i < N; i += 4)
	{
		c[i] = a[i] + b[i];
		c[i + 1] = a[i + 1] + b[i + 1];
		c[i + 2] = a[i + 2] + b[i + 2];
		c[i + 3] = a[i + 3] + b[i + 3];
	}
}

int main(){

	double *a, *b, *c;
	size_t size = sizeof(double) * N;

	a = (double *) malloc(size);
	b = (double *) malloc(size);
	c = (double *) malloc(size);

	buildup(a, b, c);
	warmup(a, b, c);
	memset(a, 0, size);
	memset(b, 0, size);
	memset(c, 0, size);

	buildup(a, b, c);
	double iStart = cpuSecond();
	operAdd(a, b, c);
	double iElaps = cpuSecond() - iStart;
	printf("Elapsed: %f\n", iElaps);
	memset(a, 0, size);
	memset(b, 0, size);
	memset(c, 0, size);

	buildup(a, b, c);
	iStart = cpuSecond();
	operAddUnrolling2(a, b, c);
	iElaps = cpuSecond() - iStart;
	printf("Elapsed: %f\n", iElaps);
	memset(a, 0, size);
	memset(b, 0, size);
	memset(c, 0, size);

	buildup(a, b, c);
	iStart = cpuSecond();
	operAddUnrolling4(a, b, c);
	iElaps = cpuSecond() - iStart;
	printf("Elapsed: %f\n", iElaps);

	free(a);
	free(b);
	free(c);

	return 0;
}
