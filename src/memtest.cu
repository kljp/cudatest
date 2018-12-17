#include <stdio.h>
#include <stdlib.h>
#include "cuda_prac.h"

void init_matrix(float *a, float *b, float *c, long long dim_x, long long dim_y, int flag){

	long long idx = 0;
	long long ix = 0;
	long long iy = 0;

	if(flag == 1)
	{
		for(iy = 0; iy < dim_y; iy++)
		{
			for(ix = 0; ix < dim_x; ix++)
			{
				idx = iy * dim_x + ix;
				a[idx] = idx;
				b[idx] = 1 / (idx + 1);
				c[idx] = a[idx] + b[idx];
			}
		}
		
		return;
	}
	

	while(ix < dim_x || iy < dim_y)
	{
		idx = iy * dim_y + ix;
		if(ix < dim_x && iy < dim_y)
		{
			a[idx] = idx;
			b[idx] = 1 / (idx + 1);
			c[idx] = a[idx] + b[idx];
			ix++;
		}
		
		else
		{
			if(iy < dim_y)
			{
				ix = 0;
				iy++;
			}

			else
				break;
		}
	}

	return;
}

int main(int argc, char** argv){

	long long N = atoi(argv[1]);
	int flag = atoi(argv[2]);
	
	float *a, *b, *c;
	size_t size = sizeof(float);
	long long dim_x = 1 << N;
	long long dim_y = 1 << N;
	
	a = (float *) malloc(dim_x * dim_y * size);
	printf("a alloc success\n");
	b = (float *) malloc(dim_x * dim_y * size);
	printf("b alloc success\n");
	c = (float *) malloc(dim_x * dim_y * size);
	printf("c alloc success\n");

	double iStart = cpuSecond();
	init_matrix(a, b, c, dim_x, dim_y, flag);
	double iElaps = cpuSecond() - iStart;
	printf("Elapsed time: %f\n", iElaps);

	free(a);
	printf("a free success\n");
	free(b);
	printf("b free success\n");
	free(c);
	printf("c free success\n");

	return 0;
}
