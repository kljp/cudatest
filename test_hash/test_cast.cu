#include <stdio.h>
#include <stdint.h>

int main(){

	int *a = (int *) malloc(sizeof(int));
	int b = reinterpret_cast<uintptr_t>(a);
	int *c = reinterpret_cast<int *>(b);

	printf("%p %x %p\n", a, b, c);

	free(a);


	return 0;
}
