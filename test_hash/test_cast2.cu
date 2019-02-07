#include <stdio.h>
#include <stdint.h>

typedef struct Node{

	int a;
	Node *hasNext;
} Node;

int main(){

	Node *node = (Node *) malloc(sizeof(Node));
	int address = reinterpret_cast<uintptr_t>(node);
	Node *next = reinterpret_cast<Node *>(address);
	printf("%p %x %p\n", node, address, next);

	free(node);

	return 0;
}
