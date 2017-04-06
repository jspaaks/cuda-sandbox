#include <stdio.h>

extern void cuda_doStuff(void);

int main( int argc, const char* argv[] )
{
	printf("Hello from main function...\n");
	cuda_doStuff();
}
