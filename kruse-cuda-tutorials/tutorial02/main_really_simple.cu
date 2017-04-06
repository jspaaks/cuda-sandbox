#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>

// this is the program that is to be run on the device for a 
// large number of threads, in our example 100
// each thread takes care of one entry in the number array,
// so in order for the thread to know which number to manipulate,
// a scheme has to be utilized in order to assign each thread a
// unique number

__global__ void incrementArrayViaCUDAdevice(int *numberArray, int N)
{
	// this is the assignment of a unique identifier.
	// blockIdx.x is the unique number of the block, in which the
	// thread is positioned, blockDim.x holds the number of threads
	// for each block and threadIdx.x is the number of the thread in
	// this block.
	int idx = blockIdx.x*blockDim.x + threadIdx.x;

	// this tells the thread to manipulate the assigned number in 
	// the array stored in device memory and increment it
	if (idx<N)
		numberArray[idx] = numberArray[idx] + 1;
}

// this is the "normal" function to be run on the CPU
// it does the exact same thing as the CUDA function above
void incrementArray(int *numberArray, int N){

	// go through every number in the array consecutively
	// and increment it
	for(int i=0; i<N; ++i)
	{
		numberArray[i] = numberArray[i] + 1;
	}
}

int main(int argc, const char* argv[] )
{
	// some arbitrary array length
	int numberOfNumbers = 100;

	// declare some arrays for storing numbers
	int *numbers1, *numbers2;

	// reserve (allocate) some working space for the numbers in device memory
	cudaMallocManaged(&numbers1, sizeof(int)*numberOfNumbers);
	cudaMallocManaged(&numbers2, sizeof(int)*numberOfNumbers);

	// fill the input array with some numbers
	for(int i=0;i<numberOfNumbers;i++) 	
	{
		numbers1[i] = i;	// this will be manipulated by the CUDA device (GPU)
		numbers2[i] = i;	// this will be manipulated by the CPU (as any standard C program would do)
	}

	// tell the device (GPU) to do its magic
	incrementArrayViaCUDAdevice<<<1, numberOfNumbers>>>(numbers1, numberOfNumbers);

	// wait for the device to finish working
	cudaDeviceSynchronize();

	// compute the same function "normally" on the CPU
	incrementArray(numbers2, numberOfNumbers);

	// check if the GPU did the same as the CPU
	bool workedCorrectly = true;
	for(int i=0;i<numberOfNumbers;i++)
	{	
		if (numbers1[i] != numbers2[i])
			workedCorrectly = 0;
	}

	if (workedCorrectly == 1)
		printf("The device performed well!\n");
	else
		printf("Something went wrong. The output numbers are not what was to be expected...\n");

	// free the space that has been used by our arrays so that
	// other programs might use it
	cudaFree(numbers1);
	cudaFree(numbers2);
	   
	return 0;
}
