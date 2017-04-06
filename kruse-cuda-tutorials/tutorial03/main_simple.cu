#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>

// this is the program that is to be run on the device for a 
// large number of threads, in our example 100
// each thread takes care of one entry in the number array,
// so in order for the thread to know which number to manipulate,
// a scheme has to be utilized in order to assign each thread a
// unique number

__global__ void processKernel(int *numberArray, int N)
{
  // this is the assignment of a unique identifier.
  // blockIdx.x is the unique number of the block, in which the
  // thread is positioned, blockDim.x holds the number of threads
  // for each block and threadIdx.x is the number of the thread in
  // this block.
  int idx = blockIdx.x*blockDim.x + threadIdx.x;

  // this tells the thread to manipulate the assigned number in 
  // the array stored in device memory
  if (idx<N)
    numberArray[idx] = numberArray[idx] + 1;
}

int main(int argc, const char* argv[] )
{
  int numberOfNumbers = 100;

  // declare some arrays for storing numbers
  int *numbers1_h, *numbers2_h, *numbers_d;	

  // define the dimensions for the device program to work upon
  // we need only one block with N threads per block (so N threads total)
  // to process our data
  int numberOfBlocks = 1;	      				
  int threadsPerBlock = numberOfNumbers;			
  int maxNumberOfThreads = numberOfNumbers;

  // reserve (allocate) some space for numbers in (h)ost memory
  numbers1_h = (int*) malloc(sizeof(int)*numberOfNumbers);		

  // reserve some space for resulting values in (h)ost memory
  numbers2_h = (int*) malloc(sizeof(int)*numberOfNumbers);

  // reserve some working space for the numbers in device memory
  cudaMalloc((void **) &numbers_d, sizeof(int)*numberOfNumbers);

  // fill the input array (in host memory) with some numbers (step 1)
  for(int i=0;i<numberOfNumbers;i++) 	
    {
      numbers1_h[i] = i;
    }

  // copy these input numbers to the device (step 2)
  cudaMemcpy(numbers_d, numbers1_h, sizeof(int)*numberOfNumbers, cudaMemcpyHostToDevice);

  // tell the device to do its magic (step 3)
  processKernel<<<numberOfBlocks, threadsPerBlock>>>(numbers_d, maxNumberOfThreads);

  // wait for the device to finish working
  cudaDeviceSynchronize();

  // copy results back to host RAM (step 4)
  cudaMemcpy(numbers2_h, numbers_d, sizeof(int)*numberOfNumbers, cudaMemcpyDeviceToHost);

  // check if the device did what it was told to do (step 5)
  int workedCorrectly = 1;
  for(int i=0;i<numberOfNumbers;i++)
    {	
      if (numbers1_h[i] + 1 != numbers2_h[i])
	workedCorrectly = 0;
    }

  if (workedCorrectly == 1)
    printf("The device performed well!\n");
  else
    printf("Something went wrong. The output numbers are not what was to be expected...\n");

  // free the space that has been used by our arrays so that
  // other programs might use it
  free(numbers1_h);
  free(numbers2_h);
  cudaFree(numbers_d);
	   
  return 0;
}
