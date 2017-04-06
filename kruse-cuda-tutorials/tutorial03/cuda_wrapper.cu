#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>

__global__ void processKernel(int *numberArray, int N)
{
  int idx = blockIdx.x*blockDim.x + threadIdx.x;

  if (idx<N)
    numberArray[idx] = numberArray[idx] + 1;
}

extern void cuda_doStuff(int *array_in, int *array_out, int N)
{
  int *numbers_d;
					  
  int numberOfBlocks = 1;	      				
  int threadsPerBlock = N;			
  int maxNumberOfThreads = N;

  cudaMalloc((void **) &numbers_d, sizeof(int)*N);

  cudaMemcpy(numbers_d, array_in, sizeof(int)*N, cudaMemcpyHostToDevice);
  processKernel<<<numberOfBlocks, threadsPerBlock>>>(numbers_d, maxNumberOfThreads);
  cudaDeviceSynchronize();
  cudaMemcpy(array_out, numbers_d, sizeof(int)*N, cudaMemcpyDeviceToHost);

  cudaFree(numbers_d);

  return;
}
