#include <stdio.h>
#include <stdlib.h>

extern void cuda_doStuff(int *array_in, int *array_out, int N);

int main( int argc, const char* argv[] )
{

  int numberOfNumbers = 100;

  int *numbers1_h, *numbers2_h;	

  numbers1_h = (int*) malloc(sizeof(int)*numberOfNumbers);
  numbers2_h = (int*) malloc(sizeof(int)*numberOfNumbers);

  for(int i=0;i<numberOfNumbers;i++) 	
    {
      numbers1_h[i] = i;
    }  

  // let the wrapper take care of communication with the device
  cuda_doStuff(numbers1_h, numbers2_h, numberOfNumbers);
  // now the data is manipulated without having to take care of
  // all the CUDA stuff here in this file. Nice, isn't it? 

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

  free(numbers1_h);
  free(numbers2_h);

  return 0;
}
