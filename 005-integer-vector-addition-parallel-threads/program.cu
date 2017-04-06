#include <stdio.h>
#include <stdlib.h>

__global__ void add(int *a, int *b, int *c) {
    // note that add has no variables in its scope, instead it reads and 
    // modifies variables that live elsewhere.
    int iElem = threadIdx.x;
    c[iElem] = a[iElem] + b[iElem];
}

void irand(int *arr, int nElems) {
    int iElem;
    for (iElem = 0; iElem < nElems; iElem+=1) {
        int r = rand() % 10;
        arr[iElem] = r;
    }
}

void zeros(int *arr, int nElems) {
    int iElem;
    for (iElem = 0; iElem < nElems; iElem+=1) {
        arr[iElem] = 0;
    }
}

void printarr(const char* formatString, int *arr, int nElems) {
    int iElem;
    for (iElem = 0; iElem < nElems; iElem += 1) {
        printf(formatString, iElem, arr[iElem]);
    }
}

int main(void) {
    
    // declare pointers to three arrays of integers in the host's memory space
    int *h_a;
    int *h_b;
    int *h_c;
    
    // declare pointers to three arrays of integers in the device's memory space
    // (the pointer itself lives in host memory, while its value points to an
    // address in the device memory?)
    int *d_a;
    int *d_b;
    int *d_c;

    // define the length of the vectors we're adding
    int nElems = 512;
    
    // define how many bytes is an integer (on this system)
    int nBytes = nElems * sizeof(int);

    // allocate nbytes memory on the device for each of the d_a, d_b, 
    // and d_c variables
    cudaMalloc((void **)&d_a, nBytes);
    cudaMalloc((void **)&d_b, nBytes);
    cudaMalloc((void **)&d_c, nBytes);

    // preallocate the memory space for h_a, h_b, and h_c
    h_a = (int *)malloc(nBytes);
    h_b = (int *)malloc(nBytes);
    h_c = (int *)malloc(nBytes);
    
    // initialize h_a, h_b with random ints, initialize h_c with zeros.
    irand(h_a, nElems);
    irand(h_b, nElems);
    zeros(h_c, nElems);
    
    // print the arrays to see what's going on
    printf("\n---- before ----\n");
    printarr("h_a[%d] = %d\n", h_a, nElems);
    printf("\n");
    printarr("h_b[%d] = %d\n", h_b, nElems);
    printf("\n");
    printarr("h_c[%d] = %d\n", h_c, nElems);
    printf("\n");
    
    // copy nBytes of memory located at &h_a on the host to variable d_a
    // on the device (then do the same for &h_b, d_b)
    cudaMemcpy(d_a, h_a, nBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, nBytes, cudaMemcpyHostToDevice);
    
    // call the integer add kernel with 1 block and nElems threads, pass
    //  it the values of d_a, d_b, as well as the (uninitialized) value 
    // of d_c
    int nBlocks = 1;
    int nThreads = nElems;
    add<<<nBlocks,nThreads>>>(d_a, d_b, d_c);

    cudaMemcpy(h_c, d_c, nBytes, cudaMemcpyDeviceToHost);

    // print the arrays to see what's going on
    printf("---- after ----\n");
    printarr("h_a[%d] = %d\n", h_a, nElems);
    printf("\n");
    printarr("h_b[%d] = %d\n", h_b, nElems);
    printf("\n");
    printarr("h_c[%d] = %d\n", h_c, nElems);
    printf("\n");

    
    // free up the memory on the device that we cudaMalloc'ed earlier.
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    
    free(h_a);
    free(h_b);
    free(h_c);
    
    return 0;

}

