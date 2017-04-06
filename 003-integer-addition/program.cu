#include <stdio.h>

__global__ void add(int *a, int *b, int *c) {
    // note that add has no variables in its scope, instead it reads and 
    // modifies variables that live elsewhere.
    *c = *a + *b;
}

int main(void) {
    
    // declare three integers in the host's memory space
    int h_a;
    int h_b;
    int h_c;
    
    // declare pointers to three integers in the device's memory space    
    int *d_a;
    int *d_b;
    int *d_c;
    
    // define how many bytes is an integer (on this system)
    int nbytes = sizeof(int);

    // allocate nbytes memory on the device for each of the d_a, d_b, 
    // and d_c variables
    cudaMalloc((void **)&d_a, nbytes);
    cudaMalloc((void **)&d_b, nbytes);
    cudaMalloc((void **)&d_c, nbytes);

    // set h_a and h_b to more or less arbitrary (but int) values
    h_a = 2;
    h_b = 7;
    
    // copy nbytes of memory located at &h_a on the host to variable d_a
    // on the device (then do the same for &h_b, d_b)
    cudaMemcpy(d_a, &h_a, nbytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, &h_b, nbytes, cudaMemcpyHostToDevice);
    
    // call the integer add kernel with 1 block and 1 thread, pass it the 
    // values of d_a, d_b, as well as the (uninitialized) value of d_c
    add<<<1,1>>>(d_a, d_b, d_c);
    
    cudaMemcpy(&h_c, d_c, nbytes, cudaMemcpyDeviceToHost);
    
    
    // free up the memory on the device that we cudaMalloc'ed earlier.
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    
    printf("h_c = %d\n", h_c);
    printf("nbytes = %d\n", nbytes);
    

}

