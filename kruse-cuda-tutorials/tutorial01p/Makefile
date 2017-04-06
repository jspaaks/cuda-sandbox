CC=g++
NVCC=nvcc
CUDA_ARCHITECTURE=20
OCELOT=`OcelotConfig -l`

all: main

main: main.o cuda_wrapper.o
	$(CC) main.o cuda_wrapper.o -o main $(OCELOT)

main.o: main.cpp
	$(CC) -std=c++0x main.cpp -c -I . 

cuda_wrapper.o: cuda_wrapper.cu
	$(NVCC) -c cuda_wrapper.cu -arch=sm_$(CUDA_ARCHITECTURE)

clean:
	rm main.o cuda_wrapper.o main