CC=g++
LINKER_DIRS=-L/usr/local/cuda-6.0/lib64
LINKER_FLAGS=-lcudart -lcuda
NVCC=nvcc
CUDA_ARCHITECTURE=35
OCELOT=

all: main_really_simple

main_really_simple:
	$(NVCC) -c main_really_simple.cu -arch=sm_$(CUDA_ARCHITECTURE)
	$(CC) main_really_simple.o -o main_really_simple $(LINKER_DIRS) $(LINKER_FLAGS) $(OCELOT)

clean:
	rm -f *.o main_really_simple
