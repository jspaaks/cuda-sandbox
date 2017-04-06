hello: hello.cu
ifneq "$(wildcard hello.out)" ""
	rm hello.out
endif
	nvcc -o hello.out hello.cu
	
.PHONY: clean

clean:
ifneq "$(wildcard hello.out)" ""
	rm hello.out
endif
