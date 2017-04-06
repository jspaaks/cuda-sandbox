program: program.cu
	nvcc -o program.out program.cu
	
.PHONY: clean

clean:
ifneq "$(wildcard program.out)" ""
	rm program.out
endif
