import numbapro
from numbapro import cuda
import numpy
import time

numbapro.check_cuda()

@numbapro.cuda.jit(argtype=[numbapro.float64[:,:], numbapro.float64[:,:], numbapro.float64[:,:]])
def cuda_sum(a, b, c):
	c = a + b

@numbapro.vectorize([numbapro.float64(numbapro.float64, numbapro.float64)], target='gpu')
def sum(a, b):
	return a + b



for i in [50, 500, 5000]:
	numElements = i

	start = time.time()
	#a = numpy.ndarray((numElements,numElements), dtype=numpy.float32)
	#b = numpy.ndarray((numElements,numElements), dtype=numpy.float32)
	a = numpy.random.uniform(1000, size=(numElements, numElements));
	b = numpy.random.uniform(1000, size=(numElements, numElements));
	
	#for i in range(0,numElements):
	#	for j in range(0,numElements):
	#		a[i][j] = i+j
	#		b[i][j] = 2*i + j
	end = time.time()
	print "Time to initialize:", end - start

	print "Adding two (", numElements, "x", numElements,")-matrices:"
	start = time.time()
	c = a + b
	end = time.time()
	time_cpu = end-start
	print "Time for CPU matrix addition:", time_cpu

	start = time.time()
	c = sum(a, b)
	end = time.time()
	time_gpu = end-start
	print "Time for CUDA matrix addition:", time_gpu
