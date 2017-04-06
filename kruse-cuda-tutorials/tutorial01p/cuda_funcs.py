import pycuda.gpuarray as gpuarray
import pycuda.driver as cuda
import pycuda.autoinit
import numpy
import time
from reikna.linalg import MatrixMul

a_gpu = gpuarray.GPUArray((5000,5000), dtype=numpy.float32)
b_gpu = gpuarray.GPUArray((5000,5000), dtype=numpy.float32)

a = numpy.ndarray((5000,5000), dtype=numpy.float32)
b = numpy.ndarray((5000,5000), dtype=numpy.float32)

for i in range(0,5000):
	for j in range(0,5000):
		a[i][j] = i+j
		b[i][j] = 2*i + j
	
a_gpu.set(a)
b_gpu.set(b)

start = time.time()
c_gpu = a_gpu*b_gpu
end = time.time()
print "Time for CUDA matrix multiplication:", end-start

start = time.time()
c = a*b
end = time.time()
print "Time for numpy matrix multiplication:", end-start 

print "Time:", time.time()


