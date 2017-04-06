import numpy as np
from numba import *
from numbapro import cuda
from timeit import default_timer as time

@cuda.jit(argtypes=[f4[:], f4[:], f4[:]])
def cuda_sum(a, b, c):
    i = cuda.grid(1)
    c[i] = a[i] + b[i]

griddim = 5000000, 1
blockdim = 32, 1, 1
N = griddim[0] * blockdim[0]
cuda_sum_configured = cuda_sum.configure(griddim, blockdim)
a = np.array(np.random.random(N), dtype=np.float32)
b = np.array(np.random.random(N), dtype=np.float32)
c_gpu = np.empty_like(a)
c_gpu = np.empty_like(a)

start_cpu = time()
c = a + b
end_cpu = time()
t_cpu = end_cpu - start_cpu
print "CPU time:", end_cpu- start_cpu

start_gpu = time()
cuda_sum_configured(a, b, c)
end_gpu = time()
t_gpu = end_gpu - start_gpu
print "GPU time:", end_gpu- start_gpu
