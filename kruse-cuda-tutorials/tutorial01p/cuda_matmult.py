from numbapro import cuda
from numba import *
import numpy as np
import math
from timeit import default_timer as time

blocksPerGrid = 50
threadsPerBlock = 32
N = blocksPerGrid * threadsPerBlock

@cuda.jit(argtypes=[f4[:,:], f4[:,:], f4[:,:]])
def cu_square_matrix_mul(A, B, C):
    threadx 	= cuda.threadIdx.x
    thready 	= cuda.threadIdx.y
    blockx 		= cuda.blockIdx.x
    blocky 		= cuda.blockIdx.y
    blockWidth 	= cuda.blockDim.x
    blockHeight = cuda.blockDim.y

    idx = threadx + blockx * blockWidth
    idy = thready + blocky * blockHeight

    if idx >= N or idy >= N:
        return

    C[idy, idx] = 0
    for i in range(N):
        C[idy, idx] += A[idy, i] * B[i, idx]


A = np.array(np.random.random((N, N)), dtype=np.float32)
B = np.array(np.random.random((N, N)), dtype=np.float32)
C = np.empty_like(A)

print "N = %d x %d" % (N, N)

s = time()
stream = cuda.stream()
with stream.auto_synchronize():
    dA = cuda.to_device(A, stream)
    dB = cuda.to_device(B, stream)
    dC = cuda.to_device(C, stream)
    cu_square_matrix_mul[(blocksPerGrid, blocksPerGrid), (threadsPerBlock, threadsPerBlock), stream](dA, dB, dC)
    dC.to_host(stream)

e = time()
tcuda = e - s

# Host compute
Amat = np.matrix(A)
Bmat = np.matrix(B)

s = time()
Cans = Amat * Bmat
e = time()
tcpu = e - s

# Check result
assert np.allclose(C, Cans)

print 'cpu: %f' % tcpu
print 'cuda: %f' % tcuda
print 'cuda speedup: %.2fx' % (tcpu / tcuda)
