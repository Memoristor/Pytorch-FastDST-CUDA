#coding=utf-8

from scipy import fftpack as fp
import torch
import time
import fdstlib


def dct2d(input, point):
    for i in range(point):
        input[i, :] = fp.dct(input[i, :], norm='ortho')
    for i in range(point):
        input[:, i] = fp.dct(input[:, i], norm='ortho')
    return input

def patch_dct2d(input, point):
    [N, C, H, W] = input.shape
    
    for n in range(N):
        for c in range(C):
            for h in range(int(H / point)):
                for w in range(int(W / point)):
                    patch = input[n, c, h * point: (h + 1) * point, w * point: (w + 1) * point]
                    input[n, c, h * point: (h + 1) * point, w * point: (w + 1) * point] = dct2d(patch, point)
    return input


input = torch.ones((1, 3, 1024, 1024)).float().cuda()
numPoint = 8


################################################################################


print('=' * 80)
print('Test DCT 2D CUDA vs CPU')

start_cuda = time.time()
dct2d_cuda = fdstlib.nativeDCT2D(input, numPoint)
end_cuda = time.time()

start_cpu = time.time()
dct2d_cpu = patch_dct2d(input.cpu().numpy(), numPoint)
dct2d_cpu = torch.from_numpy(dct2d_cpu).float().cuda()
error = torch.abs(dct2d_cuda - dct2d_cpu)
end_cpu = time.time()

print(f'... DCT-CUDA cost {end_cuda - start_cuda :4.6f} s')
print(f'... DCT-CPU cost {end_cpu - start_cpu :4.6f} s')
print(f'... DCT-CUDA output tensor size: {dct2d_cuda.size()}')
print(f'... DCT-CPU output tensor size: {dct2d_cpu.size()}')
print(f'... Max error of DCT on CUDA and CPU: {torch.max(error) :4.4f}')
print(f'... Min error of DCT on CUDA and CPU: {torch.min(error) :4.4f}')
print(f'... Average error of DCT on CUDA and CPU: {torch.mean(error) :4.4f}')

print('=' * 80)
print('Test DCT/IDCT 2D')

native_dct_2d = fdstlib.nativeDCT2D(input, numPoint)
native_idct_2d = fdstlib.nativeIDCT2D(native_dct_2d, numPoint)
error = torch.abs(native_idct_2d - input)

print(f'... DCT-CUDA output tensor size: {native_dct_2d.size()}')
print(f'... IDCT-CUDA output tensor size: {native_idct_2d.size()}')
print(f'... Max error of IDCT and input tensor: {torch.max(error) :4.4f}')
print(f'... Min error of IDCT and input tensor: {torch.min(error) :4.4f}')
print(f'... Average error of IDCT and input tensor: {torch.mean(error) :4.4f}')

print('Test DCT processing speed by CUDA')

iterations = 100
total_time = 0
for i in range(iterations):
    start = time.time()
    output = fdstlib.nativeDCT2D(input, numPoint)
    end = time.time()
    total_time += end - start
    
print(f'... DCT-CUDA average processing time: {total_time / iterations * 1e6 :4.4f} us')

iterations = 100
total_time = 0
for i in range(iterations):
    start = time.time()
    output = fdstlib.nativeIDCT2D(input, numPoint)
    end = time.time()
    total_time += end - start
    
print(f'... IDCT by CUDA average processing time: {total_time / iterations * 1e6 :4.4f} us')

################################################################################

print('=' * 80)
print('Test DST/IDST 2D')

native_dst_2d = fdstlib.nativeDST2D(input, numPoint)
native_idst_2d = fdstlib.nativeIDST2D(native_dst_2d, numPoint)
error = torch.abs(native_idst_2d - input)

print(f'... DST-CUDA output tensor size: {native_dst_2d.size()}')
print(f'... IDST-CUDA output tensor size: {native_idst_2d.size()}')
print(f'... Max error of IDST and input tensor: {torch.max(error) :4.4f}')
print(f'... Min error of IDST and input tensor: {torch.min(error) :4.4f}')
print(f'... Average error of IDST and input tensor: {torch.mean(error) :4.4f}')

print('Test DST processing speed by CUDA')

iterations = 100
total_time = 0
for i in range(iterations):
    start = time.time()
    output = fdstlib.nativeDST2D(input, numPoint)
    end = time.time()
    total_time += end - start
    
print(f'... DST-CUDA average processing time: {total_time / iterations * 1e6 :4.4f} us')

iterations = 100
total_time = 0
for i in range(iterations):
    start = time.time()
    output = fdstlib.nativeIDST2D(input, numPoint)
    end = time.time()
    total_time += end - start
    
print(f'... IDST by CUDA average processing time: {total_time / iterations * 1e6 :4.4f} us')

################################################################################

print('=' * 80)
print('Test DHT/IDHT 2D')

native_dht_2d = fdstlib.nativeDHT2D(input, numPoint)
native_idht_2d = fdstlib.nativeIDHT2D(native_dht_2d, numPoint)
error = torch.abs(native_idht_2d - input)

print(f'... DHT-CUDA output tensor size: {native_dht_2d.size()}')
print(f'... IDHT-CUDA output tensor size: {native_idht_2d.size()}')
print(f'... Max error of IDHT and input tensor: {torch.max(error) :4.4f}')
print(f'... Min error of IDHT and input tensor: {torch.min(error) :4.4f}')
print(f'... Average error of IDHT and input tensor: {torch.mean(error) :4.4f}')

print('Test DHT processing speed by CUDA')

iterations = 100
total_time = 0
for i in range(iterations):
    start = time.time()
    output = fdstlib.nativeDHT2D(input, numPoint)
    end = time.time()
    total_time += end - start
    
print(f'... DHT-CUDA average processing time: {total_time / iterations * 1e6 :4.4f} us')

iterations = 100
total_time = 0
for i in range(iterations):
    start = time.time()
    output = fdstlib.nativeIDHT2D(input, numPoint)
    end = time.time()
    total_time += end - start
    
print(f'... IDHT by CUDA average processing time: {total_time / iterations * 1e6 :4.4f} us')

################################################################################

print('=' * 80)
print('Test Sort Coefficients By Frequency')

sort_dct_frequency = fdstlib.sortCoefficients(native_dct_2d, numPoint, 0);
recover_dct_frequency = fdstlib.recoverCoefficients(sort_dct_frequency, numPoint, 0);
error = torch.abs(native_dct_2d - recover_dct_frequency)

print(f'... Sort by frequency output tensor size: {sort_dct_frequency.size()}')
print(f'... Recover by frequency output tensor size: {recover_dct_frequency.size()}')
print(f'... Max error of recovered and input tensor: {torch.max(error) :4.4f}')
print(f'... Min error of recovered and input tensor: {torch.min(error) :4.4f}')
print(f'... Average error of recovered and input tensor: {torch.mean(error) :4.4f}')

################################################################################

print('Test completed!')

