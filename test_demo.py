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


input = torch.randn((1, 3, 1024, 1024)).float().cuda()
numPoint = 8

print('=' * 80)
print('Test DCT-II 2D CUDA vs CPU')

start_cuda = time.time()
dct2d_cuda = fdstlib.nativeDCTII2D(input, numPoint, False)
end_cuda = time.time()

start_cpu = time.time()
dct2d_cpu = patch_dct2d(input.cpu().numpy(), numPoint)
dct2d_cpu = torch.from_numpy(dct2d_cpu).float().cuda()
error = torch.abs(dct2d_cuda - dct2d_cpu)
end_cpu = time.time()

print(f'... DCT-II-CUDA cost {end_cuda - start_cuda :4.6f} s')
print(f'... DCT-II-CPU cost {end_cpu - start_cpu :4.6f} s')
print(f'... DCT-II-CUDA output tensor size: {dct2d_cuda.size()}')
print(f'... DCT-II-CPU output tensor size: {dct2d_cpu.size()}')
print(f'... Max error of DCT-II on CUDA and CPU: {torch.max(error) :4.4f}')
print(f'... Min error of DCT-II on CUDA and CPU: {torch.min(error) :4.4f}')
print(f'... Average error of DCT-II on CUDA and CPU: {torch.mean(error) :4.4f}')

print('=' * 80)
print('Test DCT-II/IDCT-II 2D')

native_dctii_2d = fdstlib.nativeDCTII2D(input, numPoint, False)
native_idctii_2d = fdstlib.nativeIDCTII2D(native_dctii_2d, numPoint, False)
error = torch.abs(native_idctii_2d - input)

print(f'... DCT-II-CUDA output tensor size: {native_dctii_2d.size()}')
print(f'... IDCT-II-CUDA output tensor size: {native_idctii_2d.size()}')
print(f'... Max error of IDCT-II and input tensor: {torch.max(error) :4.4f}')
print(f'... Min error of IDCT-II and input tensor: {torch.min(error) :4.4f}')
print(f'... Average error of IDCT-II and input tensor: {torch.mean(error) :4.4f}')

print('Test DCT-II processing speed by CUDA')

iterations = 100
total_time = 0
for i in range(iterations):
    start = time.time()
    output = fdstlib.nativeDCTII2D(input, numPoint, False)
    end = time.time()
    total_time += end - start
    
print(f'... DCT-II-CUDA average processing time: {total_time / iterations * 1e6 :4.4f} us')

iterations = 100
total_time = 0
for i in range(iterations):
    start = time.time()
    output = fdstlib.nativeIDCTII2D(input, numPoint, False)
    end = time.time()
    total_time += end - start
    
print(f'... IDCT-II by CUDA average processing time: {total_time / iterations * 1e6 :4.4f} us')

print('=' * 80)