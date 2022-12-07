#coding=utf-8

from scipy import fftpack as fp
import torch
import time
import fdstlib
    
input = torch.randn((8, 3, 128, 64)).float().cuda()
numPoint = 2

print('=' * 80)
print('Test DCT-II/IDCT-II 2D')

native_dctii_2d = fdstlib.nativeDCTII2D(input, numPoint, False)
native_idctii_2d = fdstlib.nativeIDCTII2D(native_dctii_2d, numPoint, False)
error = torch.abs(native_idctii_2d - input)

print(f'... DCT-II output tensor size: {native_dctii_2d.size()}')
print(f'... IDCT-II output tensor size: {native_idctii_2d.size()}')
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
    
print(f'... DCT-II processing time: {total_time / iterations * 1e6 :4.4f} us')

iterations = 100
total_time = 0
for i in range(iterations):
    start = time.time()
    output = fdstlib.nativeIDCTII2D(input, numPoint, False)
    end = time.time()
    total_time += end - start
    
print(f'... IDCT-II processing time: {total_time / iterations * 1e6 :4.4f} us')

print('=' * 80)