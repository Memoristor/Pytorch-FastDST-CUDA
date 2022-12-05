#coding=utf-8

from torch.utils.cpp_extension import load
import torch

fdstlib = load(
    name="fdstlib",
    sources=[
        'torch_dct.cpp',
        'torch_dct_kernel.cu',
    ]
)

# import fdstlib

a = torch.randn(32).float().cuda()
b = torch.randn(32).float().cuda()
c = torch.zeros(32).float().cuda()

print(a)
print(b)

print(fdstlib.forward(a, b))

