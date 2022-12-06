#coding=utf-8

from torch.utils.cpp_extension import load
import torch

import fdstlib

input = torch.randn((1, 1, 3, 3)).float().cuda()
size = 4

output = fdstlib.nativeDCTII2DForward(input, size, False)

print(input)
print()
print(output)
