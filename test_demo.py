#coding=utf-8

from torch.utils.cpp_extension import load
import torch

import fdstlib

input = torch.randn((3, 128, 128, 3)).float().cuda()
size = 4

fdstlib.native_dctii_2d(input, size, False)

