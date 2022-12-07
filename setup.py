from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
from glob import glob
import os

setup(
    name='fdstlib',
    version="0.1.0",
    author="cjdeng",
    ext_modules=[
        CUDAExtension('fdstlib', glob(os.path.join('source', '*.c*')))
    ],
    include_dirs=[
        'include',
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)