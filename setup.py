from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
from glob import glob
import os

setup(
    name='fadst',
    version="0.2.0",
    author="cjdeng",
    author_email="cjdeng@std.uestc.edu.cn",
    ext_modules=[
        CUDAExtension(
            name='fadst', 
            sources=glob(os.path.join('source', '*.c*')),
            extra_compile_args={'cxx': ['-g'], 'nvcc': ['-O2']},
            extra_link_flags=['-Wl,--no-as-needed', '-lcuda'],
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)