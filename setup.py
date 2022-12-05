from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='fdstlib',
    version="0.1.0",
    author="cjdeng",
    ext_modules=[
        CUDAExtension('fdstlib', [
            'torch_dct.cpp',
            'torch_dct_kernel.cu',
        ])
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)