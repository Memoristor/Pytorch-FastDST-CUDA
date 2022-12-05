from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='fdstlib',
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