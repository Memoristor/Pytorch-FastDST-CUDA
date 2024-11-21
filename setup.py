from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import fnmatch
import os


def find_files(directory, pattern):
    """Find all files that matches the pattern under the directory"""
    for root, dirs, files in os.walk(os.path.expanduser(directory)):
        for basename in files:
            if fnmatch.fnmatch(basename, pattern):
                filename = os.path.join(root, basename)
                yield filename
                

setup(
    name='fadst',
    version="0.3.0",
    author="cjdeng",
    author_email="cjdeng@std.uestc.edu.cn",
    ext_modules=[
        CUDAExtension(
            name='fadst', 
            sources=list(find_files('source', '*.c*')),
            extra_compile_args={'cxx': ['-g'], 'nvcc': ['-O2']},
            extra_link_flags=['-Wl,--no-as-needed', '-lcuda'],
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)