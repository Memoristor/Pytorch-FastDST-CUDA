# Pytorch-FastDST-CUDA

> This is a Fast Discrete Signal Transform (FastDST) library for Pytorch implemented by CUDA.

## Introduction

This repository provides fast implementations of various transforms using CUDA, including Discrete Cosine Transform (DCT), Discrete Sine Transform (DST), and Discrete Hartley Transform (DHT), enabling efficient computation for large-scale data processing. It supports the direct processing of PyTorch tensors, seamlessly integrating into PyTorch workflows without requiring additional data conversions. Furthermore, the repository is designed to support operating in a distributed data-parallel mode, ensuring scalability and performance across multiple GPUs for high-performance computing tasks.

## Installation

The installation of this plugin has been successfully tested in the Ubuntu environment. It is particularly important to ensure that the CUDA Driver, CUDA Toolkit, and the corresponding version of PyTorch-GPU are installed and properly configured beforehand. Note that the versions of CUDA and PyTorch-GPU must match, such as both being CUDA 11.x or CUDA 12.x.

#### Step 1 : Install [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) and [CuDNN (Optional)](https://developer.nvidia.com/cudnn-downloads)

```shell
# Install CUDA Toolkit
# Note: The PyTorch CUDA version must be compatible with the CUDA toolkit. 
# If the Pytorch CUDA version is 11, the CUDA toolkit version should also be 11.
sudo sh cuda_12.3.0_545.23.06_linux.run

# Install CuDNN 
sudo dpkg -i cudnn-local-repo-ubuntu2204-8.9.7.29_1.0-1_amd64.deb

# Install libcudnn
apt-cache policy libcudnn8
sudo apt install libcudnn8=8.9.7.29-1+cuda12.2
sudo apt install libcudnn8-dev=8.9.7.29-1+cuda12.2
sudo apt install libcudnn8-samples=8.9.7.29-1+cuda12.2

# Install FreeImage
sudo apt-get install libfreeimage3 libfreeimage-dev

# Run demo
cp -r /usr/src/cudnn_samples_v8/ ~/Downloads/
cd ~/Downloads/cudnn_samples_v8/mnistCUDNN/
make clean && make
./mnistCUDNN
```

#### Step 2: Install packages in the python environment

```shell
# Clone the project repository
git clone https://github.com/Memoristor/Pytorch-FastDST-CUDA.git

# Pybind11 is used for Python and C++ interactions. 
# Activate your envrionment and install these packages.
pip install pytest pybind11 ninja

# Go to the project folder and install the library
cd Pytorch-FastDST-CUDA/
pip install .
```

#### Step 3: Run demos

```shell
CUDA_VISIBLE_DEVICES="0" python demo.py
```

## Example

View the function's description,
```python
import fadst

help(fadst.DCT2d)
```

Computing discrete transformations using the last two dimensions of high-dimensional tensors,

```python
import fadst
import torch

block = 4

x1 = torch.randn(1024, 1024).cuda()  # 2-D tensor
dct2d = fadst.DCT2d(x1, block)  # shape: [1024, 1024]
sorted2d = fadst.sort2d(dct2d, block)  # shape: [16, 256, 256]
recover2d = fadst.recover2d(sorted2d, block)  # shape: [1024, 1024]

x2 = torch.randn(3, 1024, 1024).cuda()  # 3-D tensor
dct2d = fadst.DCT2d(x2, block)  # shape: [3, 1024, 1024]
sorted2d = fadst.sort2d(dct2d, block)  # shape: [3, 16, 256, 256]
recover2d = fadst.recover2d(sorted2d, block)  # shape: [3, 1024, 1024]
```

Examples for double tensor and half-float tensor,
```python
import fadst
import torch

block = 4

x1 = torch.randn(1024, 1024).double().cuda()  # 2-D tensor
dct2d = fadst.DCT2d(x1, block)  # type: double

x2 = torch.randn(1024, 1024).half().cuda()  # 2-D tensor
dct2d = fadst.DCT2d(x1, block)  # type: half
```


## License

Pytorch-FastDST-CUDA has a GPL-3.0 license, as found in the [LICENSE](./LICENSE) file.
