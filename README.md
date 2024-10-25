# Pytorch-FastDST-CUDA

> This is a Fast Discrete Signal Transform (FastDST) library for Pytorch implemented by CUDA.

## Introduction

* This repository includes fast transforms using CUDA, such as DCT, DST, and DHT.

* Directly processing of tensors of the torch is allowed.

* Distributed data-parallel mode is supported.

## Development Environment

* Ubuntu 22.04.5 LTS, 8G RAM

* RTX2070s CUDA 12.3, cuDNN v8.9.7 for CUDA 12.x

* Pytorch 2.3.1+cu121

* Python 3.9

* Libtorch 1.13.0, Release version, for CUDA 11.6 (Optional)

## Installation

### Step 1 : Install [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) and [CuDNN (Optional)](https://developer.nvidia.com/cudnn-downloads)

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

### Step 2: Install packages in the python environment

```shell
# Clone the project repository
git clone https://github.com/Memoristor/Pytorch-FastDST-CUDA.git

# Pybind11 is used for Python and C++ interactions. 
# Activate your envrionment and install these packages.
conda activate <your_conda_environment>
conda install pytest pybind11 ninja

# Go to the project folder and install the library
cd Pytorch-FastDST-CUDA/
python setup.py install
```

### Step 3: Run demos

```shell
CUDA_VISIBLE_DEVICES="0" python demo.py
```

## License

Pytorch-FastDST-CUDA has a GPL-3.0 license, as found in the [LICENSE](./LICENSE) file.
