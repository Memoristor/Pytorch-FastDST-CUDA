# Pytorch-FastDST-CUDA

> This is a Fast Discrete Signal Transform (FastDST) library for Pytorch implemented by CUDA.

## Introduction

* The running environment is not strict, as long as you can run `test_demo.py` successfully.

## Environment

* Ubuntu 18.04.5 LTS (8G RAM, Not strict)
* RTX2070s (CUDA 11.0, cuDNN v8.6.0 for CUDA 11.x, Required)
* Libtorch 1.13.0 (Release version, for CUDA 11.6, Optional)
* Pytorch 1.7.0 (Not strict)
* Python 3.7.9 (Not strict)

## Installation

```shell
# Clone the project repository
git clone https://github.com/Memoristor/Pytorch-FastDST-CUDA.git

# Pybind11 is used for Python and C++ interactions. Activate your envrionment and install these packages:
conda activate <your_conda_environment>
conda install pytest pybind11

# Enter the project folder and install the library:
cd Pytorch-FastDST-CUDA/
python setup.py install

# Run a test:
python test_demo.py
```

## Usage

## Citation

## License

Pytorch-FastDST-CUDA has a GPL-3.0 license, as found in the [LICENSE](./LICENSE) file.
