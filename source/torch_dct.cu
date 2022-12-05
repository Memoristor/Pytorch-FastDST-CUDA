
#include "torch_dct.cuh"

#include <iostream>


__global__ void sumGPU(torch::Tensor a, torch::Tensor b, torch::Tensor &c) {
    int idx = threadIdx.x * blockIdx.x * blockDim.x;
    int idy = threadIdx.y * blockIdx.y * blockDim.y;


    // c[idx] = c[idx] + b[idx];
}


void test_sumGPU(torch::Tensor a, torch::Tensor b, torch::Tensor &c) {
    std::cout << "test_sumGPU" << std::endl;
}

// __global__ void nativeDctType2Size3(torch::Tensor input, uint32_t stride) {
//     CHECK_CUDA(input);


// }

// __global__ void fastDctType2Size3(torch::Tensor input, uint32_t stride) {
//     CHECK_CUDA(input);


// }