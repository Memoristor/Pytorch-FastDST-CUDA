#include <iostream>

#include <torch/torch.h>
#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include "include/utils.h"

template <typename scalar_t>
__device__ __forceinline__ torch::Tensor dctType2Size3Kernel(torch::Tensor input) {
    
    // int idx = threadIdx.x * 

}