#ifndef __CUDA_UTILS_H
#define __CUDA_UTILS_H

#include <ATen/ATen.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <torch/torch.h>

#include <iostream>

#if defined(__cplusplus)
extern "C" {
#endif

#define CHECK_CUDA_ERROR(call)                                            \
  {                                                                       \
    const cudaError_t error = call;                                       \
    \  
    if (error != cudaSuccess) {                                           \
      printf("Error: %s, %d, ", __FILE__, __LINE__);                      \
      printf("cuda: %d, reason: %s\n", error, cudaGetErrorString(error)); \
      exit(1);                                                            \
    }                                                                     \
  }

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_TENSORDIM(x, y) TORCH_CHECK(x.sizes().size() >= y, #x "'s dimension must be >= " #y)
#define CHECK_INPUT(x) \
  CHECK_CUDA(x);       \
  CHECK_CONTIGUOUS(x);

void calculateZigzag(uint *zigzag, uint points);
at::Tensor zeroPadInputTensorToFitPointSize(const at::Tensor input, const uint points);
void optimalCUDABlocksAndThreadsPerBlock(const uint numTotalThreads, dim3 &numBlocks,
                                         dim3 &threadsPerBlock);

#if defined(__cplusplus)
}
#endif

#endif /* __CUDA_UTILS_H */