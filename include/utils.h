#ifndef __CUDA_UTILS_H
#define __CUDA_UTILS_H

#include <ATen/ATen.h>
#include <torch/torch.h>
#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#if defined(__cplusplus)
extern "C" {
#endif

#define CHECK(call)                                                         \
{                                                                           \
    const cudaError_t error = call;                                         \  
    if (error != cudaSuccess)                                               \
    {                                                                       \
        printf("Error: %s, %d, ", __FILE__, __LINE__);                      \
        printf("cuda: %d, reason: %s\n", error, cudaGetErrorString(error)); \
        exit(1);                                                            \
    }                                                                       \
} 

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_4DTENSOR(x) TORCH_CHECK(x.sizes().size() == 4, #x " must be a 4-D tensor")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x); CHECK_4DTENSOR(x);

#define SORT_FREQUENCY_FIRST 1
#define SORT_CHANNELS_FIRST 0

at::Tensor zeroPadInputTensorToFitPointSize(const at::Tensor input, const uint numPoints);
void optimalCUDABlocksAndThreadsPerBlock(const uint numTotalThreads, dim3 &numBlocks, dim3 &threadsPerBlock);
void calculateZigzag(uint *zigzag, uint numPoints);

#if defined(__cplusplus)
}
#endif

#endif /* __CUDA_UTILS_H */