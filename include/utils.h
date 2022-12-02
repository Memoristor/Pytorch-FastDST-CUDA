#ifndef __CUDA_UTILS_H
#define __CUDA_UTILS_H

#include <torch/extension.h>
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

#define CHECK_CUDA(x) TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

#if defined(__cplusplus)
}
#endif

#endif /* __CUDA_UTILS_H */