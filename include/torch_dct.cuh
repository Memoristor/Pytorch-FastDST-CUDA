#ifndef __TORCH_DCT_H
#define __TORCH_DCT_H

#include <torch/torch.h>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#if defined(__cplusplus)
extern "C" {
#endif

void test_sumGPU(torch::Tensor a, torch::Tensor b, torch::Tensor &c);

#if defined(__cplusplus)
}
#endif

#endif /* __TORCH_DCT_H */