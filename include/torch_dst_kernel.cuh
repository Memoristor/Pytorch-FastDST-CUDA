#ifndef __TORCH_DST_KERNEL_CUH
#define __TORCH_DST_KERNEL_CUH

#include "utils.h"

template <typename scalar_t>
__global__ void cudaNaiveDST2DKernel(const uint totalThreads, const scalar_t* __restrict__ input,
                                     const uint points, scalar_t* __restrict__ output,
                                     const uint hDim, const uint wDim) {
  const uint idx =
      threadIdx.x + blockIdx.x * blockDim.x +
      (threadIdx.y + blockIdx.y * blockDim.y) * gridDim.x * blockDim.x +
      (threadIdx.z + blockIdx.z * blockDim.z) * gridDim.x * gridDim.y * blockDim.x * blockDim.y;

  if (idx < totalThreads) {
    const uint hwDim = hDim * wDim;
    const uint n = int(idx / hwDim);
    const uint h = int((idx % hwDim) / wDim);
    const uint w = idx % wDim;
    const uint p2 = points * points;

    for (uint k = 0; k < points; k++) {
      uint hk = h * points + k;

      for (uint v = 0; v < points; v++) {
        uint wv = w * points + v;

        uint spectralIdx = n * hwDim * p2 + hk * wDim * points + wv;

        for (uint i = 0; i < points; i++) {
          uint hi = h * points + i;
          scalar_t sin_i_k = sinf((i + 1.0f) * (k + 1.0f) * M_PI / (points + 1));

          for (uint j = 0; j < points; j++) {
            uint wj = w * points + j;
            scalar_t sin_j_v = sinf((j + 1.0f) * (v + 1.0f) * M_PI / (points + 1));

            uint specialIdx = n * hwDim * p2 + hi * wDim * points + wj;

            output[spectralIdx] += input[specialIdx] * (2.0f / (points + 1)) * sin_i_k * sin_j_v;
          }
        }
      }
    }
  }

  __syncthreads();
}

template <typename scalar_t>
__global__ void cudaNaiveIDST2DKernel(const uint totalThreads, const scalar_t* __restrict__ input,
                                      const uint points, scalar_t* __restrict__ output,
                                      const uint hDim, const uint wDim) {
  const uint idx =
      threadIdx.x + blockIdx.x * blockDim.x +
      (threadIdx.y + blockIdx.y * blockDim.y) * gridDim.x * blockDim.x +
      (threadIdx.z + blockIdx.z * blockDim.z) * gridDim.x * gridDim.y * blockDim.x * blockDim.y;

  if (idx < totalThreads) {
    const uint hwDim = hDim * wDim;
    const uint n = int(idx / hwDim);
    const uint h = int((idx % hwDim) / wDim);
    const uint w = idx % wDim;
    const uint p2 = points * points;

    for (uint i = 0; i < points; i++) {
      uint hi = h * points + i;

      for (uint j = 0; j < points; j++) {
        uint wj = w * points + j;

        uint specialIdx = n * hwDim * p2 + hi * wDim * points + wj;

        for (uint k = 0; k < points; k++) {
          uint hk = h * points + k;
          scalar_t sin_i_k = sinf((i + 1.0f) * (k + 1.0f) * M_PI / (points + 1));

          for (uint v = 0; v < points; v++) {
            uint wv = w * points + v;
            scalar_t sin_j_v = sinf((j + 1.0f) * (v + 1.0f) * M_PI / (points + 1));

            uint spectralIdx = n * hwDim * p2 + hk * wDim * points + wv;

            output[specialIdx] += input[spectralIdx] * (2.0f / (points + 1)) * sin_i_k * sin_j_v;
          }
        }
      }
    }
  }

  __syncthreads();
}

#endif /* __TORCH_DST_KERNEL_CUH */