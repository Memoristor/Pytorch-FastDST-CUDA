#ifndef __TORCH_DHT_KERNEL_CUH
#define __TORCH_DHT_KERNEL_CUH

#include "utils.h"

template <typename scalar_t>
__global__ void cudaNaiveDHT2DKernel(const uint totalThreads, const scalar_t* __restrict__ input,
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
          scalar_t sin_cos_i_k =
              cosf(2.0f * M_PI * i * k / points) + sinf(2.0f * M_PI * i * k / points);

          for (uint j = 0; j < points; j++) {
            uint wj = w * points + j;
            scalar_t sin_cos_j_v =
                cosf(2.0f * M_PI * j * v / points) + sinf(2.0f * M_PI * j * v / points);

            uint specialIdx = n * hwDim * p2 + hi * wDim * points + wj;

            output[spectralIdx] += input[specialIdx] * (1.0f / points) * sin_cos_i_k * sin_cos_j_v;
          }
        }
      }
    }
  }

  __syncthreads();
}

template <typename scalar_t>
__global__ void cudaNaiveIDHT2DKernel(const uint totalThreads, const scalar_t* __restrict__ input,
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
          scalar_t sin_cos_i_k =
              cosf(2.0f * M_PI * i * k / points) + sinf(2.0f * M_PI * i * k / points);

          for (uint v = 0; v < points; v++) {
            uint wv = w * points + v;
            scalar_t sin_cos_j_v =
                cosf(2.0f * M_PI * j * v / points) + sinf(2.0f * M_PI * j * v / points);

            uint spectralIdx = n * hwDim * p2 + hk * wDim * points + wv;

            output[specialIdx] += input[spectralIdx] * (1.0f / points) * sin_cos_i_k * sin_cos_j_v;
          }
        }
      }
    }
  }

  __syncthreads();
}

#endif /* __TORCH_DHT_KERNEL_CUH */