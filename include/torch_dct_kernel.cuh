#ifndef __TORCH_DCT_KERNEL_CUH
#define __TORCH_DCT_KERNEL_CUH

#include "device_kernel.cuh"

template <typename scalar_t>
__global__ void cudaNaiveDCT2DAndSortCoefficientsByZigzagKernel(
    const uint totalThreads, const scalar_t* __restrict__ input, const uint points,
    scalar_t* __restrict__ output, const uint hDim, const uint wDim, const bool sortbyZigzag) {
  const uint idx =
      threadIdx.x + blockIdx.x * blockDim.x +
      (threadIdx.y + blockIdx.y * blockDim.y) * gridDim.x * blockDim.x +
      (threadIdx.z + blockIdx.z * blockDim.z) * gridDim.x * gridDim.y * blockDim.x * blockDim.y;

  extern __shared__ uint zigzag[];
  bool isFirstThread = (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0);
  if (isFirstThread && sortbyZigzag) initZigzag(zigzag, points);
  __syncthreads();

  if (idx < totalThreads) {
    const uint hwDim = hDim * wDim;
    const uint n = int(idx / hwDim);
    const uint h = int((idx % hwDim) / wDim);
    const uint w = idx % wDim;
    const uint p2 = points * points;

    const scalar_t sqrt_1_2 = sqrtf(1.0f / 2);
    for (uint k = 0; k < points; k++) {
      uint hk = h * points + k;
      scalar_t lambda_k = k == 0 ? sqrt_1_2 : (scalar_t)1.0f;

      for (uint v = 0; v < points; v++) {
        uint wv = w * points + v;
        scalar_t lambda_v = v == 0 ? sqrt_1_2 : (scalar_t)1.0f;

        uint spectralIdx = (sortbyZigzag)
                               ? n * hwDim * p2 + zigzag[k * points + v] * hwDim + h * wDim + w
                               : n * hwDim * p2 + hk * wDim * points + wv;

        for (uint i = 0; i < points; i++) {
          uint hi = h * points + i;
          scalar_t cos_i_k = cosf((2.0f * i + 1.0f) * k * M_PI / (2.0f * points));

          for (uint j = 0; j < points; j++) {
            uint wj = w * points + j;
            scalar_t cos_j_v = cosf((2.0f * j + 1.0f) * v * M_PI / (2.0f * points));

            uint specialIdx = n * hwDim * p2 + hi * wDim * points + wj;

            output[spectralIdx] +=
                input[specialIdx] * (2.0f / points) * lambda_k * lambda_v * cos_i_k * cos_j_v;
          }
        }
      }
    }
  }

  __syncthreads();
}

template <typename scalar_t>
__global__ void cudaNaiveIDCT2DAndRecoverCoefficientsByZigzagKernel(
    const uint totalThreads, const scalar_t* __restrict__ input, const uint points,
    scalar_t* __restrict__ output, const uint hDim, const uint wDim, const bool recoverbyZigzag) {
  const uint idx =
      threadIdx.x + blockIdx.x * blockDim.x +
      (threadIdx.y + blockIdx.y * blockDim.y) * gridDim.x * blockDim.x +
      (threadIdx.z + blockIdx.z * blockDim.z) * gridDim.x * gridDim.y * blockDim.x * blockDim.y;

  extern __shared__ uint zigzag[];
  bool isFirstThread = (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0);
  if (isFirstThread && recoverbyZigzag) initZigzag(zigzag, points);
  __syncthreads();

  if (idx < totalThreads) {
    const uint hwDim = hDim * wDim;
    const uint n = int(idx / hwDim);
    const uint h = int((idx % hwDim) / wDim);
    const uint w = idx % wDim;
    const uint p2 = points * points;

    const scalar_t sqrt_1_2 = sqrtf(1.0f / 2);
    for (uint i = 0; i < points; i++) {
      uint hi = h * points + i;

      for (uint j = 0; j < points; j++) {
        uint wj = w * points + j;

        uint specialIdx = n * hwDim * p2 + hi * wDim * points + wj;

        for (uint k = 0; k < points; k++) {
          uint hk = h * points + k;
          scalar_t lambda_k = k == 0 ? sqrt_1_2 : (scalar_t)1.0f;
          scalar_t cos_i_k = cosf((2.0f * i + 1.0f) * k * M_PI / (2.0f * points));

          for (uint v = 0; v < points; v++) {
            uint wv = w * points + v;
            scalar_t lambda_v = v == 0 ? sqrt_1_2 : (scalar_t)1.0f;
            scalar_t cos_j_v = cosf((2.0f * j + 1.0f) * v * M_PI / (2.0f * points));

            uint spectralIdx = (recoverbyZigzag)
                                   ? n * hwDim * p2 + zigzag[k * points + v] * hwDim + h * wDim + w
                                   : n * hwDim * p2 + hk * wDim * points + wv;

            output[specialIdx] +=
                input[spectralIdx] * (2.0f / points) * lambda_k * lambda_v * cos_i_k * cos_j_v;
          }
        }
      }
    }
  }

  __syncthreads();
}

#endif /* __TORCH_DCT_KERNEL_CUH */