#ifndef __SORT_ZIGZAG_CUH
#define __SORT_ZIGZAG_CUH

#include "utils.h"

template <typename scalar_t>
__global__ void cudaSortCoefficientsByZigzagKernel(const uint totalThreads,
                                                   const scalar_t* __restrict__ input,
                                                   const uint points, scalar_t* __restrict__ output,
                                                   const uint hDim, const uint wDim,
                                                   const uint* zigzag) {
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

        uint inputIndex = n * hwDim * p2 + hk * wDim * points + wv;
        uint outputIndex = n * hwDim * p2 + zigzag[k * points + v] * hwDim + h * wDim + w;
        output[outputIndex] = input[inputIndex];
      }
    }
  }

  __syncthreads();
}

template <typename scalar_t>
__global__ void cudaRecoverCoefficientsByZigzagKernel(const uint totalThreads,
                                                      const scalar_t* __restrict__ input,
                                                      const uint hDim, const uint wDim,
                                                      scalar_t* __restrict__ output,
                                                      const uint points, const uint* zigzag) {
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

        uint outputIndex = n * hwDim * p2 + hk * wDim * points + wv;
        uint inputIndex = n * hwDim * p2 + zigzag[k * points + v] * hwDim + h * wDim + w;
        output[outputIndex] = input[inputIndex];
      }
    }
  }

  __syncthreads();
}

#endif /* __SORT_ZIGZAG_CUH */