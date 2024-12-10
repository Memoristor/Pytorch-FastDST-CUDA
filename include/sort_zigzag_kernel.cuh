#ifndef __SORT_ZIGZAG_CUH
#define __SORT_ZIGZAG_CUH

#include "device_kernel.cuh"

template <typename scalar_t>
__global__ void cudaSortCoefficientsByZigzagKernel(const uint totalThreads,
                                                   const scalar_t* __restrict__ input,
                                                   scalar_t* __restrict__ output,
                                                   const uint inputHeight, const uint inputWidth,
                                                   const uint points) {
  const uint idx =
      threadIdx.x + blockIdx.x * blockDim.x +
      (threadIdx.y + blockIdx.y * blockDim.y) * gridDim.x * blockDim.x +
      (threadIdx.z + blockIdx.z * blockDim.z) * gridDim.x * gridDim.y * blockDim.x * blockDim.y;

  extern __shared__ uint zigzag[];
  bool isFirstThread = (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0);
  if (isFirstThread) initZigzag(zigzag, points);
  __syncthreads();

  if (idx < totalThreads) {
    const uint HW = inputHeight * inputWidth;
    const uint n = int(idx / HW);
    const uint h = int((idx % HW) / inputWidth);
    const uint w = idx % inputWidth;
    const uint pref = n * HW;

    const uint bH = int(inputHeight / points);
    const uint bW = int(inputWidth / points);
    const uint bHW = bH * bW;

    const uint bh = int(h / points);
    const uint bw = int(w / points);
    const uint k = h % points;
    const uint v = w % points;

    const uint outputIdx = pref + zigzag[k * points + v] * bHW + bh * bW + bw;
    output[outputIdx] = input[idx];
  }

  __syncthreads();
}

template <typename scalar_t>
__global__ void cudaRecoverCoefficientsByZigzagKernel(const uint totalThreads,
                                                      const scalar_t* __restrict__ input,
                                                      scalar_t* __restrict__ output,
                                                      const uint outputHeight,
                                                      const uint outputWidth, const uint points) {
  const uint idx =
      threadIdx.x + blockIdx.x * blockDim.x +
      (threadIdx.y + blockIdx.y * blockDim.y) * gridDim.x * blockDim.x +
      (threadIdx.z + blockIdx.z * blockDim.z) * gridDim.x * gridDim.y * blockDim.x * blockDim.y;

  extern __shared__ uint zigzag[];
  bool isFirstThread = (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0);
  if (isFirstThread) initZigzag(zigzag, points);
  __syncthreads();

  if (idx < totalThreads) {
    const uint HW = outputHeight * outputWidth;
    const uint n = int(idx / HW);
    const uint h = int((idx % HW) / outputWidth);
    const uint w = idx % outputWidth;
    const uint pref = n * HW;

    const uint bH = int(outputHeight / points);
    const uint bW = int(outputWidth / points);
    const uint bHW = bH * bW;

    const uint bh = int(h / points);
    const uint bw = int(w / points);
    const uint k = h % points;
    const uint v = w % points;

    uint inputIdx = pref + zigzag[k * points + v] * bHW + bh * bW + bw;
    output[idx] = input[inputIdx];
  }

  __syncthreads();
}

#endif /* __SORT_ZIGZAG_CUH */