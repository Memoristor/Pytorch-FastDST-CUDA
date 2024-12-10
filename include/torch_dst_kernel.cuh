#ifndef __TORCH_DST_KERNEL_CUH
#define __TORCH_DST_KERNEL_CUH

#include "device_kernel.cuh"

template <typename scalar_t>
__global__ void cudaNaiveDST2DAndSortCoefficientsByZigzagKernel(
    const uint totalThreads, const scalar_t* __restrict__ input, scalar_t* __restrict__ output,
    const uint inputHeight, const uint inputWidth, const uint points, const bool sortbyZigzag) {
  const uint idx =
      threadIdx.x + blockIdx.x * blockDim.x +
      (threadIdx.y + blockIdx.y * blockDim.y) * gridDim.x * blockDim.x +
      (threadIdx.z + blockIdx.z * blockDim.z) * gridDim.x * gridDim.y * blockDim.x * blockDim.y;

  extern __shared__ uint zigzag[];
  bool isFirstThread = (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0);
  if (isFirstThread && sortbyZigzag) initZigzag(zigzag, points);
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
    const uint bhp = bh * points;
    const uint bwp = bw * points;

    const scalar_t var = M_PI / (points + 1);
    const scalar_t var_k = var * (k + 1.0f);
    const scalar_t var_v = var * (v + 1.0f);
    const scalar_t var_p = (2.0f / (points + 1));

    const uint spectralIdx =
        (sortbyZigzag) ? pref + zigzag[k * points + v] * bHW + bh * bW + bw : idx;

    for (uint i = 0; i < points; i++) {
      uint hi = bhp + i;
      scalar_t sin_i_k = sinf((i + 1.0f) * var_k);

      for (uint j = 0; j < points; j++) {
        uint wj = bwp + j;
        scalar_t sin_j_v = sinf((j + 1.0f) * var_v);

        uint specialIdx = pref + hi * inputWidth + wj;
        output[spectralIdx] += input[specialIdx] * var_p * sin_i_k * sin_j_v;
      }
    }
  }

  __syncthreads();
}

template <typename scalar_t>
__global__ void cudaNaiveIDST2DAndRecoverCoefficientsByZigzagKernel(
    const uint totalThreads, const scalar_t* __restrict__ input, scalar_t* __restrict__ output,
    const uint outputHeight, const uint outputWidth, const uint points,
    const bool recoverbyZigzag) {
  const uint idx =
      threadIdx.x + blockIdx.x * blockDim.x +
      (threadIdx.y + blockIdx.y * blockDim.y) * gridDim.x * blockDim.x +
      (threadIdx.z + blockIdx.z * blockDim.z) * gridDim.x * gridDim.y * blockDim.x * blockDim.y;

  extern __shared__ uint zigzag[];
  bool isFirstThread = (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0);
  if (isFirstThread && recoverbyZigzag) initZigzag(zigzag, points);
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
    const uint i = h % points;
    const uint j = w % points;
    const uint bhp = bh * points;
    const uint bwp = bw * points;

    const scalar_t var = M_PI / (points + 1);
    const scalar_t var_p = (2.0f / (points + 1));

    for (uint k = 0; k < points; k++) {
      uint hk = bhp + k;
      scalar_t var_k = var * (k + 1.0f);
      scalar_t sin_i_k = sinf((i + 1.0f) * var_k);

      for (uint v = 0; v < points; v++) {
        uint wv = bwp + v;
        scalar_t var_v = var * (v + 1.0f);
        scalar_t sin_j_v = sinf((j + 1.0f) * var_v);

        uint spectralIdx = (recoverbyZigzag) ? pref + zigzag[k * points + v] * bHW + bh * bW + bw
                                             : pref + hk * outputWidth + wv;

        output[idx] += input[spectralIdx] * var_p * sin_i_k * sin_j_v;
      }
    }
  }

  __syncthreads();
}

#endif /* __TORCH_DST_KERNEL_CUH */