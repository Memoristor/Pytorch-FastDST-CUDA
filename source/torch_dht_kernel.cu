
#include <stdio.h>

#include "../include/utils.h"

template <typename scalar_t>
__global__ void cudaNaiveDHT2DKernel(const uint numTotalThreads, const uint batchSizeDim,
                                     const uint channelDim, const uint heightDim,
                                     const uint widthDim, const scalar_t* __restrict__ input,
                                     const uint numPoints, scalar_t* __restrict__ output) {
  const uint idx =
      threadIdx.x + blockIdx.x * blockDim.x +
      (threadIdx.y + blockIdx.y * blockDim.y) * gridDim.x * blockDim.x +
      (threadIdx.z + blockIdx.z * blockDim.z) * gridDim.x * gridDim.y * blockDim.x * blockDim.y;

  if (idx < numTotalThreads) {
    const uint chwDim = channelDim * heightDim * widthDim;
    const uint hwDim = heightDim * widthDim;
    const uint n = int(idx / chwDim);
    const uint c = int((idx % chwDim) / hwDim);
    const uint h = int((idx % hwDim) / widthDim);
    const uint w = idx % widthDim;
    const uint numPointsPow2 = numPoints * numPoints;

    for (uint k = 0; k < numPoints; k++) {
      uint hk = h * numPoints + k;

      for (uint v = 0; v < numPoints; v++) {
        uint wv = w * numPoints + v;

        uint spectralIdx =
            n * chwDim * numPointsPow2 + c * hwDim * numPointsPow2 + hk * widthDim * numPoints + wv;

        for (uint i = 0; i < numPoints; i++) {
          uint hi = h * numPoints + i;
          float sin_cos_i_k =
              cosf(2.0f * M_PI * i * k / numPoints) + sinf(2.0f * M_PI * i * k / numPoints);

          for (uint j = 0; j < numPoints; j++) {
            uint wj = w * numPoints + j;
            float sin_cos_j_v =
                cosf(2.0f * M_PI * j * v / numPoints) + sinf(2.0f * M_PI * j * v / numPoints);

            uint specialIdx = n * chwDim * numPointsPow2 + c * hwDim * numPointsPow2 +
                              hi * widthDim * numPoints + wj;

            output[spectralIdx] +=
                input[specialIdx] * (1.0f / numPoints) * sin_cos_i_k * sin_cos_j_v;
          }
        }
      }
    }
  }

  __syncthreads();
}

template <typename scalar_t>
__global__ void cudaNaiveIDHT2DKernel(const uint numTotalThreads, const uint batchSizeDim,
                                      const uint channelDim, const uint heightDim,
                                      const uint widthDim, const scalar_t* __restrict__ input,
                                      const uint numPoints, scalar_t* __restrict__ output) {
  const uint idx =
      threadIdx.x + blockIdx.x * blockDim.x +
      (threadIdx.y + blockIdx.y * blockDim.y) * gridDim.x * blockDim.x +
      (threadIdx.z + blockIdx.z * blockDim.z) * gridDim.x * gridDim.y * blockDim.x * blockDim.y;

  if (idx < numTotalThreads) {
    const uint chwDim = channelDim * heightDim * widthDim;
    const uint hwDim = heightDim * widthDim;
    const uint n = int(idx / chwDim);
    const uint c = int((idx % chwDim) / hwDim);
    const uint h = int((idx % hwDim) / widthDim);
    const uint w = idx % widthDim;
    const uint numPointsPow2 = numPoints * numPoints;

    for (uint i = 0; i < numPoints; i++) {
      uint hi = h * numPoints + i;

      for (uint j = 0; j < numPoints; j++) {
        uint wj = w * numPoints + j;

        uint specialIdx =
            n * chwDim * numPointsPow2 + c * hwDim * numPointsPow2 + hi * widthDim * numPoints + wj;

        for (uint k = 0; k < numPoints; k++) {
          uint hk = h * numPoints + k;
          float sin_cos_i_k =
              cosf(2.0f * M_PI * i * k / numPoints) + sinf(2.0f * M_PI * i * k / numPoints);

          for (uint v = 0; v < numPoints; v++) {
            uint wv = w * numPoints + v;
            float sin_cos_j_v =
                cosf(2.0f * M_PI * j * v / numPoints) + sinf(2.0f * M_PI * j * v / numPoints);

            uint spectralIdx = n * chwDim * numPointsPow2 + c * hwDim * numPointsPow2 +
                               hk * widthDim * numPoints + wv;

            output[specialIdx] +=
                input[spectralIdx] * (1.0f / numPoints) * sin_cos_i_k * sin_cos_j_v;
          }
        }
      }
    }
  }

  __syncthreads();
}

at::Tensor cudaNaiveDHT2D(const at::Tensor input, const uint numPoints) {
  at::IntList inputSize = input.sizes();
  int batchSize = inputSize[0];
  int channel = inputSize[1];
  int height = inputSize[2];
  int width = inputSize[3];

  at::Tensor output = at::zeros_like(input);

  dim3 numBlocks;
  dim3 threadsPerBlock;

  uint numTotalThreads = batchSize * channel * height * width / (numPoints * numPoints);
  optimalCUDABlocksAndThreadsPerBlock(numTotalThreads, numBlocks, threadsPerBlock);

  AT_DISPATCH_FLOATING_TYPES(input.type(), "cudaNaiveDHT2D", ([&] {
                               cudaNaiveDHT2DKernel<scalar_t><<<numBlocks, threadsPerBlock>>>(
                                   numTotalThreads, batchSize, channel, height / numPoints,
                                   width / numPoints, input.data_ptr<scalar_t>(), numPoints,
                                   output.data_ptr<scalar_t>());
                             }));

  return output;
}

at::Tensor cudaNaiveIDHT2D(const at::Tensor input, const uint numPoints) {
  at::IntList inputSize = input.sizes();
  int batchSize = inputSize[0];
  int channel = inputSize[1];
  int height = inputSize[2];
  int width = inputSize[3];

  at::Tensor output = at::zeros_like(input);

  dim3 numBlocks;
  dim3 threadsPerBlock;

  uint numTotalThreads = batchSize * channel * height * width / (numPoints * numPoints);
  optimalCUDABlocksAndThreadsPerBlock(numTotalThreads, numBlocks, threadsPerBlock);

  AT_DISPATCH_FLOATING_TYPES(input.type(), "cudaNaiveIDHT2D", ([&] {
                               cudaNaiveIDHT2DKernel<scalar_t><<<numBlocks, threadsPerBlock>>>(
                                   numTotalThreads, batchSize, channel, height / numPoints,
                                   width / numPoints, input.data_ptr<scalar_t>(), numPoints,
                                   output.data_ptr<scalar_t>());
                             }));

  return output;
}
